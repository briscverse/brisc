"""
Build the tutorial subsample and publish it to the HF dataset the tutorials
link to. Kept for provenance -- the tutorials just download the finished file.

PBS vs IFN-gamma, balanced 50/50, subsampled to 100,000 cells x full genes:
24 pseudobulk samples (2 x 12 donors) instead of 1,092, and enough cells to
follow the full workflow (QC, feature selection, PCA, clustering, embedding,
pseudobulk DE, label transfer) on a laptop.

Run where the 10M source (~227 GB) lives; skips the download if it's present.
Needs huggingface_hub and a write token (HF_TOKEN or `huggingface-cli login`).
"""
import subprocess
from pathlib import Path
import polars as pl
from brisc import SingleCell
from huggingface_hub import HfApi

REPO_ID = 'dashingcell/Parse_100K_PBMC_cytokines'
N_CELLS = 100_000
CONDITIONS = ['PBS', 'IFN-gamma']
OBS_COLUMNS = ['sample', 'donor', 'cell_type', 'treatment', 'cytokine']

data_dir = Path.home() / 'single-cell' / 'Parse'
src = data_dir / 'Parse_10M_PBMC_cytokines.h5ad'
out = data_dir / f'Parse_{N_CELLS // 1000}K_PBMC_cytokines.h5ad'

CARD = """\
---
license: other
license_name: parse-biosciences-data-terms
license_link: https://www.parsebiosciences.com/datasets/
pretty_name: Parse 100K PBMC Cytokines (IFN-γ vs PBS)
tags:
- single-cell
- scRNA-seq
- PBMC
- immunology
- cytokines
- brisc
size_categories:
- 10K<n<100K
---

# Parse 100K PBMC Cytokines (IFN-γ vs PBS)

Balanced 100,000-cell subsample of the ~10 million cell cytokine stimulation
dataset from [Parse Biosciences](https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/).

In the original experiment, peripheral blood mononuclear cells (PBMCs) from
twelve healthy donors were treated with either one of 90 different cytokines or
a phosphate-buffered saline (PBS) control for 24 hours, yielding
(90 + 1) × 12 = 1,092 experimental conditions.

For the brisc tutorials, the data is restricted to the interferon-gamma (IFN-γ)
versus PBS contrast and subsampled to 50,000 cells per condition, stratified by
cell type. The result is 100,000 cells × 40,352 genes of raw counts across 24
samples (12 donors × 2 conditions) and 18 annotated cell types, with `obs`
columns `sample`, `donor`, `cell_type`, `treatment`, and `cytokine`.

## Source

- https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/
- https://www.biorxiv.org/content/10.64898/2025.12.12.693897v1

All credit to Parse Biosciences.
"""

def remove_unused_categories(s: pl.Series) -> pl.Expr:
    """polars-Enum version of pandas' remove_unused_categories: rebuild each
    Enum from the values present."""
    present = set(s.unique().to_list())
    kept = [c for c in s.dtype.categories.to_list() if c in present]
    return pl.col(s.name).cast(pl.Enum(kept))

# Fetch the 10M source once
data_dir.mkdir(parents=True, exist_ok=True)
if not src.exists():
    url = 'https://parse-wget.s3.us-west-2.amazonaws.com/10m/' + src.name
    subprocess.run(['wget', '-O', str(src), url], check=True)

# Build the balanced 100K subsample
sc = SingleCell(src, obs_columns=OBS_COLUMNS)\
    .filter_obs(pl.col('cytokine').is_in(CONDITIONS))
half = N_CELLS // 2
pbs = sc.filter_obs(pl.col('cytokine') == 'PBS')\
        .subsample_obs(n=half, by_column='cell_type', QC_column=None, seed=0)
ifn = sc.filter_obs(pl.col('cytokine') == 'IFN-gamma')\
        .subsample_obs(n=half, by_column='cell_type', QC_column=None, seed=0)
fixture = pbs.concat_obs(ifn)

# Drop unused Enum categories (sample 1,092 -> 24, cytokine 91 -> 2)
enum_cols = [name for name, dt in fixture.obs.schema.items() if dt == pl.Enum]
fixture = fixture.with_columns_obs(
    [remove_unused_categories(fixture.obs[c]) for c in enum_cols])
fixture.save(out, overwrite=True)

# Publish data + card to the HF dataset repo (no-op if unchanged)
api = HfApi()
api.create_repo(REPO_ID, repo_type='dataset', exist_ok=True)
api.upload_file(path_or_fileobj=str(out), path_in_repo=out.name,
                repo_id=REPO_ID, repo_type='dataset')
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo='README.md',
                repo_id=REPO_ID, repo_type='dataset')
print(f'Published to https://huggingface.co/datasets/{REPO_ID}')
