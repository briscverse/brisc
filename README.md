<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://brisc.run/_static/images/runner_title_wide_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://brisc.run/_static/images/runner_title_wide_light.svg">
    <img alt="brisc" src="https://brisc.run/_static/images/runner_title_wide_light.svg" width="500">
  </picture>
</p>

brisc is a high-performance library for analyzing single-cell data at scale. It prioritizes running as fast as possible on multi-core CPU systems, strict reproducibility, and a clean, user-friendly interface. On datasets of 1 to 20 million cells, it cuts the runtime of common workflows from hours to minutes.

Full documentation is available at **[brisc.run](https://brisc.run)**.

## Why brisc?

- **Blazing fast** — ground-up optimization of core algorithms and effective parallelism.
- **Deterministic** — every step gives floating-point identical results between runs, regardless of the number of threads used.
- **Complete toolkit** — preprocessing, dimensionality reduction, harmonization, label transfer, clustering, embedding, pseudobulk differential expression, and plotting.
- **Interoperable** — reads and writes `.h5ad`, `.rds`, `.h5Seurat`, and 10x files, and supports interleaving Python and R analyses via [ryp](https://github.com/Wainberg/ryp) without intermediate writes to disk.
- **Memory-efficient** — ~2× lower peak memory than Scanpy by tabulating which cells pass QC, instead of subsetting to them.
- **User-friendly** — sensible defaults, strict type-checking, and solution-focused error messages.

## Installation

brisc supports Linux, macOS, and Windows on Python 3.9+.

**conda (recommended)**

```bash
conda install -c conda-forge brisc
```

**pip**

```bash
pip install brisc
```

conda is recommended because it sets up the fast MKL BLAS and some of the R packages brisc uses. With pip, you'll need to handle those yourself: see the [installation guide](https://brisc.run/installation.html) for details, including optional R integration (for differential expression, Seurat, and SingleCellExperiment support) via [ryp](https://github.com/briscverse/ryp).

## Quick start

### Basic workflow

```python
from brisc import SingleCell

sc = SingleCell('data.h5ad')\
  .qc()\
  .hvg(batch_column='donor')\
  .normalize()\
  .pca()\
  .neighbors()\
  .shared_neighbors()\
  .cluster(resolution=[0.25, 0.5, 1, 1.5, 2])\
  .localmap()
```

### Label transfer

```python
from brisc import SingleCell

sc_ref = SingleCell('data_ref.h5ad').qc()
sc_query = SingleCell('data_query.h5ad').qc()
sc_ref, sc_query = sc_ref.hvg(sc_query)
sc_ref = sc_ref.normalize()
sc_query = sc_query.normalize()
sc_ref, sc_query = sc_ref.pca(sc_query)
sc_ref, sc_query = sc_ref.harmonize(sc_query)
sc_query = sc_query.label_transfer_from(
  sc_ref, 'cell_type')
```

### Pseudobulk differential expression

```python
from brisc import SingleCell

pb = SingleCell('data.h5ad')\
  .qc()\
  .pseudobulk('sample', 'cell_type')
de = pb\
  .qc('condition')\
  .library_size()\
  .de('~ condition + sex + pmi')
```
