# Interoperability

This tutorial covers interoperability with the wider single-cell ecosystem: reading and writing the major on-disk formats — scverse/Scanpy `.h5ad`, Seurat `.rds`/`.h5Seurat`, Bioconductor SingleCellExperiment `.rds`, and raw 10x Genomics data — and converting between brisc, AnnData, Seurat, and SingleCellExperiment objects in memory, so you can interleave Python and R tools in a single analysis without writing intermediate files to disk.

:::{note}
Reading, writing, and converting Seurat and SingleCellExperiment data goes through R via [ryp](https://github.com/Wainberg/ryp), which needs the **Seurat** and **SingleCellExperiment** R packages. Install them before running those sections — see [Installation → R packages](../installation.md#r-packages).
:::

## Loading from files

brisc can interface with files and objects from all three major single-cell ecosystems:

- **scverse/Scanpy** — AnnData `.h5ad` files and in-memory AnnData objects
- **Seurat** — `.rds` and `.h5Seurat` files, plus in-memory Seurat objects via the ryp Python-R bridge
- **Bioconductor** — SingleCellExperiment `.rds` files and in-memory SCE objects via ryp

as well as raw 10x Genomics data (`.h5` or `.mtx`/`.mtx.gz`).

The SingleCell constructor auto-detects a file's format from its extension:

```python
from brisc import SingleCell

# scverse / Scanpy
sc = SingleCell('data.h5ad')

# Seurat (requires ryp)
sc = SingleCell('seurat_obj.rds')
sc = SingleCell('seurat_obj.h5Seurat')

# Bioconductor SingleCellExperiment (requires ryp)
sc = SingleCell('sce_obj.rds')

# 10x Genomics
sc = SingleCell('raw_feature_bc_matrix.h5')

# .mtx or .mtx.gz, with barcodes.tsv(.gz) and features.tsv(.gz) in the same directory
sc = SingleCell('matrix.mtx')
sc = SingleCell('matrix.mtx.gz')
```

### Choosing the count matrix

When a file contains both raw and normalized counts, SingleCell loads raw counts by default. Use `X_key` to choose a different location:

```python
# .h5ad: use SingleCell.ls() to find the right key
SingleCell.ls('data.h5ad')

# load from a specific .h5ad slot
sc = SingleCell('data.h5ad', X_key='raw/X')

# Seurat: load normalized counts from the 'data' layer
sc = SingleCell('seurat_obj.rds', X_key='data')

# Seurat: load from a non-default assay
sc = SingleCell('seurat_obj.h5Seurat', assay='SCT')

# SingleCellExperiment: load log-normalized counts
sc = SingleCell('sce_obj.rds', X_key='logcounts')
```

### Partial loading

For large `.h5ad` and `.h5Seurat` files, you can load only the metadata columns you need:

```python
sc = SingleCell('data.h5ad',
                obs_columns=['cell_type', 'batch'],
                var_columns=['gene_symbol'])
```

You can skip loading the count matrix entirely (useful if you only need the metadata or principal components). Note that datasets loaded without `X` cannot be saved, converted, or used for analyses that require counts:

```python
sc = SingleCell('data.h5ad', X=False)
```

### Reading individual slots

You can also read {attr}`~brisc.SingleCell.obs`, {attr}`~brisc.SingleCell.var`, {attr}`~brisc.SingleCell.obsm`, {attr}`~brisc.SingleCell.varm`, or {attr}`~brisc.SingleCell.uns` from an `.h5ad` file without loading the full dataset:

```python
# polars DataFrame
obs = SingleCell.read_obs('data.h5ad', columns=['cell_type', 'batch'])

# polars DataFrame
var = SingleCell.read_var('data.h5ad')

# dict: {key: NumPy array | polars DataFrame}
obsm = SingleCell.read_obsm('data.h5ad', keys=['X_pca'])
```

## Saving to file

As with loading, format is inferred from the file extension:

```python
# scverse / Scanpy
sc.save('output.h5ad')

# Seurat v5
sc.save('output.rds')

# Seurat v3
sc.save('output.rds', v3=True)

# Seurat .h5Seurat (always v3)
sc.save('output.h5Seurat')

# SingleCellExperiment
sc.save('output_sce.rds', sce=True)

# 10x Genomics (or .mtx.gz for gzipped)
sc.save('matrix.mtx')
sc.save('matrix.mtx.gz')
```

When saving to `.rds`, the `X_key` argument controls which layer `X` is stored in:

```python
# save counts to the 'data' layer
sc.save('output.rds', X_key='data')
```

:::{note}
When saving to Seurat `.rds` or `.h5Seurat`, the `X_` prefix is automatically stripped from `obsm` keys (e.g. `X_umap` becomes `umap`) to match Seurat's conventions. Seurat also adds `orig.ident`, `nCount_RNA`, and `nFeature_RNA` by default, which can be slow; to suppress the latter two:

```python
from ryp import r
r('options(Seurat.object.assay.calcn = FALSE)')
```
:::

## In-memory conversion

### From AnnData

Pass an AnnData object directly to the SingleCell constructor. By default, raw counts are loaded from `adata.layers['UMIs']` or `adata.raw.X` if present, falling back to `adata.X`:

```python
import scanpy as sc

adata = sc.read_h5ad('data.h5ad')
sc_data = SingleCell(adata)

# explicitly choose which matrix to use
sc_data = SingleCell(adata, X=adata.layers['raw_counts'])
```

The constructor shares `adata`'s count matrix rather than copying it (the reverse of {meth}`~brisc.SingleCell.to_scanpy`), so edits to one also affect the other; use `SingleCell(adata.copy())` for an independent dataset. A dense `adata.X` is the exception: it is copied to a sparse `csr_array`, with a warning.

### To AnnData

```python
adata = sc_data.to_scanpy()
```

:::{note}
The count matrix is shared, not copied. Modifying `adata.X` will also modify the original SingleCell dataset. To avoid this, use `sc_data.copy(deep=True).to_scanpy()`.

There is no `from_scanpy()` method — {meth}`SingleCell(adata) <brisc.SingleCell.__init__>` serves that purpose.
:::

### Constructing from scratch

You can also build a SingleCell dataset from individual components:

```python
import polars as pl
from scipy.sparse import csr_array

X = csr_array([[1, 0, 3], [0, 2, 0]])
obs = pl.DataFrame({'cell_id': ['cell1', 'cell2']})
var = pl.DataFrame({'gene_id': ['g1', 'g2', 'g3']})

sc = SingleCell(X=X, obs=obs, var=var)
```

## The ryp Python-R bridge

Seurat and SingleCellExperiment `.rds` files are handled transparently via [ryp](https://github.com/Wainberg/ryp), a Python-R bridge. Loading and saving `.rds` files works like any other format. For in-memory conversion between SingleCell and R objects, use {meth}`~brisc.SingleCell.from_seurat` / {meth}`~brisc.SingleCell.to_seurat` and {meth}`~brisc.SingleCell.from_sce` / {meth}`~brisc.SingleCell.to_sce`. Unlike the AnnData path, these conversions usually copy the count matrix into or out of R's memory, so a conversion can use up to roughly double the matrix's memory. ({meth}`~brisc.SingleCell.to_seurat` and {meth}`~brisc.SingleCell.to_sce` can avoid the copy when `X` is already `float64` with 32-bit indices.)

:::{note}
R's sparse matrices use 32-bit indices, so Seurat and SingleCellExperiment objects cannot hold count matrices with more than 2,147,483,647 (INT32_MAX) non-zero elements. Large datasets may exceed this limit.
:::

### Seurat

```python
from ryp import r

# load a Seurat object in R
r('seurat_obj <- readRDS("seurat_obj.rds")')

# convert to SingleCell
sc = SingleCell.from_seurat('seurat_obj')

# convert from a specific assay/layer
sc = SingleCell.from_seurat('seurat_obj', assay='SCT', layer='data')

# convert back to Seurat (v5) in R's workspace
sc.to_seurat('seurat_out')

# convert to Seurat v3
sc.to_seurat('seurat_out', v3=True)

# or just save directly — no need for to_seurat + saveRDS
sc.save('output.rds')
```

### SingleCellExperiment

```python
from ryp import r

# load a SingleCellExperiment in R
r('sce <- readRDS("sce_obj.rds")')

# convert to SingleCell
sc = SingleCell.from_sce('sce')

# use log-normalized counts instead
sc = SingleCell.from_sce('sce', assay='logcounts')

# convert back to SCE in R's workspace
sc.to_sce('sce_out')

# or just save directly
sc.save('output_sce.rds', sce=True)
```

### Example: combining Seurat and Scanpy

A common reason to bridge ecosystems is to use tools that only exist in one. For instance, [Azimuth](https://azimuth.hubmapconsortium.org/) provides automated cell type annotation via a Seurat reference atlas (R only), while [scvi-tools](https://scvi-tools.org/) provides deep generative models for integration and differential expression (Python only). SingleCell lets you chain these without writing intermediate files:

```python
from brisc import SingleCell
from ryp import r

# start in Python
sc = SingleCell('data.h5ad').skip_qc()

# pass to R: annotate cell types with Azimuth
sc.to_seurat('obj')
r('''
library(Azimuth)
obj <- RunAzimuth(obj, reference = "pbmcref")
''')
sc = SingleCell.from_seurat('obj')
# sc.obs now contains Azimuth's predicted.celltype.l1, l2, etc.

# back in Python: run scvi integration
adata = sc.to_scanpy()
import scvi
scvi.model.SCVI.setup_anndata(adata, batch_key='batch')
model = scvi.model.SCVI(adata)
model.train()
adata.obsm['X_scVI'] = model.get_latent_representation()

# return to SingleCell for downstream analysis
sc = SingleCell(adata)
```

## Summary

| Operation | Method |
|---|---|
| Load `.h5ad` | {meth}`SingleCell('file.h5ad') <brisc.SingleCell.__init__>` |
| Load `.rds` / `.h5Seurat` | {meth}`SingleCell('file.rds') <brisc.SingleCell.__init__>` |
| Load 10x `.h5` / `.mtx` | {meth}`SingleCell('matrix.h5') <brisc.SingleCell.__init__>` |
| Load specific layer | {meth}`SingleCell('file.h5ad', X_key='raw/X') <brisc.SingleCell.__init__>` |
| Load subset of columns | {meth}`SingleCell('file.h5ad', obs_columns=[...]) <brisc.SingleCell.__init__>` |
| Load without counts | {meth}`SingleCell('file.h5ad', X=False) <brisc.SingleCell.__init__>` |
| Read `obs` only | {meth}`SingleCell.read_obs('file.h5ad') <brisc.SingleCell.read_obs>` |
| Save to any format | {meth}`sc.save('out.h5ad') <brisc.SingleCell.save>` |
| From AnnData | {meth}`SingleCell(adata) <brisc.SingleCell.__init__>` |
| To AnnData | {meth}`sc.to_scanpy() <brisc.SingleCell.to_scanpy>` |
| From Seurat (in-memory) | {meth}`SingleCell.from_seurat('obj') <brisc.SingleCell.from_seurat>` |
| To Seurat (in-memory) | {meth}`sc.to_seurat('obj') <brisc.SingleCell.to_seurat>` |
| From SCE (in-memory) | {meth}`SingleCell.from_sce('obj') <brisc.SingleCell.from_sce>` |
| To SCE (in-memory) | {meth}`sc.to_sce('obj') <brisc.SingleCell.to_sce>` |
| Construct manually | {meth}`SingleCell(X=X, obs=obs, var=var) <brisc.SingleCell.__init__>` |
