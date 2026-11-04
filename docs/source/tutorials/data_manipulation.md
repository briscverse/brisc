---
description: "Reference for brisc dataset operations: inspecting, subsetting, editing metadata, working with the count matrix, combining datasets and pseudobulking."
---

# Data Manipulation

This tutorial is a reference for the operations brisc provides on its datasets: inspecting a dataset, accessing its pieces, subsetting by cells or genes, editing cell and gene metadata, working with the count matrix, summarizing, combining datasets, and pseudobulking.

## Inspecting

```python
# shape, as a tuple of (number of cells, number of genes)
sc.shape

# number of cells
len(sc)

# column names
sc.obs.columns

# a dictionary mapping column names to their data types
sc.obs.schema

# first cell/gene, one column per line
sc.peek_obs()
sc.peek_var()
```

Inspect or partially load a file without reading the count matrix:

```python
# a file's dimensions and obs/var columns
SingleCell.ls('data.h5ad')

# read individual slots
obs = SingleCell.read_obs('data.h5ad')
var = SingleCell.read_var('data.h5ad')
obsm = SingleCell.read_obsm('data.h5ad')
varm = SingleCell.read_varm('data.h5ad')
uns = SingleCell.read_uns('data.h5ad')
```

## Accessing the pieces

```python
# the count matrix (sparse, cells × genes)
sc.X

# cell/gene metadata (polars DataFrames)
sc.obs
sc.var

# one column (a polars Series)
sc.obs['cell_type']

# cell barcodes/gene names (first columns of obs/var)
sc.obs_names
sc.var_names

# one gene across cells/one cell across genes (as a dense 1D array)
sc.gene('MALAT1')
sc.cell(sc.obs_names[0])

# pipeline outputs (after PCA/clustering; see Basic Workflow)
sc.obsm['pca']                # principal components (NumPy array)
sc.obsp['shared_neighbors']   # cell–cell graph (sparse)
sc.uns['QCed']                # pipeline flags
```

## Subsetting by cells or genes

By metadata:

```python
# filter to one condition
sc = sc.filter_obs(pl.col('cell_type') == 'CD14 Mono')

# a shorthand version of the above
sc = sc.filter_obs(cell_type='CD14 Mono')

# filter to multiple conditions
sc = sc.filter_obs(pl.col('cytokine').is_in(['IFN-gamma', 'PBS']))

# apply multiple filters at once
sc = sc.filter_obs(treatment='PBS', cell_type='CD14 Mono')

# filter genes
sc = sc.filter_var(pl.col('n_cells') > 100)
```

By position or name:

```python
# subset by position/by barcode, for all genes
sc[0]
sc[sc.obs_names[0]]

# subset to one/several genes, for all cells
sc[:, 'MALAT1']
sc[:, ['CD3D', 'CD8A', 'MS4A1']]

# subset to the first 1000 cells
sc[:1000]

# subset based on a Boolean mask of cells;
# this particular example is equivalent to
# sc = sc.filter_obs(cell_type='CD14 Mono')
sc[sc.obs['cell_type'] == 'CD14 Mono']
```

Split or sample (`split_by_var`/`subsample_var` do the same for genes):

```python
# split into a dictionary of datasets, one per unique value of the 'treatment' column
parts = sc.split_by_obs('treatment')

# subsample to 10k random cells
sc = sc.subsample_obs(n=10_000)

# subsample to 10% of each cell type
sc = sc.subsample_obs(fraction=0.1, by_column='cell_type')
```

## Editing metadata

```python
# add or overwrite a column from an expression
sc = sc.with_columns_obs(
    stimulated=pl.col('treatment') == 'cytokine')

# add a multi-branch column
sc = sc.with_columns_obs(
    lineage=pl.when(pl.col('cell_type').cast(str).str.contains('Mono|DC'))
              .then(pl.lit('myeloid'))
              .otherwise(pl.lit('lymphoid')))

# rename/drop/select/cast columns
sc = sc.rename_obs({'treatment': 'group'})
sc = sc.drop_obs('cytokine')
sc = sc.select_obs('cell_type', 'sample')   # keeps obs_names automatically
sc = sc.cast_obs({'donor': str})
```

Left-join an external DataFrame onto `obs`:

```python
donor_meta = pl.DataFrame({'donor': ['Donor1', 'Donor2'], 'age': [29, 34]})

# sc.obs['donor'] is an Enum, so match its dtype before joining
donor_meta = donor_meta.cast({'donor': sc.obs.schema['donor']})
sc = sc.join_obs(donor_meta, on='donor')

# the gene-side twin
sc = sc.join_var(gene_meta, on='gene_id')
```

Set or deduplicate cell/gene names:

```python
# set a column as the cell names (String, Enum, Categorical, or integer)
sc = sc.set_obs_names('sample')

# deduplicate cell/gene names (appends -1, -2, … to duplicates)
sc = sc.make_obs_names_unique()
sc = sc.make_var_names_unique()
```

## The count matrix

```python
# storage format: 'csr' or 'csc'
sc.X.format

# convert layout to CSR for more efficient access, if CSC
sc = sc.tocsr()

# change the matrix data type
sc = sc.cast_X('float32')

# drop X to save memory
sc = sc.drop_X()
```

## Summarizing

```python
# count of each value in a column
sc.obs['cell_type'].value_counts(sort=True)

# several stats at once
sc.obs.group_by('cell_type').agg(
    n=pl.len(),
    n_donors=pl.col('donor').n_unique())

# get columns that are constant within each sample
sc.get_sample_covariates(ID_column='sample')

# add and summarize QC metrics
sc = sc.qc_metrics(allow_float=True)
sc.obs.group_by('cell_type').agg(pl.col('num_counts').median())

# cross-tabulate two categorical columns as a heatmap
sc.plot_heatmap('cell_type', 'donor', 'composition.png')
```

## Combining and copying

```python
from brisc import concat_obs

# combine cells from multiple datasets into one, adding a column that lists each cell's source dataset
combined = concat_obs([sc_a, sc_b], dataset_column='batch')

# when genes or obs columns are inconsistent between datasets, specify flexible=True
combined = concat_obs([sc_a, sc_b], dataset_column='batch', flexible=True)

# shallow copy (default): shares X and the obsm/varm/obsp/varp/uns arrays
sc_copy = sc.copy()

# deep copy: also duplicates X and those arrays, for a fully independent dataset
sc_deep = sc.copy(deep=True)
```

## Pseudobulk

{meth}`~brisc.SingleCell.pseudobulk` returns a {class}`~brisc.Pseudobulk` dataset, with three slots — `X`, `obs`, and `var`. Access a slot, a cell type, or its pieces:

```python
# index by cell type, returning a Pseudobulk with just that one cell type
pb['CD14 Mono']

# each slot is a dictionary keyed by cell type
pb.X
pb.obs
pb.var

# one cell type's count matrix, sample metadata, and gene metadata
pb.X['CD14 Mono']
pb.obs['CD14 Mono']
pb.var['CD14 Mono']

# sample names/gene names, per cell type
pb.obs_names
pb.var_names

# shape, as a tuple of (number of samples, number of genes), per cell type
pb.shape
```

Iterate like a dictionary keyed by cell type:

```python
# the cell type names
pb.keys()

# each cell type's (X, obs, var) triple
pb.values()

# iterate over the cell types
for cell_type in pb:
    ...

# iterate over (cell type, (X, obs, var)) pairs
for cell_type, (X, obs, var) in pb.items():
    ...

# iterate over a single slot at a time, across cell types
for obs in pb.iter_obs():
    ...
for var in pb.iter_var():
    ...
```

Filter and edit within each cell type, or operate on the cell types themselves:

```python
# any *_obs/*_var method works, e.g. filter_obs
pb = pb.filter_obs(treatment='PBS')

# select/rename cell types
pb = pb.select_cell_types(['CD14 Mono', 'NK'])
pb = pb.rename_cell_types({'CD14 Mono': 'Mono'})
```

Flatten into one wide DataFrame:

```python
# one row per (sample, cell type), one column per gene
pb.to_df()
```

Normalize, then regress out covariates:

```python
# TMM-normalized library sizes, then log counts per million
pb = pb.library_size()
pb = pb.log_cpm()

# regress out covariates from the log-CPM values
pb = pb.regress_out('~ donor + log2(num_cells) + log2(library_size)')
```

This `log_cpm`/`regress_out` path is separate from {meth}`~brisc.Pseudobulk.de`, which requires raw counts and cannot run after `log_cpm`.

## DE results

{meth}`~brisc.Pseudobulk.de` returns a {class}`~brisc.DE` object whose `table` is a polars DataFrame:

```python
# one row per gene per cell type
de.table

# the significant genes (here, FDR < 0.05)
de.get_hits(significance_column='FDR', threshold=0.05)

# the number of significant genes in each cell type
de.get_num_hits()

# filter and sort like any polars DataFrame
de.table.filter(cell_type='CD14 Mono').sort('FDR').head(20)

# save, then reload
de.save('de_results')
de = DE('de_results')
```

## Custom operations

When a transformation isn't wrapped by a brisc method, apply your own function with the `pipe_*` methods:

```python
# run an arbitrary polars operation on obs (or var)
sc = sc.pipe_obs(lambda df: df.to_dummies('cell_type'))

# transform the count matrix
sc = sc.pipe_X(lambda X: X.sqrt())

# transform one entry of a dict-valued slot (here, keep the first 20 PCs);
# the other slots have *_key variants, and pipe_obsm transforms the whole dict
sc = sc.pipe_obsm_key('pca', lambda pca: pca[:, :20])

# transform the whole dataset
sc = sc.pipe(my_function)
```

A {class}`~brisc.Pseudobulk` uses `map_obs`, `map_var`, and `map_X` to apply a function within each cell type, and `pipe` to transform the whole dataset:

```python
# apply a function within each cell type
pb = pb.map_obs(lambda df: df.with_columns(
    log_counts=pl.col('num_cells').log1p()))
pb = pb.map_X(lambda X: np.log1p(X))

# transform the whole dataset
pb = pb.pipe(my_function)
```
