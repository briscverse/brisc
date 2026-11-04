---
description: "Install brisc with conda or pip on Linux, macOS or Windows, add the optional R packages for differential expression and Seurat, and pick a fast BLAS."
---

# Installation

brisc supports Linux, macOS, and Windows on Python 3.9+. Install it with conda or pip:

::::{tab-set}
:::{tab-item} conda (recommended)
```bash
conda install -c conda-forge brisc
```
:::
:::{tab-item} pip
```bash
pip install brisc
```
:::
::::

conda is recommended because it sets up the fast MKL BLAS and some of the R packages brisc uses (both covered below); with pip you must handle those yourself.

## R packages

brisc's R integration is optional — you need it only for differential expression or for working with Seurat and SingleCellExperiment objects. Skip this section if you do neither.

It runs through [ryp](https://github.com/Wainberg/ryp), which bridges Python and R via R's arrow package, so arrow is always required. Each feature then adds one package: limma for differential expression, and Seurat or SingleCellExperiment for the corresponding objects.

conda handles this for you — `conda install -c conda-forge brisc` installs R, arrow, and Seurat. With pip you do it yourself: install R using [CRAN's per-platform instructions](https://cran.r-project.org), then run `install.packages(c("arrow", "Seurat"))` in an R session.

For differential expression, install limma:

::::{tab-set}
:::{tab-item} conda (Linux/macOS)
```bash
conda install -c bioconda bioconductor-limma
```
:::
:::{tab-item} BiocManager (all platforms)
```bash
R -e "if (!require('BiocManager', quietly = TRUE)) install.packages('BiocManager'); BiocManager::install('limma')"
```
:::
::::

For SingleCellExperiment data, install it:

::::{tab-set}
:::{tab-item} conda (Linux/macOS)
```bash
conda install -c bioconda bioconductor-singlecellexperiment
```
:::
:::{tab-item} BiocManager (all platforms)
```bash
R -e 'if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager"); BiocManager::install("SingleCellExperiment")'
```
:::
::::

## BLAS

brisc's speed depends on which BLAS library is installed. Check yours:

```python
from brisc import brisc_blas
print(brisc_blas())
```

- `mkl`: ideal. Only available on x86 processors (most Linux and Windows machines, and old Macs with Intel processors).
- `accelerate`: ideal, but you must set `VECLIB_MAXIMUM_THREADS=1` (see warning box below).
- `flexiblas` or `blis`: less common but acceptable.
- `openblas`: slower, and 3 key steps (nearest-neighbor search, harmonization, and label transfer) will be capped to 64 threads. If your processor supports it, consider switching to a faster BLAS. This is straightforward with conda:
    - x86 processors (most Linux and Windows machines, and old Macs): install MKL with `conda install "libblas=*=*mkl" scipy`
    - Apple Silicon processors (new Macs): install Accelerate BLAS with `conda install "libblas=*=*newaccelerate" scipy`
- `None`: brisc can't tell which BLAS you have. This may occur on macOS < 15 with Accelerate, in which case you must set `VECLIB_MAXIMUM_THREADS=1` (see warning box below). Otherwise, install a BLAS library brisc can recognize, like MKL or Accelerate (see the `openblas` bullet above), for fast, reproducible results.

:::{important}
Mac users with Accelerate BLAS must set `VECLIB_MAXIMUM_THREADS=1` for fast, reproducible results. You can do this by:

- Setting `os.environ['VECLIB_MAXIMUM_THREADS']='1'` before importing NumPy, SciPy or brisc
- Adding `export VECLIB_MAXIMUM_THREADS=1` to your `~/.zshrc`
- Running `conda env config vars set VECLIB_MAXIMUM_THREADS=1` and then reactivating your conda environment
:::
