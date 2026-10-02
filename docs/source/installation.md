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

## BLAS and threading

Three key steps (nearest-neighbor search, harmonization, and label transfer) rely on BLAS. brisc uses scipy to call BLAS, so which BLAS library you have depends on your platform and how you installed SciPy:

- **MKL**: Intel's highly optimized BLAS, for x86 processors (most Linux and Windows machines, and old Macs that use Intel processors). Automatically installed when installing brisc through conda on x86, but can be installed manually with `conda install "libblas=*=*mkl" scipy`.
- **Accelerate**: Apple's BLAS, built into macOS. Automatically installed when installing brisc through conda on Apple Silicon Macs, but can be installed manually with `conda install "libblas=*=*newaccelerate" scipy`. pip's SciPy also uses it on all Macs running macOS 14 or later.
- **OpenBLAS**: a slower BLAS. Used by pip's SciPy on Linux, Windows, and Macs running macOS 13 or earlier, and by conda on Linux ARM machines. It only supports up to 64 threads, so brisc caps the three BLAS-reliant steps at 64 threads when using it.

To check which BLAS brisc is using:

```python
from brisc import brisc_blas
print(brisc_blas())
```

This prints `mkl`, `accelerate`, or `openblas` (or `blis` or `flexiblas` for less common setups). It prints `None` if brisc can't identify the library; for example, brisc can only detect Accelerate on macOS 15 and later.

:::{important}
**Mac users with Accelerate: set `VECLIB_MAXIMUM_THREADS=1`** for fast, reproducible results on the three BLAS-reliant steps. You can do this by:

- Setting `os.environ['VECLIB_MAXIMUM_THREADS']='1'` before importing NumPy, SciPy or brisc
- Adding `export VECLIB_MAXIMUM_THREADS=1` to your `~/.zshrc`
- Running `conda env config vars set VECLIB_MAXIMUM_THREADS=1` and then reactivating your conda environment
:::