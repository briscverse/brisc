# brisc

<div style="text-align: center; margin-bottom: 1.5rem;">
  <img src="_static/images/runner_title_wide_light.svg" alt="brisc" class="only-light" style="max-width: 80%;">
  <img src="_static/images/runner_title_wide_dark.svg" alt="brisc" class="only-dark" style="max-width: 80%;">
</div>

brisc is a high-performance library for analyzing single-cell data at scale. It prioritizes running as fast as possible on multi-core CPU systems, strict reproducibility, and a clean, user-friendly interface. On datasets of 1 to 20 million cells, it cuts the runtime of common workflows from hours to minutes. <a href="installation.html" class="hero-cta"><strong>Get started.</strong> &rsaquo;</a>

<div class="features-grid">

<div class="feature-card">
<div class="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></div>
<h3 class="feature-title">Blazing fast</h3>
<p class="feature-desc">Achieved through ground-up optimization of core algorithms and effective parallelism.</p>
</div>

<div class="feature-card">
<div class="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg></div>
<h3 class="feature-title">Deterministic</h3>
<p class="feature-desc">Every step gives floating-point identical results between runs, regardless of the number of threads used.</p>
</div>

<div class="feature-card">
<div class="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg></div>
<h3 class="feature-title">Complete toolkit</h3>
<p class="feature-desc">Supports preprocessing, dimensionality reduction, harmonization, label transfer, clustering, embedding, pseudobulk differential expression, and plotting.</p>
</div>

<div class="feature-card">
<div class="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/></svg></div>
<h3 class="feature-title">Interoperable</h3>
<p class="feature-desc">Reads and writes <code>.h5ad</code>, <code>.rds</code>, <code>.h5Seurat</code>, and 10x files; supports interleaving Python and R analyses via <a href="https://github.com/Wainberg/ryp">ryp</a> without intermediate writes to disk.</p>
</div>

<div class="feature-card">
<div class="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" x2="2" y1="12" y2="12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><line x1="6" x2="6.01" y1="16" y2="16"/><line x1="10" x2="10.01" y1="16" y2="16"/></svg></div>
<h3 class="feature-title">Memory-efficient</h3>
<p class="feature-desc">Uses ~2× lower peak memory than Scanpy by tabulating which cells pass QC, instead of subsetting to them.</p>
</div>

<div class="feature-card">
<div class="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/></svg></div>
<h3 class="feature-title">User-friendly</h3>
<p class="feature-desc">Features sensible defaults, strict type-checking, and solution-focused error messages.</p>
</div>

</div>

<div class="showcase-chart">
<div class="showcase-chart-header">
<div class="showcase-chart-title">Performance · 10M cells · Parse Biosciences PBMC</div>
<div class="showcase-chart-subtitle">192 CPUs, 749 GB RAM</div>
</div>
<div id="benchmark-chart"></div>
</div>

<div class="code-grid">
<div class="code-card code-card--left">
<div class="card-tab"><span class="card-tab-title">Basic workflow</span></div>
<div class="card-code">

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
  .pacmap()
```

</div>
</div>
<div class="code-card code-card--right">
<div class="card-tab"><span class="card-tab-title">Label transfer</span></div>
<div class="card-code">

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

</div>
</div>
<div class="code-card code-card--center">
<div class="card-tab"><span class="card-tab-title">Pseudobulk differential expression</span></div>
<div class="card-code">

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

</div>
</div>
</div>

<script>
document.body.classList.add('landing-page');
</script>

:::{toctree}
:caption: Installation
:hidden:
:maxdepth: 1

installation

:::

:::{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 2

tutorials/index

:::

:::{toctree}
:caption: API
:hidden:
:maxdepth: 4

singlecell/index
pseudobulk/index
de/index

:::
