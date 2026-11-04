:description: API reference for the brisc SingleCell class: load and save single-cell data, QC, normalization, PCA, clustering, embedding, label transfer and pseudobulking.

SingleCell
==========

.. toctree::
   :maxdepth: 2
   :hidden:

   analysis
   io
   properties
   data_access
   manipulation

.. currentmodule:: brisc

.. classdescription:: SingleCell

.. autoclass:: SingleCell
   :no-members:

Analysis
--------

.. autosummary::
   :nosignatures:

   SingleCell.qc_metrics
   SingleCell.qc
   SingleCell.skip_qc
   SingleCell.find_doublets
   SingleCell.get_sample_covariates
   SingleCell.pseudobulk
   SingleCell.hvg
   SingleCell.normalize
   SingleCell.pca
   SingleCell.neighbors
   SingleCell.shared_neighbors
   SingleCell.harmonize
   SingleCell.cluster
   SingleCell.label_transfer_from
   SingleCell.umap
   SingleCell.pacmap
   SingleCell.localmap
   SingleCell.find_markers
   SingleCell.plot_heatmap
   SingleCell.plot_markers
   SingleCell.plot_umap
   SingleCell.plot_pacmap
   SingleCell.plot_localmap
   SingleCell.plot_embedding

I/O
---

.. autosummary::
   :nosignatures:

   SingleCell.save
   SingleCell.ls
   SingleCell.read_obs
   SingleCell.read_var
   SingleCell.read_obsm
   SingleCell.read_varm
   SingleCell.read_obsp
   SingleCell.read_varp
   SingleCell.read_uns
   SingleCell.to_scanpy
   SingleCell.from_seurat
   SingleCell.to_seurat
   SingleCell.from_sce
   SingleCell.to_sce

Properties
----------

.. autosummary::
   :nosignatures:

   SingleCell.X
   SingleCell.obs
   SingleCell.var
   SingleCell.obsm
   SingleCell.varm
   SingleCell.obsp
   SingleCell.varp
   SingleCell.uns
   SingleCell.obs_names
   SingleCell.var_names
   SingleCell.num_threads
   SingleCell.shape

Data access
-----------

.. autosummary::
   :nosignatures:

   SingleCell.cell
   SingleCell.gene
   SingleCell.peek_obs
   SingleCell.peek_var

Manipulation
------------

.. autosummary::
   :nosignatures:

   SingleCell.set_obs_names
   SingleCell.set_var_names
   SingleCell.set_num_threads
   SingleCell.make_obs_names_unique
   SingleCell.make_var_names_unique
   SingleCell.filter_obs
   SingleCell.filter_var
   SingleCell.select_obs
   SingleCell.select_var
   SingleCell.select_obsm
   SingleCell.select_varm
   SingleCell.select_obsp
   SingleCell.select_varp
   SingleCell.select_uns
   SingleCell.with_columns_obs
   SingleCell.with_columns_var
   SingleCell.with_obsm
   SingleCell.with_varm
   SingleCell.with_obsp
   SingleCell.with_varp
   SingleCell.with_uns
   SingleCell.drop_X
   SingleCell.drop_obs
   SingleCell.drop_var
   SingleCell.drop_obsm
   SingleCell.drop_varm
   SingleCell.drop_obsp
   SingleCell.drop_varp
   SingleCell.drop_uns
   SingleCell.rename_obs
   SingleCell.rename_var
   SingleCell.rename_obsm
   SingleCell.rename_varm
   SingleCell.rename_obsp
   SingleCell.rename_varp
   SingleCell.rename_uns
   SingleCell.cast_X
   SingleCell.cast_obs
   SingleCell.cast_var
   SingleCell.join_obs
   SingleCell.join_var
   SingleCell.subsample_obs
   SingleCell.subsample_var
   SingleCell.tocsr
   SingleCell.tocsc
   SingleCell.copy
   SingleCell.concat_obs
   SingleCell.concat_var
   SingleCell.split_by_obs
   SingleCell.split_by_var
   SingleCell.pipe
   SingleCell.pipe_X
   SingleCell.pipe_obs
   SingleCell.pipe_var
   SingleCell.pipe_obsm
   SingleCell.pipe_obsm_key
   SingleCell.pipe_varm
   SingleCell.pipe_varm_key
   SingleCell.pipe_obsp
   SingleCell.pipe_obsp_key
   SingleCell.pipe_varp
   SingleCell.pipe_varp_key
   SingleCell.pipe_uns
   SingleCell.pipe_uns_key
