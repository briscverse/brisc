:description: API reference for the brisc DE class: differential expression results per cell type, with hit filtering and counting, saving and reloading.

DE
==

.. toctree::
   :maxdepth: 2
   :hidden:

   analysis
   io
   properties

.. currentmodule:: brisc

.. classdescription:: DE

.. autoclass:: DE
   :no-members:

Analysis
--------

.. autosummary::
   :nosignatures:

   DE.get_hits
   DE.get_num_hits
   DE.plot_voom
   DE.plot_volcano

I/O
---

.. autosummary::
   :nosignatures:

   DE.save

Properties
----------

.. autosummary::
   :nosignatures:

   DE.groups
