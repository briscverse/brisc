import os
import signal
import sys
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('brisc')
except PackageNotFoundError:
    __version__ = '0.0.0+unknown'

# Disable HDF5 file locking, which can cause issues when loading
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Python's `logging` module calls `os.register_at_fork()` to handle aspects of
# its internal locking. Unfortunately, this can cause KeyboardInterrupts to be
# repeatedly ignored in multiprocessing-based HDF5 loading with errors like:
# Exception ignored in: <function _releaseLock at 0x7fb85797c5e0>
# Traceback (most recent call last):
#   File ".../logging/__init__.py", line 243, in _releaseLock
#     def _releaseLock():
# To get around this bug, temporarily monkeypatch `os.register_at_fork()` to be
# a null-op, then `import logging`. Unfortunately, there's no way to unregister
# a callback created with `os.register_at_fork()`, so this "fix" only works
# when the `logging` module has not been imported yet.
if 'logging' not in sys.modules and hasattr(os, 'register_at_fork'):
    original_register_at_fork = os.register_at_fork
    os.register_at_fork = lambda *args, **kwargs: None
    try:
        import logging
    finally:
        os.register_at_fork = original_register_at_fork

# Ignore Ctrl + C when importing certain modules, to avoid errors due to
# incomplete imports
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
try:
    import h5py
    import numpy as np
    import polars as pl
    import pyarrow as pa
    from scipy import sparse
    from scipy.sparse._compressed import _cs_matrix
    from scipy.special import stdtrit
finally:
    signal.signal(signal.SIGINT, original_sigint_handler)

# Import threadpoolctl before anything loads an OpenMP runtime, since it sets
# KMP_DUPLICATE_LIB_OK
from .threadpoolctl import brisc_blas

# On Linux with thread pinning enabled, MKL BLAS narrows the thread affinity
# mask to one core when first executed, which would make all of brisc run
# single-threaded if not fixed. Fix this now by running a BLAS function, then
# re-expanding the mask to all available cores. (This works even if cgroups
# limit the available cores, e.g. on Slurm: the OS will take the intersection
# of what we asked for and what the cgroup allows.)
if hasattr(os, 'sched_setaffinity'):
    np.linalg.svd([[0]])
    os.sched_setaffinity(0, range(os.cpu_count()))

# Expose the public API
from .single_cell import SingleCell
from .pseudobulk import Pseudobulk
from .de import DE
from .concatenate import concat_obs, concat_var
__all__ = 'SingleCell', 'Pseudobulk', 'DE', 'brisc_blas', 'concat_obs', \
    'concat_var'