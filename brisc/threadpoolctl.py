# Adapted from threadpoolctl 3.7.0 (https://github.com/joblib/threadpoolctl):
#
# Copyright (c) 2019, threadpoolctl contributors
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of copyright holder nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Limit the number of threads used by BLAS and OpenMP libraries.

`threadpool_limits()` works like threadpoolctl's, with one difference: it
limits MKL with `MKL_Set_Num_Threads()`, which applies to every thread,
instead of `MKL_Set_Num_Threads_Local()` (threadpoolctl 3.7.0+), which applies
only to the calling thread. brisc calls BLAS from the worker threads of its
own OpenMP parallel regions, which never see a limit set only on the main
thread, so MKL would use every core on every worker, oversubscribing the
machine and changing results between `num_threads=1` and `num_threads > 1`.

`brisc_blas()` returns the name of the BLAS library brisc calls.
"""

import ctypes
import itertools
import os
import sys
import warnings
from functools import cache
from pathlib import Path

__all__ = ['threadpool_limits', 'brisc_blas']

# Intel's OpenMP runtime (which MKL uses) and LLVM's (which brisc's Windows and
# macOS builds use, as does conda's OpenBLAS on macOS) share code. By default,
# when a second copy of either starts up in a process that already has one,
# it kills the process with "OMP: Error #15: Initializing libomp..., but found
# libomp... already initialized". Setting this makes it keep running instead.
# It has to be set before the second copy starts up, so `brisc/__init__.py`
# imports this module before anything else that could load an OpenMP runtime.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')

# The longest Windows library path to look up with `GetModuleFileNameExW()`;
# threadpoolctl caps it for security reasons (threadpoolctl PR #189)
_WINDOWS_MAX_LIBRARY_PATH_LENGTH = 2600

# The RTLD_NOLOAD flag for loading shared libraries is not defined on Windows.
try:
    _RTLD_NOLOAD = os.RTLD_NOLOAD
except AttributeError:
    _RTLD_NOLOAD = ctypes.DEFAULT_MODE


class _Library:
    """
    A loaded BLAS or OpenMP library. Subclasses set `user_api` (`'blas'` or
    `'openmp'`), `name`, `filename_prefixes` (lowercase) and `check_symbols`
    (the library must export at least one), and implement `get()` and `set()`.
    Subclasses whose limit can't be undone by setting it back to what `get()`
    returned beforehand also override `limit()` and `restore()`.
    """
    user_api = 'blas'
    affixes = '', ''

    def __init__(self, *, filepath, prefix):
        self.filepath = filepath
        self.prefix = prefix
        self.dynlib = ctypes.CDLL(filepath, mode=_RTLD_NOLOAD)

    def call(self, function_name, *args):
        # Call one of the library's functions, if it has it
        prefix, suffix = self.affixes
        function = getattr(self.dynlib, f'{prefix}{function_name}{suffix}',
                           None)
        return None if function is None else function(*args)

    def limit(self, num_threads):
        state = self.get()
        self.set(num_threads)
        return state

    def restore(self, state):
        if state is not None:
            self.set(state)


class _OpenBLAS(_Library):
    name = 'openblas'
    filename_prefixes = \
        'libopenblas', 'libscipy_openblas', 'openblas', 'libblas'
    _affixes = list(itertools.product(('', 'scipy_'), ('', '64_', '_64')))
    check_symbols = [f'{prefix}openblas_get_num_threads{suffix}'
               for prefix, suffix in _affixes]

    def __init__(self, *, filepath, prefix):
        super().__init__(filepath=filepath, prefix=prefix)
        self.affixes = next(
            ((prefix, suffix) for prefix, suffix in self._affixes
             if hasattr(self.dynlib,
                        f'{prefix}openblas_get_num_threads{suffix}')),
            ('', ''))
        # OpenBLAS's OpenMP build takes its thread count from OpenMP, and its
        # `openblas_set_num_threads()` is broken before OpenBLAS 0.3.34
        self.openmp = self.call('openblas_get_parallel') == 2

    def get(self):
        return self.call('omp_get_max_threads' if self.openmp else
                         'openblas_get_num_threads')

    def set(self, num_threads):
        self.call('omp_set_num_threads' if self.openmp else
                  'openblas_set_num_threads', num_threads)


class _BLIS(_Library):
    name = 'blis'
    filename_prefixes = 'libblis', 'libblas'
    check_symbols = 'bli_thread_set_num_threads',

    def get(self):
        # -1 means BLIS's default, which is single-threaded
        num_threads = self.call('bli_thread_get_num_threads')
        return 1 if num_threads == -1 else num_threads

    def set(self, num_threads):
        self.call('bli_thread_set_num_threads', num_threads)


class _FlexiBLAS(_Library):
    name = 'flexiblas'
    filename_prefixes = 'libflexiblas',
    check_symbols = 'flexiblas_set_num_threads',

    def get(self):
        num_threads = self.call('flexiblas_get_num_threads')
        return 1 if num_threads == -1 else num_threads

    def set(self, num_threads):
        self.call('flexiblas_set_num_threads', num_threads)


class _MKL(_Library):
    name = 'mkl'
    filename_prefixes = 'libmkl_rt', 'mkl_rt', 'libblas'
    check_symbols = 'MKL_Set_Num_Threads',

    def get(self):
        return self.call('MKL_Get_Max_Threads')

    def set(self, num_threads):
        self.call('MKL_Set_Num_Threads', num_threads)

    def limit(self, num_threads):
        # A thread-local limit (e.g. one left behind by threadpoolctl 3.7.0+)
        # overrides the process-wide one, so clear the calling thread's for
        # the duration (0 means "none"). Clearing it returns its old value.
        local_num_threads = self.call('MKL_Set_Num_Threads_Local', 0)
        return super().limit(num_threads), local_num_threads

    def restore(self, state):
        num_threads, local_num_threads = state
        super().restore(num_threads)
        if local_num_threads is not None:
            self.call('MKL_Set_Num_Threads_Local', local_num_threads)


class _Accelerate(_Library):
    # Apple's Accelerate. `BLASSetThreading()` (macOS 15+) is thread-local, so
    # unlike the other libraries here, the limit only applies to the calling
    # thread: Accelerate calls made from brisc's OpenMP worker threads still
    # use Accelerate's own thread pool.
    name = 'accelerate'
    filename_prefixes = 'accelerate', 'veclib'
    check_symbols = 'BLASSetThreading',
    # Values of the `BLAS_THREADING` enum in vecLib's thread_api.h
    MULTI_THREADED, SINGLE_THREADED = 0, 1

    def get(self):
        return self.call('BLASGetThreading')

    def set(self, num_threads):
        self.call('BLASSetThreading', self.SINGLE_THREADED
                  if num_threads == 1 else self.MULTI_THREADED)

    def restore(self, state):
        if state is not None:
            self.call('BLASSetThreading', state)


class _OpenMP(_Library):
    user_api = name = 'openmp'
    filename_prefixes = 'libiomp', 'libgomp', 'libomp', 'vcomp'
    check_symbols = 'omp_set_num_threads',

    def get(self):
        return self.call('omp_get_max_threads')

    def set(self, num_threads):
        self.call('omp_set_num_threads', num_threads)


_LIBRARY_TYPES = _OpenBLAS, _BLIS, _FlexiBLAS, _MKL, _Accelerate, _OpenMP


class _LibraryFinder:
    """
    Find the supported libraries loaded into this process. Adapted from
    threadpoolctl's `ThreadpoolController`, keeping only Linux, macOS and
    Windows support.
    """
    # Class-level cache of libc and Windows system libraries, which are very
    # unlikely to be unloaded and reloaded during the lifetime of a program
    _system_libraries = {}

    def __init__(self):
        self.lib_controllers = []
        if sys.platform == 'linux':
            # Not `ctypes.util.dllist()` or `dl_iterate_phdr()`: importing
            # `ctypes.util` on CPython 3.14 isn't fork-safe with some libffi
            # builds, and `dl_iterate_phdr()` (which `dllist()` uses) can
            # deadlock with the GIL
            if os.path.exists('/proc/self/maps'):
                self._find_libraries_with_linux()
            else:
                warnings.warn(
                    '/proc/self/maps does not exist, so BLAS and OpenMP '
                    'thread limits will not be set', RuntimeWarning)
            return
        try:
            from ctypes.util import dllist  # Python 3.14+
        except ImportError:
            dllist = None
        if dllist is not None:
            self._find_libraries_with_python(dllist)
        elif sys.platform == 'darwin':
            self._find_libraries_with_dyld()
        elif sys.platform == 'win32':
            self._find_libraries_on_windows()

    def _find_libraries_with_linux(self):
        with open('/proc/self/maps') as f:
            maps = f.read()
        filepaths = set()
        for line in maps.splitlines():
            start_index = line.find('/')
            if start_index == -1 or '.so' not in line:
                continue
            filepath = line[start_index:]
            if os.path.exists(filepath):
                filepaths.add(filepath)
        for filepath in filepaths:
            self._make_controller_from_path(filepath)

    def _find_libraries_with_python(self, dllist):
        try:
            filepaths = dllist()
        except OSError as exc:
            warnings.warn(
                f'ctypes.util.dllist() failed to list loaded libraries '
                f'({exc!r}), so BLAS and OpenMP thread limits will not be set',
                RuntimeWarning)
            return
        if filepaths and filepaths[0] in ('', sys.executable):
            filepaths = filepaths[1:]
        for filepath in filepaths:
            self._make_controller_from_path(filepath)

    def _find_libraries_with_dyld(self):
        libc = self._get_libc()
        if not hasattr(libc, '_dyld_image_count'):
            warnings.warn(
                'could not find _dyld_image_count in the C standard library',
                RuntimeWarning)
            return
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        for i in range(libc._dyld_image_count()):
            self._make_controller_from_path(
                ctypes.string_at(libc._dyld_get_image_name(i)).decode('utf-8'))

    def _find_libraries_on_windows(self):
        # Prefer `CreateToolhelp32Snapshot()`, which lists the loaded modules
        # atomically, and so is more robust than `EnumProcessModulesEx()` when
        # another thread loads or unloads a DLL at the same time. Use snapshot
        # paths as-is unless empty or possibly truncated to `MAX_PATH`, in
        # which case look them up instead.
        from ctypes.wintypes import MAX_PATH
        ps_api = self._get_windll('Psapi')
        kernel_32 = self._get_windll('kernel32')
        self._setup_windows_module_apis(ps_api, kernel_32)
        h_process = kernel_32.GetCurrentProcess()
        path_buf = ctypes.create_unicode_buffer(
            _WINDOWS_MAX_LIBRARY_PATH_LENGTH)
        try:
            modules = self._snapshot_loaded_modules(kernel_32)
        except OSError:
            modules = None
        if modules is None:
            self._find_libraries_with_enum_process_modules_ex(
                ps_api, kernel_32, h_process, path_buf)
            return
        for h_module, snapshot_path in modules:
            if snapshot_path and len(snapshot_path) < MAX_PATH - 1:
                filepath = snapshot_path
            else:
                filepath = self._resolve_module_filepath(
                    ps_api, kernel_32, h_process, h_module, path_buf)
            if filepath is not None:
                self._make_controller_from_path(filepath)

    @classmethod
    def _setup_windows_module_apis(cls, ps_api, kernel_32):
        from ctypes.wintypes import BOOL, DWORD, HANDLE, HMODULE
        if getattr(cls, '_windows_module_apis_configured', False):
            return
        kernel_32.GetCurrentProcess.restype = HANDLE
        kernel_32.CreateToolhelp32Snapshot.argtypes = [DWORD, DWORD]
        kernel_32.CreateToolhelp32Snapshot.restype = HANDLE
        kernel_32.Module32FirstW.argtypes = [HANDLE, ctypes.c_void_p]
        kernel_32.Module32FirstW.restype = BOOL
        kernel_32.Module32NextW.argtypes = [HANDLE, ctypes.c_void_p]
        kernel_32.Module32NextW.restype = BOOL
        kernel_32.GetModuleFileNameW.argtypes = \
            [HMODULE, ctypes.c_wchar_p, DWORD]
        kernel_32.GetModuleFileNameW.restype = DWORD
        kernel_32.CloseHandle.argtypes = [HANDLE]
        kernel_32.CloseHandle.restype = BOOL
        ps_api.EnumProcessModulesEx.argtypes = \
            [HANDLE, ctypes.POINTER(HMODULE), DWORD, ctypes.POINTER(DWORD),
             DWORD]
        ps_api.EnumProcessModulesEx.restype = BOOL
        ps_api.GetModuleFileNameExW.argtypes = \
            [HANDLE, HMODULE, ctypes.c_wchar_p, DWORD]
        ps_api.GetModuleFileNameExW.restype = DWORD
        cls._windows_module_apis_configured = True

    def _snapshot_loaded_modules(self, kernel_32):
        # Return the loaded modules as `(hModule, snapshot_path)` pairs.
        # Retry `ERROR_BAD_LENGTH`, the documented transient error when the
        # module list changes mid-snapshot, a bounded number of times, then
        # raise `OSError` so the caller can fall back.
        from ctypes.wintypes import DWORD, HANDLE, MAX_PATH

        class MODULEENTRY32W(ctypes.Structure):
            _fields_ = [
                ('dwSize', DWORD),
                ('th32ModuleID', DWORD),
                ('th32ProcessID', DWORD),
                ('GlblcntUsage', DWORD),
                ('ProccntUsage', DWORD),
                ('modBaseAddr', ctypes.POINTER(ctypes.c_byte)),
                ('modBaseSize', DWORD),
                ('hModule', HANDLE),
                ('szModule', ctypes.c_wchar * 256),
                ('szExePath', ctypes.c_wchar * MAX_PATH)]

        TH32CS_SNAPMODULE = 0x00000008
        TH32CS_SNAPMODULE32 = 0x00000010
        ERROR_BAD_LENGTH = 0x0018
        ERROR_NO_MORE_FILES = 0x0012
        INVALID_HANDLE_VALUE = HANDLE(-1).value
        for _ in range(16):
            snap_handle = kernel_32.CreateToolhelp32Snapshot(
                TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, os.getpid())
            if snap_handle != INVALID_HANDLE_VALUE:
                break
            error = ctypes.get_last_error()
            if error != ERROR_BAD_LENGTH:
                raise OSError(f'CreateToolhelp32Snapshot failed: '
                              f'{ctypes.FormatError(error).strip()}')
        else:
            raise OSError(f'CreateToolhelp32Snapshot failed: '
                          f'{ctypes.FormatError(ERROR_BAD_LENGTH).strip()}')
        modules = []
        try:
            lib_entry = MODULEENTRY32W()
            lib_entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
            if not kernel_32.Module32FirstW(snap_handle,
                                            ctypes.byref(lib_entry)):
                error = ctypes.get_last_error()
                raise OSError(f'Module32FirstW failed: '
                              f'{ctypes.FormatError(error).strip()}')
            while True:
                modules.append((lib_entry.hModule, lib_entry.szExePath))
                if not kernel_32.Module32NextW(snap_handle,
                                               ctypes.byref(lib_entry)):
                    error = ctypes.get_last_error()
                    if error != ERROR_NO_MORE_FILES:
                        raise OSError(f'Module32NextW failed: '
                                      f'{ctypes.FormatError(error).strip()}')
                    break
        finally:
            kernel_32.CloseHandle(snap_handle)
        return modules

    def _resolve_module_filepath(self, ps_api, kernel_32, h_process, h_module,
                                 path_buf):
        # Return a module's full path, or `None` if it can't be found
        from ctypes.wintypes import MAX_PATH
        num_chars = kernel_32.GetModuleFileNameW(h_module, path_buf, MAX_PATH)
        if num_chars and num_chars < MAX_PATH - 1:
            return path_buf.value
        num_chars = ps_api.GetModuleFileNameExW(
            h_process, h_module, path_buf, _WINDOWS_MAX_LIBRARY_PATH_LENGTH)
        if num_chars and num_chars < _WINDOWS_MAX_LIBRARY_PATH_LENGTH - 1:
            return path_buf.value
        if num_chars:
            warnings.warn(
                f'ignoring a library whose path is too long, so BLAS and '
                f'OpenMP thread limits may not be set for it: '
                f'{path_buf.value!r}...', RuntimeWarning)
        else:
            warnings.warn(
                f'ignoring a library whose path could not be found, so BLAS '
                f'and OpenMP thread limits may not be set for it: '
                f'{ctypes.FormatError(ctypes.get_last_error()).strip()}',
                RuntimeWarning)
        return None

    def _find_libraries_with_enum_process_modules_ex(
            self, ps_api, kernel_32, h_process, path_buf):
        # The fallback when `CreateToolhelp32Snapshot()` fails. Adapted from
        # https://stackoverflow.com/questions/17474574 by @phihag.
        from ctypes.wintypes import DWORD, HMODULE
        LIST_LIBRARIES_ALL = 0x03
        buf_count = 256
        needed = DWORD()
        # Grow the buffer until it holds every module's handle
        while True:
            buf = (HMODULE * buf_count)()
            buf_size = ctypes.sizeof(buf)
            if not ps_api.EnumProcessModulesEx(
                    h_process, buf, buf_size, ctypes.byref(needed),
                    LIST_LIBRARIES_ALL):
                raise OSError('EnumProcessModulesEx failed')
            if buf_size >= needed.value:
                break
            buf_count = needed.value // (buf_size // buf_count)
        count = needed.value // (buf_size // buf_count)
        for h_module in map(HMODULE, buf[:count]):
            filepath = self._resolve_module_filepath(
                ps_api, kernel_32, h_process, h_module, path_buf)
            if filepath is not None:
                self._make_controller_from_path(filepath)

    def _make_controller_from_path(self, filepath):
        # Resolve symlinks, and lowercase the filename since Windows' OpenMP
        # DLL may be named vcomp, VCOMP, Vcomp, etc.
        filepath = os.path.realpath(filepath)
        filename = os.path.basename(filepath).lower()
        for controller_class in _LIBRARY_TYPES:
            prefix = next((prefix for prefix in
                           controller_class.filename_prefixes
                           if filename.startswith(prefix)), None)
            if prefix is None:
                continue
            # conda-forge used to expose OpenBLAS, BLIS and MKL on Windows as
            # `libblas.dll`, so tell them apart by their symbols. Other
            # `libblas` libraries lack the symbols needed to control them, and
            # would be duplicates.
            if prefix == 'libblas':
                if not filename.endswith('.dll'):
                    continue
                libblas = ctypes.CDLL(filepath, _RTLD_NOLOAD)
                if not any(hasattr(libblas, symbol)
                           for symbol in controller_class.check_symbols):
                    continue
            if filepath in (library.filepath
                            for library in self.lib_controllers):
                continue
            # A library with a supported prefix but none of the expected
            # symbols is a different library that happens to share the prefix
            library = controller_class(filepath=filepath, prefix=prefix)
            if any(hasattr(library.dynlib, symbol)
                   for symbol in controller_class.check_symbols):
                self.lib_controllers.append(library)

    @classmethod
    def _get_libc(cls):
        # `dlopen(NULL)` rather than `ctypes.util.find_library('c')`; see
        # `__init__()`. If libc is statically linked, the main program still
        # exports the libc symbols we need.
        libc = cls._system_libraries.get('libc')
        if libc is None:
            libc = ctypes.CDLL(None, mode=_RTLD_NOLOAD)
            cls._system_libraries['libc'] = libc
        return libc

    @classmethod
    def _get_windll(cls, dll_name):
        dll = cls._system_libraries.get(dll_name)
        if dll is None:
            dll = ctypes.WinDLL(f'{dll_name}.dll', use_last_error=True)
            cls._system_libraries[dll_name] = dll
        return dll


class threadpool_limits:
    """
    A context manager that limits BLAS and/or OpenMP libraries to `limits`
    threads, and restores their previous limits on exit.

    Args:
        limits: the maximum number of threads
        user_api: `'blas'` to limit only BLAS libraries, `'openmp'` to limit
                  only OpenMP libraries (which can also affect BLAS libraries
                  that use OpenMP), or `None` to limit both
    """
    def __init__(self, limits, user_api=None):
        if user_api not in ('blas', 'openmp', None):
            error_message = (
                f"user_api must be 'blas', 'openmp' or None, but is "
                f"{user_api!r}")
            raise ValueError(error_message)
        self._states = [
            (library, library.limit(limits))
            for library in _LibraryFinder().lib_controllers
            if user_api is None or library.user_api == user_api]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        # Restore in reverse order, in case one library is loaded under two
        # paths
        for library, state in reversed(self._states):
            library.restore(state)


def _bundling_package(filepath):
    """
    Return the package whose wheel bundles this shared library, or `None` if
    it isn't bundled with one. auditwheel (Linux) and delvewheel (Windows)
    bundle libraries in `site-packages/<package>.libs`, and delocate (macOS)
    in `site-packages/<package>/.dylibs`.
    """
    parts = Path(filepath).parts
    for index, part in enumerate(parts):
        if part.endswith('.libs'):
            return part[:-len('.libs')]
        if part == '.dylibs' and index > 0:
            return parts[index - 1]
    return None


@cache
def brisc_blas():
    """
    Return the BLAS library brisc calls: `'openblas'`, `'mkl'`, `'blis'`,
    `'flexiblas'` or `'accelerate'`, or `None` if it's none of these.

    brisc calls BLAS via `scipy.linalg.cython_blas`, so this is SciPy's BLAS.
    Other packages may load other BLAS libraries (e.g. pip's NumPy bundles its
    own OpenBLAS), so pick, in order of preference:
    1. a BLAS bundled in SciPy's own wheel (pip's SciPy bundles OpenBLAS)
    2. a BLAS not bundled with any package (e.g. conda's or the system's),
       other than Accelerate, which macOS system frameworks may load even
       when SciPy doesn't use it
    3. Accelerate
    """
    # Make sure SciPy's BLAS is loaded
    import scipy.linalg.cython_blas  # noqa: F401
    blas = [library for library in _LibraryFinder().lib_controllers
            if library.user_api == 'blas']
    unbundled = [library for library in blas
                 if _bundling_package(library.filepath) is None]
    for candidates in (
            [library for library in blas
             if _bundling_package(library.filepath) == 'scipy'],
            [library for library in unbundled
             if library.name != 'accelerate'],
            [library for library in unbundled
             if library.name == 'accelerate']):
        names = {library.name for library in candidates}
        if len(names) == 1:
            return names.pop()
        if len(names) > 1:
            error_message = (
                'could not tell which BLAS library SciPy uses, since several '
                'are loaded: ' + ', '.join(
                    f'{library.name} ({library.filepath})'
                    for library in candidates))
            raise RuntimeError(error_message)
    return None
