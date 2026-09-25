import os
import shutil
import site
import sys
import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

windows = sys.platform == 'win32'
mac = sys.platform == 'darwin'
linux = sys.platform == 'linux'
machine = platform.machine().lower()

if windows:
    # MSVC sets VSCMD_ARG_TGT_ARCH during cross-compilation (e.g. to `'arm64'`)
    win_target = os.environ.get('VSCMD_ARG_TGT_ARCH', machine).lower()
    if win_target in ['x86_64', 'amd64', 'x64']:  # 'x64' is for conda
        variants = [('x86_64_v2', ''),
                    ('x86_64_v3', '/arch:AVX2'),
                    ('x86_64_v4', '/arch:AVX512')]
    elif win_target == 'arm64':
        variants = [('', '')]
    else:
        raise RuntimeError(f'unsupported architecture {win_target!r}')
elif mac:
    # Once x86 Macs become obsolete, remove the if branch
    if os.environ.get('ARCHFLAGS', machine).endswith('x86_64'):
        variants = [('x86_64_v2', '-march=x86-64-v2'),
                    ('x86_64_v3', '-march=x86-64-v3'),
                    ('x86_64_v4', '-march=x86-64-v4')]
    else:
        variants = [('', '')]
elif linux:
    if machine in ['x86_64', 'amd64']:
        variants = [('x86_64_v2', '-march=x86-64-v2'),
                    ('x86_64_v3', '-march=x86-64-v3'),
                    ('x86_64_v4', '-march=x86-64-v4')]
    elif machine == 'aarch64':
        variants = [('', '-march=armv8.2-a'),
                    ('sve', '-march=armv8.2-a+sve')]
    else:
        raise RuntimeError(f'unsupported architecture {machine!r}')
else:
    raise RuntimeError(f'unsupported platform {sys.platform!r}')

extensions = []
source_directory = 'brisc'
source_files = os.listdir(source_directory)
pyx_files = [f for f in source_files if f.endswith('.pyx')]
pxd_files = [f for f in source_files if f.endswith('.pxd')]

try:
    for variant_name, march in variants:
        if windows:
            compiler_flags = [
                '/O2',
                '/fp:fast',
                '/std:c++17', '/openmp:llvm', '/W3', '/WX',
                # Suppress MSVC-specific warnings commonly triggered by
                # Cython's generated C++ code
                '/wd4018',             # signed/unsigned mismatch
                '/wd4060',             # switch statement lacks case/default
                '/wd4127',             # conditional expression is constant
                '/wd4146',             # unary minus applied to unsigned type
                '/wd4244',             # conversion with possible loss of data
                '/wd4267',             # conversion from size_t to int
                '/wd4305',             # truncation from double to float
                '/wd4551',             # function call missing argument list
                '/wd4700', '/wd4701',  # uninitialized variable
                '/wd4723',             # potential divide by 0 (ARM64 pedantry)
                '/wd4996']             # deprecated POSIX names
            linker_flags = ['/OPT:REF', '/OPT:ICF']
        elif mac:
            omp_base = os.environ.get('PREFIX')  # set by conda-build
            if not omp_base:
                # Point to the OpenMP folder downloaded in pyproject.toml
                # Once x86 Macs become obsolete, remove the 'openmp_x86_64'
                # branch and just do `os.path.abspath('openmp')`
                omp_base = os.path.abspath(
                    'openmp_x86_64'
                    if os.environ.get('ARCHFLAGS', machine).endswith('x86_64')
                    else 'openmp')
            compiler_flags = [
                '-O3', '-ffast-math', '-std=c++17', '-g0', '-Xpreprocessor',
                '-fopenmp', '-Wall', '-Wextra', '-Werror',
                '-Wno-uninitialized', '-Wno-ignored-qualifiers',
                '-Wno-unreachable-code', '-fvisibility=hidden',
                '-fvisibility-inlines-hidden', f'-I{omp_base}/include']
            linker_flags = ['-lomp', f'-L{omp_base}/lib',
                            f'-Wl,-rpath,{omp_base}/lib', '-Wl,-S']
        else:
            compiler_flags = [
                '-Ofast', '-funroll-loops', '-std=c++17', '-g0', '-fopenmp',
                '-Wall', '-Wextra', '-Werror', '-Wno-uninitialized',
                '-Wno-ignored-qualifiers', '-Wno-maybe-uninitialized',
                '-fvisibility=hidden', '-fvisibility-inlines-hidden']
            linker_flags = ['-fopenmp', '-s']
        if march:
            compiler_flags.append(march)
        # normalize.pyx must be compiled without fast-math semantics or
        # floating-point contraction to avoid floating-point roundoff
        # differences between vectorized and non-vectorized code paths.
        normalize_compiler_flags = [
            '-O3' if flag == '-Ofast' else '/fp:precise' if flag == '/fp:fast'
            else flag for flag in compiler_flags if flag != '-ffast-math']
        if not windows:
            normalize_compiler_flags.append('-ffp-contract=off')
        if variant_name:
            # Copy each Cython source file to a separate temporary directory
            # for each variant
            variant_dir = os.path.join(source_directory, variant_name)
            os.makedirs(variant_dir, exist_ok=True)
            with open(os.path.join(variant_dir, '__init__.py'), 'w') as f:
                pass
            for f in pyx_files + pxd_files:
                source = os.path.join(source_directory, f)
                destination = os.path.join(variant_dir, f)
                shutil.copyfile(source, destination)
            for pyx in pyx_files:
                base_name = pyx.removesuffix('.pyx')
                ext_name = f'brisc.{variant_name}.{base_name}'
                source_path = os.path.join(variant_dir, pyx)
                extensions.append(Extension(
                    name=ext_name, sources=[source_path],
                    include_dirs=[np.get_include()],
                    define_macros=[('NPY_NO_DEPRECATED_API',
                                    'NPY_1_17_API_VERSION')],
                    extra_compile_args=normalize_compiler_flags \
                        if base_name == 'normalize' else compiler_flags,
                    extra_link_args=linker_flags, language='c++'))
        else:
            for pyx in pyx_files:
                base_name = pyx.removesuffix('.pyx')
                ext_name = f'brisc.{base_name}'
                source_path = os.path.join(source_directory, pyx)
                extensions.append(Extension(
                    name=ext_name, sources=[source_path],
                    include_dirs=[np.get_include()],
                    define_macros=[('NPY_NO_DEPRECATED_API',
                                    'NPY_1_17_API_VERSION')],
                    extra_compile_args=normalize_compiler_flags \
                        if base_name == 'normalize' else compiler_flags,
                    extra_link_args=linker_flags, language='c++'))
    if any(arg in sys.argv for arg in ['egg_info', 'dist_info', 'sdist']):
        # Just a metadata pass; skip transpilation to C++ and just pass the raw
        # Extension objects; setuptools just needs their metadata
        ext_modules = extensions
    else:
        ext_modules = cythonize(extensions, compiler_directives={
            'language_level': '3', 'boundscheck': False, 'wraparound': False,
            'cdivision': True, 'initializedcheck': False,
            'warn.undeclared': False})
        # Prevent setuptools from trying to package system/NumPy headers into
        # the wheel manifest, to avoid warnings about how certain dependencies
        # "won't be automatically included in the manifest: the path must be
        # relative"
        for ext in ext_modules:
            ext.depends = [d for d in ext.depends if not os.path.isabs(d)]
    setup(ext_modules=ext_modules)
finally:
    # Remove the temporary directories
    for variant_name, _ in variants:
        if variant_name:
            variant_dir = os.path.join(source_directory, variant_name)
            shutil.rmtree(variant_dir, ignore_errors=True)
