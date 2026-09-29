from __future__ import annotations

# Python utility functions

import ctypes
import h5py
import importlib
import mmap
import numpy as np
import os
import platform
import polars as pl
import re
import signal
import sys
import warnings
from collections.abc import Iterable
from functools import reduce
from pathlib import Path
from typing import Any, Literal, NamedTuple, Sequence, Union


FLOAT_DTYPES = pl.Float16, pl.Float32, pl.Float64
INTEGER_DTYPES = pl.Int8, pl.UInt8, pl.Int16, pl.UInt16, pl.Int32, pl.UInt32, \
    pl.Int64, pl.UInt64, pl.Int128, pl.UInt128
NUMERIC_DTYPES = FLOAT_DTYPES + INTEGER_DTYPES


def import_cython(imports: dict[str, Union[str, Sequence[str]]]) -> None:
    """
    Dynamically import Cython functions into the caller's namespace based on
    CPU architecture.

    Args:
        imports: a dictionary mapping module names to a function name or
                 sequence of function names to import
    """
    variant_name = ''
    machine = platform.machine().lower()
    if machine in ('x86_64', 'amd64'):
        from numpy._core._multiarray_umath import __cpu_features__ as cpu
        variant_name = 'x86_64_v4' if cpu.get('AVX512F') else \
            'x86_64_v3' if cpu.get('AVX2') else 'x86_64_v2'
    elif machine == 'aarch64':
        from numpy._core._multiarray_umath import __cpu_features__ as cpu
        if cpu.get('SVE'):
            variant_name = 'sve'
    caller_globals = sys._getframe(1).f_globals
    prefix = f'brisc.{variant_name}' if variant_name else 'brisc'
    for module_name, function_names in imports.items():
        module = importlib.import_module(f'{prefix}.{module_name}')
        if isinstance(function_names, str):
            caller_globals[function_names] = getattr(module, function_names)
        else:
            for function_name in function_names:
                caller_globals[function_name] = getattr(module, function_name)


import_cython({'cyutils': (
    'bin_count', 'concatenate_dense', 'concatenate_indptrs_int32',
    'concatenate_indptrs_int64', 'csr_hstack',
    'get_count_at_least_threshold_csr', 'getnnz_csr',
    'getnnz_at_least_threshold_csr', 'greater_than_or_equal',
    'parallel_subset_1d_cython', 'parallel_subset_2d_cython',
    'weighted_bin_count')})


def array_equal(a1: np.ndarray, a2: np.ndarray) -> bool:
    """
    Tests whether two NumPy arrays are equal. NaNs will always compare equal.

    Args:
        a1: the first input array
        a2: the second input array

    Returns:
        Whether the two arrays are equal.
    """
    return np.array_equal(a1, a2,
                          equal_nan=a1.dtype != object and a2.dtype != object)


def bincount(x: np.ndarray[np.dtype[np.uint32]],
             *,
             num_bins: int | np.integer,
             num_threads: int | np.integer,
             counts: np.ndarray[np.dtype[np.uint32]] | None = None) -> \
        np.ndarray[np.uint32]:
    """
    A faster version of `numpy.bincount` for integer data that uses a fixed
    number of bins, lacks weights and supports multithreading.

    Args:
        x: a 1D `uint32` NumPy array
        num_bins: the number of bins
        num_threads: the number of threads to use when counting
        counts: an optional preallocated array to store the bin counts in;
                assumed to be the correct size

    Returns:
        A 1D `uint32` NumPy array of length `num_bins`, containing the bin
        counts.
    """
    if counts is None:
        counts = np.empty(num_bins, dtype=np.uint32)
    bin_count(arr=x, counts=counts, num_threads=num_threads)
    return counts


def bitview(arr: np.ndarray) -> np.ndarray:
    """
    Creates a zero-copy view of a NumPy array as unsigned integers based on its
    bit width.

    This facilitates Cython "type erasure", allowing memory-movement operations
    to be compiled for just two fused types (uint32 and uint64) rather than
    generating separate machine code for every possible 32- and 64-bit numeric
    type (float32, float64, int32, int64, uint32, uint64), vastly reducing
    Cython compilation time and binary size.

    Args:
        arr: the input array

    Returns:
        A view of the input array as `np.uint32` (if 32-bit) or `np.uint64`
        (if 64-bit).
    """
    itemsize = arr.itemsize
    if itemsize == 4:
        return arr.view(np.uint32)
    elif itemsize == 8:
        return arr.view(np.uint64)
    else:
        error_message = (
            f'X has unsupported type {arr.dtype!r} (size {itemsize} bytes); '
            'use .cast_X(dtype) to cast to a 32- or 64-bit data type')
        raise TypeError(error_message)


def bonferroni(pvalues: pl.Expr) -> pl.Expr:
    """
    Performs Bonferroni correction on a polars expression of p-values.

    Args:
        pvalues: a polars expression; may contain missing data

    Returns:
        A polars expression of Bonferroni-corrected p-values.
    """
    return (pvalues * (pvalues.len() - pvalues.null_count()))\
        .clip(upper_bound=1)


def cast_to_Enum(series: pl.Series, enum_type: pl.Enum) -> pl.Series:
    """
    Cast a polars Enum or Categorical Series to a new Enum type. All values
    present in the Series must be categories of the new Enum type (not
    checked); values that are not become `null`.

    Args:
        series: a polars Series of Enum or Categorical data type
        enum_type: the Enum data type to cast to

    Returns:
        The cast Series.
    """
    if series.dtype != pl.Enum:
        return series.cast(enum_type)

    # Look up each old category's code in the new Enum; unused old categories
    # that are absent from the new Enum map to `null`
    lookup = \
        series.dtype.categories.cast(enum_type, strict=False).to_physical()
    return lookup.gather(series.to_physical()).cat.to(enum_type)\
        .alias(series.name)


def check_bounds(variable: Any,
                 variable_name: str,
                 lower_bound: int | np.integer | None = None,
                 upper_bound: int | np.integer | None = None,
                 *,
                 left_open: bool = False,
                 right_open: bool = False) -> None:
    """
    Check whether `variable` is between lower bound and upper bound, inclusive.

    Args:
        variable: the variable to be checked
        variable_name: the name of the variable, used in the error message
        lower_bound: the smallest allowed value for variable, or None to have
                     no lower bound
        upper_bound: the largest allowed value for variable, or None to have no
                     upper bound
        left_open: if True, require variable to be strictly greater than
                   lower_bound, rather than >= lower_bound; has no effect if
                   lower_bound is None
        right_open: if True, require variable to be strictly less than
                    upper_bound, rather than <= upper_bound; has no effect if
                    upper_bound is None
    """
    if lower_bound is not None and (variable <= lower_bound if left_open
                                    else variable < lower_bound) or \
            upper_bound is not None and (variable >= upper_bound if right_open
                                         else variable > upper_bound):
        error_message = f'{variable_name} is {variable:,}, but must be'
        if lower_bound is not None:
            error_message += f' {">" if left_open else "≥"} {lower_bound:,}'
            if upper_bound is not None:
                error_message += ' and'
        if upper_bound is not None:
            error_message += f' {"<" if right_open else "≤"} {upper_bound:,}'
        raise ValueError(error_message)


def check_dtype(series: pl.Series,
                series_name: str,
                expected_dtypes: pl.datatypes.classes.DataTypeClass | str |
                                 tuple[pl.datatypes.classes.DataTypeClass |
                                       str, ...]) -> None:
    """
    Check whether `series` has the expected polars dtype.

    Args:
        series: the polars Series to be checked
        series_name: the name of the variable, used in the error message
        expected_dtypes: the expected dtype or dtypes. Specify the string
                        `'integer'` to include all integer dtypes, and
                        `'floating-point'` to include all floating-point
                        dtypes.
    """
    base_type = series.dtype.base_type()
    if not isinstance(expected_dtypes, tuple):
        expected_dtypes = expected_dtypes,
    for expected_type in expected_dtypes:
        if base_type == expected_type or expected_type == 'integer' and \
                base_type in INTEGER_DTYPES or \
                expected_type == 'floating-point' and \
                base_type in FLOAT_DTYPES:
            return
    if len(expected_dtypes) == 1:
        expected_dtypes = str(expected_dtypes[0])
    elif len(expected_dtypes) == 2:
        expected_dtypes = ' or '.join(map(str, expected_dtypes))
    else:
        expected_dtypes = ', '.join(map(str, expected_dtypes[:-1])) + \
                          ', or ' + str(expected_dtypes[-1])
    error_message = (
        f'{series_name} must be {expected_dtypes}, but has data type '
        f'{base_type!r}')
    raise TypeError(error_message)


def check_R_variable_name(
        R_variable_name: str,
        variable_name: str,
        R_keywords: set[str] = {
            'if', 'else', 'repeat', 'while', 'function', 'for', 'in', 'next',
            'break', 'TRUE', 'FALSE', 'NULL', 'Inf', 'NaN', 'NA',
            'NA_integer_', 'NA_real_', 'NA_complex_', 'NA_character_',
            '...'}) -> None:
    """
    Check whether `R_variable_name` is a valid variable name in R.

    Args:
        R_variable_name: the R variable name to be checked
        variable_name: the name of the Python variable the R variable name
                       `R_variable_name` is stored in
        R_keywords: the set of R reserved keywords to check against
    """
    if not R_variable_name:
        error_message = f'{variable_name} is an empty string'
        raise ValueError(error_message)
    if R_variable_name[0] == '.':
        if len(R_variable_name) > 1 and R_variable_name[1].isdigit():
            error_message = (
                f'{variable_name} {R_variable_name!r} starts with a period '
                f'followed by a digit, which is not a valid R variable name')
            raise ValueError(error_message)
    elif not R_variable_name[0].isidentifier():
        error_message = (
            f'{variable_name} {R_variable_name!r} must start with a letter, '
            f'number, period or underscore')
        raise ValueError(error_message)
    if not re.fullmatch(r'[\w.]*', R_variable_name[1:]):
        invalid_characters = \
            sorted(set(re.findall(r'[^\w.]',
                                  ''.join(dict.fromkeys(R_variable_name)))))
        if len(invalid_characters) == 1:
            description = f"the character '{invalid_characters[0]}'"
        else:
            description = f"the characters " + ", ".join(
                f"'{character}'" for character in invalid_characters[:-1]) + \
                f" and '{invalid_characters[-1]}'"
        error_message = (
            f'{variable_name} {R_variable_name!r} contains {description}, but '
            f'must contain only letters, numbers, periods and underscores')
        raise ValueError(error_message)
    if R_variable_name in R_keywords or (R_variable_name.startswith('..') and
                                         R_variable_name[2:].isdigit()):
        error_message = (
            f'{variable_name} {R_variable_name!r} is a reserved keyword in R, '
            f'and cannot be used as a variable name')
        raise ValueError(error_message)


def check_type(variable: Any, variable_name: str,
               expected_types: type | tuple[type, ...],
               expected_type_name: str) -> None:
    """
    Check whether `variable` has the expected type.

    Args:
        variable: the variable to be checked
        variable_name: the name of the variable, used in the error message
        expected_types: the expected type or types (specifying int, float, or
                        bool also implicitly includes their NumPy equivalents)
        expected_type_name: the name of the expected type, used in the error
                            message (e.g. 'a polars DataFrame')
    """
    if isinstance(variable, expected_types):
        return
    if not isinstance(expected_types, tuple):
        expected_types = expected_types,
    for t in expected_types:
        if t is int:
            if isinstance(variable, np.integer):
                return
        elif t is float:
            if isinstance(variable, np.floating):
                return
        elif t is bool:
            if isinstance(variable, np.bool_):
                return
    error_message = (
        f'{variable_name} must be {expected_type_name}, but has type '
        f'{type(variable).__name__!r}')
    raise TypeError(error_message)


def check_types(variable: Iterable[Any],
                variable_name: str,
                expected_types: type | tuple[type, ...],
                expected_type_name: str):
    """
    Check whether all elements of `variable` are of the expected type(s).

    Args:
        variable: the variable to be checked
        variable_name: the name of the variable, used in the error message
        expected_types: the expected type or types
        expected_type_name: the name of the expected type, used in the error
                            message (e.g. 'polars DataFrames')
    """
    if not isinstance(expected_types, tuple):
        expected_types = expected_types,
    for element in variable:
        if not isinstance(element, expected_types):
            for t in expected_types:
                if t is int:
                    if isinstance(element, np.integer):
                        break
                elif t is float:
                    if isinstance(element, np.floating):
                        break
                elif t is bool:
                    if isinstance(element, np.bool_):
                        break
            else:
                error_message = (
                    f'all elements of {variable_name} must be '
                    f'{expected_type_name}, but it contains an element of '
                    f'type {type(element).__name__!r}')
                raise TypeError(error_message)


def concatenate(arrays: Sequence[np.ndarray] | Sequence[np.ndarray],
                *,
                num_threads: int,
                output_dtype: np.typing.DTypeLike | None = None) -> \
        np.ndarray | np.ndarray:
    """
    Concatenate 1D or 2D C-contiguous dense arrays.

    Equivalent to `np.concatenate(axis=0)`, but supports multithreading.

    Args:
        arrays: the arrays to concatenate
        num_threads: the number of threads to use when concatenating
        output_dtype: an optional dtype to enforce for the output array. If
                      not specified, will be inferred via `np.result_type()`

    Returns:
        The concatenated array.
    """
    if output_dtype is None:
        output_dtype = np.result_type(*(array.dtype for array in arrays))
    return concatenate_dense([array.astype(output_dtype, copy=False)
                              for array in arrays], num_threads)


def fdr(pvalues: pl.Expr) -> pl.Expr:
    """
    Performs FDR correction on a polars expression of p-values.

    Args:
        pvalues: a polars expression; may contain missing data

    Returns:
        A polars expression of FDR q-values.
    """
    num_null = pvalues.null_count().cast(pl.Int64)
    num_non_null = pvalues.len() - pvalues.null_count()
    reverse_order = pvalues.arg_sort(descending=True, nulls_last=True)
    return (pvalues.gather(reverse_order) /
            (pl.int_range(num_non_null, -num_null, -1) / num_non_null))\
        .cum_min()\
        .gather(reverse_order.arg_sort())


def filter_columns(df: pl.DataFrame,
                   predicates: pl.Expr,
                   *more_predicates: pl.Expr) -> pl.DataFrame:
    """
    Selects columns from a polars DataFrame where all the Boolean expressions
    in `predicates` evaluate to `True`, like `filter()` but for columns instead
    of rows. Use it in method chains, e.g.
    `df.pipe(filter_columns, pl.all().n_unique() > 1)`.

    Args:
        df: a polars DataFrame
        predicates: the Boolean expressions to filter on
        *more_predicates: additional Boolean expressions, specified as
                          positional arguments

    Returns:
        `df`, filtered to the columns where all the Boolean expressions in
        `predicates` evaluate to `True`.
    """
    predicates = to_tuple(predicates) + more_predicates
    boolean_expression = reduce(lambda a, b: a & b, predicates)
    return df.pipe(lambda df: df.select(df.select(boolean_expression)
                                        .unpivot()
                                        .filter(pl.col.value)
                                        ['variable']
                                        .to_list()))


def generate_palette(num_colors: int | np.integer,
                     *,
                     lightness_range: tuple[
                         int | float | np.integer | np.floating,
                         int | float | np.integer | np.floating] =
                        (100 / 3, 200 / 3),
                     chroma_range: tuple[
                         int | float | np.integer | np.floating,
                         int | float | np.integer | np.floating] = (50, 100),
                     hue_range: tuple[
                        int | float | np.integer | np.floating,
                        int | float | np.integer | np.floating] | None = None,
                     first_color: str = '#008cb9',
                     stride: int | np.integer = 5) -> \
        np.ndarray[np.dtype[np.float32]]:
    """
    Generate a maximally perceptually distinct color palette.

    The first color in the palette is `first_color`. The second color is the
    color that's most perceptually distinct from `first_color`, i.e. has the
    largest distance from it in the perceptually uniform CAM02-UCS color space.
    The third color is the color that has the largest distance from either of
    the first two colors, i.e. the color that maximizes the minimum distance
    to any of the colors currently in the palette. And so on.

    An optimized version of github.com/taketwo/glasbey that only generates R,
    G, and B values of (0, 5, 10, ..., 255) instead of (0, 1, 2, ..., 255).
    You can change this stride (by default 5) with the `stride` paramter.

    Args:
        num_colors: the number of colors to include in the palette
        lightness_range: a two-element tuple with the lightness range of colors
                         to generate, or None to take the full range:
                         `(0, 100)`
        chroma_range: a two-element tuple with the chroma range of colors to
                      generate, or None to take the full range: `(0, 100)`.
                      Grays have low chroma, and vivid colors have high chroma.
        hue_range: a two-element tuple with the hue range of colors to
                   generate, or None to take the full range: `(0, 360)`. Red is
                   at 0°, green at 120°, and blue at 240°. Because it wraps
                   around, the first element of the tuple can be greater than
                   the second, unlike for `lightness_range` and `chroma_range`.
        first_color: the first color of the palette. Can be any valid
                     Matplotlib color, like a hex string (e.g. `'#FF0000'`), a
                     named color (e.g. 'red'), a 3- or 4-element RGB/RGBA tuple
                     of integers 0-255 or floats 0-1, or a single float 0-1 for
                     grayscale. Transparency will be ignored!
        stride: as an optimization, consider only RGB colors where R, G, and B
                are all multiples of this value. Must be a small divisor of
                255: 1, 3, 5, 15, or 17. Set to 1 for the best possible
                solution, at orders of magnitude more computational cost.

    Returns:
        A 2D NumPy array of RGB triplets for each color in the colormap, with
        `first_color` as the first color.
    """
    from .colorspacious import cspace_convert
    from matplotlib.colors import is_color_like, to_rgb

    # Check ranges
    for argument, argument_name, max_value in (
            (lightness_range, 'lightness_range', 100),
            (chroma_range, 'chroma_range', 100),
            (hue_range, 'hue_range', 360)):
        if argument is not None:
            check_type(argument, argument_name, tuple, 'a two-element tuple')
            if len(argument) != 2:
                error_message = (
                    f'{argument_name} must be a two-element tuple, but has '
                    f'{len(argument):,} elements')
                raise ValueError(error_message)
            for i in range(2):
                check_type(argument[i], f'{argument_name}[i]', (int, float),
                           f'a number between 0 and {max_value}, inclusive')
            if argument[0] < 0:
                error_message = f'{argument_name}[0] must be ≥ 0'
                raise ValueError(error_message)
            if argument[1] > max_value:
                error_message = f'{argument_name}[1] must be ≤ {max_value}'
                raise ValueError(error_message)
            if argument is not hue_range and argument[0] > argument[1]:
                error_message = \
                    f'{argument_name}[0] must be ≤ {argument_name}[1]'
                raise ValueError(error_message)

    # Check that `first_color` is a valid Matplotlib color, and convert it to
    # RGB and then to the perceptually uniform CAM02-UCS color space
    if not is_color_like(first_color):
        error_message = 'first_color is not a valid Matplotlib color'
        raise ValueError(error_message)
    first_color = to_rgb(first_color)
    first_color = cspace_convert(first_color, 'sRGB1', 'CAM02-UCS')
    if lightness_range is not None or chroma_range is not None or \
            hue_range is not None:
        lightness, chroma, hue = \
            cspace_convert(first_color, 'CAM02-UCS', 'JCh')
        if lightness_range is not None and \
                not lightness_range[0] <= lightness <= lightness_range[1]:
            error_message = (
                f'first_color has a lightness of {lightness}, outside the '
                f'specified lightness_range of {lightness_range}')
            raise ValueError(error_message)
        if chroma_range is not None and \
                not chroma_range[0] <= chroma <= chroma_range[1]:
            error_message = (
                f'first_color has a chroma of {chroma}, outside the specified '
                f'chroma_range of {chroma_range}')
            raise ValueError(error_message)
        if hue_range is not None and not (hue_range[0] <= hue <= hue_range[1]
                                          if hue_range[0] <= hue_range[1] else
                                          hue_range[0] <= hue or
                                          hue <= hue_range[1]):
            error_message = (
                f'first_color has a hue of {hue}, outside the specified '
                f'hue_range of {hue_range}')
            raise ValueError(error_message)

    # Check `stride`
    check_type(stride, 'stride', int, 'one of the integers 1, 3, 5, 15, or 17')
    if stride not in (1, 3, 5, 15, 17):
        error_message = 'stride must be 1, 3, 5, 15, or 17'
        raise ValueError(error_message)

    # Generate a lookup table with all possible RGB colors where R, G and B are
    # multiples of 5, encoded in CAM02-UCS space. Table rows correspond to
    # individual RGB colors; columns correspond to J', a', and b' components.
    rgb = np.arange(0, 256, stride)
    colors = np.empty([len(rgb)] * 3 + [3])
    colors[..., 0] = rgb[:, None, None]
    colors[..., 1] = rgb[None, :, None]
    colors[..., 2] = rgb[None, None, :]
    colors = colors.reshape(-1, 3)
    colors = cspace_convert(colors, 'sRGB255', 'CAM02-UCS')

    # Remove colors outside the specified lightness, chroma and/or hue ranges
    if lightness_range is not None or chroma_range is not None or \
            hue_range is not None:
        jch = cspace_convert(colors, 'CAM02-UCS', 'JCh')
        mask = np.ones(len(colors), dtype=bool)
        if lightness_range is not None:
            mask &= (jch[:, 0] >= lightness_range[0]) & \
                    (jch[:, 0] <= lightness_range[1])
        if chroma_range is not None:
            mask &= (jch[:, 1] >= chroma_range[0]) & \
                    (jch[:, 1] <= chroma_range[1])
        if hue_range is not None:
            if hue_range[0] <= hue_range[1]:
                mask &= (jch[:, 2] >= hue_range[0]) & \
                        (jch[:, 2] <= hue_range[1])
            else:
                mask &= (jch[:, 2] >= hue_range[0]) | \
                        (jch[:, 2] <= hue_range[1])
        colors = colors[mask]

    # Initialize the palette to `first_color`, then iteratively add the color
    # that's farthest away from all other colors (i.e. with the maximum min
    # distance to any color already in the palette)
    palette = [first_color]
    distances = np.full(len(colors), np.inf)
    while len(palette) < num_colors:
        # Update palette-colors distances to account for the color just added
        distance_to_newest_color = \
            np.linalg.norm((colors - palette[-1]), axis=1)
        np.minimum(distances, distance_to_newest_color, distances)

        # Add the color with the new maximum distance
        palette.append(colors[distances.argmax()])

    # Convert the generated palette to sRGB1 format
    palette = cspace_convert(palette, 'CAM02-UCS', 'sRGB1')

    # Clip palette to [0, 1], in case some colors are slightly out of bounds
    palette = palette.clip(0, 1)

    # Cast to float32
    palette = palette.astype(np.float32)

    return palette


def get_count_at_least_threshold(
        sparse_matrix: 'csr_array' | 'csc_array',
        threshold: int,
        axis: Literal[0] | Literal[1] | None,
        num_threads: int | np.integer,
        output: np.ndarray[np.dtype[np.bool_]] | None = None) -> \
            np.ndarray[np.dtype[np.bool_]]:
    """
    Create a mask of whether the total sum of values in a sparse array along an
    axis is at least equal to a threshold. Equivalent to
    `np.asarray(sparse_matrix.sum(axis)).flatten() >= threshold`, but performs
    the `>=` internally and avoids intermediate dense arrays.

    Args:
        sparse_matrix: a CSR or CSC sparse array or matrix
        threshold: the threshold explicitly testing for `>=`
        axis: whether to sum the values within each column (`axis=0`), or
              within each row (`axis=1`)
        num_threads: the number of threads to use when counting; only used for
                     CSR when `axis=0` and for CSC when `axis=1`
        output: an optional preallocated array to store the mask in; assumed to
                be the correct size

    Returns:
        The mask, as a 1D array.
    """
    from .sparse import csr_array
    is_csr = isinstance(sparse_matrix, csr_array)
    if axis == is_csr:
        indptr = sparse_matrix.indptr
        data = sparse_matrix.data
        if output is None:
            output = np.empty(len(indptr) - 1, dtype=bool)
        get_count_at_least_threshold_csr(data=data, indptr=indptr,
                                         threshold=threshold, output=output,
                                         num_threads=num_threads)
    else:
        counts = np.zeros(sparse_matrix.shape[is_csr],
                          dtype=sparse_matrix.dtype)
        weighted_bin_count(arr=sparse_matrix.indices,
                           weights=sparse_matrix.data, counts=counts,
                           num_threads=num_threads)
        if output is None:
            output = np.empty(len(counts), dtype=bool)
        greater_than_or_equal(nnz=counts, threshold=threshold, output=output,
                              num_threads=num_threads)
    return output


def getnnz(sparse_matrix: 'csr_array' | 'csc_array',
           axis: Literal[0] | Literal[1] | None,
           num_threads: int | np.integer,
           output: np.ndarray[np.dtype[np.integer]] | None = None) -> \
        np.ndarray[np.dtype[np.integer]]:
    """
    Count the number of stored values in a sparse array along an axis,
    including explicitly stored zeros. Matches the behavior of the
    now-deprecated `getnnz()` function for scipy sparse arrays, but differs
    from `sparse_matrix.count_nonzero()`, which excludes explicit zeros.

    Args:
        sparse_matrix: a CSR or CSC sparse array or matrix
        axis: whether to count the number of stored values within each column
              (`axis=0`), or within each row (`axis=1`)
        num_threads: the number of threads to use when counting; only used for
                     CSR when `axis=0` and for CSC when `axis=1`
        output: an optional preallocated array to store the number of stored
                values in; assumed to be the correct size

    Returns:
        The number of stored values as a 1D array.
    """
    from .sparse import csr_array
    is_csr = isinstance(sparse_matrix, csr_array)
    if axis == is_csr:
        # The code below is equivalent to
        # `output = np.diff(sparse_matrix.indptr)`
        indptr = sparse_matrix.indptr
        if output is None:
            if num_threads == 1:
                output = np.empty(len(indptr) - 1, dtype=np.uint32)
            else:
                output = numa_zeros(len(indptr) - 1, dtype=np.uint32)
        getnnz_csr(indptr=indptr, output=output, num_threads=num_threads)
        return output
    else:
        return bincount(sparse_matrix.indices,
                        num_bins=sparse_matrix.shape[is_csr],
                        num_threads=num_threads, counts=output)


def getnnz_at_least_threshold(
        sparse_matrix: 'csr_array' | 'csc_array',
        threshold: int,
        axis: Literal[0] | Literal[1] | None,
        num_threads: int | np.integer,
        output: np.ndarray[np.dtype[np.integer]] | None = None) -> \
            np.ndarray[np.dtype[np.bool_]]:
    """
    Create a mask of whether the number of non-zero values in a sparse array
    along an axis is at least equal to a threshold, counting explicitly stored
    zeros as non-zero. Equivalent to
    `getnnz(sparse_matrix, axis) >= threshold`, but performs the `>=`
    internally.

    Args:
        sparse_matrix: a CSR or CSC sparse array or matrix
        threshold: the threshold
        axis: whether to count the fraction of non-zero values within each
              column (`axis=0`), or within each row (`axis=1`)
        num_threads: the number of threads to use when counting; only used for
                     CSR when `axis=0` and for CSC when `axis=1`
        output: an optional preallocated array to store the mask in; assumed to
                be the correct size

    Returns:
        The mask, as a 1D array.
    """
    from .sparse import csr_array
    is_csr = isinstance(sparse_matrix, csr_array)
    if axis == is_csr:
        # The code below is equivalent to
        # `output = np.diff(sparse_matrix.indptr) >= threshold`
        indptr = sparse_matrix.indptr
        if output is None:
            output = np.empty(len(indptr) - 1, dtype=bool)
        getnnz_at_least_threshold_csr(
            indptr=indptr, threshold=threshold, output=output,
            num_threads=num_threads)
    else:
        nnz = bincount(sparse_matrix.indices,
                       num_bins=sparse_matrix.shape[is_csr],
                       num_threads=num_threads)
        if output is None:
            output = np.empty(len(nnz), dtype=bool)
        greater_than_or_equal(nnz=nnz, threshold=threshold, output=output,
                              num_threads=num_threads)
    return output


def is_iterable(variable: Any) -> bool:
    """
    Check if a variable is iterable, excluding strings and bytes.

    Safely evaluates to `False` for non-iterables, strings, bytes, and
    objects that define a dummy `__iter__` method that raises a `TypeError`
    (like SingleCell datasets!).

    Args:
        variable: a variable

    Returns:
        `True` if `variable` is an iterable (and not `str` or `bytes`), `False`
        otherwise.
    """
    if isinstance(variable, (str, bytes)):
        return False
    try:
        iter(variable)
        return True
    except TypeError:
        return False


def ix_symmetric(indexer: np.ndarray) -> \
        tuple[np.ndarray[np.dtype[np.integer]],
              np.ndarray[np.dtype[np.integer]]]:
    """
    A faster drop-in replacement for `np.ix_(indexer, indexer)`, where
    `indexer` is 1D and the array it indexes is 2D.

    Args:
        indexer: a NumPy array to use as the indexer

    Returns:
        The indices to index with along each of the two dimensions.
    """
    if indexer.dtype == bool:
        indexer = np.flatnonzero(indexer)
    return indexer[:, None], indexer[None]


# Count NUMA nodes by looking at which node directories exist.
# /sys/devices/system/node/node0, node1, ... are present for each NUMA node.
if sys.platform == 'linux':
    _num_NUMA_nodes = \
        max(1, len(tuple(Path('/sys/devices/system/node').glob('node[0-9]*'))))
    libc = ctypes.CDLL('libc.so.6', use_errno=True)

    # Request huge pages only when the kernel won't compact memory
    # synchronously to satisfy the request (~10x slowdowns on fragmented
    # nodes); with enabled=always it uses free huge pages anyway.
    def _thp_setting(name):
        try:
            match = re.search(r'\[([\w+]+)\]', Path(
                '/sys/kernel/mm/transparent_hugepage', name).read_text())
            return match.group(1) if match else None
        except OSError:
            return None
    _advise_hugepage = _thp_setting('enabled') == 'madvise' and \
        _thp_setting('defrag') in ('defer', 'never')


def numa_zeros(shape: int | np.integer | tuple[int | np.integer, ...],
               dtype: np.typing.DTypeLike = np.float64) -> np.ndarray:
    """
    Drop-in replacement for np.zeros() that guarantees fresh, unmapped virtual
    memory via mmap on Linux. Bypasses glibc's malloc arena to ensure strict
    NUMA first-touch placement.

    Automatically disables Transparent Huge Pages (THP) for small arrays where
    it would cause false sharing at NUMA node boundaries, and enables it for
    large arrays.

    Args:
        shape: the desired shape of the resulting array
        dtype: the desired data-type for the array

    Returns:
        A NumPy array with the desired shape and data type wrapping fresh,
        unmapped virtual memory.
    """
    if sys.platform != 'linux':
        return np.zeros(shape, dtype=dtype)
    dtype = np.dtype(dtype)
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    else:
        shape = tuple(int(s) for s in shape)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if nbytes == 0:
        return np.zeros(shape, dtype=dtype)
    mm = mmap.mmap(-1, nbytes,
                   flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                   prot=mmap.PROT_READ | mmap.PROT_WRITE)

    # Transparent Huge Pages (THP) allocate memory in 2 MB chunks. When a
    # parallel first-touch workload spans multiple NUMA nodes, each 2 MB page
    # can only be assigned to one node. At the boundaries between nodes'
    # regions, one node "wins" the entire 2 MB page, misplacing memory that
    # should belong to the neighboring node.
    #
    # There are (# NUMA nodes - 1) such boundaries, so the worst-case misplaced
    # memory is (# NUMA nodes - 1) * 2 MB. Whether this matters depends on the
    # per-node chunk size: a misplaced 2 MB page within a 200 MB chunk is
    # negligible, but within a 4 MB chunk it's catastrophic (half the node's
    # memory is on the wrong node).
    #
    # We disable THP when any single node's chunk is smaller than a fixed
    # number of huge pages. Below this point, boundary effects dominate and the
    # NUMA locality loss outweighs the TLB savings. At 16 huge pages (32 MB)
    # per node, boundary waste is at most 1/16 = 6.25% of a node's memory,
    # which is a reasonable crossover point: THP's TLB benefit (reducing TLB
    # misses by 512x for each huge page) is modest at this scale, while a
    # 6.25%+ bandwidth penalty from remote NUMA access is easily measurable.
    HUGE_PAGE_SIZE = 2_097_152
    MIN_HUGE_PAGES_PER_NODE = 16
    bytes_per_node = nbytes / _num_NUMA_nodes
    MADV_NOHUGEPAGE = 15
    MADV_HUGEPAGE = 14
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    if bytes_per_node < MIN_HUGE_PAGES_PER_NODE * HUGE_PAGE_SIZE:
        libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes),
                     ctypes.c_int(MADV_NOHUGEPAGE))
    elif _advise_hugepage:
        libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes),
                     ctypes.c_int(MADV_HUGEPAGE))

    return np.frombuffer(mm, dtype=dtype).reshape(shape)


def parallel_subset_1d(array: np.ndarray[np.dtype[np.integer | np.floating]],
                       indices: np.ndarray[np.dtype[np.integer]],
                       num_threads: int) -> \
        np.ndarray[np.dtype[np.integer | np.floating]]:
    """
    Subset a 1D NumPy array to the specified integer indices in parallel.
    Though subsetting in parallel is rarely faster except for massive arrays,
    this function is necessary to ensure optimal memory distribution of the
    subset array on NUMA machines, to avoid memory bandwidth slowdowns on
    subsequent steps.

    Args:
        array: the array to subset
        indices: the indices to subset to
        num_threads: the number of threads to use when subsetting

    Returns:
        `array[indices]`.
    """
    subset_array = numa_zeros(len(indices), dtype=array.dtype)
    parallel_subset_1d_cython(bitview(array), indices, bitview(subset_array),
                              num_threads)
    return subset_array


def parallel_subset_2d(array: np.ndarray[np.dtype[np.integer | np.floating]],
                       indices: np.ndarray[np.dtype[np.integer]],
                       num_threads: int) -> \
        np.ndarray[np.dtype[np.integer | np.floating]]:
    """
    Subset a 2D NumPy array to the specified integer indices in parallel.
    Though subsetting in parallel is rarely faster except for massive arrays,
    this function is necessary to ensure optimal memory distribution of the
    subset array on NUMA machines, to avoid memory bandwidth slowdowns on
    subsequent steps.

    Args:
        array: the array to subset
        indices: the indices to subset to
        num_threads: the number of threads to use when subsetting

    Returns:
        `array[indices]`.
    """
    subset_array = numa_zeros((len(indices), array.shape[1]),
                              dtype=array.dtype)
    parallel_subset_2d_cython(bitview(array), indices, bitview(subset_array),
                              num_threads)
    return subset_array


def plural(string: str, count: int | np.integer) -> str:
    """
    Adds an s to the end of string, unless `count` is 1 or -1.

    Args:
        string: a string
        count: a count

    Returns:
        `string`, with an s at the end if `count` is 1 or -1
    """
    return string if abs(count) == 1 else f'{string}s'


def read_dataset_worker(filename: str,
                        tasks: list[tuple[str, slice, Any]]) -> None:
    """
    Read one or more (ranges of) datasets from an HDF5 file. This function is
    run by each worker process during parallel HDF5 file reading.

    Args:
        filename: the HDF5 filename
        tasks: a list of (name, row range, destination buffer) tuples for each
               dataset to load
    """
    try:
        # To stop all workers from using the same core when `OMP_PLACES=cores`
        # is set, disable CPU pinning in the workers with
        # `os.sched_setaffinity()`. (This works even if cgroups limit the
        # available cores, e.g. on Slurm: the OS will take the intersection of
        # what we asked for and what the cgroup allows.)
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, range(os.cpu_count()))
        with h5py.File(filename, 'r') as hdf5_file:
            for dataset_name, row_slice, buffer in tasks:
                dataset = hdf5_file[dataset_name]
                dest = np.frombuffer(buffer, dtype=dataset.dtype)\
                    .reshape(dataset.shape)
                dataset.read_direct(dest, source_sel=row_slice,
                                    dest_sel=row_slice)
    except OSError as e:
        if getattr(e, 'errno', None) == 14:  # SIGBUS - likely out of memory
            sys.exit(2)  # our special error code for out of memory
        raise
    except KeyboardInterrupt:
        pass


def read_parallel_multiprocessing(
        datasets_to_preload: tuple[list[tuple[str, tuple[int, ...], np.dtype,
                                              int | None]],
                                   list[tuple[str, tuple[int, ...], np.dtype,
                                              int | None]]],
        filename: str,
        *,
        num_threads: int | np.integer) -> dict[str, np.ndarray]:
    """
    Read a sequence of HDF5 datasets into a dictionary of 1D NumPy arrays in
    parallel using the fallback multiprocessing-based reader. This is used when
    any dataset is chunked or compressed, and thus not compatible with the
    basic thread-based reader.

    Args:
        datasets_to_preload: two lists of (name, shape, dtype, file offset)
                             tuples of datasets to preload: the first is for
                             fixed-length datasets, the second for
                             variable-length string datasets.
        filename: the HDF5 filename
        num_threads: the number of worker processes to spawn when reading

    Returns:
        A dictionary of NumPy arrays with the contents of the datasets; the
        keys are taken from each dataset's `name` attribute.
    """
    # Separate the datasets to preload into fixed-length vs variable-length
    fixed_datasets, vlen_datasets = datasets_to_preload

    # If there are no fixed-length datasets to preload, skip preloading
    # entirely
    if len(fixed_datasets) == 0:
        return {}

    # `os.fork()` and the `resource` module are Unix-only, so the fork +
    # shared-memory reader below cannot run on Windows. Read the preload
    # datasets serially there instead.
    if sys.platform == 'win32':
        with h5py.File(filename) as hdf5_file:
            return {name: hdf5_file[name][:]
                    for name, *_ in (*fixed_datasets, *vlen_datasets)}

    import multiprocessing
    import resource
    from multiprocessing.sharedctypes import _new_value

    # Increase the maximum number of possible file descriptors this process
    # can use, since dataframes with hundreds of columns can easily exhaust
    # the common default limit of 1024 file descriptors (`ulimit -n`)
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit < hard_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

    # Subtract 1 from the number of workers if there are string datasets to
    # preload, since these will be loaded in the main process
    if len(vlen_datasets) > 0:
        num_threads -= 1

    # Allocate shared memory for each dataset. Use `_new_value()` instead of
    # `multiprocessing.Array()` to avoid the memset at github.com/python/
    # cpython/blob/main/Lib/multiprocessing/sharedctypes.py#L62
    buffers = {}
    tasks_by_worker = [[] for _ in range(num_threads)]
    processes = None
    try:
        for name, shape, dtype, _ in fixed_datasets:
            buffers[name] = _new_value(
                int(np.prod(shape)) * np.ctypeslib.as_ctypes_type(dtype))

        # Assign (chunks of) datasets for each worker process to read, using a
        # best-fit decreasing (BFD) bin-packing algorithm
        info = []
        total_bytes = 0
        for name, shape, dtype, _ in fixed_datasets:
            rows = shape[0]
            bytes_per_row = int(np.prod(shape[1:])) * dtype.itemsize
            bytes_per_dataset = rows * bytes_per_row
            info.append((name, rows, bytes_per_row, bytes_per_dataset))
            total_bytes += bytes_per_dataset

        target = (total_bytes + num_threads - 1) // num_threads
        target = min(max(target, 4_194_304), 67_108_864)  # 4 MB to 64 MB
        target = (target // 4096) * 4096 or 4096  # multiple of 4096 bytes

        task_records = []
        for name, rows, bytes_per_row, bytes_per_dataset in info:
            if bytes_per_dataset <= target:
                task_records.append((bytes_per_dataset, name, 0, rows))
            else:
                rows_per_chunk = min(max(1, target // bytes_per_row), rows)
                for start in range(0, rows, rows_per_chunk):
                    end = min(start + rows_per_chunk, rows)
                    chunk_bytes = (end - start) * bytes_per_row
                    task_records.append((chunk_bytes, name, start, end))

        task_records.sort(reverse=True)  # sort by decreasing size

        loads = np.zeros(num_threads, dtype=np.int64)
        for size, name, start, end in task_records:
            best_slack = None
            for index, load in enumerate(loads):
                slack = target - (load + size)
                if slack >= 0 and (best_slack is None or slack < best_slack):
                    best_index = index
                    best_slack = slack
            if best_slack is None:  # overflow → lightest bin
                best_index = loads.argmin()
            tasks_by_worker[best_index].append(
                (name, slice(start, end), buffers[name]))
            loads[best_index] += size

        with warnings.catch_warnings():
            # Ignore polars warning about using `os.fork()`
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                    module='multiprocessing')

            # Spawn `num_threads` worker processes, each of which will read
            # their assigned datasets. Because the chunks loaded by each worker
            # process are non-overlapping, there's no need to lock.
            ctx = multiprocessing.get_context('fork')
            processes = []
            for thread_index, task_list in enumerate(tasks_by_worker):
                process = ctx.Process(target=read_dataset_worker,
                                      args=(filename, task_list))
                processes.append(process)
                process.start()

            # While we are waiting for the workers to finish, preload the
            # variable-length string datasets on the main thread. (This could
            # be optimized to use the same string handling as the thread-based
            # reader, but in our experience it's not the bottleneck here.)
            with h5py.File(filename) as hdf5_file:
                arrays = {name: hdf5_file[name][:]
                          for name, shape, dtype, _ in vlen_datasets}

            # Wait until all workers finish
            for process in processes:
                process.join()
                exit_code = process.exitcode
                if exit_code == 2:  # our special exit code for out of memory
                    raise MemoryError
                elif exit_code != 0:
                    error_message = (
                        f'a dataset-reading worker process failed with exit '
                        f'code {exit_code}')
                    raise RuntimeError(error_message)
    except:
        # Avoid shared memory leaks on KeyboardInterrupt or other errors, by
        # clearing all objects that contain references to the shared-memory
        # buffers once all processes exit. The `process.pid is not None` check
        # avoids joining processes that haven't started yet, which is not
        # allowed.
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if processes is not None:
                for process in processes:
                    if process.is_alive():
                        process.kill()
                for process in processes:
                    if process.pid is not None:
                        process.join()
                processes.clear()
            buffers.clear()
            tasks_by_worker.clear()
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)
        raise

    # Wrap the shared memory in NumPy arrays
    arrays.update({
        name: np.frombuffer(buffer, dtype=dtype).reshape(shape)
        for (name, shape, dtype, _), buffer
        in zip(fixed_datasets, buffers.values())})
    return arrays


def sparse_equal(a1: 'csr_array' | 'csc_array',
                 a2: 'csr_array' | 'csc_array') -> bool:
    """
    Tests whether two SciPy sparse arrays or matrices OF THE SAME FORMAT (e.g.
    CSR) are equal. NaNs will always compare equal.

    Args:
        a1: the first input array or matrix
        a2: the second input array or matrix

    Returns:
        Whether the two arrays or matrices are equal.
    """
    return a1.shape == a2.shape and a1.nnz == a2.nnz and \
        array_equal(a1.indptr, a2.indptr) and \
        array_equal(a1.data, a2.data) and array_equal(a1.indices, a2.indices)


def sparse_major_stack(arrays: Sequence['csr_array'] | Sequence['csc_array'],
                       *,
                       num_threads: int) -> 'csr_array' | 'csc_array':
    """
    Concatenate sparse arrays along their major axis. Equivalent to
    `scipy.sparse.vstack()` for CSR arrays and `scipy.sparse.hstack()` for CSC
    arrays, but supports multithreading.

    Args:
        arrays: the sparse arrays to concatenate
        num_threads: the number of threads to use when concatenating. Does not
                     affect the concatenated array's `num_threads`; this will
                     always be the same as the first array's `num_threads`.

    Returns:
        The concatenated sparse array.
    """
    from .sparse import csc_array, csr_array
    first_array = arrays[0]
    is_csr = isinstance(first_array, csr_array)

    # Determine whether the output sparse array can get away with using int32
    # instead of int64 `indices` and `indptr`
    total_nnz = sum(int(array.indptr[-1]) for array in arrays)
    num_minor = first_array.shape[is_csr]
    index_dtype = np.int64 \
        if num_minor > 2_147_483_647 or total_nnz > 2_147_483_647 else np.int32

    # Concatenate `data`, `indices` and `indptr`
    data = concatenate([array.data for array in arrays],
                       num_threads=num_threads)
    indices = concatenate([array.indices for array in arrays],
                          num_threads=num_threads, output_dtype=index_dtype)
    indptrs = [array.indptr.astype(index_dtype, copy=False)
               for array in arrays]
    indptr = concatenate_indptrs_int64(indptrs, num_threads) \
        if index_dtype == np.int64 else \
        concatenate_indptrs_int32(indptrs, num_threads)

    # Construct the final sparse array
    num_major = len(indptr) - 1
    if is_csr:
        shape = num_major, num_minor
        result = csr_array((data, indices, indptr), shape=shape)
    else:
        shape = num_minor, num_major
        result = csc_array((data, indices, indptr), shape=shape)
    result._num_threads = first_array._num_threads
    return result


class _ArrayContainer(NamedTuple):
    # A container used in `sparse_minor_stack()` to hold a sparse array's
    # `data`, `indices` and `indptr` after casting them to the output's dtypes.
    # We don't construct a sparse array here because scipy chooses its own
    # index dtype on construction and may downcast int64 indices back to int32,
    # undoing the cast.
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray


def sparse_minor_stack(arrays: Sequence['csr_array'] | Sequence['csc_array'],
                       *,
                       num_threads: int) -> 'csr_array' | 'csc_array':
    """
    Concatenate sparse arrays along their minor axis. Equivalent to
    `scipy.sparse.hstack()` for CSR arrays and `scipy.sparse.vstack()` for CSC
    arrays, but supports multithreading.

    Args:
        arrays: the sparse arrays to concatenate
        num_threads: the number of threads to use when concatenating. Does not
                     affect the concatenated array's `num_threads`; this will
                     always be the same as the first array's `num_threads`.

    Returns:
        The concatenated sparse array.
    """
    # Get the offset of each array along the minor axis, and the total number
    # of non-zero elements across all arrays
    from .sparse import csc_array, csr_array
    first_array = arrays[0]
    is_csr = isinstance(first_array, csr_array)
    offsets = np.cumsum([0] + [array.shape[is_csr] for array in arrays[:-1]],
                        dtype=np.int64)
    total_nnz = sum(array.nnz for array in arrays)

    # Determine whether the output sparse array can get away with using int32
    # instead of int64 `indices` and `indptr`
    num_minor = int(offsets[-1]) + arrays[-1].shape[is_csr]
    index_dtype = np.int64 \
        if num_minor > 2_147_483_647 or total_nnz > 2_147_483_647 else np.int32

    # Allocate output arrays
    num_major = first_array.shape[not is_csr]
    data_dtype = np.result_type(*(array.dtype for array in arrays))
    data = np.empty(total_nnz, dtype=data_dtype)
    indices = np.empty(total_nnz, dtype=index_dtype)
    indptr = np.empty(num_major + 1, dtype=index_dtype)

    # Cast every input to the output's data types
    arrays = [_ArrayContainer(array.data.astype(data_dtype, copy=False),
                              array.indices.astype(index_dtype, copy=False),
                              array.indptr.astype(index_dtype, copy=False))
              for array in arrays]

    # Perform the concatenation, filling `data`, `indices`, and `indptr`
    csr_hstack(arrays, offsets.astype(index_dtype, copy=False), bitview(data),
               indices, indptr, num_major, num_threads)

    # Construct the final sparse array
    if is_csr:
        shape = num_major, num_minor
        result = csr_array((data, indices, indptr), shape=shape)
    else:
        shape = num_minor, num_major
        result = csc_array((data, indices, indptr), shape=shape)
    result._num_threads = first_array._num_threads
    return result


def to_tuple(variable: Any) -> tuple[Any, ...]:
    """
    Cast Iterables (except str/bytes) to tuple, but box non-Iterables (and
    str/bytes) in a length-1 tuple.

    Args:
        variable: a variable

    Returns:
        `variable` as a tuple
    """
    return tuple(variable) if is_iterable(variable) else (variable,)


def to_tuple_checked(variable: Any,
                     variable_name: str,
                     expected_types: type | tuple[type, ...],
                     expected_type_name: str) -> tuple[Any, ...]:
    """
    Like `to_tuple`, but check that `variable` or its elements are of the
    expected type(s) and that it is non-empty.

    Args:
        variable: the variable to be checked and expanded
        variable_name: the name of the variable, used in error messages
        expected_types: the expected type or types
        expected_type_name: the name of the expected type, used in error
                            messages (e.g. `'polars DataFrames'`)

    Returns:
        `variable` as a tuple.
    """
    if is_iterable(variable):
        variable = tuple(variable)
        if len(variable) == 0:
            error_message = f'{variable_name} is empty'
            raise ValueError(error_message)
        check_types(variable, variable_name, expected_types,
                    expected_type_name)
    else:
        check_type(variable, variable_name, expected_types,
                   f'{expected_type_name} (or a sequence thereof)')
        variable = variable,
    return variable
