from __future__ import annotations
import numpy as np
import os
import polars as pl
import re
import shutil
from collections.abc import Iterable
from itertools import islice, pairwise
from pathlib import Path
from scipy.special import stdtrit
from textwrap import fill
from threadpoolctl import threadpool_limits
from typing import Any, Callable, ItemsView, KeysView, Literal, Mapping, \
    NoReturn, Sequence, ValuesView
from .de import DE
from .type_aliases import Indexer, Scalar, SingleCellColumn, PseudobulkColumn
from .utils import array_equal, bonferroni, cast_to_Enum, check_bounds, \
    check_dtype, check_type, check_types, concatenate, fdr, import_cython, \
    plural, to_tuple, to_tuple_checked, FLOAT_DTYPES, INTEGER_DTYPES, \
    NUMERIC_DTYPES
from .validated_dict import Obs, Var, X_
import_cython({'cyutils': 'has_all_zero_columns',
               'norm_factors': 'calc_norm_factors'})


class Pseudobulk:
    """
    A pseudobulked single-cell dataset resulting from calling `pseudobulk()`
    on a SingleCell dataset. Has slots for:

    - `X`: a dict of NumPy arrays of counts per cell and gene for each cell
      type
    - `obs`: a dict of polars DataFrames of sample metadata for each cell type
    - `var`: a dict of polars DataFrames of gene metadata for each cell type
    - `num_threads`: the default number of threads to use for operations on the
      dataset that support multithreading (which can be overridden by
      individual functions)

    as well as `obs_names` and `var_names`, aliases for a dict of `obs[:, 0]`
    and `var[:, 0]` for each cell type.

    In many ways, Pseudobulk datasets behave like dictionaries:

    - `pb1 | pb2` combines pseudobulks with non-overlapping cell types into one
      big pseudobulk
    - `cell_type in pb` tests whether `cell_type` is a cell type in the
      pseudobulk
    - `for cell_type in pb:` and `for cell_type in pb.keys():` yield the cell
      type names
    - `for X, obs, var in pb.values():` yields each cell type's `X`, `obs`, and
      `var`
    - `for cell_type, (X, obs, var) in pseudobulk.items():` yields both the
      name and the `X`, `obs` and `var` for each cell type

    There are also custom iterators if you just want one field per cell type:

    - `for X in pseudobulk.iter_X():` yields just the `X` for each cell type
    - `for obs in pseudobulk.iter_obs():` yields just the `obs`
    - `for var in pseudobulk.iter_var():` yields just the `var`
    """
    def __init__(self,
                 source: str | Path | None = None,
                 /,
                 *,
                 X: dict[str, np.ndarray[np.dtype[np.integer | np.floating]]] |
                    None = None,
                 obs: dict[str, pl.DataFrame] | None = None,
                 var: dict[str, pl.DataFrame] | None = None,
                 num_threads: int | np.integer | None = None) -> None:
        """
        Load a saved Pseudobulk dataset, or create one from an in-memory count
        matrix + metadata for each cell type.

        Args:
            source: a directory to load a saved Pseudobulk dataset from (see
                    `save()`). Mutually exclusive with `X`, `obs`, and `var`.
            X: a {cell type: NumPy array} dictionary of counts or log CPMs.
               Mutually exclusive with `source`.
            obs: a {cell type: polars DataFrame} dict of metadata per sample,
                 when `X` is a dictionary. The first column must be String,
                 Enum, Categorical, or integer. Mutually exclusive with
                 `source`.
            var: a {cell type: polars DataFrame} dict of metadata per gene,
                 when `X` is a dictionary. The first column must be String,
                 Enum, Categorical, or integer. Mutually exclusive with
                 `source`.
            num_threads: the default number of threads to use for all
                         subsequent operations on this Pseudobulk dataset. By
                         default (`num_threads=None`), use all available cores,
                         as determined by
                         [`os.cpu_count()`](https://docs.python.org/3/library/os.html#os.cpu_count).
        """
        # Initialize this Pseudobulk dataset's `num_threads`
        if num_threads is None:
            self._num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int,
                       'a positive integer, -1, or None')
            if num_threads == 1:
                self._num_threads = 1
            else:
                num_threads = int(num_threads)
                if num_threads <= 0 and num_threads != -1:
                    error_message = (
                        f'num_threads is {num_threads:,}, but must be a '
                        f'positive integer, -1, or None')
                    raise ValueError(error_message)
                self._num_threads = \
                    os.cpu_count() if num_threads == -1 else num_threads
        if source is not None and X is not None:
            error_message = 'only one of source and X can be specified'
            raise ValueError(error_message)
        if source is not None:
            check_type(source, 'source', (str, Path),
                       'a string or pathlib.Path')
            if obs is not None:
                error_message = \
                    'obs cannot be specified when source is specified'
                raise ValueError(error_message)
            if var is not None:
                error_message = \
                    'var cannot be specified when source is specified'
                raise ValueError(error_message)
            source = str(source)
            if not os.path.isdir(source):
                if os.path.isfile(source):
                    error_message = \
                        f'{source!r} must be a directory, not a file'
                    raise NotADirectoryError(error_message)
                else:
                    error_message = \
                        f'Pseudobulk directory {source!r} does not exist'
                    raise FileNotFoundError(error_message)
            cell_types = [line.rstrip('\n') for line in
                          open(f'{source}/cell_types.txt')]
            X = {cell_type: np.load(os.path.join(
                source, f'{cell_type.replace("/", "-")}.X.npy'))
                for cell_type in cell_types}
            obs = {cell_type: pl.read_parquet(os.path.join(
                source, f'{cell_type.replace("/", "-")}.obs.parquet'))
                for cell_type in cell_types}
            var = {cell_type: pl.read_parquet(os.path.join(
                source, f'{cell_type.replace("/", "-")}.var.parquet'))
                for cell_type in cell_types}
        elif X is not None:
            check_type(X, 'X', dict, 'a dictionary')
            if obs is None:
                error_message = (
                    'obs is None, but since X is a dictionary, obs must also '
                    'be a dictionary')
                raise TypeError(error_message)
            if var is None:
                error_message = (
                    'var is None, but since X is a dictionary, var must also '
                    'be a dictionary')
                raise TypeError(error_message)
            if not X:
                error_message = 'X is an empty dictionary'
                raise ValueError(error_message)
            if X.keys() != obs.keys():
                error_message = (
                    'X and obs must have the same cell types (keys), in the '
                    'same order')
                raise ValueError(error_message)
            if X.keys() != var.keys():
                error_message = (
                    'X and var must have the same cell types (keys), in the '
                    'same order')
                raise ValueError(error_message)
            for cell_type in X:
                if not isinstance(cell_type, str):
                    error_message = (
                        f'all keys of X (cell types) must be strings, but X '
                        f'contains a key of type {type(cell_type).__name__!r}')
                    raise TypeError(error_message)
                check_type(X[cell_type], f'X[{cell_type!r}]', np.ndarray,
                           'a NumPy array')
                if X[cell_type].ndim != 2:
                    error_message = (
                        f'X[{cell_type!r}] is a {X[cell_type].ndim:,}-'
                        f'dimensional NumPy array, but must be 2-dimensional')
                    raise ValueError(error_message)
                check_type(obs[cell_type], f'obs[{cell_type!r}]', pl.DataFrame,
                           'a polars DataFrame')
                check_type(var[cell_type], f'var[{cell_type!r}]', pl.DataFrame,
                           'a polars DataFrame')
        else:
            error_message = 'either source or X must be specified'
            raise ValueError(error_message)
        for cell_type in X:
            dtype = X[cell_type].dtype
            if dtype != np.int32 and dtype != np.int64 and \
                    dtype != np.float32 and dtype != np.float64 and \
                    dtype != np.uint32 and dtype != np.uint64:
                error_message = (
                    f'X must be (u)int32/64 or float32/64, but has data type '
                    f'{str(dtype)}')
                raise TypeError(error_message)
            if len(obs[cell_type]) == 0:
                error_message = \
                    f'len(obs[{cell_type!r}]) is 0: no samples remain'
                raise ValueError(error_message)
            if len(var[cell_type]) == 0:
                error_message = \
                    f'len(var[{cell_type!r}]) is 0: no genes remain'
                raise ValueError(error_message)
            if len(obs[cell_type]) != len(X[cell_type]):
                error_message = (
                    f'len(obs[{cell_type!r}]) is {len(obs[cell_type]):,}, but '
                    f'len(X[{cell_type!r}]) is {len(X[cell_type]):,}')
                raise ValueError(error_message)
            if len(var[cell_type]) != X[cell_type].shape[1]:
                error_message = (
                    f'len(var[{cell_type!r}]) is {len(var[cell_type]):,}, but '
                    f'X[{cell_type!r}].shape[1] is {X[cell_type].shape[1]:,}')
                raise ValueError(error_message)
            obs_names_dtype = obs[cell_type][:, 0].dtype
            if obs_names_dtype not in (pl.String, pl.Enum, pl.Categorical) \
                    and obs_names_dtype not in INTEGER_DTYPES:
                error_message = (
                    f'the first column of obs[{cell_type!r}] '
                    f'({obs[cell_type].columns[0]!r}) must be String, '
                    f'Enum, Categorical, or integer, but has data type '
                    f'{obs_names_dtype.base_type()!r}')
                raise ValueError(error_message)
            var_names_dtype = var[cell_type][:, 0].dtype
            if var_names_dtype not in (pl.String, pl.Enum, pl.Categorical) \
                    and var_names_dtype not in INTEGER_DTYPES:
                error_message = (
                    f'the first column of var[{cell_type!r}] '
                    f'({var[cell_type].columns[0]!r}) must be String, '
                    f'Enum, Categorical, or integer, but has data type '
                    f'{var_names_dtype.base_type()!r}')
                raise ValueError(error_message)
        self._X = X_(X)
        self._obs = Obs(obs)
        self._var = Var(var)

    @property
    def X(self) -> dict[str, np.ndarray[np.dtype[np.integer | np.floating]]]:
        """
        A dictionary of count matrices for each cell type, as NumPy arrays.
        """
        return self._X

    @X.setter
    def X(self, X: dict[str, np.ndarray[np.dtype[np.integer |
                                                 np.floating]]]) -> None:
        check_type(X, 'X', dict, 'a dictionary')
        new_X = X_(X)
        if new_X.keys() != self._X.keys():
            error_message = 'new X does not have the same cell types as old X'
            raise ValueError(error_message)
        if tuple(new_X) != tuple(self._X):
            new_X = X_({k: new_X[k] for k in self._X.keys()})
        self._X = new_X

    @property
    def obs(self) -> dict[str, pl.DataFrame]:
        """
        A dictionary of Polars DataFrames of sample-level metadata for each
        cell type.
        """
        return self._obs

    @obs.setter
    def obs(self, obs: dict[str, pl.DataFrame]) -> None:
        check_type(obs, 'obs', dict, 'a dictionary')
        new_obs = Obs(obs)
        if new_obs.keys() != self._obs.keys():
            error_message = \
                'new obs does not have the same cell types as old obs'
            raise ValueError(error_message)
        if tuple(new_obs) != tuple(self._obs):
            new_obs = Obs({k: new_obs[k] for k in self._obs.keys()})
        self._obs = new_obs

    @property
    def var(self) -> dict[str, pl.DataFrame]:
        """
        A dictionary of Polars DataFrames of gene-level metadata for each cell
        type.
        """
        return self._var

    @var.setter
    def var(self, var: dict[str, pl.DataFrame]) -> None:
        check_type(var, 'var', dict, 'a dictionary')
        new_var = Var(var)
        if new_var.keys() != self._var.keys():
            error_message = \
                'new var does not have the same cell types as old var'
            raise ValueError(error_message)
        if tuple(new_var) != tuple(self._var):
            new_var = Var({k: new_var[k] for k in self._var.keys()})
        self._var = new_var

    @property
    def obs_names(self) -> dict[str, pl.Series]:
        """
        A shortcut to access the first column of `obs` for each cell type.
        Generally holds sample identifiers.
        """
        return {cell_type: obs[:, 0] for cell_type, obs in self._obs.items()}

    @property
    def var_names(self) -> dict[str, pl.Series]:
        """
        A shortcut to access the first column of `var` for each cell type.
        Generally holds gene names.
        """
        return {cell_type: var[:, 0] for cell_type, var in self._var.items()}

    @property
    def num_threads(self) -> int:
        """
        The default number of threads used for this Pseudobulk dataset's
        operations.
        """
        return self._num_threads

    @num_threads.setter
    def num_threads(self, num_threads: int | np.integer) -> None:
        """
        Set the default number of threads used for this Pseudobulk dataset's
        operations.

        Args:
            num_threads: the new default number of threads. Set
                         `num_threads=-1` to use all available cores, as
                         determined by
                         [`os.cpu_count()`](https://docs.python.org/3/library/os.html#os.cpu_count).
        """
        check_type(num_threads, 'num_threads', int, 'a positive integer or -1')
        if num_threads == 1:
            self._num_threads = 1
            return
        num_threads = int(num_threads)
        if num_threads <= 0 and num_threads != -1:
            error_message = (
                f'num_threads is {num_threads:,}, but must be a positive '
                f'integer or -1')
            raise ValueError(error_message)
        cpu_count = os.cpu_count()
        if num_threads > cpu_count:
            error_message = (
                f'num_threads is {num_threads:,}, but must be at most '
                f'os.cpu_count() ({cpu_count})')
            raise ValueError(error_message)
        self._num_threads = cpu_count if num_threads == -1 else num_threads

    def set_num_threads(self, num_threads: int | np.integer, /) -> Pseudobulk:
        """
        Return a new Pseudobulk dataset with a different default number of
        threads.

        Args:
            num_threads: the new default number of threads. Set
                         `num_threads=-1` to use all available cores, as
                         determined by
                         [`os.cpu_count()`](https://docs.python.org/3/library/os.html#os.cpu_count).
        """
        return Pseudobulk(X=self._X, obs=self._obs, var=self._var,
                          num_threads=num_threads)

    def _process_num_threads(self,
                             num_threads: int | np.integer | None) -> int:
        """
        Process a `num_threads` value specified by the user as an argument to a
        Pseudobulk function.

        Check that `num_threads` is a positive integer, -1 or `None`; if
        `None`, set to `self.num_threads`, and if -1, set to
        [`os.cpu_count()`](https://docs.python.org/3/library/os.html#os.cpu_count).

        Args:
            num_threads: the number of threads specified by the user

        Returns:
            The actual number of threads to use.
        """
        if num_threads is None:
            return self._num_threads
        check_type(num_threads, 'num_threads', int,
                   'a positive integer, -1, or None')
        if num_threads == -1:
            return os.cpu_count()
        else:
            num_threads = int(num_threads)
            if num_threads <= 0:
                error_message = (
                    f'num_threads is {num_threads:,}, but must be a positive '
                    f'integer, -1, or None')
                raise ValueError(error_message)
            return num_threads

    def _process_cell_types(self,
                            cell_types: str | Iterable[str] | None,
                            excluded_cell_types: str | Iterable[str] | None,
                            *,
                            return_description: bool = False) -> \
            tuple[str, ...] | tuple[tuple[str, ...], str]:
        """
        Process the `cell_types` and `excluded_cell_types` arguments of various
        Pseudobulk functions.

        Args:
            cell_types: one or more cell types to include in the calling
                        function's operation
            excluded_cell_types: one or more cell types to exclude from the
                                 calling function's operation
            return_description: whether to return a description of the cell
                                types, to use in error messages

        Returns:
            A tuple of cell-type names, or a two-element tuple of
            (cell-type names, cell-type description) if
            `return_description=True`.
        """
        if cell_types is not None:
            if excluded_cell_types is not None:
                error_message = (
                    'cell_types and excluded_cell_types cannot both be '
                    'specified')
                raise ValueError(error_message)
            is_string = isinstance(cell_types, str)
            cell_types = \
                to_tuple_checked(cell_types, 'cell_types', str, 'strings')
            for cell_type in cell_types:
                if cell_type not in self._X:
                    if is_string:
                        error_message = (
                            f'cell_types is {cell_type!r}, which is not a '
                            f'cell type in this Pseudobulk dataset')
                        raise ValueError(error_message)
                    else:
                        error_message = (
                            f'cell_types contains a cell type, {cell_type!r}, '
                            f'not present in this Pseudobulk dataset')
                        raise ValueError(error_message)
            if return_description:
                cell_type_description = 'the cell_types argument'
        elif excluded_cell_types is not None:
            is_string = isinstance(excluded_cell_types, str)
            excluded_cell_types = to_tuple_checked(
                excluded_cell_types, 'excluded_cell_types', str, 'strings')
            for cell_type in excluded_cell_types:
                if cell_type not in self._X:
                    if is_string:
                        error_message = (
                            f'excluded_cell_types is {cell_type!r}, which is '
                            f'not a cell type in this Pseudobulk dataset')
                        raise ValueError(error_message)
                    else:
                        error_message = (
                            f'excluded_cell_types contains a cell type, '
                            f'{cell_type!r}, not present in this Pseudobulk '
                            f'dataset')
                        raise ValueError(error_message)
            cell_types = tuple(cell_type for cell_type in self._X
                               if cell_type not in excluded_cell_types)
            if len(cell_types) == 0:
                error_message = \
                    'all cell types were excluded by excluded_cell_types'
                raise ValueError(error_message)
            if return_description:
                cell_type_description = (
                    'this Pseudobulk dataset (after excluding the cell types '
                    'in excluded_cell_types)')
        else:
            cell_types = tuple(self._X)
            if return_description:
                cell_type_description = 'this Pseudobulk dataset'
        return (cell_types, cell_type_description) \
            if return_description else cell_types

    def set_obs_names(self,
                      column: str,
                      /,
                      *,
                      cell_types: str | Iterable[str] | None = None,
                      excluded_cell_types: str | Iterable[str] |
                                           None = None) -> Pseudobulk:
        """
        Sets a column as the new first column of `obs`, i.e. the `obs_names`.

        Args:
            column: the column name in `obs`; must be String, Enum,
                    Categorical, or integer
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`

        Returns:
            A new Pseudobulk dataset with `column` as the first column of each
            cell type's `obs`. If `column` is already the first column for
            every cell type, return this dataset unchanged.
        """
        check_type(column, 'column', str, 'a string')
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        if all(column == self._obs[cell_type].columns[0]
               for cell_type in cell_types):
            return self
        obs = {}
        for cell_type, cell_type_obs in self._obs.items():
            if cell_type in cell_types:
                if column not in cell_type_obs:
                    error_message = \
                        f'{column!r} is not a column of obs[{cell_type!r}]'
                    raise ValueError(error_message)
                check_dtype(cell_type_obs[column], f'obs[{column!r}]',
                            (pl.String, pl.Enum, pl.Categorical, 'integer'))
                obs[cell_type] = \
                    cell_type_obs.select(column, pl.exclude(column))
            else:
                obs[cell_type] = cell_type_obs
        return Pseudobulk(X=self._X, obs=obs, var=self._var,
                          num_threads=self._num_threads)

    def set_var_names(self,
                      column: str,
                      /,
                      *,
                      cell_types: str | Iterable[str] | None = None,
                      excluded_cell_types: str | Iterable[str] |
                                           None = None) -> Pseudobulk:
        """
        Sets a column as the new first column of `var`, i.e. the `var_names`.

        Args:
            column: the column name in `var`; must be String, Enum,
                    Categorical, or integer
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`

        Returns:
            A new Pseudobulk dataset with `column` as the first column of each
            cell type's `var`. If `column` is already the first column for
            every cell type, return this dataset unchanged.
        """
        check_type(column, 'column', str, 'a string')
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        if all(column == self._var[cell_type].columns[0]
               for cell_type in cell_types):
            return self
        var = {}
        for cell_type, cell_type_var in self._var.items():
            if cell_type in cell_types:
                if column not in cell_type_var:
                    error_message = \
                        f'{column!r} is not a column of var[{cell_type!r}]'
                    raise ValueError(error_message)
                check_dtype(cell_type_var[column], f'var[{column!r}]',
                            (pl.String, pl.Enum, pl.Categorical, 'integer'))
                var[cell_type] = \
                    cell_type_var.select(column, pl.exclude(column))
            else:
                var[cell_type] = cell_type_var
        return Pseudobulk(X=self._X, obs=self._obs, var=var,
                          num_threads=self._num_threads)

    def keys(self) -> KeysView[str]:
        """
        Get a KeysView (like you would get from `dict.keys()`) of this
        Pseudobulk dataset's cell types. `for cell_type in pb.keys():` is
        equivalent to `for cell_type in pb:`.

        Returns:
            A KeysView of the cell types.
        """
        return self._X.keys()

    def values(self) -> ValuesView[tuple[np.ndarray[np.dtype[np.integer |
                                                             np.floating]],
                                       pl.DataFrame, pl.DataFrame]]:
        """
        Get a `ValuesView` (like you would get from `dict.values()`) of
        `(X, obs, var)` tuples for each cell type in this Pseudobulk dataset.

        Returns:
            A `ValuesView` of `(X, obs, var)` tuples for each cell type.
        """
        return {cell_type: (self._X[cell_type], self._obs[cell_type],
                            self._var[cell_type])
                for cell_type in self._X}.values()

    def items(self) -> ItemsView[str, tuple[np.ndarray[np.dtype[
            np.integer | np.floating]], pl.DataFrame, pl.DataFrame]]:
        """
        Get an `ItemsView` (like you would get from `dict.items()`) of
        `(cell_type, (X, obs, var))` tuples for each cell type in this
        Pseudobulk dataset.

        Yields:
            An `ItemsView` of `(cell_type, (X, obs, var))` tuples for each cell
            type.
        """
        return {cell_type: (self._X[cell_type], self._obs[cell_type],
                            self._var[cell_type])
                for cell_type in self._X}.items()

    def iter_X(self) -> Iterable[np.ndarray[np.dtype[np.integer |
                                                     np.floating]]]:
        """
        Iterate over each cell type's `X`.

        Yields:
            `X` for each cell type.
        """
        for X in self._X.values():
            yield X

    def iter_obs(self) -> Iterable[pl.DataFrame]:
        """
        Iterate over each cell type's `obs`.

        Yields:
            `obs` for each cell type.
        """
        for obs in self._obs.values():
            yield obs

    def iter_var(self) -> Iterable[pl.DataFrame]:
        """
        Iterate over each cell type's `var`.

        Yields:
            `var` for each cell type.
        """
        for var in self._var.values():
            yield var

    def __eq__(self, other: Pseudobulk) -> bool:
        """
        Test for equality with another Pseudobulk dataset.

        Args:
            other: the other Pseudobulk dataset to test for equality with

        Returns:
            Whether the two Pseudobulk datasets are identical.
        """
        if not isinstance(other, Pseudobulk):
            error_message = (
                f'the left-hand operand of `==` is a Pseudobulk dataset, but '
                f'the right-hand operand has type {type(other).__name__!r}')
            raise TypeError(error_message)
        return self._num_threads == other._num_threads and \
            tuple(self.keys()) == tuple(other.keys()) and \
            all(obs.equals(other_obs) for obs, other_obs in
                zip(self._obs.values(), other._obs.values())) and \
            all(var.equals(other_var) for var, other_var in
                zip(self._var.values(), other._var.values())) and \
            all(array_equal(X, other_X) for X, other_X in
                zip(self._X.values(), other._X.values()))

    def __or__(self, other: Pseudobulk) -> Pseudobulk:
        """
        Combine the cell types of this Pseudobulk dataset with another. The
        two datasets must have non-overlapping cell types.

        Args:
            other: the other Pseudobulk dataset to combine with this one

        Returns:
            A Pseudobulk dataset with each of the cell types in the first
            Pseudobulk dataset, followed by each of the cell types in the
            second.
        """
        if not isinstance(other, Pseudobulk):
            error_message = (
                f'the left-hand operand of `|` is a Pseudobulk dataset, but '
                f'the right-hand operand has type {type(other).__name__!r}')
            raise TypeError(error_message)
        if self.keys() & other.keys():
            error_message = (
                'the left- and right-hand operands of `|` are Pseudobulk '
                'datasets that share some cell types')
            raise ValueError(error_message)
        return Pseudobulk(X=dict(self._X) | dict(other._X),
                          obs=dict(self._obs) | dict(other._obs),
                          var=dict(self._var) | dict(other._var),
                          num_threads=self._num_threads)

    def split_by_cell_type(self,
                           cell_types: str | Iterable[str] | None = None,
                           excluded_cell_types: str | Iterable[str] |
                                                None = None) -> \
            tuple[Pseudobulk]:
        """
        Split this Pseudobulk dataset into a tuple of Pseudobulk datasets
        with one cell type each. May be useful when analyses differ so much
        by cell type that it's easier to write separate code for each cell
        type.

        Args:
            cell_types: one or more cell types to include; if `None`, include
                        all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude; mutually
                                 exclusive with `cell_types`

        Returns:
            A tuple of single-cell-type Pseudobulk datasets.
        """
        cell_types = self._process_cell_types(cell_types, excluded_cell_types)
        return tuple(Pseudobulk(X={cell_type: self._X[cell_type]},
                                obs={cell_type: self._obs[cell_type]},
                                var={cell_type: self._var[cell_type]},
                                num_threads=self._num_threads)
            for cell_type in cell_types)

    def __contains__(self, cell_type: str) -> bool:
        """
        Check if this Pseudobulk dataset contains the specified cell type.

        Args:
            cell_type: the cell type

        Returns:
            Whether the cell type is present in the Pseudobulk dataset.
        """
        check_type(cell_type, 'cell_type', str, 'a string')
        return cell_type in self._X

    @staticmethod
    def _getitem_error(item: Indexer | tuple[str, Indexer, Indexer]) -> \
            NoReturn:
        """
        Raise an error if the indexer is invalid.

        Args:
            item: the indexer
        """
        types = tuple(type(elem).__name__ for elem in to_tuple(item))
        if len(types) == 1:
            types = types[0]
        error_message = (
            f'Pseudobulk indices must be a cell-type string, a length-1 tuple '
            f'of (cell_type,), a length-2 tuple of (cell_type, samples), or a '
            f'length-3 tuple of (cell_type, samples, genes). Samples and '
            f'genes must each be a string or integer; a slice of strings or '
            f'integers; or a list, NumPy array, or polars Series of strings, '
            f'integers, or Booleans. You indexed with: {types}.')
        raise ValueError(error_message)

    @staticmethod
    def _getitem_by_string(df: pl.DataFrame, string: str) -> int:
        """
        Get the index where df[:, 0] == string, raising an error if no rows or
        multiple rows match.

        Args:
            df: a DataFrame (`obs` or `var`)
            string: the string to find the index of in the first column of df

        Returns:
            The integer index of the string within the first column of df.
        """
        first_column = df.columns[0]
        try:
            return df\
                .select(pl.int_range(pl.len(), dtype=pl.Int32)
                        .alias('_Pseudobulk_getitem'), first_column)\
                .row(by_predicate=pl.col(first_column) == string)\
                [0]
        except pl.exceptions.NoRowsReturnedError:
            raise KeyError(string)

    @staticmethod
    def _getitem_process(item: Indexer | tuple[str, Indexer, Indexer],
                         index: int,
                         df: pl.DataFrame) -> list[int] | slice | pl.Series:
        """
        Process an element of an item passed to `__getitem__()`.

        Args:
            item: the item
            index: the index of the element to process
            df: the DataFrame (`obs` or `var`) to process the element with
                respect to

        Returns:
            A new indexer indicating the rows/columns to index.
        """
        subitem = item[index]
        if isinstance(subitem, (int, np.integer)):
            return [subitem]
        elif isinstance(subitem, str):
            return [Pseudobulk._getitem_by_string(df, subitem)]
        elif isinstance(subitem, slice):
            start = subitem.start
            stop = subitem.stop
            step = subitem.step
            if isinstance(start, str):
                start = Pseudobulk._getitem_by_string(df, start)
            elif start is not None and \
                    not isinstance(start, (int, np.integer)):
                Pseudobulk._getitem_error(item)
            if isinstance(stop, str):
                stop = Pseudobulk._getitem_by_string(df, stop)
            elif stop is not None and not isinstance(stop, (int, np.integer)):
                Pseudobulk._getitem_error(item)
            if step is not None and not isinstance(step, (int, np.integer)):
                Pseudobulk._getitem_error(item)
            return slice(start, stop, step)
        elif isinstance(subitem, (tuple, list, np.ndarray, pl.Series)):
            subitem = pl.Series(subitem)
            if subitem.is_null().any():
                error_message = 'your indexer contains missing values'
                raise ValueError(error_message)
            dtype = subitem.dtype
            if dtype in (pl.String, pl.Enum, pl.Categorical):
                names_dtype = df[:, 0].dtype
                original_subitem = subitem
                if dtype != names_dtype:
                    subitem = subitem.cast(names_dtype, strict=False)
                indices = subitem\
                    .to_frame(df.columns[0])\
                    .join(df.with_columns(_Pseudobulk_index=pl.int_range(
                              pl.len(), dtype=pl.UInt32)),
                          on=df.columns[0], how='left', maintain_order='left')\
                    ['_Pseudobulk_index']
                if indices.null_count():
                    error_message = \
                        original_subitem.filter(indices.is_null())[0]
                    raise KeyError(error_message)
                return indices
            elif dtype.is_integer() or dtype == pl.Boolean:
                return subitem
            else:
                Pseudobulk._getitem_error(item)
        else:
            Pseudobulk._getitem_error(item)

    def __getitem__(self, item: Indexer | tuple[str, Indexer, Indexer]) -> \
            Pseudobulk:
        """
        Subset to specific cell type(s), sample(s), and/or gene(s).

        Index with a tuple of `(cell_types, samples, genes)`. If `samples` and
        `genes` are integers, arrays/lists/slices of integers, or arrays/lists
        of Booleans, the result will be a Pseudobulk dataset subset to
        `X[samples, genes]`, `obs[samples]`, and `var[genes]` for each of the
        cell types in `cell_types`. However, `samples` and/or `genes` can
        instead be strings (or arrays or slices of strings), in which case they
        refer to the first column of `obs` and/or `var`, respectively.

        Examples:
            Subset to one cell type:
                pb['Astro']

            Subset to multiple cell types:
                pb[['Astro', 'Micro']]

            Subset to one cell type and one sample (all genes):
                pb['Astro', 'Sample1']
                pb['Astro', 2]

            Subset to one gene (all cell types and samples):
                pb[:, :, 'APOE']
                pb[:, :, 13196]

            Subset to one cell type, sample, and gene:
                pb['Astro', 'Sample2', 'APOE']
                pb['Astro', 2, 13196]

            Subset using slices (ranges):
                pb['Astro', 'Sample2':'Sample4', 'APOE':'TREM2']
                pb['Astro', 2:10, 13196:13210]

            Subset using lists/arrays of labels or indices:
                pb['Astro', ['Sample2', 'Sample4']]
                pb['Astro', :, ['APOE', 'TREM2']]
                pb['Astro', [0, 3, 5], [10, 25, 42]]

            Subset using Boolean masks:
                pb['Astro', pb.obs['Astro']['batch'] == 'A']
                pb[:, :, pb.var['Astro']['gene_symbol'].is_in(['APOE', 'TREM2'])]

            Mix indexing types across dimensions:
                pb[['Astro', 'Micro'], 0:10, ['APOE', 'TREM2']]

        Args:
            item: the item to index with

        Returns:
            A new Pseudobulk dataset subset to the specified cell types,
            samples, and/or genes.
        """
        is_slice = False
        if isinstance(item, tuple):
            if not 1 <= len(item) <= 3:
                self._getitem_error(item)
            if isinstance(item[0], slice):
                if item[0] == slice(None):
                    cell_types = tuple(self._X)
                    is_slice = True
                else:
                    error_message = (
                        'slicing cell types is not currently supported, '
                        'except for selecting all cell types (e.g. pb[:, '
                        'samples, genes])')
                    raise ValueError(error_message)
            else:
                cell_types = to_tuple(item[0])
        elif isinstance(item, (list, np.ndarray, pl.Series)):
            cell_types = to_tuple(item)
        elif isinstance(item, str):
            cell_types = item,
        elif isinstance(item, slice):
            if item == slice(None):
                cell_types = tuple(self._X)
                is_slice = True
            else:
                error_message = (
                    'slicing cell types is not currently supported, except '
                    'for selecting all cell types (e.g. pb[:, samples, '
                    'genes])')
                raise ValueError(error_message)
        else:
            self._getitem_error(item)
        if not is_slice:
            for cell_type in cell_types:
                if cell_type not in self._X:
                    if isinstance(cell_type, str):
                        error_message = (
                            f'tried to select {cell_type!r}, which is not a '
                            f'cell type in this Pseudobulk')
                        raise ValueError(error_message)
                    else:
                        error_message = (
                            f'tried to select a non-existent cell type of '
                            f'type {type(cell_type).__name__!r}')
                        raise TypeError(error_message)
        if not isinstance(item, tuple) or len(item) == 1:
            return Pseudobulk(X={cell_type: self._X[cell_type]
                                 for cell_type in cell_types},
                              obs={cell_type: self._obs[cell_type]
                                   for cell_type in cell_types},
                              var={cell_type: self._var[cell_type]
                                   for cell_type in cell_types},
                              num_threads=self._num_threads)
        X, obs, var = {}, {}, {}
        for cell_type in cell_types:
            rows = self._getitem_process(item, 1, self._obs[cell_type])
            if isinstance(rows, pl.Series):
                obs[cell_type] = self._obs[cell_type].filter(rows) \
                    if rows.dtype == pl.Boolean else self._obs[cell_type][rows]
                rows = rows.to_numpy()
            else:
                obs[cell_type] = self._obs[cell_type][rows]
            if len(item) == 2:
                X[cell_type] = self._X[cell_type][rows]
                var[cell_type] = self._var[cell_type]
            else:
                columns = self._getitem_process(item, 2, self._var[cell_type])
                if isinstance(columns, pl.Series):
                    var[cell_type] = self._var[cell_type].filter(columns) \
                        if columns.dtype == pl.Boolean \
                        else self._var[cell_type][columns]
                    columns = columns.to_numpy()
                else:
                    var[cell_type] = self._var[cell_type][columns]
                X[cell_type] = self._X[cell_type][rows, columns] \
                    if isinstance(rows, slice) or \
                       isinstance(columns, slice) else \
                    self._X[cell_type][np.ix_(rows, columns)]
        return Pseudobulk(X=X, obs=obs, var=var, num_threads=self._num_threads)

    def sample(self, cell_type: str, sample: str, /) -> np.ndarray:
        """
        Get the row of `X[cell_type]` corresponding to a single sample, based
        on the sample's name in `obs_names`.

        Args:
            cell_type: the cell type to retrieve the row of `X` from
            sample: the name of the sample in `obs_names`

        Returns:
            The corresponding row of `X[cell_type]`, as a dense 1D NumPy array
            with zeros included.
        """
        row_index = Pseudobulk._getitem_by_string(self._obs[cell_type], sample)
        return self._X[cell_type][row_index]

    def gene(self, cell_type: str, gene: str, /) -> np.ndarray:
        """
        Get the column of `X[cell_type]` corresponding to a single gene, based
        on the gene's name in `var_names`.

        Args:
            cell_type: the cell type to retrieve the row of `X` from
            gene: the name of the gene in `var_names`

        Returns:
            The corresponding column of `X[cell_type]`, as a dense 1D NumPy
            array with zeros included.
        """
        column_index = \
            Pseudobulk._getitem_by_string(self._var[cell_type], gene)
        return self._X[cell_type][:, column_index]

    def __iter__(self) -> Iterable[str]:
        """
        Iterate over the cell types of this Pseudobulk dataset.
        `for cell_type in pb` is equivalent to `for cell_type in pb.keys()`.

        Returns:
            An iterator over the cell types.
        """
        return iter(self._X)

    def __len__(self) -> dict[str, int]:
        """
        Get the number of cell types in this Pseudobulk dataset.

        Returns:
            The number of cell types
        """
        return len(self._X)

    def __repr__(self) -> str:
        """
        Get a string representation of this Pseudobulk dataset.

        Returns:
            A string summarizing the dataset.
        """
        min_samples = min(len(obs) for obs in self._obs.values())
        max_samples = max(len(obs) for obs in self._obs.values())
        min_genes = min(len(var) for var in self._var.values())
        max_genes = max(len(var) for var in self._var.values())
        samples_string = f'{min_samples:,} {plural("sample", max_samples)}' \
            if min_samples == max_samples else \
            f'{min_samples:,}-{max_samples:,} samples'
        genes_string = f'{min_genes:,} {plural("gene", max_genes)}' \
            if min_genes == max_genes else \
            f'{min_genes:,}-{max_genes:,} genes'
        terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        return f'Pseudobulk dataset with {len(self._X):,} cell ' \
               f'{"types, each" if len(self._X) > 1 else "type,"} with ' \
               f'{samples_string} (obs) and {genes_string} (var)\n' + \
            fill(f'    Cell types: {", ".join(self._X)}',
                 width=terminal_width, subsequent_indent=' ' * 17)

    @property
    def shape(self) -> dict[str, tuple[int, int]]:
        """
        The shape of each cell type in this Pseudobulk dataset: a dictionary
        mapping each cell type to a length-2 tuple where the first element is
        the number of samples, and the second is the number of genes.
        """
        return {cell_type: X_cell_type.shape
                for cell_type, X_cell_type in self._X.items()}

    def save(self,
             directory: str | Path,
             /,
             *,
             overwrite: bool = False,
             cell_types: str | Iterable[str] | None = None,
             excluded_cell_types: str | Iterable[str] | None = None) -> None:
        """
        Saves a Pseudobulk dataset to `directory` (which must not exist unless
        `overwrite=True`, and will be created) with three files per cell type:
        the `X` at `f'{cell_type}.X.npy'`, the `obs` at
        `f'{cell_type}.obs.parquet'`, and the `var` at
        `f'{cell_type}.var.parquet'`. Also saves a text file, `cell_types.txt`,
        containing the cell types.

        Args:
            directory: the directory to save the Pseudobulk dataset to
            overwrite: if `False`, raises an error if the directory exists; if
                       `True`, overwrites files inside it as necessary
            cell_types: one or more cell types to save; if `None`, save all
                        cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from saving;
                                 mutually exclusive with `cell_types`
        """
        check_type(directory, 'directory', (str, Path),
                   'a string or pathlib.Path')
        directory = str(directory)
        check_type(overwrite, 'overwrite', bool, 'Boolean')
        if not overwrite and os.path.exists(directory):
            if os.path.isfile(directory):
                error_message = (
                    f'cannot save to the directory {directory!r} because it '
                    f'already exists as a file')
                raise FileExistsError(error_message)
            else:
                error_message = (
                    f'directory {directory!r} already exists; set '
                    f'overwrite=True to overwrite')
                raise FileExistsError(error_message)
        os.makedirs(directory, exist_ok=overwrite)
        cell_types = self._process_cell_types(cell_types, excluded_cell_types)
        with open(os.path.join(directory, 'cell_types.txt'), 'w') as f:
            print('\n'.join(cell_types), file=f)
        for cell_type in cell_types:
            escaped_cell_type = cell_type.replace('/', '-')
            np.save(os.path.join(directory, f'{escaped_cell_type}.X.npy'),
                    self._X[cell_type])
            self._obs[cell_type].write_parquet(
                os.path.join(directory, f'{escaped_cell_type}.obs.parquet'))
            self._var[cell_type].write_parquet(
                os.path.join(directory, f'{escaped_cell_type}.var.parquet'))

    def copy(self, *, deep: bool = False) -> Pseudobulk:
        """
        Make a copy of this Pseudobulk dataset.

        Args:
            deep: whether to make a deep copy instead of a shallow one. Since
                  polars DataFrames are immutable, `obs[cell_type]` and
                  `var[cell_type]` will always point to the same underlying
                  data as the original for all cell types. The only difference
                  when `deep=True` is that `X[cell_type]` will point to a fresh
                  copy of the data, rather than the same data. When
                  `deep=False`, any modifications to the underlying count
                  matrix will modify both the original and the copy.

        Returns:
            A copy of the Pseudobulk dataset.
        """
        check_type(deep, 'deep', bool, 'Boolean')
        return Pseudobulk(X={cell_type: cell_type_X.copy()
                             for cell_type, cell_type_X in self._X.items()}
                            if deep else self._X,
                          obs=self._obs, var=self._var,
                          num_threads=self._num_threads)

    def to_df(self,
              *,
              obs_columns: str | Iterable[str] | None = None,
              genes: str | Iterable[str] | None = None,
              cell_type_column: str = 'cell_type') -> pl.DataFrame:
        """
        Convert this Pseudobulk object to a polars DataFrame, with one row per
        (sample, cell type) pair and one column per gene.

        The first columns of the DataFrame will contain metadata: a `cell_type`
        column, a sample ID column (the `obs_names`), a `num_cells` column, and
        whichever additional columns are specified in `obs_columns`.

        Genes or columns of obs not present in every cell type will contain
        `null` values for cell types where they are missing.

        Args:
            obs_columns: one or more names of columns of `obs` to include in
                         the DataFrame, in addition to the cell type, the
                         sample ID, and the number of cells
            genes: one or more genes to include as columns; by default, include
                   all genes
            cell_type_column: the name of the cell-type column to be added as
                              the first column of the DataFrame

        Returns:
            A polars DataFrame containing the gene counts and metadata for each
            (sample, cell type) pair.
        """
        # Check that `cell_type_column` is a string, and not `num_cells`
        check_type(cell_type_column, 'cell_type_column', str, 'a string')
        if cell_type_column == 'num_cells':
            error_message = "the cell_type_column cannot be named 'num_cells'"
            raise ValueError(error_message)

        # Get `obs_columns` and `genes` as tuples, if they are single values,
        # and check that all their elements are strings and that
        # `cell_type_column` name is not in them
        if obs_columns is not None:
            obs_columns = to_tuple_checked(obs_columns, 'obs_columns', str,
                                           'strings')
            if cell_type_column in obs_columns:
                error_message = (
                    f'cell_type_column {cell_type_column!r} is in '
                    f'obs_columns; specify a different name for the '
                    f'cell_type_column, or remove {cell_type_column!r} from '
                    f'obs_columns')
                raise ValueError(error_message)
        if genes is not None:
            genes = to_tuple_checked(genes, 'genes', str, 'strings')
            if cell_type_column in genes:
                error_message = (
                    f'cell_type_column {cell_type_column!r} is in genes; '
                    f'specify a different name for the cell_type_column, or '
                    f'remove {cell_type_column!r} from genes')
                raise ValueError(error_message)

        # Get the DataFrame for each cell type
        dfs = []
        for cell_type, (X, obs, var) in self.items():
            columns = [pl.lit(cell_type).alias(cell_type_column),
                       obs[:, 0].name]
            if 'num_cells' in obs:
                columns.append('num_cells')
            if obs_columns is not None:
                columns += [column for column in obs_columns if column in obs]
            df = obs.select(columns)
            if genes is None:
                gene_df = pl.from_numpy(X, schema=var[:, 0].to_list())
            else:
                var_names_name = var.columns[0]
                joined = pl.DataFrame({var_names_name: genes})\
                    .join(var[:, [0]].with_row_index(), on=var_names_name,
                          how='left', maintain_order='left')\
                    .drop_nulls('index')
                cell_type_gene_indices = joined['index']
                cell_type_genes = joined['gene']
                gene_df = pl.from_numpy(X[:, cell_type_gene_indices],
                                        schema=cell_type_genes.to_list())
            df = pl.concat((df, gene_df), how='horizontal')
            dfs.append(df)

        # Concatenate across cell types
        df = pl.concat(dfs, how='diagonal_relaxed')

        # Check that all `obs_columns` and/or `genes` appear in the DataFrame
        # (i.e. were present in at least one cell type), if either argument was
        # specified
        for column_set, column_set_name, obs_or_var in \
                (obs_columns, 'obs_columns', 'obs'), (genes, 'genes', 'var'):
            if column_set is not None:
                for column in column_set:
                    if column not in df:
                        error_message = (
                            f"column {column!r} was specified in "
                            f"{column_set_name}, but did not appear in any "
                            f"cell type's {obs_or_var}")
                        raise ValueError(error_message)
        return df

    def concat_obs(self,
                   datasets: Pseudobulk | Iterable[Pseudobulk],
                   /,
                   *more_datasets: Pseudobulk,
                   dataset_column: str | None = None,
                   dataset_labels: Iterable[str] | None = None,
                   flexible: bool = False) -> Pseudobulk:
        """
        Concatenate one or more other Pseudobulk datasets with this one,
        sample-wise. All datasets must have the same cell types, and all
        datasets must have distinct `obs_names`.

        By default, all datasets must have the same `var`. They must also have
        the same columns in `obs`, with the same data types.

        Conversely, if `flexible=True`, subset to genes present in all datasets
        (according to the first column of `var`, i.e. the `var_names`) before
        concatenating. Subset to columns of `var` that are identical in all
        datasets after this subsetting. Also, subset to columns of `obs` that
        are present in all datasets, and have the same data types. All
        datasets' `obs_names` must have the same name and dtype, and similarly
        for their `var_names`.

        The one exception to the `obs` "same data type" rule: if a column is
        Enum in some datasets and Categorical in others, or Enum in all
        datasets but with different categories in each dataset, that column
        will be retained as an Enum column (with the union of the categories)
        in the concatenated `obs`.

        Args:
            datasets: one or more Pseudobulk datasets to concatenate with this
                      one
            *more_datasets: additional Pseudobulk datasets to concatenate with
                            this one, specified as positional arguments
            dataset_column: the name of an Enum column to be added to the
                            concatenated dataset's `obs` labeling which dataset
                            each cell came from. The labels themselves are
                            determined by the `dataset_labels` argument.
            dataset_labels: a sequence of labels for each dataset, used to
                            populate `dataset_column`. There must be one label
                            per dataset being concatenated. If `dataset_labels`
                            is not specified, the labels default to
                            `{dataset_column}_0`, `{dataset_column}_1`, ...,
                            `{dataset_column}_{N - 1}`. Can only be specified
                            when `dataset_column` is not `None`.
            flexible: whether to subset to genes and columns of `obs` and `var`
                      common to all datasets before concatenating, rather than
                      raising an error on any mismatches

        Returns:
            The concatenated Pseudobulk dataset.
        """
        # Check inputs
        if isinstance(datasets, Pseudobulk):
            datasets = self + (datasets,) + more_datasets
        else:
            datasets = self + tuple(datasets) + more_datasets
        if len(datasets) == 1:
            error_message = \
                'need at least one other Pseudobulk dataset to concatenate'
            raise ValueError(error_message)
        check_types(datasets[1:], 'datasets', Pseudobulk,
                    'Pseudobulk datasets')
        if dataset_column is not None:
            check_type(dataset_column, 'dataset_column', str, 'a string')
            if any(dataset_column in obs for dataset in datasets
                   for obs in dataset._obs.values()):
                error_message = (
                    f"dataset_column {dataset_column!r} is already a column "
                    f"of at least one dataset's obs in at least one cell "
                    f"type; specify a different name for dataset_column")
                raise ValueError(error_message)
            if dataset_labels is not None:
                dataset_labels = to_tuple_checked(
                    dataset_labels, 'dataset_labels', str, 'strings')
                if len(dataset_labels) != len(datasets):
                    error_message = (
                        f'dataset_labels has length {len(dataset_labels):,}, '
                        f'but there are {len(datasets):,} datasets being '
                        f'concatenated')
                    raise ValueError(error_message)
            else:
                dataset_labels = [f'dataset_{i}' for i in range(len(datasets))]
        elif dataset_labels is not None:
            error_message = (
                'when dataset_labels is specified, dataset_column must also '
                'be specified')
            raise ValueError(error_message)
        check_type(flexible, 'flexible', bool, 'Boolean')

        # Check that cell types match across all datasets
        if not (all(set(self.keys()) == set(dataset.keys())
                    for dataset in datasets[1:]) if flexible else
                all(self.keys() == dataset.keys()
                    for dataset in datasets[1:])):
            error_message = \
                'not all Pseudobulk datasets have the same cell types'
            raise ValueError(error_message)

        # Perform either flexible or non-flexible concatenation
        X = {}
        obs = {}
        var = {}
        for cell_type in self._obs:
            if flexible:
                # Check that `obs_names` and `var_names` have the same name and
                # data type for each cell type across all datasets
                obs_names_name = self._obs[cell_type][:, 0].name
                if not all(dataset._obs[cell_type][:, 0].name == obs_names_name
                           for dataset in datasets[1:]):
                    error_message = (
                        f'not all Pseudobulk datasets have the same name for '
                        f'the first column of obs (the obs_names column) for '
                        f'cell type {cell_type!r}')
                    raise ValueError(error_message)
                var_names_name = self._var[cell_type][:, 0].name
                if not all(dataset._var[cell_type][:, 0].name == var_names_name
                           for dataset in datasets[1:]):
                    error_message = (
                        f'not all Pseudobulk datasets have the same name for '
                        f'the first column of var (the var_names column) for '
                        f'cell type {cell_type!r}')
                    raise ValueError(error_message)
                obs_names_dtype = self._obs[cell_type][:, 0].dtype
                if not all(dataset._obs[cell_type][:, 0].dtype ==
                           obs_names_dtype for dataset in datasets[1:]):
                    error_message = (
                        f'not all Pseudobulk datasets have the same data type '
                        f'for the first column of obs (the obs_names column) '
                        f'for cell type {cell_type!r}')
                    raise TypeError(error_message)
                var_names_dtype = self._var[cell_type][:, 0].dtype
                if not all(dataset._var[cell_type][:, 0].dtype ==
                           var_names_dtype for dataset in datasets[1:]):
                    error_message = (
                        f'not all Pseudobulk datasets have the same data type '
                        f'for the first column of var (the var_names column) '
                        f'for cell type {cell_type!r}')
                    raise TypeError(error_message)

                # Subset to genes in common across all datasets
                genes_in_common = self._var[cell_type][:, 0]\
                    .to_frame()\
                    .filter(pl.all_horizontal(
                        self._var[cell_type][:, 0].is_in(
                            dataset._var[cell_type][:, 0])
                        for dataset in datasets[1:]))\
                    .to_series()
                if len(genes_in_common) == 0:
                    error_message = (
                        f'no genes are shared across all Pseudobulk datasets '
                        f'for cell type {cell_type!r}')
                    raise ValueError(error_message)
                cell_type_X = []
                cell_type_var = []
                for dataset in datasets:
                    if len(genes_in_common) == len(dataset._var[cell_type]) \
                            and dataset._var[cell_type][:, 0].equals(
                                genes_in_common):
                        cell_type_X.append(dataset._X[cell_type])
                        cell_type_var.append(dataset._var[cell_type])
                    else:
                        gene_indices = dataset._getitem_process(
                            (None, genes_in_common), 1,
                            dataset._var[cell_type])
                        cell_type_X.append(
                            dataset._X[cell_type][:, gene_indices.to_numpy()])
                        cell_type_var.append(
                            dataset._var[cell_type][gene_indices])

                # Subset to columns of `var` that are identical in all datasets
                # after this subsetting
                var_columns_in_common = [
                    column.name for column in cell_type_var[0][:, 1:]
                    if all(column.name in dataset_cell_type_var and
                           dataset_cell_type_var[column.name].equals(column)
                           for dataset_cell_type_var in cell_type_var[1:])]
                cell_type_var = cell_type_var[0]
                cell_type_var = cell_type_var.select(cell_type_var.columns[0],
                                                     var_columns_in_common)

                # Subset to columns of `obs` that are present in all datasets,
                # and have the same data types. Also include columns of `obs`
                # that are Enum in some datasets and Categorical in others, or
                # Enum in all datasets but with different categories in each
                # dataset; cast these to Enum.
                obs_mismatched_categoricals = {
                    column for column, dtype in self._obs[cell_type][:, 1:]
                    .select(pl.col(pl.Categorical, pl.Enum)).schema.items()
                    if all(column in dataset._obs[cell_type] and
                           dataset._obs[cell_type][column].dtype in
                           (pl.Categorical, pl.Enum)
                           for dataset in datasets[1:]) and
                       not all(dataset._obs[cell_type][column].dtype == dtype
                               for dataset in datasets[1:])}
                obs_columns_in_common = [
                    column
                    for column, dtype in islice(
                        self._obs[cell_type].schema.items(), 1, None)
                    if column in obs_mismatched_categoricals or
                       all(column in dataset._obs[cell_type] and
                           dataset._obs[cell_type][column].dtype == dtype
                           for dataset in datasets[1:])]
                cast_dict = {column: pl.Enum(
                    pl.concat([dataset._obs[cell_type][column]
                              .cat.get_categories() for dataset in datasets])
                    .unique(maintain_order=True))
                    for column in obs_mismatched_categoricals}
                cell_type_obs = [
                    dataset._obs[cell_type]
                    # the `.with_columns(...)` is a faster `.cast(cast_dict)`
                    .with_columns(cast_to_Enum(dataset._obs[cell_type][column],
                                               enum_type)
                                  .alias(column)
                                  for column, enum_type in cast_dict.items())
                    .select(dataset._obs[cell_type].columns[0],
                            obs_columns_in_common) for dataset in datasets]
            else:  # non-flexible
                # Check that all `var` are identical
                cell_type_var = self._var[cell_type]
                for dataset in datasets[1:]:
                    if not dataset._var[cell_type].equals(cell_type_var):
                        error_message = (
                            f'all Pseudobulk datasets must have the same var '
                            f'for cell type {cell_type!r}, unless '
                            f'flexible=True')
                        raise ValueError(error_message)

                # Check that all `obs` have the same columns and data types
                schema = self._obs[cell_type].schema
                for dataset in datasets[1:]:
                    if dataset._obs[cell_type].schema != schema:
                        error_message = (
                            f'all Pseudobulk datasets must have the same '
                            f'columns in obs for cell type {cell_type!r}, '
                            f'with the same data types, unless flexible=True')
                        raise ValueError(error_message)
                cell_type_X = [dataset._X[cell_type] for dataset in datasets]
                cell_type_obs = [dataset._obs[cell_type]
                                 for dataset in datasets]

            # If `dataset_column` is not `None`, add labels for each dataset
            if dataset_column is not None:
                cell_type_obs = [
                    obs_df.with_columns(pl.lit(label).alias(dataset_column))
                    for obs_df, label in zip(cell_type_obs, dataset_labels)]

            # Concatenate
            obs[cell_type] = pl.concat(cell_type_obs)
            num_unique = obs[cell_type][:, 0].n_unique()
            if num_unique < len(obs[cell_type]):
                error_message = (
                    f'obs_names contains {len(obs[cell_type]) - num_unique:,} '
                    f'duplicates after concatenation for cell type '
                    f'{cell_type!r}')
                raise ValueError(error_message)
            X[cell_type] = np.vstack(cell_type_X)
            var[cell_type] = cell_type_var
        return Pseudobulk(X=X, obs=obs, var=var, num_threads=self._num_threads)

    def concat_var(self,
                   datasets: Pseudobulk | Iterable[Pseudobulk],
                   /,
                   *more_datasets: Pseudobulk,
                   dataset_column: str | None = None,
                   dataset_labels: Iterable[str] | None = None,
                   flexible: bool = False) -> Pseudobulk:
        """
        Concatenate one or more other Pseudobulk datasets with this one,
        gene-wise. This is much less common than the sample-wise concatenation
        provided by `concat_obs()`. All datasets must have the same cell types,
        and all datasets must have distinct `var_names`.

        By default, all datasets must have the same `obs`. They must also have
        the same columns in `var`, with the same data types.

        Conversely, if `flexible=True`, subset to samples present in all
        datasets (according to the first column of `obs`, i.e. the `obs_names`)
        before concatenating. Subset to columns of `obs` that are identical in
        all datasets after this subsetting. Also, subset to columns of `var`
        that are present in all datasets, and have the same data types. All
        datasets' `obs_names` must have the same name and dtype, and similarly
        for their `var_names`.

        The one exception to the `var` "same data type" rule: if a column is
        Enum in some datasets and Categorical in others, or Enum in all
        datasets but with different categories in each dataset, that column
        will be retained as an Enum column (with the union of the categories)
        in the concatenated `var`.

        Args:
            datasets: one or more Pseudobulk datasets to concatenate with this
                      one
            *more_datasets: additional Pseudobulk datasets to concatenate with
                            this one, specified as positional arguments
            dataset_column: the name of an Enum column to be added to the
                            concatenated dataset's `var` labeling which dataset
                            each cell came from. The labels themselves are
                            determined by the `dataset_labels` argument.
            dataset_labels: a sequence of labels for each dataset, used to
                            populate `dataset_column`. There must be one label
                            per dataset being concatenated. If `dataset_labels`
                            is not specified, the labels default to
                            `{dataset_column}_0`, `{dataset_column}_1`, ...,
                            `{dataset_column}_{N - 1}`. Can only be specified
                            when `dataset_column` is not `None`.
            flexible: whether to subset to samples and columns of `obs` and
                      `var` common to all datasets before concatenating, rather
                      than raising an error on any mismatches

        Returns:
            The concatenated Pseudobulk dataset.
        """
        # Check inputs
        if isinstance(datasets, Pseudobulk):
            datasets = self + (datasets,) + more_datasets
        else:
            datasets = self + tuple(datasets) + more_datasets
        if len(datasets) == 1:
            error_message = \
                'need at least one other Pseudobulk dataset to concatenate'
            raise ValueError(error_message)
        check_types(datasets[1:], 'datasets', Pseudobulk,
                    'Pseudobulk datasets')
        if dataset_column is not None:
            check_type(dataset_column, 'dataset_column', str, 'a string')
            if any(dataset_column in var for dataset in datasets
                   for var in dataset._var.values()):
                error_message = (
                    f"dataset_column {dataset_column!r} is already a column "
                    f"of at least one dataset's var in at least one cell "
                    f"type; specify a different name for dataset_column")
                raise ValueError(error_message)
            if dataset_labels is not None:
                dataset_labels = to_tuple_checked(
                    dataset_labels, 'dataset_labels', str, 'strings')
                if len(dataset_labels) != len(datasets):
                    error_message = (
                        f'dataset_labels has length {len(dataset_labels):,}, '
                        f'but there are {len(datasets):,} datasets being '
                        f'concatenated')
                    raise ValueError(error_message)
            else:
                dataset_labels = [f'dataset_{i}' for i in range(len(datasets))]
        elif dataset_labels is not None:
            error_message = (
                'when dataset_labels is specified, dataset_column must also '
                'be specified')
            raise ValueError(error_message)
        check_type(flexible, 'flexible', bool, 'Boolean')

        # Check that cell types match across all datasets
        if not (all(set(self.keys()) == set(dataset.keys())
                    for dataset in datasets[1:]) if flexible else
                 all(self.keys() == dataset.keys()
                     for dataset in datasets[1:])):
            error_message = \
                'not all Pseudobulk datasets have the same cell types'
            raise ValueError(error_message)

        # Perform either flexible or non-flexible concatenation
        X = {}
        obs = {}
        var = {}
        for cell_type in self._var:
            if flexible:
                # Check that `var_names` and `obs_names` have the same name and
                # data type for each cell type across all datasets
                var_names_name = self._var[cell_type][:, 0].name
                if not all(dataset._var[cell_type][:, 0].name == var_names_name
                           for dataset in datasets[1:]):
                    error_message = (
                        f'not all Pseudobulk datasets have the same name for '
                        f'the first column of var (the var_names column) for '
                        f'cell type {cell_type!r}')
                    raise ValueError(error_message)
                obs_names_name = self._obs[cell_type][:, 0].name
                if not all(dataset._obs[cell_type][:, 0].name == obs_names_name
                           for dataset in datasets[1:]):
                    error_message = (
                        f'not all Pseudobulk datasets have the same name for '
                        f'the first column of obs (the obs_names column) for '
                        f'cell type {cell_type!r}')
                    raise ValueError(error_message)
                var_names_dtype = self._var[cell_type][:, 0].dtype
                if not all(dataset._var[cell_type][:, 0].dtype ==
                           var_names_dtype for dataset in datasets[1:]):
                    error_message = (
                        f'not all Pseudobulk datasets have the same data type '
                        f'for the first column of var (the var_names column) '
                        f'for cell type {cell_type!r}')
                    raise TypeError(error_message)
                obs_names_dtype = self._obs[cell_type][:, 0].dtype
                if not all(dataset._obs[cell_type][:, 0].dtype ==
                           obs_names_dtype for dataset in datasets[1:]):
                    error_message = (
                        f'not all Pseudobulk datasets have the same data type '
                        f'for the first column of obs (the obs_names column) '
                        f'for cell type {cell_type!r}')
                    raise TypeError(error_message)

                # Subset to samples in common across all datasets
                samples_in_common = self._obs[cell_type][:, 0]\
                    .to_frame()\
                    .filter(pl.all_horizontal(
                        self._obs[cell_type][:, 0].is_in(
                            dataset._obs[cell_type][:, 0])
                        for dataset in datasets[1:]))\
                    .to_series()
                if len(samples_in_common) == 0:
                    error_message = (
                        f'no samples are shared across all Pseudobulk '
                        f'datasets for cell type {cell_type!r}')
                    raise ValueError(error_message)
                cell_type_X = []
                cell_type_obs = []
                for dataset in datasets:
                    if len(samples_in_common) == len(dataset._obs[cell_type]) \
                        and dataset._obs[cell_type][:, 0].equals(
                            samples_in_common):
                        cell_type_X.append(dataset._X[cell_type])
                        cell_type_obs.append(dataset._obs[cell_type])
                    else:
                        sample_indices = dataset._getitem_process(
                            (samples_in_common,), 0, dataset._obs[cell_type])
                        cell_type_X.append(
                            dataset._X[cell_type][sample_indices.to_numpy()])
                        cell_type_obs.append(
                            dataset._obs[cell_type][sample_indices])

                # Subset to columns of `obs` that are identical in all datasets
                # after this subsetting
                obs_columns_in_common = [
                    column.name for column in cell_type_obs[0][:, 1:]
                    if all(column.name in dataset_cell_type_obs and
                           dataset_cell_type_obs[column.name].equals(column)
                           for dataset_cell_type_obs in cell_type_obs[1:])]
                cell_type_obs = cell_type_obs[0]
                cell_type_obs = cell_type_obs.select(cell_type_obs.columns[0],
                                                     obs_columns_in_common)

                # Subset to columns of `var` that are present in all datasets,
                # and have the same data types. Also include columns of `var`
                # that are Enum in some datasets and Categorical in others, or
                # Enum in all datasets but with different categories in each
                # dataset; cast these to Enum.
                var_mismatched_categoricals = {
                    column for column, dtype in self._var[cell_type][:, 1:]
                    .select(pl.col(pl.Categorical, pl.Enum)).schema.items()
                    if all(column in dataset._var[cell_type] and
                           dataset._var[cell_type][column].dtype in
                           (pl.Categorical, pl.Enum)
                           for dataset in datasets[1:]) and
                       not all(dataset._var[cell_type][column].dtype == dtype
                               for dataset in datasets[1:])}
                var_columns_in_common = [
                    column
                    for column, dtype in islice(
                        self._var[cell_type].schema.items(), 1, None)
                    if column in var_mismatched_categoricals or
                       all(column in dataset._var[cell_type] and
                           dataset._var[cell_type][column].dtype == dtype
                           for dataset in datasets[1:])]
                cast_dict = {column: pl.Enum(
                    pl.concat([dataset._var[cell_type][column]
                              .cat.get_categories() for dataset in datasets])
                    .unique(maintain_order=True))
                    for column in var_mismatched_categoricals}
                cell_type_var = [
                    dataset._var[cell_type]
                    # the `.with_columns(...)` is a faster `.cast(cast_dict)`
                    .with_columns(cast_to_Enum(dataset._var[cell_type][column],
                                               enum_type)
                                  .alias(column)
                                  for column, enum_type in cast_dict.items())
                    .select(dataset._var[cell_type].columns[0],
                            var_columns_in_common) for dataset in datasets]
            else:  # non-flexible
                # Check that all `obs` are identical
                cell_type_obs = self._obs[cell_type]
                for dataset in datasets[1:]:
                    if not dataset._obs[cell_type].equals(cell_type_obs):
                        error_message = (
                            f'all Pseudobulk datasets must have the same obs '
                            f'for cell type {cell_type!r}, unless '
                            f'flexible=True')
                        raise ValueError(error_message)

                # Check that all `var` have the same columns and data types
                schema = self._var[cell_type].schema
                for dataset in datasets[1:]:
                    if dataset._var[cell_type].schema != schema:
                        error_message = (
                            f'all Pseudobulk datasets must have the same '
                            f'columns in var for cell type {cell_type!r}, '
                            f'with the same data types, unless flexible=True')
                        raise ValueError(error_message)
                cell_type_X = [dataset._X[cell_type] for dataset in datasets]
                cell_type_var = [dataset._var[cell_type]
                                 for dataset in datasets]

            # If `dataset_column` is not `None`, add labels for each dataset
            if dataset_column is not None:
                cell_type_var = [
                    var_df.with_columns(pl.lit(label).alias(dataset_column))
                    for var_df, label in zip(cell_type_var, dataset_labels)]

            # Concatenate
            var[cell_type] = pl.concat(cell_type_var)
            num_unique = var[cell_type][:, 0].n_unique()
            if num_unique < len(var[cell_type]):
                error_message = (
                    f'var_names contains {len(var[cell_type]) - num_unique:,} '
                    f'duplicates after concatenation for cell type '
                    f'{cell_type!r}')
                raise ValueError(error_message)
            X[cell_type] = np.hstack(cell_type_X)
            obs[cell_type] = cell_type_obs
        return Pseudobulk(X=X, obs=obs, var=var, num_threads=self._num_threads)

    def _get_column(self,
                    obs_or_var_name: Literal['obs', 'var'],
                    column: PseudobulkColumn | None |
                            dict[str, PseudobulkColumn | None],
                    variable_name: str,
                    dtypes: pl.datatypes.classes.DataTypeClass | str |
                            tuple[pl.datatypes.classes.DataTypeClass | str,
                                  ...],
                    custom_error: str | None = None,
                    allow_None: bool = True,
                    allow_null: bool = False,
                    cell_types: Sequence[str] | None = None) -> \
            dict[str, pl.Series | None]:
        """
        Get a column of the same length as `obs` or `var` for each cell type.

        Args:
            obs_or_var_name: the name of the DataFrame the column is with
                             respect to, i.e. `'obs'` or `'var'`
            column: a string naming a column of each cell type's `obs`/`var`, a
                    polars expression that evaluates to a single column when
                    applied to each cell type's `obs`/`var`, a polars Series or
                    NumPy array of the same length as each cell type's
                    `obs`/`var`, or a function that takes in two arguments,
                    `self` and a cell type, and returns a polars Series or
                    NumPy array of the same length as `obs`/`var`. Or, a
                    dictionary mapping cell-type names to any of the above;
                    each cell type in this Pseudobulk dataset must be present.
                    May also be `None` (or a dictionary containing `None`
                    values) if `allow_None=True`.
            variable_name: the name of the variable corresponding to `columns`
            dtypes: the required dtype(s) of the column
            custom_error: a custom error message for when (an element of)
                          `columns` is a string and is not found in
                          `obs`/`var`; use `{}` as a placeholder for the names
                          of the column and cell type (which must appear in
                          that order)
            allow_None: whether to allow `columns` or its elements to be `None`
            allow_null: whether to allow `columns` to contain `null` values
            cell_types: a list of cell types; if `None`, use all cell types. If
                        specified and `column` is a Sequence, `column` and
                        `cell_types` should have the same length.

        Returns:
            A dictionary mapping each cell type to a polars Series of the same
            length as the cell type's `obs`/`var`. Or, if `columns` is `None`
            (or if some elements are `None`), a dict where all (or some) values
            are `None`.
        """
        obs_or_var = self._obs if obs_or_var_name == 'obs' else self._var
        if cell_types is None:
            cell_types = self._X
        if column is None:
            if not allow_None:
                error_message = f'{variable_name} is None'
                raise TypeError(error_message)
            return {cell_type: None for cell_type in cell_types}
        columns = {}
        if isinstance(column, str):
            for cell_type in cell_types:
                if column not in obs_or_var[cell_type]:
                    error_message = (
                        f'{variable_name} {column!r} is not a column of '
                        f'{obs_or_var_name}[{cell_type!r}]'
                        if custom_error is None else
                        custom_error.format(f'{column!r}', f'{cell_type!r}'))
                    raise ValueError(error_message)
                columns[cell_type] = obs_or_var[cell_type][column]
        elif isinstance(column, pl.Expr):
            for cell_type in cell_types:
                columns[cell_type] = obs_or_var[cell_type].select(column)
                if columns[cell_type].width > 1:
                    error_message = (
                        f'{variable_name} is a polars expression that expands '
                        f'to {columns[cell_type].width:,} columns rather '
                        f'than 1 for cell type {cell_type!r}')
                    raise ValueError(error_message)
                columns[cell_type] = columns[cell_type].to_series()
        elif isinstance(column, pl.Series):
            for cell_type in cell_types:
                if len(column) != len(obs_or_var[cell_type]):
                    error_message = (
                        f'{variable_name} is a polars Series of length '
                        f'{len(column):,}, which differs from the length of '
                        f'{obs_or_var_name}[{cell_type!r}] '
                        f'({len(obs_or_var[cell_type]):,})')
                    raise ValueError(error_message)
                columns[cell_type] = column
        elif isinstance(column, np.ndarray):
            for cell_type in cell_types:
                if len(column) != len(obs_or_var[cell_type]):
                    error_message = (
                        f'{variable_name} is a NumPy array of length '
                        f'{len(column):,}, which differs from the length of '
                        f'{obs_or_var_name}[{cell_type!r}] '
                        f'({len(obs_or_var[cell_type]):,})')
                    raise ValueError(error_message)
                columns[cell_type] = pl.Series(variable_name, column)
        elif callable(column):
            function = column
            for cell_type in cell_types:
                col = function(self, cell_type)
                if isinstance(col, np.ndarray):
                    if col.ndim != 1:
                        error_message = (
                            f'{variable_name} is a function that returns a '
                            f'{col.ndim:,}D NumPy array, but must return a '
                            f'polars Series or 1D NumPy array')
                        raise ValueError(error_message)
                    col = pl.Series(variable_name, col)
                elif not isinstance(col, pl.Series):
                    error_message = (
                        f'{variable_name} is a function that returns a '
                        f'variable of type {type(col).__name__}, but must '
                        f'return a polars Series or 1D NumPy array')
                    raise TypeError(error_message)
                if len(col) != len(obs_or_var[cell_type]):
                    error_message = (
                        f'{variable_name} is a function that returns a column '
                        f'of length {len(col):,} for cell type {cell_type!r}, '
                        f'which differs from the length of '
                        f'{obs_or_var_name}[{cell_type!r}] '
                        f'({len(obs_or_var[cell_type]):,})')
                    raise ValueError(error_message)
                columns[cell_type] = col
        elif isinstance(column, dict):
            if len(column) != len(cell_types):
                error_message = (
                    f'{variable_name} is a dictionary of length '
                    f'{len(column):,}, which differs from the number of cell '
                    f'types ({len(cell_types):,})')
                raise ValueError(error_message)
            column_set = set(column)
            cell_type_set = set(cell_types)
            if column_set != cell_type_set:
                overlap = len(column_set & cell_type_set)
                if overlap:
                    error_message = (
                        f'{variable_name} is a dictionary of the same length '
                        f'as the number of cell types, but only {overlap:,} '
                        f'of its {len(cell_types):,} keys '
                        f'{plural("correspond", overlap)} to cell types')
                    raise ValueError(error_message)
                else:
                    error_message = (
                        f'{variable_name} is a dictionary of the same length '
                        f'as the number of cell types, but None of its '
                        f'{len(cell_types):,} keys correspond to cell types')
                    raise ValueError(error_message)
            for cell_type, col in column.items():
                if col is None:
                    if allow_None:
                        columns[cell_type] = None
                    else:
                        error_message = \
                            f'{variable_name}[{cell_type!r}] is None'
                        raise TypeError(error_message)
                elif isinstance(col, str):
                    if col not in obs_or_var[cell_type]:
                        error_message = (
                            f'{variable_name}[{cell_type!r}] {col!r} is not a '
                            f'column of {obs_or_var_name}[{cell_type!r}]'
                            if custom_error is None else
                            custom_error.format(f'{col!r}', f'{cell_type!r}'))
                        raise ValueError(error_message)
                    columns[cell_type] = obs_or_var[cell_type][col]
                elif isinstance(col, pl.Expr):
                    col = obs_or_var[cell_type].select(col)
                    if col.width > 1:
                        error_message = (
                            f'{variable_name}[{cell_type!r}] is a polars '
                            f'expression that expands to {col.width:,} '
                            f'columns rather than 1')
                        raise ValueError(error_message)
                    columns[cell_type] = col.to_series()
                elif isinstance(col, pl.Series):
                    if len(col) != len(obs_or_var[cell_type]):
                        error_message = (
                            f'{variable_name}[{cell_type!r}] is a polars '
                            f'Series of length {len(col):,}, which differs '
                            f'from the length of '
                            f'{obs_or_var_name}[{cell_type!r}] '
                            f'({len(obs_or_var[cell_type]):,})')
                        raise ValueError(error_message)
                    columns[cell_type] = col
                elif isinstance(col, np.ndarray):
                    if len(col) != len(obs_or_var[cell_type]):
                        error_message = (
                            f'{variable_name}[{cell_type!r}] is a NumPy array '
                            f'of length {len(col):,}, which differs from the '
                            f'length of {obs_or_var_name}[{cell_type!r}] '
                            f'({len(obs_or_var[cell_type]):,})')
                        raise ValueError(error_message)
                    columns[cell_type] = pl.Series(variable_name, col)
                elif callable(col):
                    col = col(self, cell_type)
                    if isinstance(col, np.ndarray):
                        if col.ndim != 1:
                            error_message = (
                                f'{variable_name}[{cell_type!r}] is a '
                                f'function that returns a {col.ndim:,}D NumPy '
                                f'array, but must return a polars Series or '
                                f'1D NumPy array')
                            raise ValueError(error_message)
                        col = pl.Series(variable_name, col)
                    elif not isinstance(col, pl.Series):
                        error_message = (
                            f'{variable_name}[{cell_type!r}] is a function '
                            f'that returns a variable of type '
                            f'{type(col).__name__}, but must return a '
                            f'polars Series or 1D NumPy array')
                        raise TypeError(error_message)
                    if len(col) != len(obs_or_var[cell_type]):
                        error_message = (
                            f'{variable_name}[{cell_type!r}] is a function '
                            f'that returns a column of length {len(col):,}, '
                            f'which differs from the length of '
                            f'{obs_or_var_name}[{cell_type!r}] '
                            f'({len(obs_or_var[cell_type]):,})')
                        raise ValueError(error_message)
                    columns[cell_type] = col
                else:
                    error_message = (
                        f'{variable_name}[{cell_type!r}] must be a string '
                        f'column name, a polars expression or Series, a 1D '
                        f'NumPy array, or a function that returns any of '
                        f'these when applied to this Pseudobulk dataset and a '
                        f'given cell type, but has type '
                        f'{type(col).__name__!r}')
                    raise TypeError(error_message)
        else:
            error_message = (
                f'{variable_name} must be a string column name, a polars '
                f'expression or Series, a 1D NumPy array, or a function that '
                f'returns any of these when applied to this Pseudobulk '
                f'dataset and a given cell type, but has type '
                f'{type(column).__name__!r}')
            raise TypeError(error_message)

        # Check dtypes
        if not isinstance(dtypes, tuple):
            dtypes = dtypes,
        for cell_type, col in columns.items():
            base_type = col.dtype.base_type()
            for expected_type in dtypes:
                if base_type == expected_type or expected_type == 'integer' \
                        and base_type in INTEGER_DTYPES or \
                        expected_type == 'floating-point' and \
                        base_type in FLOAT_DTYPES:
                    break
            else:
                if len(dtypes) == 1:
                    dtypes = str(dtypes[0])
                elif len(dtypes) == 2:
                    dtypes = ' or '.join(map(str, dtypes))
                else:
                    dtypes = \
                        ', '.join(map(str, dtypes[:-1])) + f', or {dtypes[-1]}'
                if isinstance(column, str):
                    error_message = (
                        f'{variable_name} {obs_or_var_name}[{cell_type!r}]'
                        f'[{columns!r}] must be {dtypes}, but has data type '
                        f'{base_type!r}')
                    raise TypeError(error_message)
                else:
                    error_message = (
                        f'{variable_name} must be {dtypes}, but has data type '
                        f'{base_type!r} for cell type {cell_type!r}')
                    raise TypeError(error_message)

        # Check `null` values, if `allow_null=False`
        if not allow_null:
            for cell_type, col in columns.items():
                null_count = col.null_count()
                if null_count > 0:
                    full_variable_name = \
                        f'{variable_name} {obs_or_var_name}[{cell_type!r}]' \
                        f'[{columns!r}]' if isinstance(column, str) else \
                            variable_name
                    error_message = (
                        f'{full_variable_name} contains {null_count:,} '
                        f'{plural("null value", null_count)} for cell type '
                        f'{cell_type!r}, but must not contain any')
                    raise ValueError(error_message)
        return columns

    def _describe_column(self,
                         column_name: str,
                         column: SingleCellColumn,
                         cell_type: str):
        """
        Describe a column-name argument in an error message.

        Args:
            column_name: the name of the column-name argument
            column: the value of the column-name argument
            cell_type: the cell type where the error was triggered

        Returns:
            The column's description: just the argument's name unless the value
            (for this cell type) is a string (i.e. the column's name in `obs`
            or `var`), in which case also include the value.
        """
        if isinstance(column, Sequence):
            cell_type_index = next(i for i, cell_type_i in enumerate(self._obs)
                                   if cell_type_i == cell_type)
            column = column[cell_type_index]
        return f'{column_name} {column!r}' \
            if isinstance(column, str) else column_name

    def filter_obs(self,
                   *predicates: str | pl.Expr | pl.Series |
                                Iterable[str | pl.Expr | pl.Series] | bool |
                                list[bool] | np.ndarray[np.dtype[np.bool_]],
                   cell_types: str | Iterable[str] | None = None,
                   excluded_cell_types: str | Iterable[str] | None = None,
                   **constraints: Any) -> Pseudobulk:
        """
        Equivalent to
        [`df.filter()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.filter.html)
        from polars, but applied to both `obs` and `X` for each cell type.

        Args:
            *predicates: one or more column names, expressions that evaluate to
                         Boolean Series, Boolean Series, lists of Booleans,
                         and/or 1D Boolean NumPy arrays
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **constraints: column filters: `name=value` filters to samples
                           where the column named `name` has the value `value`

        Returns:
            A new Pseudobulk dataset filtered to samples passing all the
            Boolean filters in `predicates` and `constraints`.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        X = {}
        obs = {}
        for cell_type in self._obs:
            if cell_type in cell_types:
                obs[cell_type] = self._obs[cell_type]\
                    .with_columns(_Pseudobulk_index=pl.int_range(
                        pl.len(), dtype=pl.UInt32))\
                    .filter(*predicates, **constraints)
                X[cell_type] = self._X[cell_type][
                    obs[cell_type]['_Pseudobulk_index'].to_numpy()]
                obs[cell_type] = obs[cell_type].drop('_Pseudobulk_index')
            else:
                X[cell_type] = self._X[cell_type]
                obs[cell_type] = self._obs[cell_type]
        return Pseudobulk(X=X, obs=obs, var=self._var,
                          num_threads=self._num_threads)

    def filter_var(self,
                   *predicates: pl.Expr | pl.Series | str |
                                Iterable[pl.Expr | pl.Series | str] | bool |
                                list[bool] | np.ndarray[np.dtype[np.bool_]],
                   cell_types: str | Iterable[str] | None = None,
                   excluded_cell_types: str | Iterable[str] | None = None,
                   **constraints: Any) -> Pseudobulk:
        """
        Equivalent to
        [`df.filter()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.filter.html)
        from polars, but applied to both `var` and `X` for each cell type.

        Args:
            *predicates: one or more column names, expressions that evaluate to
                         Boolean Series, Boolean Series, lists of Booleans,
                         and/or 1D Boolean NumPy arrays
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **constraints: column filters: `name=value` filters to genes
                           where the column named `name` has the value `value`

        Returns:
            A new Pseudobulk dataset filtered to genes passing all the
            Boolean filters in `predicates` and `constraints`.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        X = {}
        var = {}
        for cell_type in self._var:
            if cell_type in cell_types:
                var[cell_type] = self._var[cell_type]\
                    .with_columns(_Pseudobulk_index=pl.int_range(
                        pl.len(), dtype=pl.UInt32))\
                    .filter(*predicates, **constraints)
                X[cell_type] = self._X[cell_type][
                    :, var[cell_type]['_Pseudobulk_index'].to_numpy()]
                var[cell_type] = var[cell_type].drop('_Pseudobulk_index')
            else:
                X[cell_type] = self._X[cell_type]
                var[cell_type] = self._var[cell_type]
        return Pseudobulk(X=X, obs=self._obs, var=var,
                          num_threads=self._num_threads)

    def select_obs(self,
                   *exprs: Scalar | pl.Expr | pl.Series |
                           Iterable[Scalar | pl.Expr | pl.Series],
                   cell_types: str | Iterable[str] | None = None,
                   excluded_cell_types: str | Iterable[str] | None = None,
                   **named_exprs: Scalar | pl.Expr | pl.Series) -> Pseudobulk:
        """
        Equivalent to
        [`df.select()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.select.html)
        from polars, but applied to each cell type's `obs`. `obs_names` will be
        automatically included as the first column, if not included explicitly.

        Args:
            *exprs: column(s) to select, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **named_exprs: additional columns to select, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.

        Returns:
            A new Pseudobulk dataset with
            `obs[cell_type]=obs[cell_type].select(*exprs, **named_exprs)` for
            all cell types in `obs`, and `obs_names` as the first column unless
            already included explicitly.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        obs = {}
        for cell_type, cell_type_obs in self._obs.items():
            if cell_type in cell_types:
                new_cell_type_obs = cell_type_obs.select(*exprs, **named_exprs)
                if cell_type_obs.columns[0] not in new_cell_type_obs:
                    new_cell_type_obs = \
                        new_cell_type_obs.select(cell_type_obs[:, 0], pl.all())
                obs[cell_type] = new_cell_type_obs
            else:
                obs[cell_type] = cell_type_obs
        return Pseudobulk(X=self._X, obs=obs, var=self._var,
                          num_threads=self._num_threads)

    def select_var(self,
                   *exprs: Scalar | pl.Expr | pl.Series |
                           Iterable[Scalar | pl.Expr | pl.Series],
                   cell_types: str | Iterable[str] | None = None,
                   excluded_cell_types: str | Iterable[str] | None = None,
                   **named_exprs: Scalar | pl.Expr | pl.Series) -> Pseudobulk:
        """
        Equivalent to
        [`df.select()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.select.html)
        from polars, but applied to each cell type's `var`. `var_names` will be
        automatically included as the first column, if not included explicitly.

        Args:
            *exprs: column(s) to select, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **named_exprs: additional columns to select, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.

        Returns:
            A new Pseudobulk dataset with
            `var[cell_type]=var[cell_type].select(*exprs, **named_exprs)` for
            all cell types in `var`, and `var_names` as the first column unless
            already included explicitly.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        var = {}
        for cell_type, cell_type_var in self._var.items():
            if cell_type in cell_types:
                new_cell_type_var = cell_type_var.select(*exprs, **named_exprs)
                if cell_type_var.columns[0] not in new_cell_type_var:
                    new_cell_type_var = \
                        new_cell_type_var.select(cell_type_var[:, 0], pl.all())
                var[cell_type] = new_cell_type_var
            else:
                var[cell_type] = cell_type_var
        return Pseudobulk(X=self._X, obs=self._obs, var=var,
                          num_threads=self._num_threads)

    def select_cell_types(self,
                          cell_types: str | Iterable[str],
                          /,
                          *more_cell_types: str) -> Pseudobulk:
        """
        Create a new Pseudobulk dataset subset to the cell type(s) in
        `cell_types` and `more_cell_types`.

        Args:
            cell_types: cell type(s) to select
            *more_cell_types: additional cell types to select, specified as
                              positional arguments

        Returns:
            A new Pseudobulk dataset subset to the specified cell type(s).
        """
        cell_types = to_tuple_checked(cell_types, 'cell_types', str, 'strings')
        check_types(more_cell_types, 'more_cell_types', str, 'strings')
        cell_types += more_cell_types
        for cell_type in cell_types:
            if cell_type not in self._X:
                error_message = (
                    f'tried to select {cell_type!r}, which is not a cell type '
                    f'in this Pseudobulk')
                raise ValueError(error_message)
        return Pseudobulk(X={cell_type: self._X[cell_type]
                             for cell_type in cell_types},
                          obs={cell_type: self._obs[cell_type]
                               for cell_type in cell_types},
                          var={cell_type: self._var[cell_type]
                               for cell_type in cell_types},
                          num_threads=self._num_threads)

    def with_columns_obs(self,
                         *exprs: Scalar | pl.Expr | pl.Series |
                                 Iterable[Scalar | pl.Expr | pl.Series],
                         cell_types: str | Iterable[str] | None = None,
                         excluded_cell_types: str | Iterable[str] |
                                              None = None,
                         **named_exprs: Scalar | pl.Expr | pl.Series) -> \
            Pseudobulk:
        """
        Equivalent to
        [`df.with_columns()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_columns.html)
        from polars, but applied to each cell type's `obs`.

        Args:
            *exprs: column(s) to add, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **named_exprs: additional columns to add, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.

        Returns:
            A new Pseudobulk dataset with
            `obs[cell_type]=obs[cell_type].with_columns(*exprs, **named_exprs)`
            for all cell types.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        return Pseudobulk(
            X=self._X,
            obs={cell_type: obs.with_columns(*exprs, **named_exprs)
                 if cell_type in cell_types else obs
                 for cell_type, obs in self._obs.items()},
            var=self._var, num_threads=self._num_threads)

    def with_columns_var(self,
                         *exprs: Scalar | pl.Expr | pl.Series |
                                 Iterable[Scalar | pl.Expr | pl.Series],
                         cell_types: str | Iterable[str] | None = None,
                         excluded_cell_types: str | Iterable[str] |
                                              None = None,
                         **named_exprs: Scalar | pl.Expr | pl.Series) -> \
            Pseudobulk:
        """
        Equivalent to
        [`df.with_columns()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_columns.html)
        rom polars, but applied to each cell type's `var`.

        Args:
            *exprs: column(s) to add, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **named_exprs: additional columns to add, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.

        Returns:
            A new Pseudobulk dataset with
            `var[cell_type]=var[cell_type].with_columns(*exprs, **named_exprs)`
            for all cell types.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        return Pseudobulk(
            X=self._X, obs=self._obs,
            var={cell_type: var.with_columns(*exprs, **named_exprs)
                 if cell_type in cell_types else var
                 for cell_type, var in self._var.items()},
            num_threads=self._num_threads)

    def drop_obs(self,
                 columns: pl.type_aliases.ColumnNameOrSelector |
                          Iterable[pl.type_aliases.ColumnNameOrSelector],
                 /,
                 *more_columns: pl.type_aliases.ColumnNameOrSelector,
                 cell_types: str | Iterable[str] | None = None,
                 excluded_cell_types: str | Iterable[str] | None = None) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with `columns` and `more_columns`
        removed from `obs`.

        Args:
            columns: columns(s) to drop
            *more_columns: additional columns to drop, specified as
                           positional arguments
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`

        Returns:
            A new Pseudobulk dataset with the column(s) removed.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        columns = to_tuple(columns) + more_columns
        return Pseudobulk(X=self._X,
                          obs={cell_type: obs.drop(columns)
                                          if cell_type in cell_types else obs
                               for cell_type, obs in self._obs.items()},
                          var=self._var, num_threads=self._num_threads)

    def drop_var(self,
                 columns: pl.type_aliases.ColumnNameOrSelector |
                          Iterable[pl.type_aliases.ColumnNameOrSelector],
                 /,
                 *more_columns: pl.type_aliases.ColumnNameOrSelector,
                 cell_types: str | Iterable[str] | None = None,
                 excluded_cell_types: str | Iterable[str] | None = None) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with `columns` and `more_columns`
        removed from `var`.

        Args:
            columns: columns(s) to drop
            *more_columns: additional columns to drop, specified as
                           positional arguments
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`

        Returns:
            A new Pseudobulk dataset with the column(s) removed.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        columns = to_tuple(columns) + more_columns
        return Pseudobulk(X=self._X, obs=self._obs,
                          var={cell_type: var.drop(columns)
                                          if cell_type in cell_types else var
                               for cell_type, var in self._var.items()},
                          num_threads=self._num_threads)

    def drop_cell_types(self,
                        cell_types: str | Iterable[str],
                        /,
                        *more_cell_types: str) -> Pseudobulk:
        """
        Create a new Pseudobulk dataset with `cell_types` and `more_cell_types`
        removed. Raises an error if all cell types would be dropped.

        Args:
            cell_types: cell type(s) to drop
            *more_cell_types: additional cell types to drop, specified as
                              positional arguments

        Returns:
            A new Pseudobulk dataset with the cell type(s) removed.
        """
        cell_types = to_tuple_checked(cell_types, 'cell_types', str, 'strings')
        check_types(more_cell_types, 'more_cell_types', str, 'strings')
        cell_types = set(cell_types) | set(more_cell_types)
        original_cell_types = set(self)
        if not cell_types < original_cell_types:
            if cell_types == original_cell_types:
                error_message = 'all cell types would be dropped'
                raise ValueError(error_message)
            for cell_type in cell_types:
                if cell_type not in original_cell_types:
                    error_message = (
                        f'tried to drop {cell_type!r}, which is not a cell '
                        f'type in this Pseudobulk')
                    raise ValueError(error_message)
        new_cell_types = \
            [cell_type for cell_type in self if cell_type not in cell_types]
        return Pseudobulk(X={cell_type: self._X[cell_type]
                             for cell_type in new_cell_types},
                          obs={cell_type: self._obs[cell_type]
                               for cell_type in new_cell_types},
                          var={cell_type: self._var[cell_type]
                               for cell_type in new_cell_types},
                          num_threads=self._num_threads)

    def rename_obs(self,
                   mapping: dict[str, str] | Callable[[str], str],
                   /,
                   *,
                   cell_types: str | Iterable[str] | None = None,
                   excluded_cell_types: str | Iterable[str] | None = None) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with column(s) of `obs` renamed for
        each cell type.

        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`

        Returns:
            A new Pseudobulk dataset with the column(s) of `obs` renamed.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        return Pseudobulk(
            X=self._X,
            obs={cell_type: obs.rename(mapping)
                            if cell_type in cell_types else obs
                 for cell_type, obs in self._obs.items()},
            var=self._var, num_threads=self._num_threads)

    def rename_var(self,
                   mapping: dict[str, str] | Callable[[str], str],
                   /,
                   *,
                   cell_types: str | Iterable[str] | None = None,
                   excluded_cell_types: str | Iterable[str] | None = None) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with column(s) of `var` renamed for
        each cell type.

        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`

        Returns:
            A new Pseudobulk dataset with the column(s) of `var` renamed.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        return Pseudobulk(
            X=self._X, obs=self._obs,
            var={cell_type: var.rename(mapping)
                            if cell_type in cell_types else var
                 for cell_type, var in self._var.items()},
            num_threads=self._num_threads)

    def rename_cell_types(self,
                          mapping: dict[str, str] | Callable[[str], str],
                          /) -> Pseudobulk:
        """
        Create a new Pseudobulk dataset with cell type(s) renamed.

        Args:
            mapping: the renaming to apply, either as a dictionary with old
                     cell type names as keys and new names as values, or a
                     function that takes an old name and returns a new name.
                     If `mapping` is a dictionary, cell types missing from its
                     keys will retain their original names.

        Returns:
            A new Pseudobulk dataset with the cell type(s) renamed.
        """
        if isinstance(mapping, dict):
            check_types(mapping.keys(), 'mapping.keys()', str, 'strings')
            check_types(mapping.values(), 'mapping.values()', str, 'strings')
            for key, new_key in mapping.items():
                if key not in self._X:
                    error_message = \
                        f'tried to rename {key!r}, which is not a cell type'
                    raise ValueError(error_message)
                if new_key in self._X:
                    error_message = (
                        f'tried to rename cell type {key!r} to {new_key!r}, '
                        f'but cell type {new_key!r} already exists')
                    raise ValueError(error_message)
            new_cell_types = [mapping.get(cell_type, cell_type)
                              for cell_type in self._X]
        elif isinstance(mapping, Callable):
            new_cell_types = []
            for cell_type in self._X:
                new_cell_type = mapping(cell_type)
                if not isinstance(new_cell_type, str):
                    error_message = (
                        f'tried to rename cell type {cell_type!r} to a '
                        f'non-string value of type '
                        f'{type(new_cell_type).__name__!r}')
                    raise TypeError(error_message)
                if new_cell_type in self._X:
                    error_message = (
                        f'tried to rename cell type {cell_type!r} to '
                        f'{new_cell_type!r}, but cell type {new_cell_type!r} '
                        f'already exists')
                    raise ValueError(error_message)
                new_cell_types.append(new_cell_type)
        else:
            error_message = (
                f'mapping must be a dictionary or function, but has type '
                f'{type(mapping).__name__!r}')
            raise TypeError(error_message)
        if len(set(new_cell_types)) < len(self._X):
            error_message = \
                'renaming would map multiple cell types to the same new name'
            raise ValueError(error_message)
        return Pseudobulk(X={new_cell_type: X
                             for new_cell_type, X in
                             zip(new_cell_types, self._X.values())},
                          obs={new_cell_type: obs
                               for new_cell_type, obs in
                               zip(new_cell_types, self._obs.values())},
                          var={new_cell_type: var
                               for new_cell_type, var in
                               zip(new_cell_types, self._var.values())},
                          num_threads=self._num_threads)

    def cast_X(self,
               dtype: np._typing.DTypeLike,
               /,
               *,
               cell_types: str | Iterable[str] | None = None,
               excluded_cell_types: str | Iterable[str] |
                                    None = None) -> Pseudobulk:
        """
        Cast each cell type's `X` to the specified data type.

        Args:
            dtype: a NumPy data type
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`

        Returns:
            A new Pseudobulk dataset with each cell type's `X` cast to the
            specified data type.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        return Pseudobulk(X={cell_type: X.astype(dtype)
                                        if cell_type in cell_types else X
                             for cell_type, X in self._X.items()},
                          obs=self._obs, var=self._var,
                          num_threads=self._num_threads)

    def cast_obs(self,
                 dtypes: Mapping[pl.type_aliases.ColumnNameOrSelector |
                                 pl.type_aliases.PolarsDataType,
                                 pl.type_aliases.PolarsDataType] |
                         pl.type_aliases.PolarsDataType,
                 /,
                 *,
                 cell_types: str | Iterable[str] | None = None,
                 excluded_cell_types: str | Iterable[str] | None = None,
                 strict: bool = True) -> Pseudobulk:
        """
        Cast column(s) of each cell type's `obs` to the specified data type(s).

        Args:
            dtypes: a mapping of column names (or selectors) to data types, or
                    a single data type to which all columns will be cast
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            strict: whether to raise an error if a cast could not be performed
                    (for instance, due to numerical overflow)

        Returns:
            A new Pseudobulk dataset with column(s) of each cell type's `obs`
            cast to the specified data type(s).
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        return Pseudobulk(X=self._X,
                          obs={cell_type: obs.cast(dtypes, strict=strict)
                                          if cell_type in cell_types else obs
                               for cell_type, obs in self._obs.items()},
                          var=self._var, num_threads=self._num_threads)

    def cast_var(self,
                 dtypes: Mapping[pl.type_aliases.ColumnNameOrSelector |
                                 pl.type_aliases.PolarsDataType,
                                 pl.type_aliases.PolarsDataType] |
                         pl.type_aliases.PolarsDataType,
                 /,
                 *,
                 cell_types: str | Iterable[str] | None = None,
                 excluded_cell_types: str | Iterable[str] | None = None,
                 strict: bool = True) -> Pseudobulk:
        """
        Cast column(s) of each cell type's `var` to the specified data type(s).

        Args:
            dtypes: a mapping of column names (or selectors) to data types, or
                    a single data type to which all columns will be cast
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            strict: whether to raise an error if a cast could not be performed
                    (for instance, due to numerical overflow)

        Returns:
            A new Pseudobulk dataset with column(s) of each cell type's `var`
            cast to the specified data type(s).
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        return Pseudobulk(X=self._X,
                          obs=self._obs,
                          var={cell_type: var.cast(dtypes, strict=strict)
                                          if cell_type in cell_types else var
                               for cell_type, var in self._var.items()},
                          num_threads=self._num_threads)

    def join_obs(self,
                 other: pl.DataFrame,
                 /,
                 *,
                 cell_types: str | Iterable[str] | None = None,
                 excluded_cell_types: str | Iterable[str] | None = None,
                 on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
                 left_on: str | pl.Expr | Sequence[str | pl.Expr] |
                          None = None,
                 right_on: str | pl.Expr | Sequence[str | pl.Expr] |
                           None = None,
                 suffix: str = '_right',
                 validate: Literal['m:m', 'm:1', '1:m', '1:1'] = 'm:m',
                 nulls_equal: bool = False,
                 coalesce: bool = True) -> Pseudobulk:
        """
        Left-join each cell type's `obs` with another DataFrame, using the same
        logic as
        [`df.join()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html).

        Args:
            other: a polars DataFrame to join each cell type's `obs` with
            on: the name(s) of the join column(s) in both DataFrames
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            left_on: the name(s) of the join column(s) in `obs`
            right_on: the name(s) of the join column(s) in `other`
            suffix: a suffix to append to columns with a duplicate name
            validate: checks whether the join is of the specified type. Can be:

                      - 'm:m' (many-to-many): the default, no checks performed.
                      - '1:1' (one-to-one): check that none of the values in
                        the join column(s) appear more than once in `obs` or
                        more than once in `other`.
                      - '1:m' (one-to-many): check that none of the values in
                        the join column(s) appear more than once in `obs`.
                      - 'm:1' (many-to-one): check that none of the values in
                        the join column(s) appear more than once in `other`.

            nulls_equal: whether to include `null` as a valid value to join on.
                        By default, `null` values will never produce matches.
            coalesce: if `True`, coalesce each of the pairs of join columns
                      (the columns in `on` or `left_on`/`right_on`) from `obs`
                      and `other` into a single column, filling missing values
                      from one with the corresponding values from the other.
                      If `False`, include both as separate columns, adding
                      `suffix` to the join columns from `other`.

        Returns:
            A new Pseudobulk dataset with the columns from `other` joined to
            each cell type's `obs`.

        Note:
            If a column of `on`, `left_on` or `right_on` is Enum in `obs` and
            Categorical in `other` (or vice versa), or Enum in both but with
            different categories in each, that pair of columns will be
            automatically cast to a common Enum data type (with the union of
            the categories) before joining.
        """
        check_type(other, 'other', pl.DataFrame, 'a polars DataFrame')
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        if on is None:
            if left_on is None and right_on is None:
                error_message = (
                    f"either 'on' or both of 'left_on' and 'right_on' must be "
                    f"specified")
                raise ValueError(error_message)
            elif left_on is None:
                error_message = \
                    'right_on is specified, so left_on must be specified'
                raise ValueError(error_message)
            elif right_on is None:
                error_message = \
                    'left_on is specified, so right_on must be specified'
                raise ValueError(error_message)
        else:
            if left_on is not None:
                error_message = "'on' is specified, so 'left_on' must be None"
                raise ValueError(error_message)
            if right_on is not None:
                error_message = "'on' is specified, so 'right_on' must be None"
                raise ValueError(error_message)
        obs = {}
        for cell_type, cell_type_obs in self._obs.items():
            if cell_type not in cell_types:
                obs[cell_type] = cell_type_obs
                continue
            left = cell_type_obs
            right = other
            if on is None:
                left_columns = left.select(left_on)
                right_columns = right.select(right_on)
            else:
                left_columns = left.select(on)
                right_columns = right.select(on)
            left_cast_dict = {}
            right_cast_dict = {}
            for left_column, right_column in zip(left_columns, right_columns):
                left_dtype = left_column.dtype
                right_dtype = right_column.dtype
                if left_dtype == right_dtype:
                    continue
                if (left_dtype == pl.Enum or left_dtype == pl.Categorical) \
                        and (right_dtype == pl.Enum or
                             right_dtype == pl.Categorical):
                    common_dtype = \
                        pl.Enum(pl.concat([left_column.cat.get_categories(),
                                           right_column.cat.get_categories()])
                                .unique(maintain_order=True))
                    left_cast_dict[left_column.name] = common_dtype
                    right_cast_dict[right_column.name] = common_dtype
                else:
                    error_message = (
                        f'obs[{cell_type!r}][{left_column.name!r}] has data '
                        f'type {left_dtype.base_type()!r}, but '
                        f'other[{cell_type!r}][{right_column.name!r}] has '
                        f'data type {right_dtype.base_type()!r}')
                    raise TypeError(error_message)
            if left_cast_dict is not None:
                left = left.cast(left_cast_dict)
                right = right.cast(right_cast_dict)
            obs[cell_type] = \
                left.join(right, on=on, how='left', left_on=left_on,
                          right_on=right_on, suffix=suffix, validate=validate,
                          nulls_equal=nulls_equal, coalesce=coalesce,
                          maintain_order='left')
            if len(obs[cell_type]) > len(self._obs[cell_type]):
                other_on = to_tuple(right_on if right_on is not None else on)
                assert other.select(other_on).is_duplicated().any()
                duplicate_column = other_on[0] if len(other_on) == 1 else \
                    next(column for column in other_on
                         if other[column].is_duplicated().any())
                error_message = (
                    f'other[{duplicate_column!r}] contains duplicate values, '
                    f'so it must be deduplicated before being joined on')
                raise ValueError(error_message)
        return Pseudobulk(X=self._X, obs=obs, var=self._var,
                          num_threads=self._num_threads)

    def join_var(self,
                 other: pl.DataFrame,
                 /,
                 *,
                 cell_types: str | Iterable[str] | None = None,
                 excluded_cell_types: str | Iterable[str] | None = None,
                 on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
                 left_on: str | pl.Expr | Sequence[str | pl.Expr] |
                          None = None,
                 right_on: str | pl.Expr | Sequence[str | pl.Expr] |
                           None = None,
                 suffix: str = '_right',
                 validate: Literal['m:m', 'm:1', '1:m', '1:1'] = 'm:m',
                 nulls_equal: bool = False,
                 coalesce: bool = True) -> Pseudobulk:
        """
        Left-join each cell type's `var` with another DataFrame, using the same
        logic as
        [`df.join()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html).

        Args:
            other: a polars DataFrame to join each cell type's `var` with
            on: the name(s) of the join column(s) in both DataFrames
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            left_on: the name(s) of the join column(s) in `var`
            right_on: the name(s) of the join column(s) in `other`
            suffix: a suffix to append to columns with a duplicate name
            validate: checks whether the join is of the specified type. Can be:

                      - 'm:m' (many-to-many): the default, no checks performed.
                      - '1:1' (one-to-one): check that none of the values in
                        the join column(s) appear more than once in `var` or
                        more than once in `other`.
                      - '1:m' (one-to-many): check that none of the values in
                        the join column(s) appear more than once in `var`.
                      - 'm:1' (many-to-one): check that none of the values in
                        the join column(s) appear more than once in `other`.

            nulls_equal: whether to include `null` as a valid value to join on.
                        By default, `null` values will never produce matches.
            coalesce: if `True`, coalesce each of the pairs of join columns
                      (the columns in `on` or `left_on`/`right_on`) from `var`
                      and `other` into a single column, filling missing values
                      from one with the corresponding values from the other.
                      If `False`, include both as separate columns, adding
                      `suffix` to the join columns from `other`.

        Returns:
            A new Pseudobulk dataset with the columns from `other` joined to
            each cell type's `var`.

        Note:
            If a column of `on`, `left_on` or `right_on` is Enum in `var` and
            Categorical in `other` (or vice versa), or Enum in both but with
            different categories in each, that pair of columns will be
            automatically cast to a common Enum data type (with the union of
            the categories) before joining.
        """
        check_type(other, 'other', pl.DataFrame, 'a polars DataFrame')
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        if on is None:
            if left_on is None and right_on is None:
                error_message = (
                    "either 'on' or both of 'left_on' and 'right_on' must be "
                    "specified")
                raise ValueError(error_message)
            elif left_on is None:
                error_message = \
                    'right_on is specified, so left_on must be specified'
                raise ValueError(error_message)
            elif right_on is None:
                error_message = \
                    'left_on is specified, so right_on must be specified'
                raise ValueError(error_message)
        else:
            if left_on is not None:
                error_message = "'on' is specified, so 'left_on' must be None"
                raise ValueError(error_message)
            if right_on is not None:
                error_message = "'on' is specified, so 'right_on' must be None"
                raise ValueError(error_message)
        var = {}
        for cell_type, cell_type_var in self._var.items():
            if cell_type not in cell_types:
                var[cell_type] = cell_type_var
                continue
            left = cell_type_var
            right = other
            if on is None:
                left_columns = left.select(left_on)
                right_columns = right.select(right_on)
            else:
                left_columns = left.select(on)
                right_columns = right.select(on)
            left_cast_dict = {}
            right_cast_dict = {}
            for left_column, right_column in zip(left_columns, right_columns):
                left_dtype = left_column.dtype
                right_dtype = right_column.dtype
                if left_dtype == right_dtype:
                    continue
                if (left_dtype == pl.Enum or left_dtype == pl.Categorical) \
                        and (right_dtype == pl.Enum or
                             right_dtype == pl.Categorical):
                    common_dtype = \
                        pl.Enum(pl.concat([left_column.cat.get_categories(),
                                           right_column.cat.get_categories()])
                                .unique(maintain_order=True))
                    left_cast_dict[left_column.name] = common_dtype
                    right_cast_dict[right_column.name] = common_dtype
                else:
                    error_message = (
                        f'var[{cell_type!r}][{left_column.name!r}] has data '
                        f'type {left_dtype.base_type()!r}, but '
                        f'other[{cell_type!r}][{right_column.name!r}] has '
                        f'data type {right_dtype.base_type()!r}')
                    raise TypeError(error_message)
            if left_cast_dict is not None:
                left = left.cast(left_cast_dict)
                right = right.cast(right_cast_dict)
            var[cell_type] = \
                left.join(right, on=on, how='left', left_on=left_on,
                          right_on=right_on, suffix=suffix, validate=validate,
                          nulls_equal=nulls_equal, coalesce=coalesce,
                          maintain_order='left')
            if len(var[cell_type]) > len(self._var[cell_type]):
                other_on = to_tuple(right_on if right_on is not None else on)
                assert other.select(other_on).is_duplicated().any()
                duplicate_column = other_on[0] if len(other_on) == 1 else \
                    next(column for column in other_on
                         if other[column].is_duplicated().any())
                error_message = (
                    f'other[{duplicate_column!r}] contains duplicate values, '
                    f'so it must be deduplicated before being joined on')
                raise ValueError(error_message)
        return Pseudobulk(X=self._X, obs=self._obs, var=var,
                          num_threads=self._num_threads)

    def peek_obs(self, cell_type: str | None = None, /, *, row: int = 0) -> \
            None:
        """
        Print a row of `obs` (the first row, by default) for a cell type (the
        first cell type, by default) with each column on its own line.

        Args:
            cell_type: the cell type to print the row for, or `None` to use the
                       first cell type
            row: the index of the row to print
        """
        if cell_type is None:
            cell_type = next(iter(self._obs))
        else:
            check_type(cell_type, 'cell_type', str, 'a string')
        check_type(row, 'row', int, 'an integer')
        with pl.Config(tbl_rows=-1):
            print(self._obs[cell_type][row]
                  .with_columns(pl.col(pl.Enum, pl.Categorical)
                                .cast(pl.String))
                  .unpivot(variable_name='column'))

    def peek_var(self, cell_type: str | None = None, /, *, row: int = 0) -> \
            None:
        """
        Print a row of `var` (the first row, by default) for a cell type (the
        first cell type, by default) with each column on its own line.

        Args:
            cell_type: the cell type to print the row for, or `None` to use the
                       first cell type
            row: the index of the row to print
        """
        if cell_type is None:
            cell_type = next(iter(self._var))
        else:
            check_type(cell_type, 'cell_type', str, 'a string')
        check_type(row, 'row', int, 'an integer')
        with pl.Config(tbl_rows=-1):
            print(self._var[cell_type][row]
                  .with_columns(pl.col(pl.Enum, pl.Categorical)
                                .cast(pl.String))
                  .unpivot(variable_name='column'))

    def subsample_obs(self,
                      *,
                      n: int | np.integer | None = None,
                      fraction: int | float | np.integer | np.floating |
                                None = None,
                      cell_types: str | Iterable[str] | None = None,
                      excluded_cell_types: str | Iterable[str] | None = None,
                      by_column: PseudobulkColumn | None |
                                 dict[str, PseudobulkColumn | None] = None,
                      subsample_column: str | None = None,
                      seed: int | np.integer = 0,
                      overwrite: bool = False) -> Pseudobulk:
        """
        Subsample a specific number or fraction of samples.

        Args:
            n: the number of samples to return; mutually exclusive with
               `fraction`
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            fraction: the fraction of samples to return; mutually exclusive
                      with `n`
            by_column: an optional String, Enum, Categorical, or integer column
                       of `obs` to subsample by. Can be `None`, a column name,
                       a polars expression, a polars Series, a 1D NumPy array,
                       or a function that takes in this Pseudobulk dataset and
                       a cell type and returns a polars Series or 1D NumPy
                       array. Or, a dictionary mapping cell-type names to any
                       of the above; each cell type in this Pseudobulk dataset
                       must be present. Specifying `by_column` ensures that the
                       same fraction of cells with each value of `by_column`
                       are subsampled. When combined with `n`, to make sure the
                       total number of samples is exactly `n`, some of the
                       smallest groups may be oversampled by one element, or
                       some of the largest groups can be undersampled by one
                       element. Can contain `null` entries: the corresponding
                       samples will not be included in the result.
            subsample_column: an optional name of a Boolean column to add to
                              obs indicating the subsampled samples; if `None`,
                              subset to these samples instead
            seed: the random seed to use when subsampling
            overwrite: if `True`, overwrite `subsample_column` if already
                       present in `obs`, instead of raising an error. Must be
                       `False` when `subsample_column` is `None`.

        Returns:
            A new Pseudobulk dataset subset to the subsampled cells, or if
            `subsample_column` is specified, the full dataset with
            `subsample_column` added to `obs`.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        check_type(overwrite, 'overwrite', bool, 'Boolean')
        if subsample_column is not None:
            check_type(subsample_column, 'subsample_column', str, 'a string')
            if not overwrite:
                for cell_type in cell_types:
                    if subsample_column in self._obs[cell_type]:
                        error_message = (
                            f'subsample_column {subsample_column!r} is '
                            f'already a column of obs[{cell_type!r}]; did you '
                            f'already run subsample_obs()? Set overwrite=True '
                            f'to overwrite')
                        raise ValueError(error_message)
        elif overwrite:
            error_message = \
                'overwrite must be False when subsample_column is None'
            raise ValueError(error_message)
        if n is not None and fraction is not None:
            error_message = 'only one of n and fraction can be specified'
            raise ValueError(error_message)
        if n is not None:
            check_type(n, 'n', int, 'a positive integer')
            check_bounds(n, 'n', 1)
        elif fraction is not None:
            check_type(fraction, 'fraction', float,
                       'a floating-point number between 0 and 1')
            check_bounds(fraction, 'fraction', 0, 1, left_open=True,
                         right_open=True)
        else:
            error_message = 'either n or fraction must be specified'
            raise ValueError(error_message)
        by_column = self._get_column(
            'obs', by_column, 'by_column',
            (pl.String, pl.Enum, pl.Categorical, 'integer'), allow_null=True)
        check_type(seed, 'seed', int, 'an integer')
        by = lambda expr, cell_type: \
            expr if by_column[cell_type] is None else \
            expr.over(by_column[cell_type])
        if by_column is not None and n is not None:
            # Reassign `n` to be a vector of sample sizes per group, broadcast
            # to the length of `obs`. The total sample size should exactly
            # match the original `n`; if necessary, oversample the smallest
            # groups or undersample the largest groups to make this happen.
            cell_type_n = {}
            for cell_type in cell_types:
                cell_type_by_column = by_column[cell_type]
                if cell_type_by_column is None:
                    cell_type_n[cell_type] = n
                else:
                    by_frame = cell_type_by_column.to_frame()
                    by_name = cell_type_by_column.name
                    num_non_null = len(cell_type_by_column) - \
                                   cell_type_by_column.null_count()
                    group_counts = by_frame\
                        .drop_nulls(by_name)\
                        .group_by(by_name)\
                        .agg(pl.len(), n=(n / num_non_null * pl.len())
                                         .round().cast(pl.Int32))
                    diff = n - group_counts['n'].sum()
                    if diff != 0:
                        group_counts = group_counts\
                            .sort('len', descending=diff < 0)\
                            .with_columns(n=pl.col.n +
                                            pl.int_range(pl.len(),
                                                         dtype=pl.Int32)
                                            .lt(abs(diff)).cast(pl.Int32) *
                                            pl.lit(diff).sign())
                    cell_type_n[cell_type] = by_frame\
                        .join(group_counts, on=by_name, how='left',
                              maintain_order='left')['n']\
                        .fill_null(0)
        expressions = {}
        for cell_type in cell_types:
            rank = pl.col('_Pseudobulk_rand').rank('ordinal')\
                .pipe(by, cell_type=cell_type)
            if fraction is not None:
                expression = (rank - 1)\
                    .lt(fraction * pl.len().pipe(by, cell_type=cell_type))
            else:
                expression = rank\
                    .le(cell_type_n[cell_type] if by_column is not None else n)
            if by_column is not None:
                expression &= by_column[cell_type].is_not_null()
            expressions[cell_type] = expression
        obs = {}
        rng = np.random.default_rng(seed)
        if subsample_column is None:
            X = {}
            for cell_type, cell_type_obs in self._obs.items():
                if cell_type in cell_types:
                    cell_type_obs = cell_type_obs\
                        .with_columns(
                            _Pseudobulk_index=pl.int_range(
                                pl.len(), dtype=pl.UInt32),
                            _Pseudobulk_rand=rng.random(len(cell_type_obs)))\
                        .filter(expressions[cell_type])
                    X[cell_type] = self._X[cell_type][
                        cell_type_obs['_Pseudobulk_index'].to_numpy()]
                    obs[cell_type] = cell_type_obs\
                        .drop('_Pseudobulk_index', '_Pseudobulk_rand')
                else:
                    X[cell_type] = self._X[cell_type]
                    obs[cell_type] = cell_type_obs
            return Pseudobulk(X=X, obs=obs, var=self._var,
                              num_threads=self._num_threads)
        else:
            for cell_type, cell_type_obs in self._obs.items():
                if cell_type in cell_types:
                    obs[cell_type] = cell_type_obs\
                        .with_columns(_Pseudobulk_rand=
                                      rng.random(len(cell_type_obs)))\
                        .with_columns(expressions[cell_type]
                                      .alias(subsample_column))\
                        .drop('_Pseudobulk_rand')
                else:
                    obs[cell_type] = cell_type_obs
            return Pseudobulk(X=self._X, obs=obs, var=self._var,
                              num_threads=self._num_threads)

    def subsample_var(self,
                      *,
                      n: int | np.integer | None = None,
                      fraction: int | float | np.integer | np.floating |
                                None = None,
                      cell_types: str | Iterable[str] | None = None,
                      excluded_cell_types: str | Iterable[str] | None = None,
                      by_column: PseudobulkColumn | None |
                                 dict[str, PseudobulkColumn | None] = None,
                      subsample_column: str | None = None,
                      seed: int | np.integer = 0,
                      overwrite: bool = False) -> Pseudobulk:
        """
        Subsample a specific number or fraction of genes.

        Args:
            n: the number of genes to return; mutually exclusive with
               `fraction`
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            fraction: the fraction of genes to return; mutually exclusive with
                      `n`
            by_column: an optional String, Enum, Categorical, or integer column
                       of `var` to subsample by. Can be `None`, a column name,
                       a polars expression, a polars Series, a 1D NumPy array,
                       or a function that takes in this Pseudobulk dataset and
                       a cell type and returns a polars Series or 1D NumPy
                       array. Or, a dictionary mapping cell-type names to any
                       of the above; each cell type in this Pseudobulk dataset
                       must be present. Specifying `by_column` ensures that the
                       same fraction of cells with each value of `by_column`
                       are subsampled. When combined with `n`, to make sure the
                       total number of samples is exactly `n`, some of the
                       smallest groups may be oversampled by one element, or
                       some of the largest groups may be undersampled by one
                       element. Can contain `null` entries: the corresponding
                       genes will not be included in the result.
            subsample_column: an optional name of a Boolean column to add to
                              var indicating the subsampled genes; if `None`,
                              subset to these genes instead
            seed: the random seed to use when subsampling
            overwrite: if `True`, overwrite `subsample_column` if already
                       present in `var`, instead of raising an error. Must be
                       `False` when `subsample_column` is `None`.

        Returns:
            A new Pseudobulk dataset subset to the subsampled genes, or if
            `subsample_column` is specified, the full dataset with
            `subsample_column` added to `var`.
        """
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        check_type(overwrite, 'overwrite', bool, 'Boolean')
        if subsample_column is not None:
            check_type(subsample_column, 'subsample_column', str, 'a string')
            if not overwrite:
                for cell_type in cell_types:
                    if subsample_column in self._var[cell_type]:
                        error_message = (
                            f'subsample_column {subsample_column!r} is '
                            f'already a column of var[{cell_type!r}]; did you '
                            f'already run subsample_var()? Set overwrite=True '
                            f'to overwrite')
                        raise ValueError(error_message)
        elif overwrite:
            error_message = \
                'overwrite must be False when subsample_column is None'
            raise ValueError(error_message)
        if n is not None and fraction is not None:
            error_message = 'only one of n and fraction can be specified'
            raise ValueError(error_message)
        if n is not None:
            check_type(n, 'n', int, 'a positive integer')
            check_bounds(n, 'n', 1)
        elif fraction is not None:
            check_type(fraction, 'fraction', float,
                       'a floating-point number between 0 and 1')
            check_bounds(fraction, 'fraction', 0, 1, left_open=True,
                         right_open=True)
        else:
            error_message = 'either n or fraction must be specified'
            raise ValueError(error_message)
        by_column = self._get_column(
            'var', by_column, 'by_column',
            (pl.String, pl.Enum, pl.Categorical, 'integer'), allow_null=True)
        check_type(seed, 'seed', int, 'an integer')
        by = lambda expr, cell_type: \
            expr if by_column[cell_type] is None else \
            expr.over(by_column[cell_type])
        if by_column is not None and n is not None:
            # Reassign `n` to be a vector of sample sizes per group, broadcast
            # to the length of `var`. The total sample size should exactly
            # match the original `n`; if necessary, oversample the smallest
            # groups or undersample the largest groups to make this happen.
            cell_type_n = {}
            for cell_type in cell_types:
                cell_type_by_column = by_column[cell_type]
                if cell_type_by_column is None:
                    cell_type_n[cell_type] = n
                else:
                    by_frame = cell_type_by_column.to_frame()
                    by_name = cell_type_by_column.name
                    num_non_null = len(cell_type_by_column) - \
                                   cell_type_by_column.null_count()
                    group_counts = by_frame\
                        .drop_nulls(by_name)\
                        .group_by(by_name)\
                        .agg(pl.len(), n=(n / num_non_null * pl.len())
                                         .round().cast(pl.Int32))
                    diff = n - group_counts['n'].sum()
                    if diff != 0:
                        group_counts = group_counts\
                            .sort('len', descending=diff < 0)\
                            .with_columns(n=pl.col.n +
                                            pl.int_range(pl.len(),
                                                         dtype=pl.Int32)
                                            .lt(abs(diff)).cast(pl.Int32) *
                                            pl.lit(diff).sign())
                    cell_type_n[cell_type] = by_frame\
                        .join(group_counts, on=by_name, how='left',
                              maintain_order='left')['n']\
                        .fill_null(0)
        expressions = {}
        for cell_type in cell_types:
            rank = pl.col('_Pseudobulk_rand').rank('ordinal')\
                .pipe(by, cell_type=cell_type)
            if fraction is not None:
                expression = (rank - 1)\
                    .lt(fraction * pl.len().pipe(by, cell_type=cell_type))
            else:
                expression = rank\
                    .le(cell_type_n[cell_type] if by_column is not None else n)
            if by_column is not None:
                expression &= by_column[cell_type].is_not_null()
            expressions[cell_type] = expression
        var = {}
        rng = np.random.default_rng(seed)
        if subsample_column is None:
            X = {}
            for cell_type, cell_type_var in self._var.items():
                if cell_type in cell_types:
                    cell_type_var = cell_type_var\
                        .with_columns(
                            _Pseudobulk_index=pl.int_range(
                                pl.len(), dtype=pl.UInt32),
                            _Pseudobulk_rand=rng.random(len(cell_type_var)))\
                        .filter(expressions[cell_type])
                    X[cell_type] = self._X[cell_type][
                        :, cell_type_var['_Pseudobulk_index'].to_numpy()]
                    var[cell_type] = cell_type_var\
                        .drop('_Pseudobulk_index', '_Pseudobulk_rand')
                else:
                    X[cell_type] = self._X[cell_type]
                    var[cell_type] = cell_type_var
            return Pseudobulk(X=X, obs=self._obs, var=var,
                              num_threads=self._num_threads)
        else:
            for cell_type, cell_type_var in self._var.items():
                if cell_type in cell_types:
                    var[cell_type] = cell_type_var\
                        .with_columns(_Pseudobulk_rand=
                                      rng.random(len(cell_type_var)))\
                        .with_columns(expressions[cell_type]
                                      .alias(subsample_column))\
                        .drop('_Pseudobulk_rand')
                else:
                    var[cell_type] = cell_type_var
            return Pseudobulk(X=self._X, obs=self._obs, var=var,
                              num_threads=self._num_threads)

    def pipe(self,
             function: Callable[[Pseudobulk, ...], Any],
             /,
             *args: Any,
             **kwargs: Any) -> Any:
        """
        Apply a function to a Pseudobulk dataset.

        `pb.pipe(func)` is equivalent to `func(pb)`. `pb.pipe(func, 1, a=2)` is
        equivalent to `func(pb, 1, a=2)`.

        Args:
            function: the function to apply
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            The result of applying the function to this Pseudobulk dataset.
        """
        # Check that `function` is callable
        if not callable(function):
            error_message = (
                f'function is not callable; it has type '
                f'{type(function).__name__}')
            raise TypeError(error_message)

        return function(self, *args, **kwargs)

    def pipe_X(self,
               function: Callable[[dict[str, np.ndarray[np.dtype[
                                      np.integer | np.floating]]], ...],
                                  dict[str, np.ndarray[np.dtype[
                                      np.integer | np.floating]]]],
               /,
               *args: Any,
               **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to a Pseudobulk dataset's `X`.

        `pb = pb.pipe_X(func)` is equivalent to `pb.X = func(pb.X)`.
        `pb = pb.pipe_X(func, 1, a=2)` is equivalent to
        `pb.X = func(pb.X, 1, a=2)`.

        To apply a function to each cell type's `X`, rather than to `X` as a
        whole, use `map_X()`.

        Args:
            function: the function to apply to `X`. It must take the old `X` as
                      its first argument and return the new `X`. The function
                      may also take other arguments after `X`, which can be
                      specified via `args` and `kwargs`.
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            `X`.
        """
        # Check that `function` is callable
        if not callable(function):
            error_message = (
                f'function is not callable; it has type '
                f'{type(function).__name__}')
            raise TypeError(error_message)

        return Pseudobulk(X=function(self._X, *args, **kwargs),
                          obs=self._obs, var=self._var,
                          num_threads=self._num_threads)

    def pipe_obs(self,
                 function: Callable[[dict[str, pl.DataFrame], ...],
                                    dict[str, pl.DataFrame]],
                 /,
                 *args: Any,
                 **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to a Pseudobulk dataset's `obs`.

        `pb = pb.pipe_obs(func)` is equivalent to `pb.obs = func(pb.obs)`.
        `pb = pb.pipe_obs(func, 1, a=2)` is equivalent to
        `pb.obs = func(pb.obs, 1, a=2)`.

        To apply a function to each cell type's `obs`, rather than to `obs` as
        a whole, use `map_obs()`.

        Args:
            function: the function to apply to `obs`. It must take the old
                      `obs` as its first argument and return the new `obs`. The
                      function may also take other arguments after `obs`, which
                      can be specified via `args` and `kwargs`.
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            obs.
        """
        # Check that `function` is callable
        if not callable(function):
            error_message = (
                f'function is not callable; it has type '
                f'{type(function).__name__}')
            raise TypeError(error_message)

        return Pseudobulk(X=self._X, obs=function(self._obs, *args, **kwargs),
                          var=self._var, num_threads=self._num_threads)

    def pipe_var(self,
                 function: Callable[[dict[str, pl.DataFrame], ...],
                                    dict[str, pl.DataFrame]],
                 /,
                 *args: Any,
                 **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to a Pseudobulk dataset's `var`.

        `pb = pb.pipe_var(func)` is equivalent to `pb.var = func(pb.var)`.
        `pb = pb.pipe_var(func, 1, a=2)` is equivalent to
        `pb.var = func(pb.var, 1, a=2)`.

        To apply a function to each cell type's `var`, rather than to `var` as
        a whole, use `map_var()`.

        Args:
            function: the function to apply to `var`. It must take the old
                      `var` as its first argument and return the new `var`. The
                      function may also take other arguments after `var`, which
                      can be specified via `args` and `kwargs`.
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            var.
        """
        # Check that `function` is callable
        if not callable(function):
            error_message = (
                f'function is not callable; it has type '
                f'{type(function).__name__}')
            raise TypeError(error_message)

        return Pseudobulk(X=self._X, obs=self._obs,
                          var=function(self._var, *args, **kwargs),
                          num_threads=self._num_threads)

    def map_X(self,
              function: Callable[[np.ndarray[np.dtype[np.integer |
                                                      np.floating]], ...],
                                 np.ndarray[np.dtype[np.integer |
                                                     np.floating]]],
              /,
              *args: Any,
              cell_types: str | Iterable[str] | None = None,
              excluded_cell_types: str | Iterable[str] | None = None,
              **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to each cell type's `X`.

        To apply a function to `X` as a whole, rather than each cell type's
        `X`, use `pipe_X()`.

        Args:
            function: the function to apply to each cell type's `X`. It must
                      take the old `X` for a cell type and return the new `X`.
                      The function may also take other arguments after `X`,
                      which can be specified via `args` and `kwargs`.
            *args: the positional arguments to the function
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            each cell type's `X`.
        """
        # Check that `function` is callable
        if not callable(function):
            error_message = (
                f'function is not callable; it has type '
                f'{type(function).__name__}')
            raise TypeError(error_message)

        # Get the cell type(s) to operate on
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))

        return Pseudobulk(X={cell_type: function(X, *args, **kwargs)
                                        if cell_type in cell_types else X
                             for cell_type, X in self._X.items()},
                          obs=self._obs, var=self._var,
                          num_threads=self._num_threads)

    def map_obs(self,
                function: Callable[[pl.DataFrame, ...], pl.DataFrame],
                /,
                *args: Any,
                cell_types: str | Iterable[str] | None = None,
                excluded_cell_types: str | Iterable[str] | None = None,
                **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to each cell type's `obs`.

        To apply a function to `obs` as a whole, rather than each cell type's
        `obs`, use `pipe_obs()`.

        Args:
            function: the function to apply to each cell type's `obs`. It must
                      take the old `obs` for a cell type and return the new
                      `obs`. The function may also take other arguments after
                      `obs`, which can be specified via `args` and `kwargs`.
            *args: the positional arguments to the function
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            each cell type's `obs`.
        """
        # Check that `function` is callable
        if not callable(function):
            error_message = (
                f'function is not callable; it has type '
                f'{type(function).__name__}')
            raise TypeError(error_message)

        # Get the cell type(s) to operate on
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))

        return Pseudobulk(X=self._X,
                          obs={cell_type: function(obs, *args, **kwargs)
                                          if cell_type in cell_types else obs
                               for cell_type, obs in self._obs.items()},
                          var=self._var, num_threads=self._num_threads)

    def map_var(self,
                function: Callable[[pl.DataFrame, ...], pl.DataFrame],
                /,
                *args: Any,
                cell_types: str | Iterable[str] | None = None,
                excluded_cell_types: str | Iterable[str] | None = None,
                **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to each cell type's `var`.

        To apply a function to `var` as a whole, rather than each cell type's
        `var`, use `pipe_var()`.

        Args:
            function: the function to apply to each cell type's `var`. It must
                      take the old `var` for a cell type and return the new
                      `var`. The function may also take other arguments after
                      `var`, which can be specified via `args` and `kwargs`.
            *args: the positional arguments to the function
            cell_types: one or more cell types to operate on; if `None`,
                        operate on all cell types. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from the
                                 operation; mutually exclusive with
                                 `cell_types`
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            each cell type's `var`.
        """
        # Check that `function` is callable
        if not callable(function):
            error_message = (
                f'function is not callable; it has type '
                f'{type(function).__name__}')
            raise TypeError(error_message)

        # Get the cell type(s) to operate on
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))

        return Pseudobulk(X=self._X, obs=self._obs,
                          var={cell_type: function(var, *args, **kwargs)
                                          if cell_type in cell_types else var
                               for cell_type, var in self._var.items()},
                          num_threads=self._num_threads)

    @staticmethod
    def _too_few_samples(obs: pl.DataFrame,
                         group_column: pl.Series | None,
                         min_samples: int | np.integer,
                         cell_type: str,
                         verbose: bool,
                         after_filtering: bool = False) -> bool:
        """
        Skip cell types with fewer than `min_samples` samples, or with fewer
        than `min_samples` samples in any group if `group_column` is not None.
        When `group_column` is not None, also skip cell types where all samples
        have the same value of `group_column`.

        Args:
            obs: the cell type's `obs`, after applying one or more QC filters
            group_column: the column with sample group information (e.g. which
                          samples are disease cases and which are controls),
                          after applying one or more QC filters. The samples
                          in this column must be the same as those in `obs`.
            min_samples: filter to cell types with at least this many samples
                         in each group, or with at least this many total
                         samples if `group_column` is `None`
            cell_type: the name of the cell type
            verbose: whether to explain why the cell type is being skipped, if
                     it is
            after_filtering: whether this function is being run after sample
                             filtering

        Returns:
            Whether this cell type has too few samples and should be skipped.
        """
        num_samples = len(obs)
        if num_samples == 0:
            if verbose:
                print(f'[{cell_type}] Skipping this cell type because '
                      f'it has 0 samples'
                      f'{" after filtering" if after_filtering else ""}.')
            return True
        elif group_column is None:
            if num_samples < min_samples:
                if verbose:
                    print(f'[{cell_type}] Skipping this cell type because '
                          f'it has only {num_samples:,} '
                          f'{plural("sample", num_samples)}'
                          f'{" after filtering" if after_filtering else ""}, '
                          f'which is fewer than min_samples '
                          f'({min_samples:,})')
                return True
        else:
            value_counts = group_column.value_counts()
            if len(value_counts) == 1:
                if verbose:
                    print(f'[{cell_type}] Skipping this cell type because all '
                          f'samples have the same value of group_column, '
                          f'namely {value_counts[0, 0]!r}')
                return True
            too_small_groups = value_counts\
                .filter(pl.col.count < min_samples)\
                .drop_nulls()
            if len(too_small_groups) > 0:
                if verbose:
                    count = too_small_groups['count'][-1]
                    group_description = (
                        f'{count:,} {plural("sample", count)} where '
                        f'group_column = {too_small_groups.to_series()[-1]!r}')
                    if len(too_small_groups) > 1:
                        group_description = (
                            ', '.join(f'{count:,} {plural("sample", count)} '
                                      f'where group_column = {group!r}'
                                      for group, count in
                                      too_small_groups[:-1].iter_rows()) +
                            f' and {group_description}')
                    print(f'[{cell_type}] Skipping this cell type because it '
                          f'has only {group_description}'
                          f'{" after filtering" if after_filtering else ""}, '
                          f'which '
                          f'{"is" if len(too_small_groups) == 1 else "are"} '
                          f'fewer than min_samples ({min_samples:,})')
                return True
        return False

    def qc(self,
           group_column: PseudobulkColumn | None |
                         dict[str, PseudobulkColumn | None],
           /,
           *,
           custom_filter: PseudobulkColumn | None |
                          dict[str, PseudobulkColumn | None] = None,
           min_samples: int | np.integer = 2,
           min_cells: int | np.integer | None = 10,
           max_standard_deviations: int | float | np.integer | np.floating |
                                    None = 3,
           min_nonzero_fraction: int | float | np.integer | np.floating |
                                 None = 0.8,
           cell_types: str | Iterable[str] | None = None,
           excluded_cell_types: str | Iterable[str] | None = None,
           error_if_negative_counts: bool = True,
           allow_float: bool = False,
           verbose: bool = False) -> Pseudobulk:
        """
        Subsets each cell type to samples passing quality control (QC). If
        samples fall into discrete groups (e.g. disease cases versus controls),
        these should be specified via the `group_column` argument.

        Filters, in order, to:

        - samples that pass the `custom_filter` (if specified), have
          non-missing values for `group_column` (if specified), and have at
          least `min_cells` cells of that type (default: 10)
        - samples where the number of genes with 0 counts is at most
          `max_standard_deviations` standard deviations above the mean
          (default: 3)
        - genes with at least 1 count in `100 * min_nonzero_fraction`%
          (default: 80%) of samples (in every group, if `group_column` is
          specified)

        If at any point during this filtering process, there are fewer than
        `min_samples` (default: 2) samples (in any group, if `group_column` is
        specified), or `group_column` is specified and all samples have the
        same value of `group_column`, the cell type is filtered out entirely.

        Args:
            group_column: an optional String, Categorical, Enum, Boolean, or
                          integer column of `obs` with sample group
                          information, e.g. which samples are disease cases and
                          which are controls. If specified, the
                          `min_nonzero_fraction` and `min_samples` filters must
                          pass for every group, rather than merely passing for
                          the dataset as a whole. Set to `None` if samples do
                          not fall into discrete groups. Can be `None`, a
                          column name, a polars expression, a polars Series, a
                          1D NumPy array, or a function that takes in this
                          Pseudobulk dataset and a cell type and returns a
                          polars Series or 1D NumPy array. Or, a dictionary
                          mapping cell-type names to any of the above; each
                          cell type in this Pseudobulk dataset must be present.
                          Can contain `null` entries: the corresponding samples
                          will be deemed to fail QC.
            custom_filter: an optional Boolean column of `obs` containing a
                           filter to apply on top of the other QC filters;
                           `True` elements will be kept. Can be `None`, a
                           column name, a polars expression, a polars Series, a
                           1D NumPy array, or a function that takes in this
                           Pseudobulk dataset and a cell type and returns a
                           polars Series or 1D NumPy array. Or, a dictionary
                           mapping cell-type names to any of the above; each
                           cell type in this Pseudobulk dataset must be
                           present.
            min_samples: filter to cell types with at least this many samples
                         in every group, or with at least this many total
                         samples if `group_column` is `None`
            min_cells: if not `None`, filter to samples with ≥ this many cells
                       of each cell type
            max_standard_deviations: if not `None`, filter to samples where the
                                     number of genes with 0 counts is at most
                                     this many standard deviations above the
                                     mean
            min_nonzero_fraction: if not `None`, filter to genes with at least
                                  one count in this fraction of samples in each
                                  group, or if `group_column` is `None`, at
                                  least one count in this fraction of samples
                                  overall. Note: `min_nonzero_fraction=0`
                                  filters out only genes with all-zero counts,
                                  while `min_nonzero_fraction=None` does not
                                  filter out any genes.
            cell_types: one or more cell types to QC; if `None`, QC all cell
                        types. Mutually exclusive with `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude from QC;
                                 mutually exclusive with `cell_types`
            error_if_negative_counts: if `True`, raise an error if any counts
                                      are negative
            allow_float: if `False`, raise an error if `self.X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts); if `True`, disable this sanity check
            verbose: whether to print how many samples and genes were filtered
                     out at each step of the QC process

        Returns:
            A new Pseudobulk dataset with each cell type's `X`, `obs` and `var`
            subset to samples and genes passing QC.

        Note:
            This function may give an incorrect output if the count matrix
            contains negative values: this is not checked for, due to speed
            considerations.
        """
        # Check inputs
        cell_types = \
            set(self._process_cell_types(cell_types, excluded_cell_types))
        group_column = self._get_column(
            'obs', group_column, 'group_column',
            (pl.String, pl.Enum, pl.Categorical, pl.Boolean, 'integer'),
            allow_null=True)
        custom_filter = self._get_column(
            'obs', custom_filter, 'custom_filter', pl.Boolean)
        check_type(min_samples, 'min_samples', int,
                   'an integer greater than or equal to 2')
        check_bounds(min_samples, 'min_samples', 2)
        if min_cells is not None:
            check_type(min_cells, 'min_cells', int, 'a positive integer')
            check_bounds(min_cells, 'min_cells', 1)
        if max_standard_deviations is not None:
            check_type(max_standard_deviations, 'max_standard_deviations',
                       (int, float), 'a positive number')
            check_bounds(max_standard_deviations, 'max_standard_deviations', 0,
                         left_open=True)
        if min_nonzero_fraction is not None:
            check_type(min_nonzero_fraction, 'min_nonzero_fraction',
                       (int, float), 'a number between 0 and 1, inclusive')
            check_bounds(min_nonzero_fraction, 'min_nonzero_fraction', 0, 1)
        check_type(error_if_negative_counts, 'error_if_negative_counts', bool,
                   'Boolean')
        check_type(allow_float, 'allow_float', bool, 'Boolean')
        check_type(verbose, 'verbose', bool, 'Boolean')

        # Check that `group_column` is None when neither the
        # `min_nonzero_fraction` nor the `min_samples` filters will be applied
        if group_column is not None and min_nonzero_fraction is None and \
                min_samples is None:
            error_message = (
                'group_column must be None when min_nonzero_fraction and '
                'min_samples are both None')
            raise ValueError(error_message)

        # If `error_if_negative_counts=True`, raise an error if `X` has any
        # negative values for any cell type
        if error_if_negative_counts:
            for cell_type in cell_types:
                if self._X[cell_type].min() < 0:
                    error_message = f'X[{cell_type!r}] has negative counts'
                    raise ValueError(error_message)

        # If `allow_float=False`, raise an error if `X` is floating-point for
        # any cell type
        if not allow_float:
            for cell_type in cell_types:
                dtype = self._X[cell_type].dtype
                if np.issubdtype(dtype, np.floating):
                    error_message = (
                        f"qc() requires raw counts but X[{cell_type!r}] "
                        f"has data type {str(dtype)!r}, a floating-point data "
                        f"type; if you are sure that all values are raw "
                        f"integer counts, i.e. that (X[{cell_type!r}].data == "
                        f"X[{cell_type!r}].data.astype(int)).all(), then set "
                        f"allow_float=True (or just cast X to an integer data "
                        f"type).")
                    raise TypeError(error_message)

        # QC each cell type
        X_qced, obs_qced, var_qced = {}, {}, {}
        at_least_one_cell_type_passes_QC = False
        for cell_type in self._X:
            X = self._X[cell_type]
            obs = self._obs[cell_type]
            var = self._var[cell_type]
            if cell_type not in cell_types:
                X_qced[cell_type] = X
                obs_qced[cell_type] = obs
                var_qced[cell_type] = var
                if verbose:
                    if excluded_cell_types is not None:
                        print(f'\n[{cell_type}] Skipping this cell type due '
                              f'to being present in excluded_cell_types')
                    else:
                        print(f'\n[{cell_type}] Skipping this cell type due '
                              f'to being absent from cell_types')
                continue
            if verbose:
                print(f'\n[{cell_type}] Starting with {len(obs):,} '
                      f'{plural("sample", len(obs))} and {len(var):,} '
                      f'{plural("gene", len(var))}.')

            # Get the group column for this cell type.
            groups = \
                group_column[cell_type] if group_column is not None else None

            # Check if we have enough samples for this cell type
            if Pseudobulk._too_few_samples(obs, groups, min_samples, cell_type,
                                           verbose):
                continue

            # Get a mask of samples passing the custom filter, if specified
            if custom_filter is not None:
                if verbose:
                    print(f'[{cell_type}] Applying the custom filter...')
                sample_mask = custom_filter[cell_type]
                if sample_mask is not None:
                    if verbose:
                        num_samples = sample_mask.sum()
                        remain_string = 'sample remains' \
                            if num_samples == 1 else 'samples remain'
                        print(
                            f'[{cell_type}] {num_samples:,} {remain_string} '
                            f'after applying the custom filter.')
            else:
                sample_mask = None

            # If `groups` is not `None` and some samples have missing groups,
            # get a mask of samples with non-missing groups
            if groups is not None:
                null_count = groups.null_count()
                if null_count:
                    if verbose:
                        print(f'[{cell_type}] Filtering to samples with '
                              f'non-missing values for group_column...')
                    if sample_mask is None:
                        sample_mask = groups.is_not_null()
                    else:
                        sample_mask &= groups.is_not_null()
                    if verbose:
                        num_samples = sample_mask.sum()
                        remain_string = 'sample remains' \
                            if num_samples == 1 else 'samples remain'
                        print(f'[{cell_type}] {num_samples:,} {remain_string} '
                              f'after filtering to samples with non-missing '
                              f'values for group_column.')

            # Get a mask of samples with at least `min_cells` cells of this
            # cell type, if `min_cells` was specified. Combine this with the
            # sample mask from `custom_filter` above, if both were specified.
            if min_cells is not None:
                if verbose:
                    print(f'[{cell_type}] Filtering to samples with at least '
                          f'{min_cells:,} {cell_type} '
                          f'{plural("cell", min_cells)}...')
                if sample_mask is None:
                    sample_mask = obs['num_cells'] >= min_cells
                else:
                    sample_mask &= obs['num_cells'] >= min_cells
                if verbose:
                    num_samples = sample_mask.sum()
                    remain_string = 'sample remains' \
                        if num_samples == 1 else 'samples remain'
                    print(f'[{cell_type}] {num_samples:,} {remain_string} '
                          f'after filtering to samples with at least '
                          f'{min_cells:,} {cell_type} '
                          f'{plural("cell", min_cells)}.')

            # Now apply the sample mask, which contains the samples passing the
            # custom filter and/or `min_cells` filter
            if sample_mask is not None:
                obs = obs.filter(sample_mask)
                if groups is not None:
                    groups = groups.filter(sample_mask)
                # Check if we still have enough samples for this cell type,
                # after applying these three filters
                if Pseudobulk._too_few_samples(obs, groups, min_samples,
                                               cell_type, verbose,
                                               after_filtering=True):
                    continue
                X = X[sample_mask.to_numpy()]

            # Filter to samples where the number of genes with 0 counts is less
            # than `max_standard_deviations` standard deviations above the mean
            if max_standard_deviations is not None:
                if verbose:
                    print(f'[{cell_type}] Filtering to samples where the '
                          f'number of genes with 0 counts is '
                          f'<{max_standard_deviations} standard deviations '
                          f'above the mean...')
                num_zero_counts = X.shape[1] - np.count_nonzero(X, axis=1)
                sample_mask_NumPy = \
                    num_zero_counts < num_zero_counts.mean() + \
                    max_standard_deviations * num_zero_counts.std()
                sample_mask = pl.Series(sample_mask_NumPy)
                obs = obs.filter(sample_mask)
                if groups is not None:
                    groups = groups.filter(sample_mask)
                if verbose:
                    remain_string = 'sample remains' \
                        if len(obs) == 1 else 'samples remain'
                    print(f'[{cell_type}] {len(obs):,} {remain_string} after '
                          f'filtering to samples where the number of genes '
                          f'with 0 counts is <{max_standard_deviations} '
                          f'standard deviations above the mean.')

                # Check if we have still enough samples for this cell type,
                # after applying this filter
                if Pseudobulk._too_few_samples(obs, groups, min_samples,
                                               cell_type, verbose,
                                               after_filtering=True):
                    continue
                X = X[sample_mask_NumPy]

            # Filter to genes with at least 1 count in
            # `100 * min_nonzero_fraction`% of samples (or samples in each
            # group, if `group_column` is not None for this cell type)
            if min_nonzero_fraction is not None:
                if groups is not None:
                    if verbose:
                        print(f'[{cell_type}] Filtering to genes with at '
                              f'least one count in '
                              f'{100 * min_nonzero_fraction}% of samples in '
                              f'each group...')
                    gene_mask = np.logical_and.reduce([
                        np.quantile(X[mask.to_numpy()],
                                    1 - min_nonzero_fraction, axis=0) > 0
                        for mask in groups.to_dummies().cast(pl.Boolean)])
                    X = X[:, gene_mask]
                    var = var.filter(gene_mask)
                    if verbose:
                        remain_string = 'gene remains' \
                            if len(var) == 1 else 'genes remain'
                        print(f'[{cell_type}] {len(var):,} {remain_string} '
                              f'after filtering to genes with at least one '
                              f'count in {100 * min_nonzero_fraction}% of '
                              f'samples in each group.')
                else:
                    if verbose:
                        print(f'[{cell_type}] Filtering to genes with at '
                              f'least one count in '
                              f'{100 * min_nonzero_fraction}% of samples...')
                    gene_mask = np.quantile(X, 1 - min_nonzero_fraction,
                                            axis=0) > 0
                    X = X[:, gene_mask]
                    var = var.filter(gene_mask)
                    if verbose:
                        remain_string = 'gene remains' \
                            if len(var) == 1 else 'genes remain'
                        print(f'[{cell_type}] {len(var):,} {remain_string} '
                              f'after filtering to genes with at least one '
                              f'count in {100 * min_nonzero_fraction}% of '
                              f'samples.')
            X_qced[cell_type] = np.ascontiguousarray(X)
            obs_qced[cell_type] = obs
            var_qced[cell_type] = var
            at_least_one_cell_type_passes_QC = True
        if not at_least_one_cell_type_passes_QC:
            error_message = (
                'all cell types were skipped due to having too few samples '
                'passing one or more QC filters')
            if not verbose:
                error_message += \
                    '; re-run with verbose=True to see which filter(s)'
            raise ValueError(error_message)
        return Pseudobulk(X=X_qced, obs=obs_qced, var=var_qced,
                          num_threads=self._num_threads)

    @staticmethod
    def _library_size(X: np.ndarray[np.dtype[np.integer | np.floating]],
                      cell_type: str,
                      *,
                      logratio_trim: int | float = 0.3,
                      sum_trim: int | float = 0.05,
                      A_cutoff: int | float = -1e10,
                      num_threads: int) -> \
            np.ndarray[np.dtype[np.float32]]:
        """
        Calculate normalization factor-adjusted library sizes for each sample
        in each cell type, via the approach of edgeR's
        [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors).
         Used by `library_size()`.

        Uses the same method as
        [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors)
        with the default `method='TMM'`. However, results differ from edgeR due
        to the presence of a floating-point bug in edgeR's
        [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors)
        implementation. When calculating `logR`, the log2 ratio of
        `count / library_size` for a gene between a particular sample and a
        "reference" sample, the numerator and denominator of the ratio both
        involve a division by their sample's library size. In principle, these
        divisions by library size are equivalent to multiplying by the same
        constant across genes, namely the ratio of the two samples'
        library sizes. But in practice, even if two genes have the same count
        ratio between the two samples, they may still have slightly different
        `count / library_size` ratios due to floating-point roundoff, leading
        to these genes erroneously being assigned different `logR` ranks
        instead of being treated as tied. Our implementation fixes this bug by
        changing the order of operations so that the library size ratio is
        calculated first, then multiplied by the count ratio. Because this bug
        affects which genes are included in the trimmed mean, its impact can be
        relatively large, sometimes leading to a >1% error in edgeR's estimated
        library size relative to our correct implementation.

        Does not support the `lib.size` and `refColumn` arguments to
        [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors);
        these are both assumed to be `NULL` (the default) and will always be
        calculated internally. The `doWeighting` argument is also not supported
        and is assumed to be `TRUE` (the default), so asymptotic binomial
        precision weights will always be used.

        Args:
            X: a matrix of raw (read) counts. `X` is assumed to have the
               opposite orientation from the original
               [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors):
               samples are rows and genes are columns.
            cell_type: the cell type `X` is from, used in error messages
            logratio_trim: the amount of trim to use on log-ratios ("M"
                           values); must be greater than 0 and less than 1
            sum_trim: the amount of trim to use on the combined absolute levels
                      ("A" values); must be greater than 0 and less than 1
            A_cutoff: the cutoff on "A" values to use before trimming
            num_threads: the number of threads to use when calculating
                         normalization factors. Parallelization occurs within
                         each cell type, across samples.

        Returns:
            The norm factor-corrected library sizes: raw library sizes (column
            sums) times norm factors.
        """
        # Degenerate cases
        num_samples, num_genes = X.shape
        if num_samples == 1 or num_genes == 0:
            return X.sum(axis=1).astype(np.float32)

        # Raise an error if `X` is not C-contiguous
        if not X.flags['C_CONTIGUOUS']:
            error_message = (
                f'X[{cell_type!r}] is not C-contiguous; did you forget to run '
                f'Pseudobulk.qc()?')
            raise ValueError(error_message)

        # Raise an error if there are any all-zero columns (genes)
        if has_all_zero_columns(X=X):
            error_message = (
                f'[{cell_type}] some genes have all-zero counts; did you '
                f'forget to run Pseudobulk.qc()?')
            raise ValueError(error_message)

        # Calculate raw library sizes
        library_size = X.sum(axis=1)

        # Raise an error if any raw library sizes are 0
        if library_size.min() == 0:
            error_message = (
                'some samples have all-zero counts; did you forget to run '
                'Pseudobulk.qc()?')
            raise ValueError(error_message)

        # Determine which sample is the reference sample
        f75 = np.quantile(X, 0.75, axis=1) / library_size
        if np.median(f75) < 1e-20:
            ref_sample = np.argmax(np.sqrt(X).sum(axis=1))
        else:
            ref_sample = np.argmin(np.abs(f75 - f75.mean()))

        # Calculate norm factors
        norm_factors = np.empty(num_samples, dtype=np.float32)
        calc_norm_factors(X=X, logratio_trim=logratio_trim, sum_trim=sum_trim,
                          A_cutoff=A_cutoff, ref_sample=ref_sample,
                          norm_factors=norm_factors, library_size=library_size,
                          num_threads=num_threads)

        return norm_factors  # this is actually `library_size * norm_factors`

    def library_size(self,
                     *,
                     library_size_column: str = 'library_size',
                     cell_types: str | Iterable[str] | None = None,
                     excluded_cell_types: str | Iterable[str] | None = None,
                     logratio_trim: int | float | np.integer |
                                    np.floating = 0.3,
                     sum_trim: int | float | np.integer | np.floating = 0.05,
                     A_cutoff: int | float | np.integer |
                               np.floating = -1e10,
                     allow_float: bool = False,
                     overwrite: bool = False,
                     num_threads: int | np.integer | None = None) -> \
            Pseudobulk:
        """
        Calculate normalization factor-adjusted library sizes for each sample
        in each cell type, via the approach of edgeR's
        [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors).

        Uses the same method as
        [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors)
        with the default `method='TMM'`. However, results differ from edgeR due
        to the presence of a floating-point bug in edgeR's
        [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors)
        implementation. When calculating `logR`, the log2 ratio of
        `count / library_size` for a gene between a particular sample and a
        "reference" sample, the numerator and denominator of the ratio both
        involve a division by their sample's library size. In principle, these
        divisions by library size are equivalent to multiplying by the same
        constant across genes, namely the ratio of the two samples'
        library sizes. But in practice, even if two genes have the same count
        ratio between the two samples, they may still have slightly different
        `count / library_size` ratios due to floating-point roundoff, leading
        to these genes erroneously being assigned different `logR` ranks
        instead of being treated as tied. Our implementation fixes this bug by
        changing the order of operations so that the library size ratio is
        calculated first, then multiplied by the count ratio. Because this bug
        affects which genes are included in the trimmed mean, its impact can be
        relatively large, sometimes leading to a >1% error in edgeR's estimated
        library size relative to our correct implementation.

        Does not support the `lib.size` and `refColumn` arguments to
        [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors);
        these are both assumed to be `NULL` (the default) and will always be
        calculated internally. The `doWeighting` argument is also not supported
        and is assumed to be `TRUE` (the default), so asymptotic binomial
        precision weights will always be used.

        Args:
            library_size_column: the name of a floating-point column to add to
                                 `obs` containing each sample's library size
            cell_types: one or more cell types to calculate library sizes for;
                        if `None`, calculate library sizes for all cell types.
                        Mutually exclusive with `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude when
                                 calculating library sizes; mutually exclusive
                                 with `cell_types`
            logratio_trim: the amount of trim to use on log-ratios ("M"
                           values); must be greater than 0 and less than 1
            sum_trim: the amount of trim to use on the combined absolute levels
                      ("A" values); must be greater than 0 and less than 1
            A_cutoff: the cutoff on "A" values to use before trimming
            allow_float: if `False`, raise an error if `self.X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts); if `True`, disable this sanity check
            overwrite: if `True`, overwrite `library_size_column` if already
                       present in `obs`, instead of raising an error.
            num_threads: the number of threads to use when calculating library
                         sizes. Set `num_threads=-1` to use all available
                         cores, as determined by
                         [`os.cpu_count()`](https://docs.python.org/3/library/os.html#os.cpu_count),
                         or leave unset to use `self.num_threads` cores. Does
                         not affect the returned Pseudobulk dataset's
                         `num_threads`; this will always be the same as the
                         original dataset's `num_threads`.

        Returns:
            A new Pseudobulk dataset where `obs[library_size_column]` contains
            the norm factor-corrected library sizes for each cell type: raw
            library sizes (column sums) times norm factors.
        """
        # Get the list of cell types to calculate library sizes for
        cell_types, cell_type_description = \
            self._process_cell_types(cell_types, excluded_cell_types,
                                     return_description=True)

        # Check that `library_size_column` is a string
        check_type(library_size_column, 'library_size_column', str, 'a string')

        # Check that `overwrite` is Boolean
        check_type(overwrite, 'overwrite', bool, 'Boolean')

        # Check that `library_size_column` is not already a column of `obs` for
        # any cell type, unless `overwrite=True`
        if not overwrite:
            for cell_type, obs in self._obs.items():
                if library_size_column in obs:
                    error_message = (
                        f'library_size_column {library_size_column!r} is '
                        f'already a column of obs for cell type '
                        f'{cell_type!r}; did you already run library_size()? '
                        f'Set overwrite=True to overwrite.')
                    raise ValueError(error_message)

        # Check that `logratio_trim`, `sum_trim`, and `A_cutoff` are
        # floating-point numbers with the correct ranges
        check_type(logratio_trim, 'logratio_trim', float,
                   'a floating-point number')
        check_bounds(logratio_trim, 'logratio_trim', 0, 1, left_open=True,
                     right_open=True)
        check_type(sum_trim, 'sum_trim', float, 'a floating-point number')
        check_bounds(sum_trim, 'sum_trim', 0, 1, left_open=True,
                     right_open=True)
        check_type(A_cutoff, 'A_cutoff', float, 'a floating-point number')

        # Check that `allow_float` is Boolean
        check_type(allow_float, 'allow_float', bool, 'Boolean')

        # If `allow_float=False`, raise an error if `X` is floating-point for
        # any cell type
        if not allow_float:
            for cell_type in cell_types:
                dtype = self._X[cell_type].dtype
                if np.issubdtype(dtype, np.floating):
                    error_message = (
                        f"library_size() requires raw counts but "
                        f"X[{cell_type!r}] has data type {str(dtype)!r}, a "
                        f"floating-point data type; if you are sure that all "
                        f"values are raw integer counts, i.e. that "
                        f"(X[{cell_type!r}].data == "
                        f"X[{cell_type!r}].data.astype(int)).all(), then set "
                        f"allow_float=True (or just cast X to an integer data "
                        f"type).")
                    raise TypeError(error_message)

        # Check that `num_threads` is a positive integer, -1 or `None`; if
        # `None`, set to `self.num_threads`, and if -1, set to
        # `os.cpu_count()`.
        num_threads = self._process_num_threads(num_threads)

        # Compute library sizes for each cell type
        library_sizes = [Pseudobulk._library_size(
            X=self._X[cell_type], cell_type=cell_type,
            logratio_trim=logratio_trim, sum_trim=sum_trim,
            A_cutoff=A_cutoff, num_threads=num_threads)
            for cell_type in cell_types]

        # Add library sizes to each cell type's `obs`
        obs = self._obs.copy()
        for cell_type, library_size in zip(cell_types, library_sizes):
            obs[cell_type] = obs[cell_type]\
                .with_columns(pl.lit(library_size).alias(library_size_column))

        # Return a new Pseudobulk dataset with the residuals
        return Pseudobulk(X=self._X, obs=obs, var=self._var,
                          num_threads=self._num_threads)

    def cpm(self,
            *,
            library_size_column: PseudobulkColumn = 'library_size',
            allow_float: bool = False) -> Pseudobulk:
        """
        Calculate counts per million for each cell type.

        Must be run after `library_size()`. Must not be run before de(), since
        `de()` already normalizes the data internally.

        Args:
            library_size_column: a floating-point column of `obs` containing
                                 each sample's library size. Can be a column
                                 name, a polars expression, a polars Series, a
                                 1D NumPy array, or a function that takes in
                                 this Pseudobulk dataset and a cell type and
                                 returns a polars Series or 1D NumPy array. Or,
                                 a dictionary mapping cell-type names to any of
                                 the above; each cell type in this Pseudobulk
                                 dataset must be present.
            allow_float: if `False`, raise an error if `self.X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts); if `True`, disable this sanity check
        Returns:
            A new Pseudobulk dataset containing the CPMs.
        """
        # Get the library size column
        library_sizes = self._get_column(
            'obs', library_size_column, 'library_size_column',
            'floating-point',
            custom_error='library_size_column {} is not a column of obs[{}]; '
                         'did you forget to run library_size()?',
            allow_None=False)

        # Check that `allow_float` is Boolean
        check_type(allow_float, 'allow_float', bool, 'Boolean')

        # If `allow_float=False`, raise an error if `X` is floating-point for
        # any cell type
        if not allow_float:
            for cell_type, X in self._X.items():
                dtype = X.dtype
                if np.issubdtype(dtype, np.floating):
                    error_message = (
                        f"cpm() requires raw counts but X[{cell_type!r}] "
                        f"has data type {str(dtype)!r}, a floating-point data "
                        f"type; if you are sure that all values are raw "
                        f"integer counts, i.e. that (X[{cell_type!r}].data == "
                        f"X[{cell_type!r}].data.astype(int)).all(), then set "
                        f"allow_float=True (or just cast X to an integer data "
                        f"type).")
                    raise TypeError(error_message)

        # Calculate CPMs
        CPMs = {}
        for cell_type, X in self._X.items():
            library_size = library_sizes[cell_type].to_numpy()
            CPMs[cell_type] = X / library_size[:, None] * 1e6
        return Pseudobulk(X=CPMs, obs=self._obs, var=self._var,
                          num_threads=self._num_threads)

    def log_cpm(self,
                *,
                library_size_column: PseudobulkColumn = 'library_size',
                prior_count: int | float | np.integer | np.floating = 2,
                allow_float: bool = False) -> Pseudobulk:
        """
        Calculate log counts per million for each cell type.

        Must be run after `library_size()`. Must not be run before `de()`,
        since `de()` already normalizes the data internally.

        Results were verified to match edgeR to within floating-point error.

        Args:
            library_size_column: a floating-point column of `obs` containing
                                 each sample's library size. Can be a column
                                 name, a polars expression, a polars Series, a
                                 1D NumPy array, or a function that takes in
                                 this Pseudobulk dataset and a cell type and
                                 returns a polars Series or 1D NumPy array. Or,
                                 a dictionary mapping cell-type names to any of
                                 the above; each cell type in this Pseudobulk
                                 dataset must be present.
            allow_float: if `False`, raise an error if `self.X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts); if `True`, disable this sanity check
            prior_count: the pseudocount to add before log-transforming. The
                         corresponding argument in edgeR, `prior.count`, now
                         defaults to 2 instead of the old default of 0.5.

        Returns:
            A new Pseudobulk dataset containing the log(CPMs).
        """
        # Get the library size column
        library_sizes = self._get_column(
            'obs', library_size_column, 'library_size_column',
            'floating-point',
            custom_error='library_size_column {} is not a column of obs[{}]; '
                         'did you forget to run library_size()?',
            allow_None=False)

        # Check that `prior_count` is a positive number
        check_type(prior_count, 'prior_count', (int, float),
                   'a positive number')
        check_bounds(prior_count, 'prior_count', 0, left_open=True)

        # Check that `allow_float` is Boolean
        check_type(allow_float, 'allow_float', bool, 'Boolean')

        # If `allow_float=False`, raise an error if `X` is floating-point for
        # any cell type
        if not allow_float:
            for cell_type, X in self._X.items():
                dtype = X.dtype
                if np.issubdtype(dtype, np.floating):
                    error_message = (
                        f"log_cpm() requires raw counts but X[{cell_type!r}] "
                        f"has data type {str(dtype)!r}, a floating-point data "
                        f"type; if you are sure that all values are raw "
                        f"integer counts, i.e. that (X[{cell_type!r}].data == "
                        f"X[{cell_type!r}].data.astype(int)).all(), then set "
                        f"allow_float=True (or just cast X to an integer data "
                        f"type).")
                    raise TypeError(error_message)

        # Get the log CPMs; this code is based on the R translation of edgeR's
        # C++ `cpm()` code at bioinformatics.stackexchange.com/a/4990.
        log_cpms = {}
        for cell_type, X in self._X.items():
            library_size = library_sizes[cell_type].to_numpy(writable=True)
            pseudocount = prior_count * library_size / library_size.mean()
            library_size += 2 * pseudocount
            log_cpms[cell_type] = np.log2(X + pseudocount[:, None]) - \
                np.log2(library_size[:, None]) + np.log2(1e6)
        return Pseudobulk(X=log_cpms, obs=self._obs, var=self._var,
                          num_threads=self._num_threads)

    @staticmethod
    def _get_unique_variables(formulas: str | Iterable[str],
                              composite: bool = False) -> list[str]:
        """
        Get a list of the unique variables referenced in one or more R
        formulas. Include backtick-quoted variable names that contain spaces
        or other characters that would otherwise be invalid in R variables.
        Do not include R functions (e.g. `exp(x1)` adds `'x1'` to the list, but
        not `exp`) or numbers.

        Args:
            formulas: one or more R formulas, represented as Python strings
            composite: if `True`, avoid splitting "composite" variables like
                       `x1:x2`, so that the unique variables are columns of the
                       design matrix. If `False`, split these into their
                       components, so that the unique variables are columns of
                       obs.

        Returns:
            A list of the unique variables in `formula`, in order of first
            appearance.
        """
        if isinstance(formulas, str):
            formulas = formulas,
        pattern = rf'[+\-*/^()]|`[^`]+`|[\w{":" if composite else ""}.]+'
        seen = set()
        unique_variables = [
            token[1:-1] if token[0] == '`' else token
            for formula in formulas
            for token, next_token in pairwise(re.findall(pattern, formula) +
                                              [''])
            if (token not in seen and not seen.add(token) and
                not re.fullmatch(r'\d+\.?\d*', token) and
                (token[0] == '`' or
                 re.fullmatch(rf'[\w{":" if composite else ""}.]*', token)) and
                next_token != '(')]
        return unique_variables

    @staticmethod
    def _process_formula_variables(formula: str,
                                   cell_type: str,
                                   obs: pl.DataFrame) -> list[str]:
        """
        Check that all variables referenced in `formula` are Categorical, Enum,
        Boolean, integer, or floating-point columns of `obs`. Make a set of
        these variables. Used by `de()` and `regress_out()`.

        Args:
            formula: the formula to process
            cell_type: the cell type the formula is for, used in error messages
            obs: the `obs` for this cell type

        Returns:
            A list of the unique variables in `formula`, in order of first
            appearance.
        """
        unique_formula_variables = \
            Pseudobulk._get_unique_variables(formula)
        valid_dtypes = \
            NUMERIC_DTYPES + (pl.String, pl.Enum, pl.Categorical, pl.Boolean)
        for variable in unique_formula_variables:
            if variable not in obs:
                error_message = (
                    f'formula contains the variable {variable!r}, which is '
                    f'not the name of a column of obs[{cell_type!r}]')
                raise ValueError(error_message)
            base_type = obs[variable].dtype.base_type()
            if base_type not in valid_dtypes:
                error_message = (
                    f'all columns of obs referenced in formula must be '
                    f'Categorical, Enum, Boolean, integer, or floating-point, '
                    f'but it contains the variable {variable!r} and '
                    f'obs[{cell_type!r}][{variable!r}] has data type '
                    f'{base_type!r}')
                raise TypeError(error_message)
        return unique_formula_variables

    @staticmethod
    def _create_design_matrix(formula: str,
                              cell_type: str,
                              obs: pl.DataFrame,
                              obs_names: pl.Series,
                              obs_columns: list[str],
                              categorical_columns: str | tuple[str] | None,
                              ordinal_columns: str | tuple[str] | None,
                              prefix: str,
                              strict: bool) -> bool:
        """
        Create a design matrix from a formula. Used by `de()` and
        `regress_out()`.

        Adds variables called `f'{prefix}.formula'`, `f'{prefix}.obs'`, and
        `f'{prefix}.design.matrix'` to the ryp R workspace, which need to be
        deleted by the calling function.

        Args:
            formula: the formula to construct the design matrix from
            cell_type: the cell type the formula is for
            obs: the `obs` for this cell type
            obs_names: the `obs_names` for this cell type
            obs_columns: the columns of `obs` that need to be converted to R
            categorical_columns: one or more names of integer columns of `obs`
                                 to treat as categorical (i.e. convert to
                                 unordered factors)
            ordinal_columns: one or more names of integer, String, Categorical,
                             or Enum columns of `obs` to treat as ordinal (i.e.
                             convert to ordered factors)
            prefix: a prefix to use for the three variables to be added to the
                    ryp R workspace
            strict: whether to raise an error if the design matrix is
                    rank-deficient

        Returns:
            Whether the design matrix has more rows than columns. When
            `strict=True`, an error will be raised instead of returning
            `False`.
        """
        from ryp import r, to_py, to_r, \
            _bytestring_to_character_vector, _rlib, _RMemory

        # Convert the formula to R
        to_r(formula, f'{prefix}.formula')
        r(f'{prefix}.formula = as.formula({prefix}.formula)')

        # Subset `obs` to just the columns we need
        obs = obs.select(*obs_columns)

        # Check that these columns do not contain any `null` values
        for column in obs:
            null_count = column.null_count()
            if null_count > 0:
                error_message = (
                    f'{column.name} contains {null_count:,} '
                    f'{plural("null value", null_count)} for cell type '
                    f'{cell_type!r}, but must not contain any')
                raise ValueError(error_message)

        # Check that all columns in `categorical_columns` are integer columns
        # of `obs`
        if categorical_columns is None:
            categorical_columns = ()
        else:
            for column in categorical_columns:
                if column not in obs:
                    error_message = (
                        f'one of the columns in categorical_columns, '
                        f'{column!r}, is not a column of obs[{cell_type!r}]')
                    raise ValueError(error_message)
                base_type = obs[column].dtype.base_type()
                if base_type not in INTEGER_DTYPES:
                    error_message = (
                        f'all columns in categorical_columns must be integer, '
                        f'but one of the columns is {column!r} and '
                        f'obs[{cell_type!r}][{column!r}] has data type '
                        f'{base_type!r}')
                    raise TypeError(error_message)

        # Check that all columns in `ordinal_columns` are integer, String,
        # Categorical, or Enum columns of `obs`
        if ordinal_columns is None:
            ordinal_columns = ()
        else:
            for column in ordinal_columns:
                if column not in obs:
                    error_message = (
                        f'one of the columns in ordinal_columns, {column!r}, '
                        f'is not a column of obs[{cell_type!r}]')
                    raise ValueError(error_message)
                base_type = obs[column].dtype.base_type()
                if base_type not in (pl.String, pl.Categorical, pl.Enum) and \
                        base_type not in INTEGER_DTYPES:
                    error_message = (
                        f'all columns in ordinal_columns must be integer, '
                        f'String, Categorical, or Enum, but one of the '
                        f'columns is {column!r} and '
                        f'obs[{cell_type!r}][{column!r}] has data type '
                        f'{base_type!r}')
                    raise TypeError(error_message)

        # Check that no column is specified in both `categorical_columns` and
        # `ordinal_columns`
        if categorical_columns is not None and ordinal_columns is not None:
            for column in categorical_columns:
                if column in ordinal_columns:
                    error_message = (
                        f'the column {column!r} is specified in both '
                        f'categorical_columns and ordinal_columns for cell '
                        f'type {cell_type!r}, but a column cannot be both '
                        f'categorical and ordinal')
                    raise ValueError(error_message)

        # Determine which columns of `obs` will become unordered factors and
        # which will become ordered factors. String, Categorical, and Enum
        # columns become unordered factors, unless listed in `ordinal_columns`.
        # Integer columns become unordered factors only if listed in
        # `categorical_columns`. Any column listed in `ordinal_columns` becomes
        # an ordered factor.
        unordered_columns = [
            column for column in obs.columns
            if column not in ordinal_columns and
            (obs[column].dtype.base_type() in
             (pl.String, pl.Categorical, pl.Enum) or
             column in categorical_columns)]
        ordered_columns = list(ordinal_columns)

        # Convert all factor columns to Enum, so that `to_r` converts them to
        # ordered factors; we strip the `'ordered'` class from the unordered
        # ones below. Columns that are already Enum are left untouched, to
        # preserve their existing level order (which matters for ordinal Enum
        # columns); other columns get sorted unique values as their levels.
        columns_to_cast = [column for column in
                           unordered_columns + ordered_columns
                           if obs[column].dtype.base_type() != pl.Enum]
        if columns_to_cast:
            # Compute the levels before any dtype change, so that integer
            # factors keep numeric rather than lexicographic level order
            # (1, 2, 10 rather than 1, 10, 2), then stringify them.
            levels = {row[0]: row[1] for row in
                      obs.select(
                          pl.selectors.by_name(columns_to_cast)
                          .unique()
                          .sort()
                          .implode()
                          .list.drop_nulls())
                     .unpivot()
                     .cast({'value': pl.List(pl.String)})
                     .rows()}
            # Integer factor columns have to become String before the Enum
            # cast: casting an integer column straight to an Enum whose
            # categories are strings raises InvalidOperationError, which made
            # `categorical_columns` unusable for any integer column.
            integer_factors = [column for column in columns_to_cast
                               if obs[column].dtype.base_type()
                               in INTEGER_DTYPES]
            if integer_factors:
                obs = obs.cast({column: pl.String
                                for column in integer_factors})
            obs = obs.cast({column: pl.Enum(categories)
                            for column, categories in levels.items()})

        # Cast the integer columns that are staying numeric to Float64. `to_r`
        # maps polars integers to R's `integer64` (bit64) class, which
        # `model.matrix()` does not recognize: it reinterprets the 64-bit
        # payload as a double, so an age of 76 becomes 3.75e-322. Depending on
        # the data that yields either a rank-deficient design matrix or NaN
        # coefficients, and neither failure mode points at the dtype.
        numeric_integer_columns = [
            column for column in obs.columns
            if column not in unordered_columns
            and column not in ordered_columns
            and obs[column].dtype.base_type() in INTEGER_DTYPES]
        if numeric_integer_columns:
            obs = obs.cast({column: pl.Float64
                            for column in numeric_integer_columns})

        # Convert the selected columns of `obs` to R
        obs_name = f'{prefix}.obs'
        to_r(obs, obs_name, rownames=obs_names)

        # Convert the unordered-factor columns from ordered to unordered
        # factors by removing the `'ordered'` class and leaving only the
        # `'factor'` class. This has to be done through the R C API to be
        # in-place.
        if unordered_columns:
            R_obs = _rlib.Rf_findVar(_rlib.Rf_install(obs_name.encode()),
                                     _rlib.R_GlobalEnv)
            with _RMemory(_rlib) as rmemory:
                new_class = \
                    _bytestring_to_character_vector(b'factor', rmemory)
                for column in unordered_columns:
                    column_index = obs.columns.index(column)
                    R_column = _rlib.VECTOR_ELT(R_obs, column_index)
                    _rlib.Rf_setAttrib(R_column, _rlib.R_ClassSymbol,
                                       new_class)
        # Create the design matrix
        r(f'{prefix}.design.matrix = model.matrix('
          f'{prefix}.formula, {prefix}.obs)')

        # Remove all-zero columns; these can arise from Enum columns with empty
        # categories, or combinations of variables in an interaction term with
        # zero frequency. Keep the `assign` attribute, which we will need
        # later.
        r(f'''{prefix}.design.matrix <- (function(X) {{
              keep <- colSums(X) != 0
              X2 <- X[, keep, drop = FALSE]
              attr(X2, "assign") <- attr(X, "assign")[keep]
              X2
            }})({prefix}.design.matrix)''')

        # Check that the design matrix has more rows than columns
        height = to_py(f'nrow({prefix}.design.matrix)')
        width = to_py(f'ncol({prefix}.design.matrix)')
        if width >= height:
            if strict:
                error_message = (
                    f'the design matrix must have more rows (samples) than '
                    f'columns (covariates + intercept), but has {height:,} '
                    f'{plural("row", height)} and {width:,} '
                    f'{plural("column", width)} for cell type {cell_type!r}. '
                    f'Either reduce the number of covariates, or exclude this '
                    f'cell type with e.g. excluded_cell_types={cell_type!r}.')
                raise ValueError(error_message)
            else:
                return False

        # Check that the design matrix is non-empty (which can happen e.g. if
        # the user specifies `formula='~0'`)
        if width == 0:
            error_message = 'the design matrix is empty'
            raise ValueError(error_message)

        return True

    @staticmethod
    def _regress_out(X: np.ndarray[np.dtype[np.integer | np.floating]],
                     obs: pl.DataFrame,
                     cell_type: str,
                     cell_type_index: int,
                     formula: str,
                     categorical_columns: str | tuple[str] | None,
                     ordinal_columns: str | tuple[str] | None,
                     error_if_int: bool,
                     verbose: bool) -> np.ndarray[np.dtype[np.integer |
                                                           np.floating]]:
        """
        Regress out covariates from `obs` for a single cell type. Used by
        `regress_out()`.

        Args:
            X: the `X` for this cell type
            obs: the `obs` for this cell type
            cell_type: the cell type covariates will be regressed out for
            cell_type_index: the integer index of the cell type in `cell_types`
            formula: a string representation of an R formula specifying the
                     design matrix to regress out in terms of columns of `obs`,
                     e.g. `'~ disease_status + age + sex'`. Will be converted
                     into an R formula object with R's
                     [`as.formula()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html)
                     function and then expanded into a design matrix with R's
                     [`model.matrix()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/model.matrix.html)
                     function. Must begin with a tilde (`~`). May also be a
                     dictionary mapping cell-type names to formulas; each cell
                     type in this Pseudobulk dataset must be present.
            categorical_columns: one or more names of integer columns of `obs`
                                 to treat as categorical (i.e. convert to
                                 unordered factors)
            ordinal_columns: one or more names of integer, String, Categorical,
                             or Enum columns of `obs` to treat as ordinal (i.e.
                             convert to ordered factors)
            error_if_int: if `True`, raise an error if `self.X.dtype` is
                          integer (indicating the user may not have run
                          `log_cpm()` yet)
            verbose: whether to print out details of the regressing-out process

        Returns:
            `X` with covariates regressed out.
        """
        # Import ryp
        from ryp import r, to_py

        # If `error_if_int=True`, raise an error if `X` has an integer dtype
        if error_if_int and np.issubdtype(X.dtype, np.integer):
            error_message = (
                f'X[{cell_type!r}] has data type {str(X.dtype)!r}, an integer '
                f'data type; did you forget to run log_cpm() before '
                f'regress_out()?')
            raise ValueError(error_message)

        # Check that all variables referenced by `formula` are Categorical,
        # Enum, Boolean, integer, or floating-point columns of `obs`. Make a
        # set of these variables.
        if verbose:
            print(f'[{cell_type}] Validating formula...')
        obs_columns = \
            Pseudobulk._process_formula_variables(formula, cell_type, obs)

        # Make a unique prefix for all R variables for this cell type, to
        # avoid name conflicts with other cell types when multithreading
        # and with other R objects the user might have defined in the ryp R
        # workspace
        prefix = f'.Pseudobulk.{cell_type_index}'

        # Create the design matrix
        try:
            if verbose:
                print(f'[{cell_type}] Creating design matrix...')
            obs_names = obs[:, 0]
            if obs_names.dtype in INTEGER_DTYPES:
                obs_names = obs_names.cast(pl.String)
            Pseudobulk._create_design_matrix(formula, cell_type, obs,
                                             obs_names, obs_columns,
                                             categorical_columns,
                                             ordinal_columns, prefix,
                                             strict=True)
            design_matrix = to_py(f'{prefix}.design.matrix', format='numpy')
        finally:
            r(f'rm(list = Filter(exists, c("{prefix}.obs", '
              f'"{prefix}.formula", "{prefix}.design.matrix")))')

        # Regress out the design matrix; silence warnings with `rcond=None`
        if verbose:
            print(f'[{cell_type}] Regressing out...')
        beta, _, rank, _ = np.linalg.lstsq(design_matrix, X, rcond=None)

        # Check that the design matrix is full-rank
        if rank < design_matrix.shape[1]:
            error_message = (
                f'the design matrix is rank-deficient for cell type '
                f'{cell_type!r} (rank {rank} with {design_matrix.shape[1]} '
                f'columns); some of your covariates are linear '
                f'combinations of other covariates')
            raise ValueError(error_message)

        # Calculate the residuals
        residuals = X - design_matrix @ beta
        return residuals

    def regress_out(self,
                    formula: str | dict[str, str],
                    /,
                    *,
                    categorical_columns: str | Iterable[str] | None |
                                         dict[str, str | Iterable[str] |
                                                   None] = None,
                    ordinal_columns: str | Iterable[str] | None |
                                     dict[str, str | Iterable[str] |
                                     None] = None,
                    cell_types: str | Iterable[str] | None = None,
                    excluded_cell_types: str | Iterable[str] | None = None,
                    error_if_int: bool = True,
                    verbose: bool = False,
                    num_threads: int | np.integer | None = None) -> Pseudobulk:
        """
        Regress out covariates from `obs`. Must be run after `log_cpm()`.

        *To avoid confounding due to library size or the number of cells
        included in the pseudobulk, we strongly recommend including
        `'+ log2(num_cells) + log2(library_size)'` to the `formula`, where
        `'library_size'` is a column of `obs` that can be added by running
        `library_size()` before this function.*

        The design matrix is constructed via the
        [`model.matrix()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/model.matrix.html)
        R function. String, Categorical, and Enum columns of `obs` referenced
        in `formula` are converted to unordered factors, which by default are
        one-hot encoded into `N - 1` columns of the design matrix, where `N` is
        the number of unique values (`contr.treatment` in R). Use the
        `ordinal_columns` argument to treat specific columns as ordered factors
        (i.e. ordinal variables) instead. Use the `categorical_columns`
        argument to treat specific integer columns as unordered factors (i.e.
        categorical variables).

        The way that ordered and unordered factors are encoded can also be
        changed globally. For example, to use Helmert contrasts for ordered
        factors:

        ```
        from ryp import r
        r('options(contrasts=c(unordered="contr.treatment", '
          '                    ordered="contr.helmert"))')
        ```

        To view the current value of the `contrasts` option, use:

        ```r
        from ryp import r
        r('getOption("contrasts")')
        ```

        Args:
            formula: a string representation of an R formula specifying the
                     design matrix to regress out in terms of columns of `obs`,
                     e.g. `'~ disease_status + age + sex'`. Will be converted
                     into an R formula object with R's
                     [`as.formula()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html)
                     function and then expanded into a design matrix with R's
                     [`model.matrix()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/model.matrix.html)
                     function. Must begin with a tilde (`~`). May also be a
                     dictionary mapping cell-type names to formulas; each cell
                     type in this Pseudobulk dataset must be present.
            categorical_columns: one or more names of integer columns of `obs`
                                 to treat as categorical (i.e. convert to
                                 unordered factors), or a dictionary mapping
                                 cell-type names to names of integer columns
            ordinal_columns: one or more names of integer, String, Categorical,
                             or Enum columns of `obs` to treat as ordinal (i.e.
                             convert to ordered factors), or a dictionary
                             mapping cell-type names to names of such columns.
                             By default, ordered factors are assumed to have
                             equally spaced levels and are expanded into
                             `N - 1` columns in the design matrix, where each
                             column represents a polynomial term of increasing
                             degree (linear, quadratic, cubic, etc.) calculated
                             from these equally spaced levels (`contr.poly` in
                             R).
            cell_types: one or more cell types to regress the covariates out
                        of; if `None`, regress covariates out of all cell
                        types. Mutually exclusive with `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude when
                                 regressing out covariates; mutually exclusive
                                 with `cell_types`
            error_if_int: if `True`, raise an error if `self.X.dtype` is
                          integer (indicating the user may not have run
                          `log_cpm()` yet)
            verbose: whether to print out details of the regressing-out process
            num_threads: the number of threads to use when regressing out. Set
                         `num_threads=-1` to use all available cores, as
                         determined by
                         [`os.cpu_count()`](https://docs.python.org/3/library/os.html#os.cpu_count),
                         or leave unset to use `self.num_threads` cores. Does
                         not affect the returned Pseudobulk dataset's
                         `num_threads`; this will always be the same as the
                         original dataset's `num_threads`.

        Returns:
            A new Pseudobulk dataset with covariates regressed out.
        """
        from ryp import r

        # Get the list of cell types to regress out covariates for
        cell_types, cell_type_description = \
            self._process_cell_types(cell_types, excluded_cell_types,
                                     return_description=True)

        # Check that `formula` is a string or a dictionary mapping cell types
        # to strings, and that each string is a valid R formula
        check_type(formula, 'formula', (str, dict),
                   'a string or dictionary of strings')
        formula_is_dict = isinstance(formula, dict)
        if formula_is_dict:
            for key, value in formula.items():
                if not isinstance(key, str):
                    error_message = (
                        f'when formula is a dictionary, all its keys must be '
                        f'strings (cell types), but it contains a key of type '
                        f'{type(key).__name__!r}')
                    raise TypeError(error_message)
                check_type(value, f'formula[{key!r}]', str, 'a string')
                if not value.lstrip().startswith('~'):
                    error_message = \
                        f'formula[{key!r}] must start with a tilde (~)'
                    raise ValueError(error_message)
                try:
                    r(f'invisible(as.formula({value!r}))')
                except RuntimeError as e:
                    error_message = \
                        f'formula[{key!r}] is not a valid R formula'
                    raise ValueError(error_message) from e
            if tuple(formula) != cell_types:
                error_message = (
                    f'formula is a dictionary, but does not have the same '
                    f'cell types (keys) as {cell_type_description}, or has '
                    f'the same cell types in a different order')
                raise ValueError(error_message)
            formulas = formula
        else:
            if not formula.lstrip().startswith('~'):
                error_message = 'formula must start with a tilde (~)'
                raise ValueError(error_message)
            try:
                r(f'invisible(as.formula({formula!r}))')
            except RuntimeError as e:
                error_message = 'formula is not a valid R formula'
                raise ValueError(error_message) from e

        # Check that `categorical_columns` is one or more strings or `None`, or
        # a dictionary mapping cell types to one or more strings or `None`.
        # Convert it (or its values, if a dictionary) to tuples.
        categorical_columns_is_dict = isinstance(categorical_columns, dict)
        if categorical_columns is not None:
            if categorical_columns_is_dict:
                all_categorical_columns = {}
                for key, value in categorical_columns.items():
                    if not isinstance(key, str):
                        error_message = (
                            f'when categorical_columns is a dictionary, all '
                            f'its keys must be strings (cell types), but it '
                            f'contains a key of type {type(key).__name__!r}')
                        raise TypeError(error_message)
                    if value is not None:
                        value_name = f'categorical_columns[{key!r}]'
                        value = \
                            to_tuple_checked(value, value_name, str, 'strings')
                    all_categorical_columns[key] = value
                if tuple(categorical_columns) != cell_types:
                    error_message = (
                        f'categorical_columns is a dictionary, but does not '
                        f'have the same cell types (keys) as '
                        f'{cell_type_description}, or has the same cell types '
                        f'in a different order')
                    raise ValueError(error_message)
            else:
                categorical_columns = to_tuple_checked(
                    categorical_columns, 'categorical_columns', str, 'strings')

        # Check that `ordinal_columns` is one or more strings or `None`, or a
        # dictionary mapping cell types to one or more strings or `None`.
        # Convert it (or its values, if a dictionary) to tuples.
        ordinal_columns_is_dict = isinstance(ordinal_columns, dict)
        if ordinal_columns is not None:
            if ordinal_columns_is_dict:
                all_ordinal_columns = {}
                for key, value in ordinal_columns.items():
                    if not isinstance(key, str):
                        error_message = (
                            f'when ordinal_columns is a dictionary, all its '
                            f'keys must be strings (cell types), but it '
                            f'contains a key of type {type(key).__name__!r}')
                        raise TypeError(error_message)
                    if value is not None:
                        value_name = f'ordinal_columns[{key!r}]'
                        value = \
                            to_tuple_checked(value, value_name, str, 'strings')
                    all_ordinal_columns[key] = value
                if tuple(ordinal_columns) != cell_types:
                    error_message = (
                        f'ordinal_columns is a dictionary, but does not have '
                        f'the same cell types (keys) as '
                        f'{cell_type_description}, or has the same cell types '
                        f'in a different order')
                    raise ValueError(error_message)
            else:
                ordinal_columns = to_tuple_checked(
                    ordinal_columns, 'ordinal_columns', str, 'strings')

        # Check that `error_if_int` and `verbose` are Boolean
        check_type(error_if_int, 'error_if_int', bool, 'Boolean')
        check_type(verbose, 'verbose', bool, 'Boolean')

        # Check that `num_threads` is a positive integer, -1 or `None`; if
        # `None`, set to `self.num_threads`, and if -1, set to
        # `os.cpu_count()`.
        num_threads = self._process_num_threads(num_threads)

        # Compute residuals for each cell type
        with threadpool_limits(1):
            residuals = {
                cell_type: Pseudobulk._regress_out(
                    X=self._X[cell_type], obs=self._obs[cell_type],
                    cell_type=cell_type, cell_type_index=cell_type_index,
                    formula=formulas[cell_type] if formula_is_dict else
                            formula,
                    categorical_columns=all_categorical_columns[cell_type]
                                        if categorical_columns_is_dict else
                                        categorical_columns,
                    ordinal_columns=all_ordinal_columns[cell_type]
                                    if ordinal_columns_is_dict else
                                    ordinal_columns,
                    error_if_int=error_if_int, verbose=verbose)
                    for cell_type_index, cell_type in enumerate(cell_types)}

        # Return a new Pseudobulk dataset with the residuals
        return Pseudobulk(X=residuals, obs=self._obs, var=self._var,
                          num_threads=self._num_threads)

    # A slightly reformatted version of the voomByGroup source code from
    # github.com/YOU-k/voomByGroup/blob/main/voomByGroup.R, which is available
    # under the MIT license. Copyright (c) 2023 Yue You. Also added
    # `drop=FALSE` to the `countsi` subsetting to avoid an error with length-1
    # groups.
    _voomByGroup_source_code = r'''
    voomByGroup <- function (counts, group = NULL, design = NULL,
                             lib.size = NULL, dynamic = NULL,
                             normalize.method = "none", span = 0.5,
                             save.plot = FALSE, print = TRUE, plot = c("none",
                             "all", "separate", "combine"),
                             col.lines = NULL, pos.legend = c("inside",
                             "outside", "none"), fix.y.axis = FALSE, ...) {
      out <- list()
      if (is(counts, "DGEList")) {
        out$genes <- counts$genes
        out$targets <- counts$samples
        if(is.null(group))
          group <- counts$samples$group
        if (is.null(lib.size))
          lib.size <- with(counts$samples, lib.size * norm.factors)
        counts <- counts$counts
      }
      else {
        isExpressionSet <-
          suppressPackageStartupMessages(is(counts, "ExpressionSet"))
        if (isExpressionSet) {
          if (length(Biobase::fData(counts)))
            out$genes <- Biobase::fData(counts)
          if (length(Biobase::pData(counts)))
            out$targets <- Biobase::pData(counts)
          counts <- Biobase::exprs(counts)
        }
        else {
          counts <- as.matrix(counts)
        }
      }
      if (nrow(counts) < 2L)
        stop("Need at least two genes to fit a mean-variance trend")
      # Library size
      if(is.null(lib.size))
        lib.size <- colSums(counts)
      # Group
      if(is.null(group))
        group <- rep("Group1", ncol(counts))
      group <- as.factor(group)
      intgroup <- as.integer(group)
      levgroup <- levels(group)
      ngroups <- length(levgroup)
      # Design matrix
      if (is.null(design)) {
        design <- matrix(1L, ncol(counts), 1)
        rownames(design) <- colnames(counts)
        colnames(design) <- "GrandMean"
      }
      # Dynamic
      if (is.null(dynamic)) {
        dynamic <- rep(FALSE, ngroups)
      }
      # voom by group
      if(print)
        cat("Group:\n")
      E <- w <- counts
      xy <- line <- as.list(rep(NA, ngroups))
      names(xy) <- names(line) <- levgroup
      for (lev in 1L:ngroups) {
        if(print)
          cat(lev, levgroup[lev], "\n")
        i <- intgroup == lev
        countsi <- counts[, i, drop = FALSE]
        libsizei <- lib.size[i]
        designi <- design[i, , drop = FALSE]
        QR <- qr(designi)
        if(QR$rank<ncol(designi))
          designi <- designi[,QR$pivot[1L:QR$rank], drop = FALSE]
        if(ncol(designi)==ncol(countsi))
          designi <- matrix(1L, ncol(countsi), 1)
        voomi <- voom(counts = countsi, design = designi, lib.size = libsizei,
                      normalize.method = normalize.method, span = span,
                      plot = FALSE, save.plot = TRUE, ...)
        E[, i] <- voomi$E
        w[, i] <- voomi$weights
        xy[[lev]] <- voomi$voom.xy
        line[[lev]] <- voomi$voom.line
      }
      #voom overall
      if (TRUE %in% dynamic){
        voom_all <- voom(counts = counts, design = design, lib.size = lib.size,
                         normalize.method = normalize.method, span = span,
                         plot = FALSE, save.plot = TRUE, ...)
        E_all <- voom_all$E
        w_all <- voom_all$weights
        xy_all <- voom_all$voom.xy
        line_all <- voom_all$voom.line
        dge <- DGEList(counts)
        disp <- estimateCommonDisp(dge)
        disp_all <- disp$common
      }
      # Plot, can be "both", "none", "separate", or "combine"
      plot <- plot[1]
      if(plot!="none"){
        disp.group <- c()
        for (lev in levgroup) {
          dge.sub <- DGEList(counts[,group == lev])
          disp <- estimateCommonDisp(dge.sub)
          disp.group[lev] <- disp$common
        }
        if(plot %in% c("all", "separate")){
          if (fix.y.axis == TRUE) {
            yrange <- sapply(levgroup, function(lev){
              c(min(xy[[lev]]$y), max(xy[[lev]]$y))
            }, simplify = TRUE)
            yrange <- c(min(yrange[1,]) - 0.1, max(yrange[2,]) + 0.1)
          }
          for (lev in 1L:ngroups) {
            if (fix.y.axis == TRUE){
              plot(xy[[lev]], xlab = "log2( count size + 0.5 )",
                   ylab = "Sqrt( standard deviation )", pch = 16, cex = 0.25,
                   ylim = yrange)
            } else {
              plot(xy[[lev]], xlab = "log2( count size + 0.5 )",
                   ylab = "Sqrt( standard deviation )", pch = 16, cex = 0.25)
            }
            title(paste("voom: Mean-variance trend,", levgroup[lev]))
            lines(line[[lev]], col = "red")
            legend("topleft", bty="n", paste("BCV:",
              round(sqrt(disp.group[lev]), 3)), text.col="red")
          }
        }

        if(plot %in% c("all", "combine")){
          if(is.null(col.lines))
            col.lines <- 1L:ngroups
          if(length(col.lines)<ngroups)
            col.lines <- rep(col.lines, ngroups)
          xrange <- unlist(lapply(line, `[[`, "x"))
          xrange <- c(min(xrange)-0.3, max(xrange)+0.3)
          yrange <- unlist(lapply(line, `[[`, "y"))
          yrange <- c(min(yrange)-0.1, max(yrange)+0.3)
          plot(1L,1L, type="n", ylim=yrange, xlim=xrange,
               xlab = "log2( count size + 0.5 )",
               ylab = "Sqrt( standard deviation )")
          title("voom: Mean-variance trend")
          if (TRUE %in% dynamic){
            for (dy in which(dynamic)){
              line[[dy]] <- line_all
              disp.group[dy] <- disp_all
              levgroup[dy] <- paste0(levgroup[dy]," (all)")
            }
          }
          for (lev in 1L:ngroups)
            lines(line[[lev]], col=col.lines[lev], lwd=2)
          pos.legend <- pos.legend[1]
          disp.order <- order(disp.group, decreasing = TRUE)
          text.legend <-
            paste(levgroup, ", BCV: ", round(sqrt(disp.group), 3), sep="")
          if(pos.legend %in% c("inside", "outside")){
            if(pos.legend=="outside"){
              plot(1,1, type="n", yaxt="n", xaxt="n", ylab="", xlab="",
                   frame.plot=FALSE)
              legend("topleft", text.col=col.lines[disp.order],
                     text.legend[disp.order], bty="n")
            } else {
              legend("topright", text.col=col.lines[disp.order],
                     text.legend[disp.order], bty="n")
            }
          }
        }
      }
      # Output
      if (TRUE %in% dynamic){
        E[,intgroup %in% which(dynamic)] <-
          E_all[,intgroup %in% which(dynamic)]
        w[,intgroup %in% which(dynamic)] <-
          w_all[,intgroup %in% which(dynamic)]
      }
      out$E <- E
      out$weights <- w
      out$design <- design
      if(save.plot){
        out$voom.line <- line
        out$voom.xy <- xy
      }
      new("EList", out)
    }
    '''

    @staticmethod
    def _de(X: np.ndarray[np.dtype[np.integer | np.floating]],
            obs: pl.DataFrame,
            var: pl.DataFrame,
            cell_type: str,
            cell_type_index: int,
            formula: str,
            coefficient: str | int | np.integer |
                         Iterable[str | int | np.integer] | None,
            contrasts: dict[str, str] | None,
            group: Literal[False] | pl.Series | None,
            categorical_columns: str | tuple[str] | None,
            ordinal_columns: str | tuple[str] | None,
            library_size: pl.Series,
            strict: bool,
            robust: bool,
            return_voom_info: bool,
            verbose: bool) -> pl.DataFrame | \
                              tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Compute differential expression for a single cell type. Used by `de()`.

        Args:
            X: the cell type's `X`
            obs: the cell type's `obs`
            var: the cell type's `var`
            cell_type: the cell type DE will be calculated for
            cell_type_index: the integer index of the cell type in `cell_types`
            formula: a string representation of an R formula specifying the DE
                     design in terms of columns of `obs`, e.g.
                     `'~ disease_status + age + sex'`. Will be converted into
                     an R formula object with R's
                     [`as.formula()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html)
                     function and then expanded into a design matrix with R's
                     [`model.matrix()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/model.matrix.html)
                     function. Must begin with a tilde (`~`).
            coefficient: the name or 0-based index of a coefficient in the
                         design matrix to report DE with respect to, or a
                         sequence of names or indices to report DE with respect
                         to multiple coefficients. Negative indices work in the
                         usual Python way. If `None`, defaults to the first
                         non-intercept column of the design matrix: column 1 if
                         the design matrix has an intercept, or column 0 if it
                         does not. Mutually exclusive with `contrasts`.
            contrasts: an optional dictionary mapping contrast names to string
                       representations of R formulas specifying contrasts
                       between names of columns in the design matrix (e.g.
                       `{'Drug A': 'treatmentDrugA - treatmentControl'}`); the
                       contrast names (keys of the dictionary) will appear in
                       the `'Coefficient'` column of the output DE object. Or,
                       a dictionary mapping cell-type names to these
                       dictionaries; each cell type in this Pseudobulk dataset
                       must be present. If specified, DE will be performed with
                       respect to each contrast by running limma's
                       [`makeContrasts()`](https://www.rdocumentation.org/packages/limma/topics/makeContrasts)
                       and
                       [`contrasts.fit()`](https://www.rdocumentation.org/packages/limma/topics/contrasts.fit)
                       functions after
                       [`lmFit()`](https://www.rdocumentation.org/packages/limma/topics/lmFit).
                       Mutually exclusive with `coefficient`. Notably,
                       contrasts can be specified that involve column names
                       that would not be valid R variable names, and backticks
                       are optional:
                       `{'CD8 vs CD4': 'CD8+ T-cells - CD4+ T-cells'}` is valid
                       even though the two column names `'CD8+ T-cells'` and
                       `'CD4+ T-cells'` are not escaped with backticks.
            group: if `group=False`, force the use of voom instead of
                   voomByGroup. If `group=None`, group on the unique
                   combinations of values of the categorical columns of `obs`
                   referenced in `coefficient`, or the columns of `obs`
                   referenced in `contrasts`. Here, categorical columns are
                   those that are String, Categorical, Enum, or Boolean and not
                   specified in `ordinal_columns`, or those that are integer
                   and specified in `categorical_columns`. If `group` is a
                   column, force the use of voomByGroup and group on the unique
                   values of that column. When using voomByGroup, the same
                   groups are also used as the `group` argument to
                   [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors)
                   when normalizing by library size. All groups must have at
                   least two samples.
            categorical_columns: one or more names of integer columns of `obs`
                                 to treat as categorical (i.e. convert to
                                 unordered factors)
            ordinal_columns: one or more names of integer, String, Categorical,
                             or Enum columns of `obs` to treat as ordinal (i.e.
                             convert to ordered factors)
            library_size: a floating-point polars Series containing each
                          sample's library size
            strict: whether to raise an error if the design matrix does not
                    have more rows than columns and/or is rank-deficient for
                    any cell type. If `strict=False`, cell types where this is
                    the case will be skipped, and an error will only be raised
                    if every cell type is skipped.
            robust: whether to specify `robust=True` in limma's
                    [`eBayes()`](https://www.rdocumentation.org/packages/limma/topics/ebayes)
                    function. You may wish to specify this if your dataset
                    contains outliers.
            return_voom_info: whether to include the voom weights and voom plot
                              data in the returned DE object; set to `False`
                              for reduced runtime if you do not need to use the
                              voom weights or generate voom plots
            verbose: whether to print out details of the DE estimation

        Returns:
            A DataFrame of the DE results for this cell type. Or, if
            `return_voom_info=True`, a tuple of three DataFrames: the DE
            results, the voom weights, and the voom plot info for this cell
            type.
        """
        # Import ryp and limma, and source voomByGroup code
        from ryp import r, to_py, to_r
        r('suppressPackageStartupMessages(library(limma))')
        r(Pseudobulk._voomByGroup_source_code)

        # Check that all variables referenced by `formula` are Categorical,
        # Enum, Boolean, integer, or floating-point columns of `obs`. Make a
        # set of these variables so we can subset to them before converting
        # `obs` to R.
        if verbose:
            print(f'\n[{cell_type}] Validating formula...')
        obs_columns = \
            Pseudobulk._process_formula_variables(formula, cell_type, obs)

        # Make a unique prefix for all R variables for this cell type, to
        # avoid name conflicts with other cell types when multithreading
        # and with other R objects the user might have defined in the ryp R
        # workspace
        prefix = f'.Pseudobulk.{cell_type_index}'
        try:
            # Get obs and var names
            obs_names = obs[:, 0]
            if obs_names.dtype in INTEGER_DTYPES:
                obs_names = obs_names.cast(pl.String)
            var_names = var[:, 0]
            if var_names.dtype in INTEGER_DTYPES:
                var_names = var_names.cast(pl.String)

            # Create the design matrix; return if `strict=False` and there are
            # not strictly more rows than columns. (Don't need to explicitly
            # check for `strict=False`, since if `strict=True`,
            # `_create_design_matrix()` will raise an error internally.)
            if verbose:
                print(f'[{cell_type}] Creating design matrix...')
            more_rows_than_columns = Pseudobulk._create_design_matrix(
                formula, cell_type, obs, obs_names, obs_columns,
                categorical_columns, ordinal_columns, prefix, strict)
            if not more_rows_than_columns:
                if verbose:
                    print(f'[{cell_type}] Design matrix does not have more '
                          f'rows than columns; skipping DE')
                return None
            design_matrix_columns = to_py(f'colnames({prefix}.design.matrix)',
                                          squeeze=False)

            # Check that the design matrix is full-rank
            rank = to_py(f'qr({prefix}.design.matrix)$rank')
            width = len(design_matrix_columns)
            if rank < width:
                if strict:
                    error_message = (
                        f'the design matrix is rank-deficient for cell type '
                        f'{cell_type!r} (rank {rank:,} with {width:,} '
                        f'columns); some of your covariates are linear '
                        f'combinations of other covariates')
                    raise ValueError(error_message)
                else:
                    if verbose:
                        print(f'[{cell_type}] Design matrix is '
                              f'rank-deficient; skipping DE')
                    return None

            # If `contrasts` was specified, validate `contrasts`; otherwise,
            # validate `coefficient`
            if contrasts is None:
                if verbose:
                    print(f'[{cell_type}] Validating coefficient...')

                # If `coefficient` was not specified, default to the first
                # non-intercept column of the design matrix: column 1 if the
                # design matrix has an intercept (the usual case), or column 0
                # if it does not (e.g. when the formula contains `~0` or `-1`)
                if coefficient is None:
                    coefficient = (1,) \
                        if '(Intercept)' in design_matrix_columns else (0,)

                # Check that all string entries of `coefficient` are names of
                # columns of the design matrix
                for coef in coefficient:
                    if isinstance(coef, str) and \
                            coef not in design_matrix_columns:
                        error_message = (
                            f'coefficient {coef!r} is not a column of the '
                            f'design matrix for cell type {cell_type!r}. The '
                            f'design matrix has ')
                        if len(design_matrix_columns) == 1:
                            error_message += \
                                f'one column: {design_matrix_columns[0]!r}'
                        else:
                            all_but_last = ', '.join(
                                f'{column!r}'
                                for column in design_matrix_columns[:-1])
                            error_message += (
                                f'{len(design_matrix_columns):,} columns: '
                                f'{all_but_last} and '
                                f'{design_matrix_columns[-1]!r}')
                        error_message += '.'
                        raise ValueError(error_message)

                # Extract the name of the design matrix column corresponding to
                # each integer in `coefficient`; make sure none of the integers
                # are ≥ the design matrix width
                for coef in coefficient:
                    if isinstance(coef, (int, np.integer)) and coef >= width:
                        contains_string = 'is' \
                            if len(coefficient) == 1 else 'contains the number'
                        error_message = (
                            f'coefficient {contains_string} {coef}, which is '
                            f'more than the number of columns of the design '
                            f'matrix ({width:,}) minus 1 for cell type '
                            f'{cell_type!r}')
                        raise ValueError(error_message)
                coefficient = [design_matrix_columns[coef]
                               if isinstance(coef, (int, np.integer)) else coef
                               for coef in coefficient]

                # Convert `coefficient` to R
                to_r(pl.Series(coefficient), f'{prefix}.coef')
            else:
                if verbose:
                    print(f'[{cell_type}] Validating contrasts...')

                # Check that all variables referenced in `contrasts` are in the
                # design matrix. Use `composite=True` to avoid splitting e.g.
                # `x1:x2` into `x1` and `x2`.
                contrasts = {contrast_name: contrast.replace(' ', '')
                             for contrast_name, contrast in contrasts.items()}
                valid_dtypes = INTEGER_DTYPES + (
                    pl.String, pl.Enum, pl.Categorical, pl.Boolean)
                unique_contrast_variables = \
                    Pseudobulk._get_unique_variables(contrasts.values(),
                                                     composite=True)
                for variable in unique_contrast_variables:
                    if variable not in design_matrix_columns:
                        error_message = (
                            f'a contrast contains the variable {variable!r}, '
                            f'which is not the name of a column of the design '
                            f'matrix for cell type {cell_type!r}. The design '
                            f'matrix has ')
                        if len(design_matrix_columns) == 1:
                            error_message += \
                                f'one column: {design_matrix_columns[0]!r}'
                        else:
                            all_but_last = ', '.join(
                                f'{column!r}'
                                for column in design_matrix_columns[:-1])
                            error_message += (
                                f'{len(design_matrix_columns):,} columns '
                                f'{all_but_last} and '
                                f'{design_matrix_columns[-1]!r}')
                        error_message += '.'
                        raise ValueError(error_message)
            if group is None or contrasts is not None:
                # Get the list of columns of `obs` referenced in `coefficient`
                # or `contrasts`. This can be done by:
                # 1. Getting the columns of the design matrix referenced in
                #    `coefficient` or `contrasts`. For `coefficient`, this is
                #    just the entries of `coefficient` themselves. For
                #    `contrasts`, this is the `unique_contrast_variables`
                #    variable we defined above.
                referenced_design_matrix_columns = coefficient \
                    if contrasts is None else unique_contrast_variables

                # 2. Mapping each column of the design matrix listed in
                #    `coefficient` or `contrasts` back to the integer index of
                #    the term in `formula` it was derived from, using the
                #    design matrix's `assign` attribute. This is 0-based, where
                #    0 indicates the intercept.
                assign = to_py(f'attr({prefix}.design.matrix, "assign")',
                               squeeze=False)
                referenced_term_indices = assign\
                    .filter(design_matrix_columns
                            .is_in(referenced_design_matrix_columns))

                # 3. Using R's `terms()` function to expand the formula into
                #    a list of its component terms. This does not include the
                #    intercept.
                term_labels = \
                    to_py(f'attr(terms({prefix}.formula), "term.labels")',
                          squeeze=False)

                # 4. Pulling out the terms corresponding to the indices from
                #    step 2, to get the terms of `formula` referenced in
                #    `coefficient` or `contrasts`. Since
                #    `referenced_term_indices` includes the intercept but
                #    `term_labels` does not, there's an off-by-one issue. To
                #    fix it, remove any 0s present in `referenced_term_indices`
                #    (we don't care about references to the intercept, since we
                #    are looking for columns of obs referenced in the formula,
                #    and the intercept is not a column of obs), then subtract 1
                #    from `referenced_term_indices`.
                referenced_terms = term_labels[referenced_term_indices.filter(
                    referenced_term_indices != 0) - 1]

                # 5. Getting the unique variables in these terms. These are the
                #    columns of `obs` referenced in `coefficient` or
                #    `contrasts`.
                referenced_columns = \
                    Pseudobulk._get_unique_variables(referenced_terms)

            # If `contrasts` was specified, check that all columns of `obs`
            # referenced in `contrasts` are categorical, now that we know which
            # columns are referenced
            if contrasts is not None:
                for column in referenced_columns:
                    base_type = obs[column].dtype.base_type()
                    if base_type not in valid_dtypes:
                        error_message = (
                            f'all columns of obs referenced in contrasts must '
                            f'be String, Categorical, Enum, Boolean, or '
                            f'integer, but a contrast references the column '
                            f'obs[{cell_type!r}][{column!r}], which has data '
                            f'type {base_type!r}')
                        raise TypeError(error_message)
                    if ordinal_columns is not None and \
                            column in ordinal_columns:
                        error_message = (
                            f'a contrast references the column '
                            f'obs[{cell_type!r}][{column!r}], but it is '
                            f'specified in ordinal_columns and so is treated '
                            f'as ordinal rather than categorical; all columns '
                            f'referenced in contrasts must be categorical')
                        raise TypeError(error_message)
                    if base_type in INTEGER_DTYPES and \
                            (categorical_columns is None or
                             column not in categorical_columns):
                        error_message = (
                            f'a contrast references the column '
                            f'obs[{cell_type!r}][{column!r}] with data type '
                            f'{base_type!r}, but all columns referenced in '
                            f'contrasts must be categorical and integer '
                            f'columns are not treated as categorical unless '
                            f'specified in categorical_columns; did you '
                            f'forget to add {column!r} to '
                            f'categorical_columns?')
                        raise TypeError(error_message)

            # If `group=None`, group on the unique combinations of values of
            # the categorical columns of `obs` referenced in `coefficient` or
            # `contrasts`
            if group is None:
                if verbose:
                    print(f'[{cell_type}] Defining groups...')
                if contrasts is not None:
                    # If using `contrasts`, always use voomByGroup, and group
                    # on the unique combinations of the columns referenced in
                    # the contrasts (recall that we already checked that they
                    # are categorical)
                    group_columns = referenced_columns
                else:
                    # If using `coefficient`, group on the unique combinations
                    # of the categorical columns referenced in the
                    # coefficients. If no columns are categorical, use voom
                    # instead of voomByGroup.
                    categorical_selector = \
                        pl.selectors.by_dtype(pl.String, pl.Categorical,
                                              pl.Enum, pl.Boolean)
                    if categorical_columns is not None:
                        categorical_selector |= \
                            pl.selectors.by_name(categorical_columns)
                    if ordinal_columns is not None:
                        categorical_selector -= \
                            pl.selectors.by_name(ordinal_columns)
                    group_columns = pl.selectors.expand_selector(
                        obs,
                        pl.selectors.by_name(referenced_columns) &
                        categorical_selector)
                if len(group_columns) > 0:
                    # Create a descriptive name for each combination of values
                    # in the `group_columns`
                    group = obs\
                        .select(pl.format(', '.join(
                            f'{column} = {{}}' for column in group_columns),
                            *group_columns))\
                        .to_series()
                    group = group\
                        .cast(pl.Enum(group.unique().sort().to_list()))

                    # If there's only one group (i.e. every sample has the same
                    # value of `group_columns`), disable grouping
                    single_group = len(group.cat.get_categories()) == 1
                    if single_group:
                        group = None

                    # If `verbose=True`, print whether and how we're grouping
                    if verbose:
                        if len(group_columns) == 1:
                            group_column_description = f'{group_columns[0]!r}'
                        elif len(group_columns) == 2:
                            group_column_description = \
                                f'{group_columns[0]!r} and ' \
                                f'{group_columns[1]!r}'
                        else:
                            group_column_description = \
                                ', '.join(map(repr, group_columns[:-1])) + \
                                f', and {group_columns[-1]!r}'
                        if single_group:
                            if len(group_columns) == 1:
                                print(f'[{cell_type}] Not grouping since the '
                                      f'group column '
                                      f'{group_column_description} has the '
                                      f'same value for every sample')
                            else:
                                print(f'[{cell_type}] Not grouping since the '
                                      f'group columns '
                                      f'{group_column_description} have the '
                                      f'same values for every sample')
                        else:
                            print(f'[{cell_type}] Grouping on the '
                                  f'{group_column_description} '
                                  f'{plural("column", len(group_columns))} of '
                                  f'obs.')
                else:
                    if verbose:
                        print(f'[{cell_type}] Not grouping since coefficient '
                              f'does not reference any categorical variables.')
                    group = None

            # If grouping, check that all groups have at least two samples
            grouping = group is not None and group is not False
            if grouping:
                group_counts = group.value_counts().sort('count')
                if group_counts['count'][0] == 1:
                    error_message = (
                        f'all groups must have at least two samples, but '
                        f'group {group_counts[0, 0]!r} has only one '
                        f'sample for cell type {cell_type}')
                    raise ValueError(error_message)

            # Convert the expression matrix and library sizes to R
            if verbose:
                if grouping:
                    print(f'[{cell_type}] Converting the expression matrix, '
                          f'library sizes and groups to R...')
                else:
                    print(f'[{cell_type}] Converting the expression matrix '
                          f'and library sizes to R...')
            to_r(X.T, f'{prefix}.X.T', rownames=var_names, colnames=obs_names)
            to_r(library_size, f'{prefix}.library.size', rownames=obs_names)
            to_r(group, f'{prefix}.group')

            # Run voom
            to_r(return_voom_info, 'save.plot')
            if grouping:
                if verbose:
                    print(f'[{cell_type}] Running voomByGroup...')
                r(f'{prefix}.voom.result = voomByGroup('
                  f'{prefix}.X.T, {prefix}.group, {prefix}.design.matrix, '
                  f'{prefix}.library.size, save.plot=save.plot, print=FALSE)')
            else:
                if verbose:
                    print(f'[{cell_type}] Running voom...')
                r(f'{prefix}.voom.result = voom('
                  f'{prefix}.X.T, {prefix}.design.matrix, '
                  f'{prefix}.library.size, save.plot=save.plot)')
            if return_voom_info:
                voom_weights = \
                    to_py(f'{prefix}.voom.result$weights', index='gene')
                if grouping:
                    voom_plot_data = var_names.to_frame('gene')
                    for group_name in group.unique(maintain_order=True):
                        group_voom_plot_data = pl.DataFrame({
                            'gene': to_py(
                                f'names({prefix}.voom.result$voom.xy$'
                                f'`{group_name}`$x)')} | {
                            f'{prop}_{dim}_{group_name}': to_py(
                                f'{prefix}.voom.result$voom.{prop}$'
                                f'`{group_name}`${dim}', index=False)
                            for prop in ('xy', 'line')
                            for dim in ('x', 'y')})
                        voom_plot_data = voom_plot_data\
                            .join(group_voom_plot_data, on='gene', how='left',
                                  maintain_order='left')
                else:
                    voom_plot_data = pl.DataFrame({
                        'gene': to_py(
                            f'names({prefix}.voom.result$voom.xy$x)')} | {
                        f'{prop}_{dim}': to_py(
                            f'{prefix}.voom.result$voom.{prop}${dim}',
                            index=False)
                        for prop in ('xy', 'line') for dim in ('x', 'y')})

            # Run `lmFit()`
            if verbose:
                print(f'[{cell_type}] Running lmFit...')
            r(f'{prefix}.lmFit.result = lmFit('
              f'{prefix}.voom.result, {prefix}.design.matrix)')
            # Handle contrasts, if specified
            if contrasts is not None:
                if verbose:
                    print(f'[{cell_type}] Making contrasts...')
                # limma's `makeContrasts()` requires the contrast levels to be
                # syntactically valid names in R. But the user specified the
                # contrasts using the original column names (which may not be
                # valid R names), and we want the final result to be in terms
                # of them too. This requires manually escaping the column names
                # within each contrast, then running `makeContrasts()`, then
                # converting back to the original column names.

                # Store the original design matrix column names, then escape
                # them with `make.names()`
                r(f'{prefix}.original = colnames({prefix}.design.matrix)')
                r(f'{prefix}.escaped = make.names({prefix}.original)')
                r(f'colnames({prefix}.design.matrix) = {prefix}.escaped')

                # Convert the original and escaped column names back to Python.
                # Check for name collisions. Make sure no original column name
                # is numeric (which may lead to ambiguity, e.g. what does
                # `2 * X1 - X2` mean when `2` is a column name?). Map original
                # to escaped column names.
                original = to_py(f'{prefix}.original')
                escaped = to_py(f'{prefix}.escaped')
                seen = {}
                for orig, esc in zip(original, escaped):
                    try:
                        float(orig)
                    except ValueError:
                        pass
                    else:
                        error_message = (
                            f'[{cell_type}] column name {orig!r} is numeric, '
                            f'which is disallowed when using contrasts '
                            f'because it risks causing ambiguity; rename or '
                            f'drop it')
                        raise ValueError(error_message)
                    if esc in seen:
                        error_message = (
                            f'[{cell_type}] {seen[esc]!r} and {orig!r} both '
                            f'map to the same R identifier {esc!r} according '
                            f'to make.names(), which is disallowed when using '
                            f'contrasts; rename or drop one to avoid this '
                            f'name collision')
                        raise ValueError(error_message)
                    seen[esc] = orig
                original_to_escaped = dict(zip(original, escaped))

                # Escape the column names within each contrast by mapping them
                # through the `original_to_escaped` mapping. This is
                # challenging because e.g. `'CD8+ T-cells'` could be referring
                # to either a single column called `'CD8+ T-cells'` or a
                # contrast between two columns called `'CD8+ T'` and `'cells'`.
                # So we need to refererence our list of what the original
                # columns were, and detect instances of them within each
                # contrast. Additional complications:
                # - If there's a column called `'24h'` (which R escapes to
                #   `'X24h'` because R variables can't start with numbers) and
                #   another called `'CD24high'`, blindly replacing `'24h'` with
                #   `'X24h'` everywhere will corrupt `'CD24high'` to
                #   `'CDX24high'`. So  naive exact substring matching (e.g.
                #  `polars.Series.str.contains_any()`) will fail. We need to
                #   use regex instead, using negative lookbehind/lookahead to
                #   forbid matches that split valid R identifiers in two. Valid
                #   R identifier characters are `[\w.]`, i.e. word characters
                #   or a dot.
                # - The original columns may be substrings of each other, so
                #   try to match the longest column names first. We can do this
                #   by sorting the column names in descending order of length
                #   before putting them into the regex.
                pattern = re.compile('|'.join(
                    rf'(?<![\w.](?=[\w.])){re.escape(c)}(?!(?<=[\w.])[\w.])'
                    for c in sorted(original, key=len, reverse=True)))
                escaped_contrasts = [
                    pattern.sub(lambda match: original_to_escaped[
                                    match.group(0)], contrast)
                    for contrast in contrasts.values()]

                # Convert the escaped contrasts to R
                to_r(pl.Series(escaped_contrasts), f'{prefix}.contrasts')

                # Make contrasts, using the escaped column names as the levels
                # (to match the escaping we just applied to the contrasts
                # themselves)
                r(f'{prefix}.contrasts = makeContrasts('
                  f'contrasts={prefix}.contrasts, levels={prefix}.escaped)')

                # Rename the design matrix colnames (and the contrast rownames)
                # to the original design matrix colnames
                r(f'rownames({prefix}.contrasts) = {prefix}.original')
                r(f'colnames({prefix}.design.matrix) = {prefix}.original')

                # Set the colnames of the contrasts to `contrasts.keys()`, so
                # they display as those names in the output DE table
                to_r(pl.Series(contrasts.keys()), f'{prefix}.coef')
                r(f'colnames({prefix}.contrasts) = {prefix}.coef')

                # Fit contrasts
                if verbose:
                    print(f'[{cell_type}] Fitting contrasts...')
                r(f'{prefix}.lmFit.result = contrasts.fit('
                  f'{prefix}.lmFit.result, {prefix}.contrasts)')

            # Run `eBayes()`
            if verbose:
                print(f'[{cell_type}] Running eBayes...')
            to_r(robust, f'{prefix}.robust')
            r(f'{prefix}.eBayes.result = eBayes('
              f'{prefix}.lmFit.result, trend=FALSE, robust={prefix}.robust)')

            # Make a table of the DE results
            if verbose:
                print(f'[{cell_type}] Collating results...')
            gene = to_py(f'rownames({prefix}.eBayes.result)')
            logFC = to_py(f'{prefix}.eBayes.result$coefficients['
                          f',{prefix}.coef, drop=FALSE]', index=False)
            SE = to_py(f'{prefix}.eBayes.result$s2.post').sqrt() * \
                 to_py(f'{prefix}.eBayes.result$stdev.unscaled['
                          f',{prefix}.coef, drop=FALSE]', index=False)
            margin_error = \
                SE * stdtrit(to_py(f'{prefix}.eBayes.result$df.total'), 0.975)
            LCI = logFC - margin_error
            UCI = logFC + margin_error
            AveExpr = to_py(f'{prefix}.eBayes.result$Amean', index=False)
            p = to_py(f'{prefix}.eBayes.result$p.value['
                      f',{prefix}.coef, drop=FALSE]', index=False)
            DE_results = pl.concat([
                pl.DataFrame({
                    'coefficient': coef, 'gene': gene, 'logFC': logFC[coef],
                    'SE': SE[coef], 'LCI': LCI[coef], 'UCI': UCI[coef],
                    'AveExpr': AveExpr, 'p': p[coef]})
                .with_columns(Bonferroni=bonferroni(pl.col.p),
                              FDR=fdr(pl.col.p))
                for coef in (
                    contrasts if contrasts is not None else coefficient)])
        finally:
            r(f'rm(list = Filter(exists, c("{prefix}.obs", '
              f'"{prefix}.formula", "{prefix}.design.matrix", '
              f'"{prefix}.library.size", "{prefix}.X.T", "{prefix}.group", '
              f'"{prefix}.voom.result", "{prefix}.lmFit.result", '
              f'"{prefix}.contrasts", "{prefix}.original", '
              f'"{prefix}.escaped", "{prefix}.robust", '
              f'"{prefix}.eBayes.result", "{prefix}.coef")))')
        return (DE_results, voom_weights, voom_plot_data) \
            if return_voom_info else DE_results

    def de(self,
           formula: str | dict[str, str],
           /,
           *,
           coefficient: str | int | np.integer |
                        Iterable[str | int | np.integer] |
                        dict[str, str | int | np.integer |
                                  Iterable[str | int | np.integer] | None] |
                        None = None,
           contrasts: dict[str, str] | None |
                      dict[str, dict[str, str] | None] = None,
           group: Literal[False] | PseudobulkColumn | None |
                  dict[str, Literal[False] | str | pl.Expr | pl.Series |
                            np.ndarray | None] = None,
           categorical_columns: str | Iterable[str] | None |
                                dict[str, str | Iterable[str] | None] = None,
           ordinal_columns: str | Iterable[str] | None |
                            dict[str, str | Iterable[str] | None] = None,
           cell_types: str | Iterable[str] | None = None,
           excluded_cell_types: str | Iterable[str] | None = None,
           library_size_column: PseudobulkColumn = 'library_size',
           strict: bool = False,
           robust: bool = False,
           return_voom_info: bool = True,
           allow_float: bool = False,
           verbose: bool = False) -> DE:
        """
        Perform differential expression (DE) on a Pseudobulk dataset with
        limma-voom. The DE design is specified via an R formula string that
        references columns from `obs`.

        Must be run after `library_size()`. Requires raw counts, so must not be
        run after `cpm()` or `log_cpm()`.

        To avoid confounding due to library size or the number of cells
        included in the pseudobulk, we strongly recommend including
        `'+ log2(num_cells) + log2(library_size)'` to the `formula`, where
        `'library_size'` is a column of `obs` that will be added when running
        `library_size()` before this function.

        By default, DE is reported with respect to the first non-intercept
        column of the design matrix, which is usually just the first term in
        the formula. This can be changed via the `coefficient` and `contrasts`
        arguments.

        By default, voomByGroup is used instead of voom when reporting DE with
        respect to a categorical variable, e.g. when comparing disease cases to
        healthy controls. The unique values of `coefficient` are used as the
        groups. This can be changed via the `group` argument.

        The design matrix is constructed via the
        [`model.matrix()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/model.matrix.html)
        R function. String, Categorical, and Enum columns of `obs` referenced
        in `formula` are converted to unordered factors, which by default are
        one-hot encoded into `N - 1` columns of the design matrix, where `N` is
        the number of unique values (`contr.treatment` in R). Use the
        `ordinal_columns` argument to treat specific columns as ordered factors
        (i.e. ordinal variables) instead. Use the `categorical_columns`
        argument to treat specific integer columns as unordered factors (i.e.
        categorical variables).

        The way that ordered and unordered factors are encoded can also be
        changed globally. For example, to use Helmert contrasts for ordered
        factors:

        ```
        from ryp import r
        r('options(contrasts=c(unordered="contr.treatment", '
          '                    ordered="contr.helmert"))')
        ```

        To view the current value of the `contrasts` option, use:

        ```r
        from ryp import r
        r('getOption("contrasts")')
        ```

        Args:
            formula: a string representation of an R formula specifying the DE
                     design in terms of columns of `obs`, e.g.
                     `'~ disease_status + age + sex'`. Will be converted into
                     an R formula object with R's
                     [`as.formula()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html)
                     function and then expanded into a design matrix with R's
                     [`model.matrix()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/model.matrix.html)
                     function. Must begin with a tilde (`~`).
                     May also be a dictionary mapping cell-type names to
                     formulas; each cell type in this Pseudobulk dataset must
                     be present.
            coefficient: the name or 0-based index of a coefficient in the
                         design matrix to report DE with respect to, or a
                         sequence of names or indices to report DE with respect
                         to multiple coefficients. Or, a dictionary mapping
                         cell-type names to any of the above, or to None to use
                         the default for that cell type; each cell type in this
                         Pseudobulk dataset must be present. Negative indices
                         work in the usual Python way. If not specified, or set
                         to `None` for a given cell type, defaults to the first
                         non-intercept column of the design matrix: column 1 if
                         the formula includes an intercept (the usual case), or
                         column 0 if it does not (e.g. when the formula
                         contains `~0` or `-1`). Mutually exclusive with
                         `contrasts`.
            contrasts: an optional dictionary mapping contrast names to string
                       representations of R formulas specifying contrasts
                       between names of columns in the design matrix (e.g.
                       `{'Drug A': 'treatmentDrugA - treatmentControl'}`); the
                       contrast names (keys of the dictionary) will appear in
                       the `'Coefficient'` column of the output DE object. Or,
                       a dictionary mapping cell-type names to these
                       dictionaries; each cell type in this Pseudobulk dataset
                       must be present. If specified, DE will be performed with
                       respect to each contrast by running limma's
                       [`makeContrasts()`](https://www.rdocumentation.org/packages/limma/topics/makeContrasts)
                       and
                       [`contrasts.fit()`](https://www.rdocumentation.org/packages/limma/topics/contrasts.fit)
                       functions after
                       [`lmFit()`](https://www.rdocumentation.org/packages/limma/topics/lmFit).
                       Mutually exclusive with `coefficient`. Notably,
                       contrasts can be specified that involve column names
                       that would not be valid R variable names, and backticks
                       are optional:
                       `{'CD8 vs CD4': 'CD8+ T-cells - CD4+ T-cells'}` is valid
                       even though the two column names `'CD8+ T-cells'` and
                       `'CD4+ T-cells'` are not escaped with backticks.
            group: if `group=False`, force the use of voom instead of
                   voomByGroup. If `group=None`, group on the unique
                   combinations of values of the categorical columns of `obs`
                   referenced in `coefficient`, or the columns of `obs`
                   referenced in `contrasts`. Here, categorical columns are
                   those that are String, Categorical, Enum, or Boolean and not
                   specified in `ordinal_columns`, or those that are integer
                   and specified in `categorical_columns`. If `group` is a
                   column, force the use of voomByGroup and group on the unique
                   values of that column. `group` can also be a dictionary
                   mapping cell-type names to `False`, `None`, or a column for
                   each cell type. When using voomByGroup, the same groups are
                   also used as the `group` argument to
                   [`calcNormFactors()`](https://www.rdocumentation.org/packages/edgeR/topics/calcNormFactors)
                   when normalizing by library size. All groups must have at
                   least two samples.
            categorical_columns: one or more names of integer columns of `obs`
                                 to treat as categorical (i.e. convert to
                                 unordered factors), or a dictionary mapping
                                 cell-type names to names of integer columns
            ordinal_columns: one or more names of integer, String, Categorical,
                             or Enum columns of `obs` to treat as ordinal (i.e.
                             convert to ordered factors), or a dictionary
                             mapping cell-type names to names of such columns.
                             By default, ordered factors are assumed to have
                             equally spaced levels and are expanded into
                             `N - 1` columns in the design matrix, where each
                             column represents a polynomial term of increasing
                             degree (linear, quadratic, cubic, etc.) calculated
                             from these equally spaced levels (`contr.poly` in
                             R).
            cell_types: one or more cell types to test for DE; if `None`, test
                        all cell types. If specified and using dictionaries for
                        `formula`, `coefficient`, `contrasts`, or `group`, the
                        keys of these dictionaries must match the cell types
                        specified here. Mutually exclusive with
                        `excluded_cell_types`.
            excluded_cell_types: one or more cell types to exclude when testing
                                 for DE. If specified and using dictionaries
                                 for `formula`, `coefficient`, `contrasts`, or
                                 `group`, the keys of these dictionaries must
                                 also exclude these cell types. Mutually
                                 exclusive with `cell_types`.
            library_size_column: a floating-point column of `obs` containing
                                 each sample's library size. Can be a column
                                 name, a polars expression, a polars Series, a
                                 1D NumPy array, or a function that takes in
                                 this Pseudobulk dataset and a cell type and
                                 returns a polars Series or 1D NumPy array. Or,
                                 a dictionary mapping cell-type names to any of
                                 the above; each cell type in this Pseudobulk
                                 dataset must be present.
            strict: whether to raise an error if the design matrix does not
                    have more rows than columns and/or is rank-deficient for
                    any cell type. If `strict=False`, cell types where this is
                    the case will be skipped, and an error will only be raised
                    if every cell type is skipped.
            robust: whether to specify `robust=True` in limma's
                    [`eBayes()`](https://www.rdocumentation.org/packages/limma/topics/ebayes)
                    function. You may wish to specify this if your dataset
                    contains outliers.
            return_voom_info: whether to include the voom weights and voom plot
                              data in the returned DE object; set to `False`
                              for reduced runtime if you do not need to use the
                              voom weights or generate voom plots
            allow_float: if `False`, raise an error if `self.X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts, e.g. due to accidentally having run
                         `log_cpm()` already); if `True`, disable this sanity
                         check
            verbose: whether to print out details of the DE estimation

        Returns:
            A DE object with a `table` attribute containing a polars DataFrame
            of the DE results, with columns:

            - cell_type: the cell type in which DE was tested
            - coefficient: the coefficient (or contrast) for which DE was
                           tested
            - gene: the gene for which DE was tested
            - logFC: the log2 fold change of the gene, i.e. its effect size
            - SE: the standard error of the effect size
            - LCI: the lower 95% confidence interval of the effect size
            - UCI: the upper 95% confidence interval of the effect size
            - AveExpr: the gene's average expression in this cell type, in log
                       CPM
            - p: the DE p-value
            - Bonferroni: the Bonferroni-corrected DE p-value
            - FDR: the FDR q-value for the DE

            If `return_voom_info=True`, the DE object also includes a
            `voom_weights` attribute containing a {cell_type: DataFrame}
            dictionary of voom weights, and a `voom_plot_data` attribute
            containing a {cell_type: DataFrame} dictionary of info necessary to
            construct a voom plot with `DE.plot_voom()`.

        Warning:
            When specifying explicit `contrasts`, you must understand how R's
            `model.matrix()` names its columns under default treatment coding.
            If your formula is `~ treatment` and the `treatment` column has
            values `'Control'` and `'DrugA'`, R treats `'Control'` as the
            baseline and creates a single binary column named
            `'treatmentDrugA'`, where the column name `'treatment'` is
            concatenated with the value `'DrugA'` with no space in between.
            Attempting to contrast `'DrugA - Control'` will crash because
            neither column exists.

            To contrast specific groups intuitively, you should omit the
            intercept from your formula by adding `0 +`
            (e.g. `~ 0 + treatment`). This forces R to create a column for
            every level (e.g., `'treatmentControl'` and `'treatmentDrugA'`).
            You can then explicitly contrast them:
            `contrasts={'Drug A': 'treatmentDrugA - treatmentControl'}`.
        """
        from ryp import r

        # Get the list of cell types to compute DE for
        cell_types, cell_type_description = \
            self._process_cell_types(cell_types, excluded_cell_types,
                                     return_description=True)

        # Check that `formula` is a string or a dictionary mapping cell types
        # to strings, and that each formula is a valid R formula
        check_type(formula, 'formula', (str, dict),
                   'a string or dictionary of strings')
        formula_is_dict = isinstance(formula, dict)
        if formula_is_dict:
            for key, value in formula.items():
                if not isinstance(key, str):
                    error_message = (
                        f'when formula is a dictionary, all its keys must be '
                        f'strings (cell types), but it contains a key of type '
                        f'{type(key).__name__!r}')
                    raise TypeError(error_message)
                check_type(value, f'formula[{key!r}]', str, 'a string')
                if not value.lstrip().startswith('~'):
                    error_message = \
                        f'formula[{key!r}] must start with a tilde (~)'
                    raise ValueError(error_message)
                try:
                    r(f'invisible(as.formula({value!r}))')
                except RuntimeError as e:
                    error_message = \
                        f'formula[{key!r}] is not a valid R formula'
                    raise ValueError(error_message) from e
            if tuple(formula) != cell_types:
                error_message = (
                    f'formula is a dictionary, but does not have the same '
                    f'cell types (keys) as {cell_type_description}, or has '
                    f'the same cell types in a different order')
                raise ValueError(error_message)
            formulas = formula
        else:
            if not formula.lstrip().startswith('~'):
                error_message = 'formula must start with a tilde (~)'
                raise ValueError(error_message)
            try:
                r(f'invisible(as.formula({formula!r}))')
            except RuntimeError as e:
                error_message = 'formula is not a valid R formula'
                raise ValueError(error_message) from e

        # Check that `coefficient` is one or more strings or integers, or a
        # dictionary mapping cell types to one or more strings or integers.
        # Convert it (or its values, if a dictionary) to tuples, storing the
        # result as a new variable, `coefficients`.
        coefficient_is_dict = isinstance(coefficient, dict)
        if coefficient is None:
            # Leave `coefficients` as None; `_de()` will default it to the
            # first non-intercept column of each cell type's design matrix
            coefficients = None
        elif coefficient_is_dict:
            coefficients = {}
            for cell_type, value in coefficient.items():
                if not isinstance(cell_type, str):
                    error_message = (
                        f'when coefficient is a dictionary, all its keys must '
                        f'be strings (cell types), but it contains a key of '
                        f'type {type(cell_type).__name__!r}')
                    raise TypeError(error_message)
                if value is None:
                    coefficients[cell_type] = None
                else:
                    coefficients[cell_type] = to_tuple_checked(
                        value, f'coefficient[{cell_type!r}]', (str, int),
                        'strings or integers')
            if tuple(coefficients) != cell_types:
                error_message = (
                    f'coefficient is a dictionary, but does not have the same '
                    f'cell types (keys) as {cell_type_description}, or has '
                    f'the same cell types in a different order')
                raise ValueError(error_message)
        else:
            coefficients = to_tuple_checked(coefficient, 'coefficient',
                                            (str, int), 'strings or integers')

        # Check that `contrasts` is `None`, a dictionary mapping strings to
        # strings, or a dictionary mapping cell types to `None` or dictionaries
        # mapping strings to strings. Check that each contrast is a valid
        # R formula.
        contrasts_is_nested_dict = False
        if contrasts is not None:
            check_type(contrasts, 'contrasts', dict, 'a dictionary')
            if coefficient is not None:
                error_message = \
                    'coefficient and contrasts cannot both be specified'
                raise ValueError(error_message)
            for key in contrasts:
                if not isinstance(key, str):
                    error_message = (
                        f'all keys of contrasts must be strings, but it '
                        f'contains a key of type {type(key).__name__!r}')
                    raise TypeError(error_message)
            if all(isinstance(value, str) for value in contrasts.values()):
                # `contrasts` is a dictionary mapping strings to strings
                for key, value in contrasts.items():
                    try:
                        r(f'invisible(as.formula({f"~{value}"!r}))')
                    except RuntimeError as e:
                        error_message = \
                            f'contrasts[{key!r}] is not a valid R formula'
                        raise ValueError(error_message) from e
            elif all(isinstance(value, dict) or value is None
                     for value in contrasts.values()):
                # `contrasts` is a dictionary mapping cell types to `None` or
                # dictionaries mapping strings to strings
                contrasts_is_nested_dict = True
                if tuple(contrasts) != cell_types:
                    error_message = (
                        f'contrasts is a dictionary of dictionaries, but does '
                        f'not have the same cell types (keys) as '
                        f'{cell_type_description}, or has the same cell types '
                        f'in a different order')
                    raise ValueError(error_message)
                for key, value in contrasts.items():
                    if value is not None:
                        for inner_key, inner_value in value.items():
                            if not isinstance(inner_key, str):
                                error_message = (
                                    f'all keys of contrasts[{key!r}] must be '
                                    f'strings, but it contains a key of type '
                                    f'{type(inner_key).__name__!r}')
                                raise TypeError(error_message)
                            if not isinstance(inner_value, str):
                                error_message = (
                                    f'all values of contrasts[{key!r}] must '
                                    f'be strings, but it contains a value of '
                                    f'type {type(inner_value).__name__!r}')
                                raise TypeError(error_message)
                            try:
                                r(f'invisible('
                                  f'as.formula({f"~{inner_value}"!r}))')
                            except RuntimeError as e:
                                error_message = (
                                    f'contrasts[{key!r}][{inner_key!r}] is '
                                    f'not a valid R formula')
                                raise ValueError(error_message) from e
                all_contrasts = contrasts
            else:
                error_message = (
                    'contrasts.values() must either be all strings or all '
                    'dictionaries/None')
                raise TypeError(error_message)

        # Check that `group` is `False`, `None`, a categorical column, or a
        # dictionary mapping cell types to `False`, `None`, or categorical
        # columns. If `group` is an Enum and has unused levels, drop them.
        grouping = group is not None and group is not False
        if grouping:
            if group is True:
                error_message = \
                    'group must be None, False, or a column of obs, not True'
                raise TypeError(error_message)
            if isinstance(group, dict):
                for key in group:
                    if not isinstance(key, str):
                        error_message = (
                            f'when group is a dictionary, all its keys must '
                            f'be strings (cell types), but it contains a key '
                            f'of type {type(key).__name__!r}')
                        raise TypeError(error_message)
                if tuple(group) != cell_types:
                    error_message = (
                        f'group is a dictionary, but does not have the same '
                        f'cell types (keys) as {cell_type_description}, or '
                        f'has the same cell types in a different order')
                    raise ValueError(error_message)
                groups = {}
                for cell_type, column in group.items():
                    obs = self._obs[cell_type]
                    if column is None:
                        groups[cell_type] = None
                        continue
                    elif column is False:
                        groups[cell_type] = False
                        continue
                    elif isinstance(column, str):
                        if column not in obs:
                            error_message = (
                                f'group[{cell_type!r}] is {column!r}, which '
                                f'is not a column of obs[{cell_type!r}]')
                            raise ValueError(error_message)
                        column = obs[column]
                    elif isinstance(column, pl.Expr):
                        column = obs.select(column)
                        if column.width > 1:
                            error_message = (
                                f'group[{cell_type!r}] is a polars expression '
                                f'that expands to {column.width:,} columns '
                                f'rather than 1')
                            raise ValueError(error_message)
                        column = column.to_series()
                    elif isinstance(column, pl.Series):
                        if len(column) != len(obs):
                            error_message = (
                                f'group[{cell_type!r}] is a polars Series of '
                                f'length {len(column):,}, which differs from '
                                f'the length of obs[{cell_type!r}] '
                                f'({len(obs):,})')
                            raise ValueError(error_message)
                    elif isinstance(column, np.ndarray):
                        if len(column) != len(obs):
                            error_message = (
                                f'group[{cell_type!r}] is a NumPy array of '
                                f'length {len(column):,}, which differs from '
                                f'the length of obs[{cell_type!r}] '
                                f'({len(obs):,})')
                            raise ValueError(error_message)
                        column = pl.Series('group', column)
                    else:
                        error_message = (
                            f'group[{cell_type!r}] must be None, False, a '
                            f'string column name, a polars expression or '
                            f'Series, or a 1D NumPy array, but has type '
                            f'{type(column).__name__!r}')
                        raise TypeError(error_message)

                    # Check dtype
                    base_type = column.dtype.base_type()
                    if base_type not in (pl.String, pl.Enum, pl.Categorical,
                                         pl.Boolean) and \
                            base_type not in INTEGER_DTYPES:
                        error_message = (
                            f'group[{cell_type!r}] must be String, '
                            f'Categorical, Enum, Boolean, or integer, but has '
                            f'data type {base_type!r}')
                        raise TypeError(error_message)

                    # Remove unused categories, if Enum
                    if base_type == pl.Enum:
                        unique_values = column.unique()
                        if len(unique_values) < \
                                len(column.cat.get_categories()):
                            column = column.cast(
                                pl.Enum(unique_values.cast(pl.String)))

                    # Check `null` values
                    null_count = column.null_count()
                    if null_count > 0:
                        error_message = (
                            f'group[{cell_type!r}] contains {null_count:,} '
                            f'{plural("null value", null_count)}, but must '
                            f'not contain any')
                        raise ValueError(error_message)

                    # Reassign the result
                    groups[cell_type] = column
            else:
                groups = self._get_column(
                    'obs', group, 'group',
                    (pl.String, pl.Enum, pl.Categorical, pl.Boolean,
                     'integer'))

                for cell_type, column in groups.items():
                    if column is not None and column is not False:
                        null_count = column.null_count()
                        if null_count > 0:
                            error_message = (
                                f'group contains {null_count:,} '
                                f'{plural("null value", null_count)} for cell '
                                f'type {cell_type!r}, but must not contain '
                                f'any. Specify the same group column via the '
                                f'group_column argument to Pseudobulk.qc(), '
                                f'explicitly remove nulls from your group '
                                f'column with fill_null(), or specify a '
                                f'different group column (or none at all, to '
                                f'group automatically).')
                            raise ValueError(error_message)

                    # Remove unused categories, if Enum
                    if column.dtype == pl.Enum:
                        unique_values = column.unique()
                        if len(unique_values) < \
                                len(column.cat.get_categories()):
                            groups[cell_type] = column.cast(
                                pl.Enum(unique_values.cast(pl.String)))

        # Check that `categorical_columns` is one or more strings or `None`, or
        # a dictionary mapping cell types to one or more strings or `None`.
        # Convert it (or its values, if a dictionary) to tuples.
        categorical_columns_is_dict = isinstance(categorical_columns, dict)
        if categorical_columns is not None:
            if categorical_columns_is_dict:
                all_categorical_columns = {}
                for key, value in categorical_columns.items():
                    if not isinstance(key, str):
                        error_message = (
                            f'when categorical_columns is a dictionary, all '
                            f'its keys must be strings (cell types), but it '
                            f'contains a key of type {type(key).__name__!r}')
                        raise TypeError(error_message)
                    if value is not None:
                        value_name = f'categorical_columns[{key!r}]'
                        value = \
                            to_tuple_checked(value, value_name, str, 'strings')
                    all_categorical_columns[key] = value
                if tuple(categorical_columns) != cell_types:
                    error_message = (
                        f'categorical_columns is a dictionary, but does not '
                        f'have the same cell types (keys) as '
                        f'{cell_type_description}, or has the same cell types '
                        f'in a different order')
                    raise ValueError(error_message)
            else:
                categorical_columns = to_tuple_checked(
                    categorical_columns, 'categorical_columns', str, 'strings')

        # Check that `ordinal_columns` is one or more strings or `None`, or a
        # dictionary mapping cell types to one or more strings or `None`.
        # Convert it (or its values, if a dictionary) to tuples.
        ordinal_columns_is_dict = isinstance(ordinal_columns, dict)
        if ordinal_columns is not None:
            if ordinal_columns_is_dict:
                all_ordinal_columns = {}
                for key, value in ordinal_columns.items():
                    if not isinstance(key, str):
                        error_message = (
                            f'when ordinal_columns is a dictionary, all its '
                            f'keys must be strings (cell types), but it '
                            f'contains a key of type {type(key).__name__!r}')
                        raise TypeError(error_message)
                    if value is not None:
                        value_name = f'ordinal_columns[{key!r}]'
                        value = \
                            to_tuple_checked(value, value_name, str, 'strings')
                    all_ordinal_columns[key] = value
                if tuple(ordinal_columns) != cell_types:
                    error_message = (
                        f'ordinal_columns is a dictionary, but does not have '
                        f'the same cell types (keys) as '
                        f'{cell_type_description}, or has the same cell types '
                        f'in a different order')
                    raise ValueError(error_message)
            else:
                ordinal_columns = to_tuple_checked(
                    ordinal_columns, 'ordinal_columns', str, 'strings')

        # Get the library size column
        library_sizes = self._get_column(
            'obs', library_size_column, 'library_size_column',
            'floating-point',
            custom_error='library_size_column {} is not a column of obs[{}]; '
                         'did you forget to run library_size()?',
            allow_None=False)

        # Check that Boolean arguments are Boolean
        check_type(strict, 'strict', bool, 'Boolean')
        check_type(robust, 'robust', bool, 'Boolean')
        check_type(return_voom_info, 'return_voom_info', bool, 'Boolean')
        check_type(allow_float, 'allow_float', bool, 'Boolean')
        check_type(verbose, 'verbose', bool, 'Boolean')

        # If `allow_float=False`, raise an error if `X` is floating-point for
        # any cell type
        if not allow_float:
            for cell_type in cell_types:
                dtype = self._X[cell_type].dtype
                if np.issubdtype(dtype, np.floating):
                    error_message = (
                        f"DE() requires raw counts but X[{cell_type!r}] "
                        f"has data type {str(dtype)!r}, a floating-point data "
                        f"type; if you are sure that all values are raw "
                        f"integer counts, i.e. that (X[{cell_type!r}].data == "
                        f"X[{cell_type!r}].data.astype(int)).all(), then set "
                        f"allow_float=True (or just cast X to an integer data "
                        f"type).")
                    raise TypeError(error_message)

        # Compute DE for each cell type
        with threadpool_limits(1):
            DE_results = {}
            if return_voom_info:
                voom_weights = {}
                voom_plot_data = {}
            for cell_type_index, cell_type in enumerate(cell_types):
                results = self._de(
                    X=self._X[cell_type],
                    obs=self._obs[cell_type],
                    var=self._var[cell_type],
                    cell_type=cell_type,
                    cell_type_index=cell_type_index,
                    formula=formulas[cell_type]
                            if formula_is_dict else formula,
                    coefficient=coefficients[cell_type]
                                if coefficient_is_dict else coefficients,
                    contrasts=all_contrasts[cell_type]
                              if contrasts_is_nested_dict else contrasts,
                    group=groups[cell_type] if grouping else group,
                    categorical_columns=all_categorical_columns[cell_type]
                                        if categorical_columns_is_dict else
                                        categorical_columns,
                    ordinal_columns=all_ordinal_columns[cell_type]
                                    if ordinal_columns_is_dict else
                                    ordinal_columns,
                    library_size=library_sizes[cell_type],
                    strict=strict,
                    robust=robust,
                    return_voom_info=return_voom_info,
                    verbose=verbose)
                if results is None:
                    continue
                if return_voom_info:
                    DE_results[cell_type], voom_weights[cell_type], \
                        voom_plot_data[cell_type] = results
                else:
                    DE_results[cell_type] = results

        if not strict and not DE_results:
            error_message = (
                'the design matrix does not have more rows than columns '
                'and/or is rank-deficient for every cell type; set '
                'strict=True for a more detailed explanation of the problem')
            raise ValueError(error_message)

        # Concatenate across cell types
        table = pl.concat([
            cell_type_DE_results
            .select(pl.lit(cell_type).alias('cell_type'), pl.all())
            for cell_type, cell_type_DE_results in DE_results.items()])
        if return_voom_info:
            return DE(table=table, voom_weights=voom_weights,
                      voom_plot_data=voom_plot_data)
        else:
            return DE(table=table)
