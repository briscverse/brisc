from __future__ import annotations
import numpy as np
import polars as pl
from collections.abc import Iterable
from scipy import sparse
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self  # only available in Python 3.11+
from .sparse import csr_array, csc_array
from .utils import array_equal, check_type, sparse_equal, INTEGER_DTYPES, \
    X_DTYPES


class ValidatedDict(dict):
    """
    A dictionary subclass that provides input validation via a
    `_validate()` method (implemented by subclasses).
    """

    def __init__(self, dictionary, /) -> None:
        super().__init__()
        for key, value in dictionary.items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        value = self._validate(key, value)
        super().__setitem__(key, value)

    def _validate(self, key: str, value: Any) -> Any:
        raise NotImplementedError

    def copy(self: Self) -> Self:
        return type(self)(self)

    def setdefault(self, key: str, default: Any = None, /) -> Any:
        if key in self:
            return self[key]
        self[key] = default
        return default

    def update(self,
               other: dict[str, Any] | Iterable[tuple[str, Any]],
               /,
               **kwargs: Any) -> None:
        temp = {}
        temp.update(other, **kwargs)
        for key, value in temp.items():
            self[key] = value

    def __ior__(self: Self, other: dict[str, Any]) -> Self:
        if not isinstance(other, dict):
            return NotImplemented
        self.update(other)
        return self

    def __or__(self: Self, other: dict[str, Any]) -> Self:
        if not isinstance(other, dict):
            return NotImplemented
        new = self.copy()
        new.update(other)
        return new

    def __ror__(self: Self, other: dict[str, Any]) -> Self:
        if not isinstance(other, dict):
            return NotImplemented
        new = self.copy()
        for key, value in other.items():
            if key not in new:
                new[key] = value
        return new

    def equals(self: Self, other: dict[str, Any]) -> bool:
        if not isinstance(other, dict):
            return False
        return self.keys() == other.keys() and \
            all(type(other[key]) is type(value) and
                (array_equal(other[key], value)
                 if isinstance(value, np.ndarray) else
                 other[key].equals(value)
                 if isinstance(value, pl.DataFrame) else
                 sparse_equal(other[key], value)
                 if isinstance(value, (csr_array, csc_array)) else
                 other[key] == value)
                for key, value in self.items())


class LengthValidatedDict(ValidatedDict):
    """
    A subclass of ValidatedDict that stores a `_length` (the number of cells
    for `obs` and `obsm`, or the number of genes for `var` and `varm`), which
    is used during validation by `_validate()`.
    """
    def __init__(self, dictionary, /, *, length) -> None:
        self._length = length
        super().__init__(dictionary)

    def copy(self: Self) -> Self:
        return type(self)(self, length=self._length)


class FixedKeyValidatedDict(ValidatedDict):
    """
    A subclass of ValidatedDict that prevents keys from being added or deleted,
    once initialized by the constructor. The prevention of key deletion is
    handled explicitly here, while the prevention of key addition is delegated
    to `_validate()` for `X`, `Obs` and `Var`.
    """
    def __init__(self, dictionary, /):
        self._initialized = False
        super().__init__(dictionary)
        self._initialized = True

    def __delitem__(self, key: str) -> None:
        error_message = (
            'manual key deletion is not allowed; use drop_cell_types() to '
            'delete cell types')
        raise RuntimeError(error_message)

    def pop(self, key: str, *args: Any) -> Any:
        error_message = (
            'manual key deletion is not allowed; use drop_cell_types() to '
            'delete cell types')
        raise RuntimeError(error_message)

    def popitem(self) -> tuple[str, Any]:
        error_message = (
            'manual key deletion is not allowed; use drop_cell_types() to '
            'delete cell types')
        raise RuntimeError(error_message)

    def clear(self) -> None:
        error_message = (
            'manual key deletion is not allowed; use drop_cell_types() to '
            'delete cell types')
        raise RuntimeError(error_message)


class Obsm(LengthValidatedDict):
    def _validate(self,
                  key: str,
                  value: np.ndarray | pl.DataFrame) -> \
            np.ndarray | pl.DataFrame:
        if not isinstance(key, str):
            error_message = (
                f'all keys of obsm must be strings, but new key has type '
                f'{type(key).__name__!r}')
            raise TypeError(error_message)
        if isinstance(value, np.ndarray):
            if value.ndim != 2:
                error_message = (
                    f'all values of obsm must be 2D NumPy arrays or '
                    f'polars DataFrames, but obsm[{key!r}] is a '
                    f'{value.ndim:,}D NumPy array')
                raise ValueError(error_message)
        elif not isinstance(value, pl.DataFrame):
            error_message = (
                f'all values of obsm must be NumPy arrays or polars '
                f'DataFrames, but obsm[{key!r}] has type '
                f'{type(value).__name__!r}')
            raise TypeError(error_message)
        if len(value) != self._length:
            error_message = (
                f'there are {self._length:,} cells, but the length of new '
                f'obsm[{key!r}] is {len(value):,}')
            raise ValueError(error_message)
        return value


class Varm(LengthValidatedDict):
    def _validate(self,
                  key: str,
                  value: np.ndarray | pl.DataFrame) -> \
            np.ndarray | pl.DataFrame:
        if not isinstance(key, str):
            error_message = (
                f'all keys of varm must be strings, but new key has type '
                f'{type(key).__name__!r}')
            raise TypeError(error_message)
        if isinstance(value, np.ndarray):
            if value.ndim != 2:
                error_message = (
                    f'all values of varm must be 2D NumPy arrays or '
                    f'polars DataFrames, but varm[{key!r}] is a '
                    f'{value.ndim:,}D NumPy array')
                raise ValueError(error_message)
        elif not isinstance(value, pl.DataFrame):
            error_message = (
                f'all values of varm must be NumPy arrays or polars '
                f'DataFrames, but varm[{key!r}] has type '
                f'{type(value).__name__!r}')
            raise TypeError(error_message)
        if len(value) != self._length:
            error_message = (
                f'there are {self._length:,} genes, but the length of new '
                f'varm[{key!r}] is {len(value):,}')
            raise ValueError(error_message)
        return value


class Obsp(LengthValidatedDict):
    def _validate(self,
                  key: str,
                  value: sparse.csr_array | sparse.csc_array |
                         sparse.csr_matrix | sparse.csc_matrix) -> \
            csr_array | csc_array:
        if not isinstance(key, str):
            error_message = (
                f'all keys of obsp must be strings, but new key has type '
                f'{type(key).__name__!r}')
            raise TypeError(error_message)
        if isinstance(value, (csr_array, csc_array)):
            pass
        elif isinstance(value, (sparse.csr_array, sparse.csr_matrix)):
            value = csr_array(value)
        elif isinstance(value, (sparse.csc_array, sparse.csc_matrix)):
            value = csc_array(value)
        else:
            error_message = (
                f'every value of obsp must be a csr_array, csc_array, '
                f'csr_matrix, or csc_matrix, but obsp[{key!r}] has type '
                f'{type(value).__name__!r}')
            raise TypeError(error_message)
        for dim in range(2):
            if value.shape[dim] != self._length:
                error_message = (
                    f'there are {self._length:,} cells, but new '
                    f'obsp[{key!r}].shape[{dim}] is {value.shape[dim]:,}')
                raise ValueError(error_message)
        return value


class Varp(LengthValidatedDict):
    def _validate(self,
                  key: str,
                  value: sparse.csr_array | sparse.csc_array |
                         sparse.csr_matrix | sparse.csc_matrix) -> \
            csr_array | csc_array:
        if not isinstance(key, str):
            error_message = (
                f'all keys of varp must be strings, but new key has type '
                f'{type(key).__name__!r}')
            raise TypeError(error_message)
        if isinstance(value, (csr_array, csc_array)):
            pass
        elif isinstance(value, (sparse.csr_array, sparse.csr_matrix)):
            value = csr_array(value)
        elif isinstance(value, (sparse.csc_array, sparse.csc_matrix)):
            value = csc_array(value)
        else:
            error_message = (
                f'every value of varp must be a csr_array, csc_array, '
                f'csr_matrix, or csc_matrix, but varp[{key!r}] has type '
                f'{type(value).__name__!r}')
            raise TypeError(error_message)
        for dim in range(2):
            if value.shape[dim] != self._length:
                error_message = (
                    f'there are {self._length:,} genes, but '
                    f'varp[{key!r}].shape[{dim}] is {value.shape[dim]:,}')
                raise ValueError(error_message)
        return value


class Uns(ValidatedDict):
    _valid_uns_types = str, int, np.integer, float, np.floating, bool, \
        np.bool_, np.ndarray

    def __setitem__(self, key: str, value: UnsItem | UnsDict) -> None:
        if isinstance(value, dict):
            value = Uns._copy_uns(value)
        super().__setitem__(key, value)

    def copy(self, *, deep: bool = False) -> Uns:
        """
        Make a copy of `uns`.

        Args:
            deep: whether to make a deep or shallow copy of `uns`; if
                  `deep=True`, copy the underlying NumPy arrays if any are
                  present

        Returns:
            A copy of `uns`.
        """
        copied_uns = Uns._copy_uns(self, deep=deep)
        new = dict.__new__(Uns)
        for key, value in copied_uns.items():
            new[key] = value
        return new

    @staticmethod
    def _copy_uns(uns: UnsDict, *, deep: bool = False) -> UnsDict:
        copied_uns = {}
        if deep:
            for key, value in uns.items():
                if isinstance(value, dict):
                    copied_uns[key] = Uns._copy_uns(value, deep=deep)
                elif isinstance(value, np.ndarray):
                    copied_uns[key] = value.copy()
                else:
                    copied_uns[key] = value
        else:
            for key, value in uns.items():
                if isinstance(value, dict):
                    copied_uns[key] = Uns._copy_uns(value, deep=deep)
                else:
                    copied_uns[key] = value
        return copied_uns

    @staticmethod
    def _iter_uns(uns: UnsDict, *, prefix: str = 'uns') -> \
            Iterable[tuple[str, UnsItem]]:
        """
        Recurse through `uns`, yielding tuples of a string describing each
        key (e.g. `"uns['a']['b']"`) and the corresponding value.

        Args:
            uns: an `uns` dictionary
            prefix: the prefix to prepend to each key; applied recursively

        Yields:
            Length-2 tuples where the first element is a string describing
            each key, and the second element is the corresponding value.
        """
        for key, value in uns.items():
            key = f'{prefix}[{key!r}]'
            if isinstance(value, dict):
                yield from Uns._iter_uns(value, prefix=key)
            else:
                yield key, value

    def equals(self: Self, other: dict[str, Any]) -> bool:
        if not isinstance(other, dict):
            return False
        return Uns._equals_recursive(self, other)

    @staticmethod
    def _equals_recursive(uns: UnsDict, other_uns: UnsDict) -> bool:
        return uns.keys() == other_uns.keys() and all(
            isinstance(value, dict) and isinstance(other_value, dict) and
            Uns._equals_recursive(value, other_value) or
            isinstance(value, np.ndarray) and
            isinstance(other_value, np.ndarray) and
            array_equal(value, other_value) or
            not isinstance(other_value, (dict, np.ndarray)) and
            value == other_value
            for key, value, other_value in
            ((key, value, other_uns[key]) for key, value in uns.items()))

    @staticmethod
    def _validate_recursive(uns: UnsDict, *, prefix: str = 'uns') -> None:
        """
        Recurse through `uns`, raising an error if keys are not strings or
        values are not dictionaries or valid uns types.

        Args:
            uns: an `uns` dictionary
            prefix: the prefix to prepend to each key; applied recursively
        """
        for key, value in uns.items():
            if not isinstance(key, str):
                error_message = (
                    f'all keys of uns must be strings, but a subkey of '
                    f'{prefix} has type {type(key).__name__!r}')
                raise TypeError(error_message)
            current_prefix = f'{prefix}[{key!r}]'
            if isinstance(value, dict):
                Uns._validate_recursive(value, prefix=current_prefix)
            elif not isinstance(value, Uns._valid_uns_types):
                error_message = (
                    f'all values of uns must be scalars (strings, '
                    f'numbers or Booleans) or NumPy arrays, or nested '
                    f'dictionaries thereof, but {current_prefix} has '
                    f'type {type(value).__name__!r}')
                raise TypeError(error_message)

    def _validate(self, key: str, value: UnsItem | UnsDict) -> \
            UnsItem | UnsDict:
        if not isinstance(key, str):
            error_message = (
                f'all keys of uns must be strings, but new key has type '
                f'{type(key).__name__!r}')
            raise TypeError(error_message)
        if isinstance(value, dict):
            Uns._validate_recursive(value, prefix=f'uns[{key!r}]')
        elif not isinstance(value, Uns._valid_uns_types):
            error_message = (
                f'all values of uns must be scalars (strings, numbers '
                f'or Booleans) or NumPy arrays, or nested dictionaries '
                f'thereof, but uns[{key!r}] has type '
                f'{type(value).__name__!r}')
            raise TypeError(error_message)
        return value


class X_(FixedKeyValidatedDict):
    def _validate(self,
                  key: str,
                  value: np.ndarray[np.dtype[np.integer | np.floating]]) -> \
            np.ndarray[np.dtype[np.integer | np.floating]]:
        # Check that the key is a string
        if not isinstance(key, str):
            error_message = (
                f'all keys of obs (cell types) must be strings, but new '
                f'key has type {type(key).__name__!r}')
            raise TypeError(error_message)

        # Check that the value is a 2D NumPy array
        check_type(value, f'X[{key!r}]', np.ndarray, 'a NumPy array')
        if value.ndim != 2:
            error_message = (
                f'X[{key!r}] must be 2-dimensional, but is '
                f'{value.ndim:,}-dimensional')
            raise ValueError(error_message)

        # Use `getattr(self, '_initialized', False)` instead of
        # `self._initialized` because the constructor is not called when the
        # object is unpickled (which happens during parallel execution)
        if getattr(self, '_initialized', False):
            # Check that `key` is an existing cell type in `X`: once the
            # Pseudobulk dataset is initialized, the cell types are fixed
            if key not in self:
                error_message = (
                    f'{key!r} is not a cell type in this Pseudobulk '
                    f'dataset')
                raise ValueError(error_message)

            # Check that the old and new shapes match: once the Pseudobulk
            # dataset is initialized, the lengths are fixed
            old = self[key]
            new_shape = value.shape
            old_shape = old.shape
            if new_shape != old_shape:
                error_message = (
                    f'new X[{key!r}] is {new_shape[0]:,} × '
                    f'{new_shape[1]:,}, but old X[{key!r}] is '
                    f'{old_shape[0]:,} × {old_shape[1]:,}')
                raise ValueError(error_message)

        # Check that the data type is (u)int16/32/64 or float32/64
        dtype = value.dtype
        if dtype not in X_DTYPES:
            error_message = (
                f'X must be (u)int16/32/64 or float32/64, but new '
                f'X[{key!r}] has data type {str(dtype)}')
            raise TypeError(error_message)

        return value


class Obs(FixedKeyValidatedDict):
    def _validate(self,
                  key: str,
                  value: pl.DataFrame) -> pl.DataFrame:

        # Check that the key is a string
        if not isinstance(key, str):
            error_message = (
                f'all keys of obs (cell types) must be strings, but new '
                f'key has type {type(key).__name__!r}')
            raise TypeError(error_message)

        # Check that the value is a polars DataFrame
        check_type(value, f'obs[{key!r}]', pl.DataFrame,
                   'a polars DataFrame')

        # Use `getattr(self, '_initialized', False)` instead of
        # `self._initialized` because the constructor is not called when the
        # object is unpickled (which happens during parallel execution)
        if getattr(self, '_initialized', False):
            # Check that `key` is an existing cell type in `obs`: once the
            # Pseudobulk dataset is initialized, the cell types are fixed
            if key not in self:
                error_message = (
                    f'{key!r} is not a cell type in this Pseudobulk '
                    f'dataset')
                raise ValueError(error_message)

            # Check that the old and new lengths match: once the Pseudobulk
            # dataset is initialized, the lengths are fixed
            old = self[key]
            if len(value) != len(old):
                error_message = (
                    f'new obs[{key!r}] has length {len(value):,}, but old '
                    f'obs[{key!r}] has length {len(old):,}')
                raise ValueError(error_message)

        # Check that the first column is String, Enum, Categorical, or integer
        dtype = value[:, 0].dtype
        if dtype not in (pl.String, pl.Enum, pl.Categorical) and \
                dtype not in INTEGER_DTYPES:
            error_message = (
                f'the first column of obs must be String, Enum, Categorical, '
                f'or integer, but the first column of the new obs[{key!r}] '
                f'({value.columns[0]!r}) has data type {dtype.base_type()!r}')
            raise ValueError(error_message)

        return value


class Var(FixedKeyValidatedDict):
    def _validate(self,
                  key: str,
                  value: pl.DataFrame) -> pl.DataFrame:

        # Check that the key is a string
        if not isinstance(key, str):
            error_message = (
                f'all keys of var (cell types) must be strings, but new '
                f'key has type {type(key).__name__!r}')
            raise TypeError(error_message)

        # Check that the value is a polars DataFrame
        check_type(value, f'var[{key!r}]', pl.DataFrame,
                   'a polars DataFrame')

        # Use `getattr(self, '_initialized', False)` instead of
        # `self._initialized` because the constructor is not called when the
        # object is unpickled (which happens during parallel execution)
        if getattr(self, '_initialized', False):
            # Check that `key` is an existing cell type in `var`: once the
            # Pseudobulk dataset is initialized, the cell types are fixed
            if key not in self:
                error_message = (
                    f'{key!r} is not a cell type in this Pseudobulk '
                    f'dataset')
                raise ValueError(error_message)

            # Check that the old and new lengths match: once the Pseudobulk
            # dataset is initialized, the lengths are fixed
            old = self[key]
            if len(value) != len(old):
                error_message = (
                    f'new var[{key!r}] has length {len(value):,}, but old '
                    f'var[{key!r}] has length {len(old):,}')
                raise ValueError(error_message)

        # Check that the first column is String, Enum, Categorical, or integer
        dtype = value[:, 0].dtype
        if dtype not in (pl.String, pl.Enum, pl.Categorical) and \
                dtype not in INTEGER_DTYPES:
            error_message = (
                f'the first column of var must be String, Enum, Categorical, '
                f'or integer, but the first column of the new var[{key!r}] '
                f'({value.columns[0]!r}) has data type {dtype.base_type()!r}')
            raise ValueError(error_message)

        return value
