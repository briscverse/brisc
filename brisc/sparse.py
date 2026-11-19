from __future__ import annotations

# Custom sparse array classes that support parallel operations

import numpy as np
import os
import operator
from scipy import sparse
from scipy.sparse._compressed import _cs_matrix
from typing import Any
from .utils import bitview, check_type, import_cython, sparse_equal, \
    X_DTYPES
import_cython({'cysparse': (
    'cast', 'check_bounds_only', 'check_bounds_and_negative_indices',
    'check_bounds_and_wrap_around', 'csr_column_index1', 'csr_column_index2',
    'csr_eliminate_zeros', 'csr_has_canonical_format',
    'csr_has_sorted_indices', 'csr_outer_index1', 'csr_outer_index2',
    'csr_row_index', 'csr_row_index_check', 'csr_row_slice',
    'csr_row_slice_check', 'csr_sample_values', 'csr_sample_values_canonical',
    'csr_sample_values_sorted', 'csr_sort_indices', 'csr_tocsc',
    'get_csr_submatrix1', 'get_csr_submatrix1_check',
    'get_csr_submatrix1_sorted', 'get_csr_submatrix2',
    'get_csr_submatrix2_sorted', 'wrap_around')})


def get_csr_submatrix(n_row, n_col, Ap, Aj, Ax, ir0, ir1, ic0, ic1,
                      num_threads, has_canonical_format, has_sorted_indices):
    # Allocate indptr
    Bp = np.empty(ir1 - ir0 + 1, dtype=Ap.dtype)

    # Count non-zeros and populate indptr.
    # The _sorted path uses binary search and works whenever indices are sorted.
    # The _check path discovers canonicalness/sortedness on the result matrix,
    # more or less for free.
    if has_canonical_format:
        get_csr_submatrix1_sorted(n_row, n_col, Ap, Aj, ir0, ir1, ic0, ic1, Bp,
                                  num_threads)
        has_sorted_indices = True
    elif has_canonical_format is None:
        has_canonical_format, has_sorted_indices = get_csr_submatrix1_check(
            n_row, n_col, Ap, Aj, ir0, ir1, ic0, ic1, Bp, num_threads)
    elif has_sorted_indices:
        # Not canonical (has duplicates) but sorted; binary search still works
        get_csr_submatrix1_sorted(n_row, n_col, Ap, Aj, ir0, ir1, ic0, ic1, Bp,
                                  num_threads)
    else:
        get_csr_submatrix1(n_row, n_col, Ap, Aj, ir0, ir1, ic0, ic1, Bp,
                           num_threads)

    # Allocate indices and data
    new_nnz = Bp[-1]
    Bj = np.empty(new_nnz, dtype=Aj.dtype)
    Bx = np.empty(new_nnz, dtype=Ax.dtype)

    # Populate indices and data
    if has_sorted_indices:
        get_csr_submatrix2_sorted(n_row, n_col, Ap, Aj, bitview(Ax), ir0, ir1,
                                  ic0, ic1, Bp, Bj, bitview(Bx), num_threads)
    else:
        get_csr_submatrix2(n_row, n_col, Ap, Aj, bitview(Ax), ir0, ir1, ic0,
                           ic1, Bp, Bj, bitview(Bx), num_threads)
    return Bp, Bj, Bx, has_canonical_format, has_sorted_indices


def isintlike(x) -> bool:
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    # Fast-path check to eliminate non-scalar values. operator.index would
    # catch this case too, but the exception catching is slow.
    if np.ndim(x) != 0:
        return False
    try:
        operator.index(x)
    except (TypeError, ValueError):
        try:
            loose_int = bool(int(x) == x)
        except (TypeError, ValueError):
            return False
        if loose_int:
            error_message = \
                'inexact indices into sparse matrices are not allowed'
            raise ValueError(error_message)
        return loose_int
    return True


def _process_slice(sl, num):
    if sl is None:
        i0, i1 = 0, num
    elif isinstance(sl, slice):
        i0, i1, stride = sl.indices(num)
        if stride != 1:
            error_message = 'slicing with step != 1 not supported'
            raise ValueError(error_message)
        i0 = min(i0, i1)  # give an empty slice when i0 > i1
    elif isintlike(sl):
        if sl < 0:
            sl += num
        i0, i1 = sl, sl + 1
        if i0 < 0 or i1 > num:
            error_message = f'index out of bounds: 0 <= {i0} < {i1} <= {num}'
            raise IndexError(error_message)
    else:
        error_message = 'expected slice or scalar'
        raise TypeError(error_message)

    return i0, i1


class cs_matrix(_cs_matrix):
    _num_threads = os.cpu_count()
    _has_sorted_indices = None
    _has_canonical_format = None

    def __getitem__(self, key):
        index, new_shape = self._validate_indices(key)
        INT_TYPES = int, np.integer

        # 1D array
        if len(index) == 1:
            idx = index[0]
            if isinstance(idx, np.ndarray):
                if idx.shape == ():
                    idx = idx.item()
            if isinstance(idx, INT_TYPES):
                res = self._get_int(idx)
            elif isinstance(idx, slice):
                res = self._get_slice(idx)
            else:  # assume array idx
                res = self._get_array(idx)

            # handle np.newaxis in idx when result would otherwise be a scalar
            if res.shape == () and new_shape != ():
                if len(new_shape) == 1:
                    return self.__class__([res], shape=new_shape,
                                          dtype=self.dtype)
                if len(new_shape) == 2:
                    return self.__class__([[res]], shape=new_shape,
                                          dtype=self.dtype)
            return res.reshape(new_shape)

        # 2D array
        row, col = index

        # Dispatch to specialized methods.
        if isinstance(row, INT_TYPES):
            if isinstance(col, INT_TYPES):
                res = self._get_intXint(row, col)
            elif isinstance(col, slice):
                res = self._get_intXslice(row, col)
            elif col.ndim == 1:
                res = self._get_intXarray(row, col)
            elif col.ndim == 2:
                res = self._get_intXarray(row, col)
            else:
                error_message = 'index results in >2 dimensions'
                raise IndexError(error_message)
        elif isinstance(row, slice):
            if isinstance(col, INT_TYPES):
                res = self._get_sliceXint(row, col)
            elif isinstance(col, slice):
                if row == slice(None) and row == col:
                    res = self.copy()
                else:
                    res = self._get_sliceXslice(row, col)
            elif col.ndim == 1:
                res = self._get_sliceXarray(row, col)
            else:
                error_message = 'index results in >2 dimensions'
                raise IndexError(error_message)
        else:
            if isinstance(col, INT_TYPES):
                res = self._get_arrayXint(row, col)
            elif isinstance(col, slice):
                res = self._get_arrayXslice(row, col)
            # arrayXarray preprocess
            elif row.ndim == 2 and row.shape[1] == 1 and \
                    (col.ndim == 1 or col.shape[0] == 1):
                # outer indexing
                res = self._get_columnXarray(row[:, 0], col.ravel())
            else:
                # inner indexing
                row, col = self._broadcast_arrays(row, col)
                if row.shape != col.shape:
                    error_message = 'number of row and column indices differ'
                    raise IndexError(error_message)
                if row.size == 0:
                    res = self.__class__(np.atleast_2d(row).shape,
                                         dtype=self.dtype)
                else:
                    res = self._get_arrayXarray(row, col)
        if getattr(res, 'shape', ()) != new_shape:
            res = res.reshape(new_shape)
        return res

    def shrink_indices(self, copy: bool = False) -> None | cs_matrix:
        """
        Convert indices and indptr from int64 to int32, raising an error if the
        matrix is too large or has too many non-zero elements to be
        representable with int32 indices/indptr.

        Args:
            copy: whether to return a copy of the matrix with shrunk indices,
                  rather than modifying the matrix in-place.

        Returns:
            Nothing (if `copy=False`), or a copy of the matrix with shrunk
            indices (if `copy=True`).
        """
        if self.indices.dtype == np.int32:
            error_message = (
                'indices and indptr are already int32, so shrink_indices '
                'would have no effect')
            raise TypeError(error_message)
        if self.nnz > 2_147_483_647:
            error_message = (
                'X has more than 2,147,483,647 (INT32_MAX) non-zero entries, '
                'so its indices cannot be shrunk')
            raise ValueError(error_message)
        if self.shape[0] > 2_147_483_647:
            error_message = (
                'X has more than 2,147,483,647 (INT32_MAX) cells, so its '
                'indices cannot be shrunk')
            raise ValueError(error_message)
        if self.shape[1] > 2_147_483_647:
            error_message = (
                'X has more than 2,147,483,647 (INT32_MAX) genes, so its '
                'indices cannot be shrunk')
            raise ValueError(error_message)
        indices = np.empty_like(self.indices, dtype=np.int32)
        indptr = np.empty_like(self.indptr, dtype=np.int32)
        cast(self.indices, indices, self._num_threads)
        cast(self.indptr, indptr, self._num_threads)
        if copy:
            result = \
                self.__class__((self.data, indices, indptr), shape=self.shape)
            result._num_threads = self._num_threads
            result._has_sorted_indices = self._has_sorted_indices
            result._has_canonical_format = self._has_canonical_format
            return result
        else:
            self.indices = indices
            self.indptr = indptr
            return None

    @staticmethod
    def _broadcast_arrays(a, b):
        """
        Same as np.broadcast_arrays(a, b) but old writeability rules.

        NumPy >= 1.17.0 transitions broadcast_arrays to return
        read-only arrays. Set writeability explicitly to avoid warnings.
        Retain the old writeability rules, as our Cython code assumes
        the old behavior.
        """
        x, y = np.broadcast_arrays(a, b)
        x.flags.writeable = a.flags.writeable
        y.flags.writeable = b.flags.writeable
        return x, y

    @staticmethod
    def _compatible_boolean_index(idx, desired_ndim):
        """Check for boolean array or array-like. peek before asarray for
        array-like"""
        # use attribute ndim to indicate a compatible array and check dtype
        # if not, look at 1st element as quick rejection of bool, else slower
        # asanyarray
        if not hasattr(idx, 'ndim'):
            # is first element boolean?
            try:
                ix = next(iter(idx), None)
                for _ in range(desired_ndim):
                    if isinstance(ix, bool):
                        break
                    ix = next(iter(ix), None)
                else:
                    return None
            except TypeError:
                return None
            # since first is boolean, construct array and check all elements
            idx = np.asanyarray(idx)

        if idx.dtype.kind == 'b':
            return idx
        return None

    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.
        """
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError) as e:
            error_message = 'invalid index'
            raise IndexError(error_message) from e

        if x.ndim not in (1, 2):
            error_message = 'Index dimension must be 1 or 2'
            raise IndexError(error_message)

        if x.size == 0:
            return x

        # Check bounds and wrap negative indices around
        num_threads = self._num_threads
        if x.dtype == np.uint32 or x.dtype == np.uint64:
            # Unsigned arrays can't have negative indices, so just check bounds
            out_of_bounds = check_bounds_only(x.ravel(), length, num_threads)
            if out_of_bounds:
                error_message = 'index out of range'
                raise IndexError(error_message)
        elif x is idx or not idx.flags.owndata:
            # Need to copy the array if there are negative indices
            out_of_bounds, negative_indices = \
                check_bounds_and_negative_indices(x.ravel(), length,
                                                  num_threads)
            if out_of_bounds:
                error_message = 'index out of range'
                raise IndexError(error_message)
            if negative_indices:
                x = x.copy()
                wrap_around(x.ravel(), length, num_threads)
        else:
            # No need to copy if there are negative indices, so check bounds
            # and wrap around negative indices in a single step
            out_of_bounds = \
                check_bounds_and_wrap_around(x.ravel(), length, num_threads)
            if out_of_bounds:
                error_message = 'index out of range'
                raise IndexError(error_message)
        return x

    def _validate_indices(self, key):
        """Returns two tuples: (index tuple, requested shape tuple)"""
        # single ellipsis
        if key is Ellipsis:
            return (slice(None),) * self.ndim, self.shape

        if not isinstance(key, tuple):
            key = [key]

        ellps_pos = None
        index_1st = []
        prelim_ndim = 0
        for i, idx in enumerate(key):
            if idx is ...:
                if ellps_pos is not None:
                    error_message = \
                        'an index can only have a single ellipsis'
                    raise IndexError(error_message)
                ellps_pos = i
            elif idx is None:
                index_1st.append(idx)
            elif isinstance(idx, slice) or isintlike(idx):
                index_1st.append(idx)
                prelim_ndim += 1
            else:
                ix = self._compatible_boolean_index(idx, self.ndim)
                if ix is not None:
                    index_1st.append(ix)
                    prelim_ndim += ix.ndim
                elif sparse.issparse(idx):
                    raise IndexError(
                        'Indexing with sparse matrices is not supported '
                        'except boolean indexing where matrix and index '
                        'are equal shapes.')
                else:  # dense array
                    index_1st.append(np.asarray(idx))
                    prelim_ndim += 1
        ellip_slices = (self.ndim - prelim_ndim) * [slice(None)]
        if ellip_slices:
            if ellps_pos is None:
                index_1st.extend(ellip_slices)
            else:
                index_1st = index_1st[:ellps_pos] + ellip_slices + \
                            index_1st[ellps_pos:]

        # second pass (have processed ellipsis and preprocessed arrays)
        idx_shape = []
        index_ndim = 0
        index = []
        array_indices = []
        for i, idx in enumerate(index_1st):
            if idx is None:
                idx_shape.append(1)
            elif isinstance(idx, slice):
                index.append(idx)
                Ms = self._shape[index_ndim]
                len_slice = len(range(*idx.indices(Ms)))
                idx_shape.append(len_slice)
                index_ndim += 1
            elif isintlike(idx):
                N = self._shape[index_ndim]
                if not -N <= idx < N:
                    error_message = f'index ({idx}) out of range'
                    raise IndexError(error_message)
                idx = int(idx + N if idx < 0 else idx)
                index.append(idx)
                index_ndim += 1
            # bool array (checked in first pass)
            elif idx.dtype.kind == 'b':
                ix = idx
                tmp_ndim = index_ndim + ix.ndim
                mid_shape = self._shape[index_ndim:tmp_ndim]
                if ix.shape != mid_shape:
                    error_message = (
                        f'bool index {i} has shape {ix.shape} instead of '
                        f'{mid_shape}')
                    raise IndexError(error_message)
                index.extend(ix.nonzero())
                array_indices.extend(range(index_ndim, tmp_ndim))
                index_ndim = tmp_ndim
            # (u)int32/64 integer array
            elif idx.dtype == np.int32 or idx.dtype == np.int64 or \
                    idx.dtype == np.uint32 or idx.dtype == np.uint64:
                N = self._shape[index_ndim]
                idx = self._asindices(idx, N)
                index.append(idx)
                array_indices.append(index_ndim)
                index_ndim += 1
            else:
                error_message = f'invalid index dtype {idx.dtype}'
                raise TypeError(error_message)
        if index_ndim > self.ndim:
            error_message = (
                f'invalid index ndim. Array is {self.ndim}D. Index needs '
                f'{index_ndim}D')
            raise IndexError(error_message)
        if len(array_indices) > 1:
            idx_arrays = \
                self._broadcast_arrays(*(index[i] for i in array_indices))
            if any(idx_arrays[0].shape != ix.shape
                   for ix in idx_arrays[1:]):
                shapes = ' '.join(str(ix.shape) for ix in idx_arrays)
                error_message = (
                    f'shape mismatch: indexing arrays could not be broadcast '
                    f'together with shapes {shapes}')
                raise IndexError(error_message)
            idx_shape = list(idx_arrays[0].shape) + idx_shape
        elif len(array_indices) == 1:
            arr_index = array_indices[0]
            arr_shape = list(index[arr_index].shape)
            idx_shape = idx_shape[:arr_index] + arr_shape + \
                        idx_shape[arr_index:]
        ndim = len(idx_shape)
        if ndim > 2:
            error_message = \
                f'Only 1D or 2D arrays allowed. Index makes {ndim}D'
            raise IndexError(error_message)
        return tuple(index), tuple(idx_shape)

    @property
    def num_threads(self) -> int:
        """
        The number of threads used for sparse array operations.
        """
        return self._num_threads

    @num_threads.setter
    def num_threads(self, num_threads: int | np.integer) -> None:
        """
        Set the number of threads used for sparse array operations.

        Args:
            num_threads: the new number of threads to use for sparse array
                         operations. Set `num_threads=-1` to use all available
                         cores, as determined by `os.cpu_count()`.
        """
        check_type(num_threads, 'num_threads', int, 'a positive integer or -1')
        cpu_count = os.cpu_count()
        if num_threads == -1:
            num_threads = cpu_count
        else:
            num_threads = int(num_threads)
            if num_threads <= 0:
                error_message = (
                    f'num_threads is {num_threads:,}, but must be a positive '
                    f'integer or -1')
                raise ValueError(error_message)
            if num_threads > cpu_count:
                error_message = (
                    f'num_threads is {num_threads:,}, but must be at most '
                    f'os.cpu_count() ({cpu_count})')
                raise ValueError(error_message)
        self._num_threads = num_threads

    def _get_intXint(self, row, col):
        major, minor = self._swap((row, col))
        indptr = self.indptr
        start, end = indptr[major], indptr[major + 1]
        indices = self.indices
        data = self.data
        if self._has_sorted_indices is True:
            offset = np.searchsorted(indices[start:end], minor) + start
            if self._has_canonical_format:
                if offset < end and indices[offset] == minor:
                    return data[offset]
                return self.dtype.type(0)
            else:
                # sorted with duplicates: sum all
                result = self.dtype.type(0)
                while offset < end and indices[offset] == minor:
                    result += data[offset]
                    offset += 1
                return result
        else:
            result = self.dtype.type(0)
            for k in range(start, end):
                if indices[k] == minor:
                    result += data[k]
            return result

    def _get_sliceXslice(self, row, col):
        major, minor = self._swap((row, col))
        if major.step in (1, None) and minor.step in (1, None):
            return self._get_submatrix(major, minor, copy=True)
        return self._major_slice(major)._minor_slice(minor)

    def _get_arrayXarray(self, row, col):
        # inner indexing
        M, N = self._swap(self.shape)
        major, minor = self._swap((row, col))
        val = np.empty(major.size, dtype=self.dtype)
        if self._has_canonical_format:
            csr_sample_values_canonical(M, self.indptr, self.indices,
                                        bitview(self.data), major.size,
                                        major.ravel(), minor.ravel(),
                                        bitview(val), self._num_threads)
        elif self._has_sorted_indices:
            csr_sample_values_sorted(M, self.indptr, self.indices, self.data,
                                     major.size, major.ravel(),
                                     minor.ravel(), val, self._num_threads)
        else:
            csr_sample_values(M, self.indptr, self.indices, self.data,
                              major.size, major.ravel(), minor.ravel(), val,
                              self._num_threads)
        if major.ndim == 1:
            return val
        result = self.__class__(val.reshape(major.shape))
        result._num_threads = self._num_threads
        result._has_sorted_indices = self._has_sorted_indices
        result._has_canonical_format = self._has_canonical_format
        return result

    def _get_columnXarray(self, row, col):
        # outer indexing
        new_shape = len(row), len(col)
        row, col = self._swap((row, col))
        indices = self.indices
        indptr = self.indptr
        M, N = self._swap(self._shape_as_2d)
        if len(col) == 0:
            result = self.__class__(new_shape, dtype=self.dtype)
            result._num_threads = self._num_threads
            result._has_sorted_indices = self._has_sorted_indices
            result._has_canonical_format = self._has_canonical_format
            return result

        # pass 1: count column index entries and compute new indptr
        index_dtype = indices.dtype
        col_offsets = np.empty(N, dtype=index_dtype)
        res_indptr = np.empty(len(row) + 1, dtype=index_dtype)
        csr_outer_index1(row, col, N, indptr, indices, col_offsets,
                         res_indptr, self._num_threads)

        # pass 2: copy indices/data for selected column indices
        col_order = np.argsort(col)
        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=index_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        csr_outer_index2(row, col_order, col_offsets, len(indices), indptr,
                         indices, bitview(self.data), res_indptr, res_indices,
                         bitview(res_data), self._num_threads)
        result = self.__class__((res_data, res_indices, res_indptr),
                                shape=new_shape)
        result._num_threads = self._num_threads
        # `result` is sorted only if `col` is sorted, and canonical only if
        # strictly monotonically increasing
        col_sorted = (col[:-1] <= col[1:]).all()
        col_strictly_increasing = col_sorted and (col[:-1] < col[1:]).all()
        result._has_sorted_indices = \
            self._has_sorted_indices if col_sorted else False
        result._has_canonical_format = \
            self._has_canonical_format if col_strictly_increasing else False
        return result

    def _major_index_fancy(self, idx):
        """Index along the major axis where idx is an array of ints.
        """
        data = self.data
        indices = self.indices
        indptr = self.indptr
        idx = idx.ravel()
        N = self._swap(self._shape_as_2d)[1]
        M = len(idx)
        new_shape = self._swap((M, N)) if self.ndim == 2 else (M,)
        if M == 0:
            result = self.__class__(new_shape, dtype=self.dtype)
            result._num_threads = self._num_threads
            result._has_sorted_indices = self._has_sorted_indices
            result._has_canonical_format = self._has_canonical_format
            return result
        row_nnz = self.indptr[idx + 1] - self.indptr[idx]
        index_dtype = indices.dtype
        res_indptr = np.empty(M + 1, dtype=index_dtype)
        res_indptr[0] = 0
        np.cumsum(row_nnz, out=res_indptr[1:])

        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=index_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        if self._has_canonical_format:
            csr_row_index(M, idx, indptr, indices, bitview(data), res_indptr,
                          res_indices, bitview(res_data), self._num_threads)
            has_canonical_format, has_sorted_indices = True, True
        elif self._has_canonical_format is None:
            has_canonical_format, has_sorted_indices = csr_row_index_check(
                M, idx, indptr, indices, bitview(data), res_indptr,
                res_indices, bitview(res_data), self._num_threads)
            # Propagate: can't confirm parent is canonical from a subset, but
            # can confirm it's non-canonical
            if has_canonical_format is False:
                self._has_canonical_format = False
            if has_sorted_indices is False:
                self._has_sorted_indices = False
        else:
            csr_row_index(M, idx, indptr, indices, bitview(data), res_indptr,
                          res_indices, bitview(res_data), self._num_threads)
            has_canonical_format = False
            has_sorted_indices = self._has_sorted_indices

        result = self.__class__((res_data, res_indices, res_indptr),
                                shape=new_shape)
        result._num_threads = self._num_threads
        # `result` is sorted only if `idx` is sorted, and canonical only if
        # strictly monotonically increasing
        idx_sorted = (idx[:-1] <= idx[1:]).all()
        idx_strictly_increasing = idx_sorted and (idx[:-1] < idx[1:]).all()
        result._has_sorted_indices = \
            self._has_sorted_indices if idx_sorted else False
        result._has_canonical_format = \
            self._has_canonical_format if idx_strictly_increasing else False
        return result

    def _major_slice(self, idx, copy=False):
        """Index along the major axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(self._shape_as_2d)
        start, stop, step = idx.indices(M)
        M = len(range(start, stop, step))
        new_shape = self._swap((M, N)) if self.ndim == 2 else (M,)
        if M == 0:
            result = self.__class__(new_shape, dtype=self.dtype)
            result._num_threads = self._num_threads
            result._has_sorted_indices = self._has_sorted_indices
            result._has_canonical_format = self._has_canonical_format
            return result

        # Work out what slices are needed for `row_nnz`
        # start,stop can be -1, only if step is negative
        start0, stop0 = start, stop
        if stop == -1 and start >= 0:
            stop0 = None
        start1, stop1 = start + 1, stop + 1

        row_nnz = self.indptr[start1:stop1:step] - \
            self.indptr[start0:stop0:step]
        index_dtype = self.indices.dtype
        res_indptr = np.empty(M + 1, dtype=index_dtype)
        res_indptr[0] = 0
        np.cumsum(row_nnz, out=res_indptr[1:])

        if step == 1:
            all_idx = slice(self.indptr[start], self.indptr[stop])
            res_indices = np.array(self.indices[all_idx], copy=copy)
            res_data = np.array(self.data[all_idx], copy=copy)
            has_canonical_format = self._has_canonical_format
            has_sorted_indices = self._has_sorted_indices
        else:
            nnz = res_indptr[-1]
            res_indices = np.empty(nnz, dtype=index_dtype)
            res_data = np.empty(nnz, dtype=self.dtype)
            if self._has_canonical_format:
                csr_row_slice(start, stop, step, self.indptr, self.indices,
                              bitview(self.data), res_indptr, res_indices,
                              bitview(res_data), self._num_threads)
                has_canonical_format = True
                has_sorted_indices = True
            elif self._has_canonical_format is None:
                has_canonical_format, has_sorted_indices = csr_row_slice_check(
                    start, stop, step, self.indptr, self.indices,
                    bitview(self.data), res_indptr, res_indices,
                    bitview(res_data), self._num_threads)
                # Propagate non-canonicalness to parent
                if has_canonical_format is False:
                    self._has_canonical_format = False
                if has_sorted_indices is False:
                    self._has_sorted_indices = False
            else:
                csr_row_slice(start, stop, step, self.indptr, self.indices,
                              bitview(self.data), res_indptr, res_indices,
                              bitview(res_data), self._num_threads)
                has_canonical_format = False
                has_sorted_indices = self._has_sorted_indices

        result = self.__class__((res_data, res_indices, res_indptr),
                                shape=new_shape)
        result._num_threads = self._num_threads
        result._has_sorted_indices = has_sorted_indices
        result._has_canonical_format = has_canonical_format
        return result

    def _minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """
        indices = self.indices
        indptr = self.indptr
        idx = idx.ravel()

        M, N = self._swap(self._shape_as_2d)
        k = len(idx)
        new_shape = self._swap((M, k)) if self.ndim == 2 else (k,)
        if k == 0:
            result = self.__class__(new_shape, dtype=self.dtype)
            result._num_threads = self._num_threads
            result._has_sorted_indices = self._has_sorted_indices
            result._has_canonical_format = self._has_canonical_format
            return result

        # pass 1: count idx entries and compute new indptr
        index_dtype = indices.dtype
        col_offsets = np.empty(N, dtype=index_dtype)
        res_indptr = np.empty_like(indptr, dtype=index_dtype)
        csr_column_index1(k, idx, M, N, indptr, indices, col_offsets,
                          res_indptr, self._num_threads)

        # pass 2: copy indices/data for selected idxs
        col_order = np.argsort(idx)
        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=index_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        csr_column_index2(col_order, col_offsets, len(indices),
                          indptr, indices, bitview(self.data), res_indptr,
                          res_indices, bitview(res_data), self._num_threads)
        result = self.__class__((res_data, res_indices, res_indptr),
                                shape=new_shape)
        result._num_threads = self._num_threads
        # `result` is sorted only if `idx` is sorted, and canonical only if
        # strictly monotonically increasing
        idx_sorted = (idx[:-1] <= idx[1:]).all()
        idx_strictly_increasing = idx_sorted and (idx[:-1] < idx[1:]).all()
        result._has_sorted_indices = \
            self._has_sorted_indices if idx_sorted else False
        result._has_canonical_format = \
            self._has_canonical_format if idx_strictly_increasing else False
        return result

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(self._shape_as_2d)
        start, stop, step = idx.indices(N)
        N = len(range(start, stop, step))
        if N == 0:
            result = self.__class__(self._swap((M, N)), dtype=self.dtype)
            result._num_threads = self._num_threads
            result._has_sorted_indices = self._has_sorted_indices
            result._has_canonical_format = self._has_canonical_format
            return result
        if step == 1:
            return self._get_submatrix(minor=idx, copy=copy)
        return self._minor_index_fancy(np.arange(start, stop, step))

    def _get_submatrix(self, major=None, minor=None, copy=False):
        """Return a submatrix of this matrix.

        major, minor: None, int, or slice with step 1
        """
        M, N = self._swap(self._shape_as_2d)
        i0, i1 = _process_slice(major, M)
        j0, j1 = _process_slice(minor, N)

        if i0 == 0 and j0 == 0 and i1 == M and j1 == N:
            return self.copy() if copy else self

        indptr, indices, data, has_canonical_format, has_sorted_indices = \
            get_csr_submatrix(M, N, self.indptr, self.indices, self.data, i0,
                              i1, j0, j1, self._num_threads,
                              self._has_canonical_format,
                              self._has_sorted_indices)

        # Propagate discovered status back to parent.
        # _check scans rows [i0, i1) of the parent, so:
        # - full range → result determines parent completely
        # - partial range, found non-canonical → parent is non-canonical
        # - partial range, found canonical → can't conclude about parent
        if self._has_canonical_format is None:
            if i0 == 0 and i1 == M:
                self._has_canonical_format = has_canonical_format
                self._has_sorted_indices = has_sorted_indices
            else:
                if has_canonical_format is False:
                    self._has_canonical_format = False
                if has_sorted_indices is False:
                    self._has_sorted_indices = False

        shape = self._swap((i1 - i0, j1 - j0))
        if self.ndim == 1:
            shape = shape[1],
        result = self.__class__((data, indices, indptr), shape=shape,
                                dtype=self.dtype)
        result._num_threads = self._num_threads
        result._has_sorted_indices = has_sorted_indices
        result._has_canonical_format = has_canonical_format
        return result

    def eliminate_zeros(self):
        """Remove zero entries from the array/matrix

        This is an *in place* operation.
        """
        M, N = self._swap(self._shape_as_2d)
        csr_eliminate_zeros(M, N, self.indptr, self.indices, self.data)
        self.indices = self.indices[:self.nnz]
        self.data = self.data[:self.nnz]

    @property
    def has_canonical_format(self) -> bool:
        """Whether the array/matrix has sorted indices and no duplicates

        Returns
            - `True`: if the above applies
            - `False`: otherwise

        `has_canonical_format` implies `has_sorted_indices`, so if the latter
        flag is `False`, so will the former be; if the former is found `True`,
        the latter flag is also set.
        """
        # first check to see if result was cached
        if self._has_sorted_indices is False:
            # not sorted => not canonical
            self._has_canonical_format = False
        elif self._has_canonical_format is None:
            self._has_canonical_format = bool(
                csr_has_canonical_format(
                    len(self.indptr) - 1, self.indptr, self.indices,
                    self._num_threads))
            if self._has_canonical_format:
                self._has_sorted_indices = True
        return self._has_canonical_format

    @has_canonical_format.setter
    def has_canonical_format(self, val: bool):
        self._has_canonical_format = bool(val)
        if val:
            self._has_sorted_indices = True

    @property
    def has_sorted_indices(self) -> bool:
        """Whether the indices are sorted

        Returns
            - True: if the indices of the array/matrix are in sorted order
            - False: otherwise
        """
        # first check to see if result was cached
        if self._has_sorted_indices is None:
            self._has_sorted_indices = bool(
                csr_has_sorted_indices(
                    len(self.indptr) - 1, self.indptr, self.indices,
                    self._num_threads))
        return self._has_sorted_indices

    @has_sorted_indices.setter
    def has_sorted_indices(self, val: bool):
        self._has_sorted_indices = bool(val)

    def sort_indices(self):
        """Sort the indices of this array/matrix *in place*
        """
        if not self.has_sorted_indices:
            self._has_canonical_format = csr_sort_indices(
                len(self.indptr) - 1, self.indptr, self.indices,
                bitview(self.data), self._num_threads)
            self._has_sorted_indices = True

    def astype(self, dtype, casting='unsafe', copy=True):
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self.copy() if copy else self
        old_data = self.data
        old_dtype = old_data.dtype
        if casting == 'unsafe' and old_dtype in X_DTYPES and \
                dtype in X_DTYPES:
            # Fast path
            new_data = np.empty_like(old_data, dtype=dtype)
            cast(old_data, new_data, self._num_threads)
            result = self._with_data(new_data, copy=copy)
        else:
            # Slow path
            result = self._with_data(
                old_data.astype(dtype, casting=casting, copy=True),
                copy=copy)
        result._num_threads = self._num_threads
        result._has_sorted_indices = self._has_sorted_indices
        result._has_canonical_format = self._has_canonical_format
        return result

    def copy(self):
        result = self._with_data(self.data.copy(), copy=True)
        result._num_threads = self._num_threads
        result._has_sorted_indices = self._has_sorted_indices
        result._has_canonical_format = self._has_canonical_format
        return result


class csr_array(cs_matrix, sparse.csr_array):
    def tocsc(self, copy=False):
        M, N = self.shape
        index_dtype = self.indices.dtype
        indptr = np.empty(N + 1, dtype=index_dtype)
        indices = np.empty(self.nnz, dtype=index_dtype)
        data = np.empty(self.nnz, dtype=self.dtype)

        csr_tocsc(M, N, self.indptr, self.indices, bitview(self.data), indptr,
                  indices, bitview(data), self._num_threads)

        result = csc_array((data, indices, indptr), shape=self.shape)
        result._num_threads = self._num_threads
        # Conversion visits rows in order, so result indices are always sorted.
        # Thus, the result is canonical iff the input had no duplicate
        # (row, col) pairs; if the input is canonical that's guaranteed,
        # otherwise unknown.
        result._has_sorted_indices = True
        result._has_canonical_format = \
            True if self._has_canonical_format else self._has_canonical_format
        return result

    def equals(self, other: Any) -> bool:
        """
        Determine whether this `csr_array` equals another object.

        Args:
            other: another object

        Returns:
            `True` if `other` is a `csr_array` and has `data`, `indices` and
            `indptr` all equal to this one's, `False` otherwise. NaNs will
            always compare equal.
        """
        return isinstance(other, csr_array) and sparse_equal(self, other)

    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        if self.ndim == 1:
            if i not in (0, -1):
                error_message = f'index ({i}) out of range'
                raise IndexError(error_message)
            return self.reshape((1, self.shape[0]), copy=True)

        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            error_message = f'index ({i}) out of range'
            raise IndexError(error_message)
        indptr, indices, data, has_canonical_format, has_sorted_indices = \
            get_csr_submatrix(M, N, self.indptr, self.indices, self.data, i,
                              i + 1, 0, N, self._num_threads,
                              self._has_canonical_format,
                              self._has_sorted_indices)
        # Propagate non-canonicalness discovered on this row
        if self._has_canonical_format is None:
            if has_canonical_format is False:
                self._has_canonical_format = False
            if has_sorted_indices is False:
                self._has_sorted_indices = False
        result = self.__class__((data, indices, indptr), shape=(1, N),
                                dtype=self.dtype)
        result._num_threads = self._num_threads
        result._has_sorted_indices = has_sorted_indices
        result._has_canonical_format = has_canonical_format
        return result

    def _getcol(self, i):
        """Returns a copy of column i. A (m x 1) sparse array (column vector).
        """
        if self.ndim == 1:
            error_message = \
                'getcol not provided for 1d arrays. Use indexing A[j]'
            raise ValueError(error_message)
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            error_message = f'index ({i}) out of range'
            raise IndexError(error_message)
        indptr, indices, data, has_canonical_format, has_sorted_indices = \
            get_csr_submatrix(M, N, self.indptr, self.indices, self.data, 0, M,
                              i, i + 1, self._num_threads,
                              self._has_canonical_format,
                              self._has_sorted_indices)
        # Full row scan - propagate to parent unconditionally
        if self._has_canonical_format is None:
            self._has_canonical_format = has_canonical_format
            self._has_sorted_indices = has_sorted_indices
        result = self.__class__((data, indices, indptr), shape=(M, 1),
                                dtype=self.dtype)
        result._num_threads = self._num_threads
        result._has_sorted_indices = has_sorted_indices
        result._has_canonical_format = has_canonical_format
        return result

    def _get_int(self, idx):
        spot = np.flatnonzero(self.indices == idx)
        if spot.size:
            return self.data[spot[0]]
        return self.data.dtype.type(0)

    def _get_slice(self, idx):
        if idx == slice(None):
            return self.copy()
        if idx.step in (1, None):
            ret = self._get_submatrix(0, idx, copy=True)
            return ret.reshape(ret.shape[-1])
        return self._minor_slice(idx)

    def _get_intXarray(self, row, col):
        return self._getrow(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(row, col, copy=True)

        M, N = self.shape
        start, stop, stride = col.indices(N)

        ii, jj = self.indptr[row:row+2]
        row_indices = self.indices[ii:jj]
        row_data = self.data[ii:jj]

        if stride > 0:
            ind = (row_indices >= start) & (row_indices < stop)
        else:
            ind = (row_indices <= start) & (row_indices > stop)

        if abs(stride) > 1:
            ind &= (row_indices - start) % stride == 0

        row_indices = (row_indices[ind] - start) // stride
        row_data = row_data[ind]
        row_indptr = np.array([0, len(row_indices)])

        if stride < 0:
            row_data = row_data[::-1]
            row_indices = abs(row_indices[::-1])

        shape = (1, max(0, int(np.ceil(float(stop - start) / stride))))
        result = self.__class__((row_data, row_indices, row_indptr),
                                shape=shape, dtype=self.dtype)
        result._num_threads = self._num_threads
        result._has_sorted_indices = self._has_sorted_indices
        result._has_canonical_format = self._has_canonical_format
        return result

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        return self._major_slice(row)._get_submatrix(minor=col)

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        return self._major_index_fancy(row)._get_submatrix(minor=col)

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            col = np.arange(*col.indices(self.shape[1]))
            return self._get_arrayXarray(row, col)
        return self._major_index_fancy(row)._get_submatrix(minor=col)


class csc_array(cs_matrix, sparse.csc_array):
    def tocsr(self, copy=False):
        M, N = self.shape
        index_dtype = self.indices.dtype
        indptr = np.empty(M + 1, dtype=index_dtype)
        indices = np.empty(self.nnz, dtype=index_dtype)
        data = np.empty(self.nnz, dtype=self.dtype)
        # csc_tocsr(M, N, ...) is equivalent to csr_tocsc(N, M, ...)
        csr_tocsc(N, M, self.indptr, self.indices, bitview(self.data), indptr,
                  indices, bitview(data), self._num_threads)
        result = csr_array((data, indices, indptr), shape=self.shape)
        result._num_threads = self._num_threads
        result._has_sorted_indices = True
        result._has_canonical_format = \
            True if self._has_canonical_format else self._has_canonical_format
        return result

    def equals(self, other: Any) -> bool:
        """
        Determine whether this `csc_array` equals another object.

        Args:
            other: another object

        Returns:
            `True` if `other` is a `csc_array` and has `data`, `indices` and
            `indptr` all equal to this one's, `False` otherwise. NaNs will
            always compare equal.
        """
        return isinstance(other, csc_array) and sparse_equal(self, other)

    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            error_message = f'index ({i}) out of range'
            raise IndexError(error_message)
        return self._get_submatrix(minor=i).tocsr()

    def _getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSC matrix (column vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            error_message = f'index ({i}) out of range'
            raise IndexError(error_message)
        return self._get_submatrix(major=i, copy=True)

    def _get_intXarray(self, row, col):
        return self._major_index_fancy(col)._get_submatrix(minor=row)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._major_slice(col)._get_submatrix(minor=row)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._get_submatrix(major=col)._minor_slice(row)

    def _get_sliceXarray(self, row, col):
        return self._major_index_fancy(col)._minor_slice(row)

    def _get_arrayXint(self, row, col):
        return self._get_submatrix(major=col)._minor_index_fancy(row)

    def _get_arrayXslice(self, row, col):
        return self._major_slice(col)._minor_index_fancy(row)