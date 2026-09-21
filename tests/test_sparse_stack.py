"""sparse_minor_stack() / sparse_major_stack() with mixed input dtypes.

`indices`/`indptr` are int64 only when an array needs them, so stacking a small
array with a large one -- or a subset of a >2^31 non-zero array, which keeps its
parent's index dtype, with a freshly built one -- mixes int32 and int64.
`csr_hstack()` reinterpret-casts each input's raw buffer to the fused types
resolved from the OUTPUT arrays, so a mismatched input is read at the wrong
stride: out-of-bounds reads and writes, not a wrong answer or a slow path.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from brisc.sparse import csc_array, csr_array
from brisc.utils import sparse_major_stack, sparse_minor_stack


def _build(cls, rows, cols, index_dtype, data_dtype, seed):
    fmt = "csr" if cls is csr_array else "csc"
    ref = sp.random(rows, cols, density=0.3, format=fmt,
                    dtype=data_dtype, random_state=seed)
    out = cls((ref.data,
               ref.indices.astype(index_dtype),
               ref.indptr.astype(index_dtype)), shape=ref.shape)
    out._num_threads = 1
    return ref, out


@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("dtypes", [(np.int32, np.int64), (np.int64, np.int32),
                                    (np.int32, np.int32), (np.int64, np.int64)],
                         ids=["32+64", "64+32", "32+32", "64+64"])
@pytest.mark.parametrize("shape", [(50, 40, 20), (1000, 500, 200)],
                         ids=["small", "large"])
def test_sparse_minor_stack_mixed_index_dtypes(dtypes, shape, num_threads):
    """Stacking CSC arrays along the minor (row) axis == scipy.sparse.vstack."""
    rows_a, rows_b, cols = shape
    ref_a, a = _build(csc_array, rows_a, cols, dtypes[0], np.float32, 0)
    ref_b, b = _build(csc_array, rows_b, cols, dtypes[1], np.float32, 1)

    got = sparse_minor_stack([a, b], num_threads=num_threads)
    want = sp.vstack([ref_a, ref_b], format="csc")

    assert got.shape == want.shape
    assert got.nnz == want.nnz
    np.testing.assert_array_equal(got.indptr, want.indptr)
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_allclose(got.data, want.data)


@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("dtypes", [(np.int32, np.int64), (np.int64, np.int32)],
                         ids=["32+64", "64+32"])
def test_sparse_major_stack_mixed_index_dtypes(dtypes, num_threads):
    """The major-axis counterpart already casts; pin the behaviour."""
    ref_a, a = _build(csr_array, 200, 50, dtypes[0], np.float32, 0)
    ref_b, b = _build(csr_array, 100, 50, dtypes[1], np.float32, 1)

    got = sparse_major_stack([a, b], num_threads=num_threads)
    want = sp.vstack([ref_a, ref_b], format="csr")

    assert got.shape == want.shape
    assert got.nnz == want.nnz
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_allclose(got.data, want.data)


@pytest.mark.parametrize("num_threads", [1, 2])
def test_sparse_minor_stack_mixed_data_dtypes(num_threads):
    """`data` is reinterpret-cast the same way `indices`/`indptr` are."""
    ref_a, a = _build(csc_array, 200, 50, np.int32, np.float32, 0)
    ref_b, b = _build(csc_array, 100, 50, np.int32, np.float64, 1)

    got = sparse_minor_stack([a, b], num_threads=num_threads)
    want = sp.vstack([ref_a, ref_b.astype(np.float32)], format="csc")

    assert got.nnz == want.nnz
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_allclose(got.data, want.data)
