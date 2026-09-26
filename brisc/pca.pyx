# Functionality for principal component analysis

from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libcpp.algorithm cimport fill
from libcpp.cmath cimport abs, sqrt
from libcpp.vector cimport vector
from scipy.linalg.cython_lapack cimport sgesdd
from .cyutils cimport get_thread_offsets, norm, random_normal, sgemm, sgemv, \
    signed_integer, srand, uninitialized_vector


cdef inline void clipped_stddev_csr(const float[::1] data,
                                    const signed_integer[::1] indices,
                                    const signed_integer[::1] indptr,
                                    const unsigned long long num_cells,
                                    const unsigned num_genes,
                                    float clip_val,
                                    float[::1] clipped_stddev):
    # Compute `X.std(axis=0)` where `X` is CSR, clipping to a minimum of
    # `clip_val`. Only used when `num_threads=1` and `match_parallel=False`.

    cdef unsigned long long num_elements, i
    cdef unsigned gene
    cdef double value, \
        inv_num_pairs_of_cells = 1.0 / (num_cells * (num_cells - 1))
    cdef vector[double] sum_buffer, sum_of_squares_buffer
    sum_buffer.resize(num_genes)
    sum_of_squares_buffer.resize(num_genes)
    cdef double[::1] sum = <double[:num_genes]> sum_buffer.data(), \
        sum_of_squares = <double[:num_genes]> sum_of_squares_buffer.data()

    # Iterate over all elements of the count matrix, ignoring which cell
    # they're from
    num_elements = indices.shape[0]
    for i in range(num_elements):
        gene = indices[i]
        value = <double> data[i]
        sum[gene] += value
        sum_of_squares[gene] += value * value

    # Calculate standard deviations from the sums and squared sums
    for gene in range(num_genes):
        clipped_stddev[gene] = <float> sqrt(inv_num_pairs_of_cells * (
            num_cells * sum_of_squares[gene] - sum[gene] * sum[gene]))
        if clipped_stddev[gene] < clip_val:
            clipped_stddev[gene] = clip_val


cdef inline void clipped_stddev_csc(const float[::1] data,
                                    const signed_integer[::1] indices,
                                    const signed_integer[::1] indptr,
                                    const unsigned long long num_cells,
                                    const unsigned num_genes,
                                    float clip_val,
                                    float[::1] clipped_stddev,
                                    unsigned num_threads,
                                    const unsigned* thread_offsets) \
        noexcept nogil:
    # Compute `X.std(axis=0)` where `X` is CSC, clipping to a minimum of
    # `clip_val`.

    cdef unsigned gene, thread_index, start_col, end_col
    cdef unsigned long long i
    cdef double value, sum, sum_of_squares, \
        inv_num_pairs_of_cells = 1.0 / (num_cells * (num_cells - 1))

    num_threads = min(num_threads, num_genes)
    if num_threads <= 1:
        for gene in range(num_genes):
            # Calculate the sum and squared sum for this gene, across cells
            # with non-zero counts for the gene
            sum = 0
            sum_of_squares = 0
            for i in range(<unsigned long long> indptr[gene],
                           <unsigned long long> indptr[gene + 1]):
                value = <double> data[i]
                sum += value
                sum_of_squares += value * value

            # Calculate the scaled variance from the sum and squared sum
            clipped_stddev[gene] = <float> sqrt(inv_num_pairs_of_cells * (
                num_cells * sum_of_squares - sum * sum))
            if clipped_stddev[gene] < clip_val:
                clipped_stddev[gene] = clip_val
    else:
        with parallel(num_threads=num_threads):
            thread_index = threadid()
            start_col = thread_offsets[thread_index]
            end_col = thread_offsets[thread_index + 1]
            for gene in range(start_col, end_col):
                sum = 0
                sum_of_squares = 0
                for i in range(<unsigned long long> indptr[gene],
                               <unsigned long long> indptr[gene + 1]):
                    value = <double> data[i]
                    sum = sum + value
                    sum_of_squares = sum_of_squares + value * value
                clipped_stddev[gene] = <float> sqrt(inv_num_pairs_of_cells * (
                    num_cells * sum_of_squares - sum * sum))
                if clipped_stddev[gene] < clip_val:
                    clipped_stddev[gene] = clip_val


cdef void matvec_csr_fast(const float[::1] data,
                          const signed_integer[::1] indices,
                          const signed_integer[::1] indptr,
                          const float[::1] V,
                          const float[::1] clipped_stddev,
                          float* num_cells_buffer,
                          float* num_genes_buffer,
                          const unsigned long long num_cells,
                          const unsigned num_genes,
                          const unsigned num_threads,
                          const unsigned* thread_offsets,
                          const unsigned long long chunk_size,
                          float* block_sums) noexcept nogil:
    # Compute `num_cells_buffer = scale(X) @ V`, where `X` is CSR and `V`
    # and `num_cells_buffer` are vectors. Also has a parallel version,
    # `matvec_csr_parallel()`. `num_threads`, `thread_offsets`, `chunk_size`,
    # and `block_sums` are unused, but needed to match the signature of
    # `matvec_csr_parallel()`.

    cdef unsigned i
    cdef unsigned long long j
    cdef float dot_product, mean = 0

    # Variance scaling
    for i in range(num_genes):
        num_genes_buffer[i] = V[i] / clipped_stddev[i]

    # Matrix-vector multiplication and mean calculation
    for i in range(num_cells):
        dot_product = 0
        for j in range(<unsigned long long> indptr[i],
                       <unsigned long long> indptr[i + 1]):
            dot_product += data[j] * num_genes_buffer[indices[j]]
        num_cells_buffer[i] = dot_product
        mean += dot_product
    mean /= num_cells

    # Mean scaling
    for i in range(num_cells):
        num_cells_buffer[i] -= mean


cdef void matvec_csr(const float[::1] data,
                     const signed_integer[::1] indices,
                     const signed_integer[::1] indptr,
                     const float[::1] V,
                     const float[::1] clipped_stddev,
                     float* num_cells_buffer,
                     float* num_genes_buffer,
                     const unsigned long long num_cells,
                     const unsigned num_genes,
                     const unsigned num_threads,
                     const unsigned* thread_offsets,
                     const unsigned long long chunk_size,
                     float* block_sums) noexcept nogil:
    # Compute `num_cells_buffer = scale(X) @ V`, where `X` is CSR and `V`
    # and `num_cells_buffer` are vectors. Also has a parallel version,
    # `matvec_csr_parallel()`. `num_threads`, `thread_offsets`, `chunk_size`,
    # and `block_sums` are unused, but needed to match the signature of
    # `matvec_csr_parallel()`.

    cdef unsigned i
    cdef unsigned long long j
    cdef float dot_product, mean = 0

    # Variance scaling
    for i in range(num_genes):
        num_genes_buffer[i] = V[i] / clipped_stddev[i]

    # Matrix-vector multiplication
    for i in range(num_cells):
        dot_product = 0
        for j in range(<unsigned long long> indptr[i],
                       <unsigned long long> indptr[i + 1]):
            dot_product += data[j] * num_genes_buffer[indices[j]]
        num_cells_buffer[i] = dot_product

    # Mean calculation; unlike `matvec_csr_fast()`, this is a separate
    # loop to maintain the parallel version's floating-point behavior.
    for i in range(num_cells):
        mean += num_cells_buffer[i]
    mean /= num_cells

    # Mean scaling
    for i in range(num_cells):
        num_cells_buffer[i] -= mean


cdef void matvec_csr_parallel(const float[::1] data,
                              const signed_integer[::1] indices,
                              const signed_integer[::1] indptr,
                              const float[::1] V,
                              const float[::1] clipped_stddev,
                              float* num_cells_buffer,
                              float* num_genes_buffer,
                              const unsigned long long num_cells,
                              const unsigned num_genes,
                              const unsigned num_threads,
                              const unsigned* thread_offsets,
                              const unsigned long long chunk_size,
                              float* block_sums) noexcept nogil:
    # Compute `num_cells_buffer = scale(X) @ V`, where `X` is CSR and `V`
    # and `num_cells_buffer` are vectors. Also has a single-threaded
    # version, `matvec_csr()`.

    cdef unsigned i, thread_index, start_row, end_row
    cdef unsigned long long j, block, block_start, block_end, \
        num_blocks = (num_cells + chunk_size - 1) / chunk_size
    cdef float dot_product, block_partial, mean = 0

    # Variance scaling (single-threaded since `num_genes` is just 2000
    # by default)
    for i in range(num_genes):
        num_genes_buffer[i] = V[i] / clipped_stddev[i]

    # Matrix-vector multiplication
    with parallel(num_threads=num_threads):
        thread_index = threadid()
        start_row = thread_offsets[thread_index]
        end_row = thread_offsets[thread_index + 1]
        for i in range(start_row, end_row):
            dot_product = 0
            for j in range(<unsigned long long> indptr[i],
                           <unsigned long long> indptr[i + 1]):
                dot_product = \
                    dot_product + data[j] * num_genes_buffer[indices[j]]
            num_cells_buffer[i] = dot_product

    # Mean calculation via fixed-block deterministic parallel reduction;
    # block boundaries are independent of thread count
    for block in prange(num_blocks, num_threads=num_threads):
        block_start = block * chunk_size
        block_end = min(block_start + chunk_size, num_cells)
        block_partial = 0
        for j in range(block_start, block_end):
            block_partial = block_partial + num_cells_buffer[j]
        block_sums[block] = block_partial
    for block in range(num_blocks):
        mean += block_sums[block]
    mean /= num_cells

    # Mean scaling
    for i in prange(num_cells, num_threads=num_threads):
        num_cells_buffer[i] -= mean


cdef void matvec_csc(const float[::1] data,
                     const signed_integer[::1] indices,
                     const signed_integer[::1] indptr,
                     const float[::1] V,
                     const float[::1] clipped_stddev,
                     float* num_cells_buffer,
                     float* num_genes_buffer,
                     const unsigned long long num_cells,
                     const unsigned num_genes,
                     const unsigned num_threads,
                     const unsigned* thread_offsets,
                     const unsigned long long chunk_size,
                     float* block_sums) noexcept nogil:
    # Compute `num_cells_buffer = scale(X) @ V`, where `X` is CSC and `V`
    # and `num_cells_buffer` are vectors. Does not have a parallel version.
    # `num_threads`, `thread_offsets`, `chunk_size`, and `block_sums` are
    # unused, but needed to match the signature of `matvec_csr_parallel()`.

    cdef unsigned i
    cdef unsigned long long j
    cdef float num_genes_buffer_i, mean = 0

    # Variance scaling
    for i in range(num_genes):
        num_genes_buffer[i] = V[i] / clipped_stddev[i]

    # Matrix-vector multiplication
    fill(num_cells_buffer, num_cells_buffer + num_cells, 0)
    for i in range(num_genes):
        num_genes_buffer_i = num_genes_buffer[i]
        for j in range(<unsigned long long> indptr[i],
                       <unsigned long long> indptr[i + 1]):
            num_cells_buffer[indices[j]] += data[j] * num_genes_buffer_i

    # Mean calculation
    for i in range(num_cells):
        mean += num_cells_buffer[i]
    mean /= num_cells

    # Mean scaling
    for i in range(num_cells):
        num_cells_buffer[i] -= mean


cdef void rmatvec_csr(const float[::1] data,
                      const signed_integer[::1] indices,
                      const signed_integer[::1] indptr,
                      const float[::1] V,
                      const float[::1] clipped_stddev,
                      float* num_cells_buffer,
                      float* num_genes_buffer,
                      const unsigned long long num_cells,
                      const unsigned num_genes,
                      const unsigned num_threads,
                      const unsigned* thread_offsets,
                      const unsigned long long chunk_size,
                      float* block_sums) noexcept nogil:
    # Compute `num_cells_buffer = scale(X).T @ V`, where `X` is CSR and `V`
    # and `num_cells_buffer` are vectors. Does not have a parallel version.
    # `num_threads`, `thread_offsets`, `chunk_size`, and `block_sums` are
    # unused, but needed to match the signature of `rmatvec_csc_parallel()`.

    cdef unsigned i
    cdef unsigned long long j
    cdef float num_cells_buffer_i, mean = 0

    # Mean scaling
    for i in range(num_cells):
        mean += V[i]
    mean /= num_cells
    for i in range(num_cells):
        num_cells_buffer[i] = V[i] - mean

    # Matrix-vector multiplication
    fill(num_genes_buffer, num_genes_buffer + num_genes, 0)
    for i in range(num_cells):
        num_cells_buffer_i = num_cells_buffer[i]
        for j in range(<unsigned long long> indptr[i],
                       <unsigned long long> indptr[i + 1]):
            num_genes_buffer[indices[j]] += data[j] * num_cells_buffer_i

    # Variance scaling
    for i in range(num_genes):
        num_genes_buffer[i] /= clipped_stddev[i]


cdef void rmatvec_csc(const float[::1] data,
                      const signed_integer[::1] indices,
                      const signed_integer[::1] indptr,
                      const float[::1] V,
                      const float[::1] clipped_stddev,
                      float* num_cells_buffer,
                      float* num_genes_buffer,
                      const unsigned long long num_cells,
                      const unsigned num_genes,
                      const unsigned num_threads,
                      const unsigned* thread_offsets,
                      const unsigned long long chunk_size,
                      float* block_sums) noexcept nogil:
    # Compute `num_cells_buffer = scale(X).T @ V`, where `X` is CSC and `V`
    # and `num_cells_buffer` are vectors. Also has a parallel version,
    # `rmatvec_csc_parallel()`. `num_threads`, `thread_offsets`, `chunk_size`,
    # and `block_sums` are unused, but needed to match the signature of
    # `rmatvec_csc_parallel()`.

    cdef unsigned i
    cdef unsigned long long j
    cdef float dot_product, mean = 0

    # Mean calculation
    for i in range(num_cells):
        mean += V[i]
    mean /= num_cells

    # Mean scaling
    for i in range(num_cells):
        num_cells_buffer[i] = V[i] - mean

    # Matrix-vector multiplication
    for i in range(num_genes):
        dot_product = 0
        for j in range(<unsigned long long> indptr[i],
                       <unsigned long long> indptr[i + 1]):
            dot_product += data[j] * num_cells_buffer[indices[j]]
        num_genes_buffer[i] = dot_product

    # Variance scaling
    for i in range(num_genes):
        num_genes_buffer[i] /= clipped_stddev[i]


cdef void rmatvec_csc_parallel(const float[::1] data,
                               const signed_integer[::1] indices,
                               const signed_integer[::1] indptr,
                               const float[::1] V,
                               const float[::1] clipped_stddev,
                               float* num_cells_buffer,
                               float* num_genes_buffer,
                               const unsigned long long num_cells,
                               const unsigned num_genes,
                               const unsigned num_threads,
                               const unsigned* thread_offsets,
                               const unsigned long long chunk_size,
                               float* block_sums) noexcept nogil:
    # Compute `num_cells_buffer = scale(X).T @ V`, where `X` is CSC and `V`
    # and `num_cells_buffer` are vectors. Also has a single-threaded
    # version, `rmatvec_csc()`.

    cdef unsigned i, thread_index, start_col, end_col
    cdef unsigned long long j, block, block_start, block_end, \
        num_blocks = (num_cells + chunk_size - 1) / chunk_size
    cdef float dot_product, block_partial, mean = 0

    # Mean calculation via fixed-block deterministic parallel reduction
    for block in prange(num_blocks, num_threads=num_threads):
        block_start = block * chunk_size
        block_end = min(block_start + chunk_size, num_cells)
        block_partial = 0
        for j in range(block_start, block_end):
            block_partial = block_partial + V[j]
        block_sums[block] = block_partial
    for block in range(num_blocks):
        mean += block_sums[block]
    mean /= num_cells

    # Mean scaling
    for i in prange(num_cells, num_threads=num_threads):
        num_cells_buffer[i] = V[i] - mean

    # Matrix-vector multiplication
    with parallel(num_threads=num_threads):
        thread_index = threadid()
        start_col = thread_offsets[thread_index]
        end_col = thread_offsets[thread_index + 1]
        for i in range(start_col, end_col):
            dot_product = 0
            for j in range(<unsigned long long> indptr[i],
                           <unsigned long long> indptr[i + 1]):
                dot_product = \
                    dot_product + data[j] * num_cells_buffer[indices[j]]
            num_genes_buffer[i] = dot_product

    # Variance scaling (single-threaded since `num_genes` is just 2000
    # by default)
    for i in range(num_genes):
        num_genes_buffer[i] /= clipped_stddev[i]


cdef void matvec_csr_parallel_raw(
        const float[::1] data,
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const float[::1] V,
        const float[::1] clipped_stddev,
        float* num_cells_buffer,
        float* num_genes_buffer,
        const unsigned long long num_cells,
        const unsigned num_genes,
        const unsigned num_threads,
        const unsigned* thread_offsets) noexcept nogil:
    # Compute the sparse matrix-vector product
    # `num_cells_buffer = X @ (V / stddev)` without mean-centering. The caller
    # handles mean-centering subsequently.

    cdef unsigned i, thread_index, start_row, end_row
    cdef unsigned long long j
    cdef float dot_product

    # Variance scaling
    for i in range(num_genes):
        num_genes_buffer[i] = V[i] / clipped_stddev[i]

    # Matrix-vector multiplication
    with parallel(num_threads=num_threads):
        thread_index = threadid()
        start_row = thread_offsets[thread_index]
        end_row = thread_offsets[thread_index + 1]
        for i in range(start_row, end_row):
            dot_product = 0
            for j in range(<unsigned long long> indptr[i],
                           <unsigned long long> indptr[i + 1]):
                dot_product = \
                    dot_product + data[j] * num_genes_buffer[indices[j]]
            num_cells_buffer[i] = dot_product


cdef void rmatvec_csc_parallel_raw(
        const float[::1] data,
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const float[::1] V,
        const float[::1] clipped_stddev,
        float* num_genes_buffer,
        const unsigned long long num_cells,
        const unsigned num_genes,
        const unsigned num_threads,
        const unsigned* thread_offsets) noexcept nogil:
    # Compute `num_genes_buffer = (X.T @ V) / stddev` where `V` is already
    # zero-mean, skipping mean-centering and the copy into a temporary buffer.

    cdef unsigned i, thread_index, start_col, end_col
    cdef unsigned long long j
    cdef float dot_product

    # Matrix-vector multiplication, reading `V` directly
    with parallel(num_threads=num_threads):
        thread_index = threadid()
        start_col = thread_offsets[thread_index]
        end_col = thread_offsets[thread_index + 1]
        for i in range(start_col, end_col):
            dot_product = 0
            for j in range(<unsigned long long> indptr[i],
                           <unsigned long long> indptr[i + 1]):
                dot_product = \
                    dot_product + data[j] * V[indices[j]]
            num_genes_buffer[i] = dot_product

    # Variance scaling
    for i in range(num_genes):
        num_genes_buffer[i] /= clipped_stddev[i]


cdef inline int svd_workspace_size(float[::1, :] B,
                                   float[::1, :] U,
                                   float[::1] S,
                                   float[::1, :] Vt,
                                   int* iwork) noexcept nogil:
    # Get the optimal workspace size for SVD with `sgesdd()`. Assumes `m < n`,
    # which it is for `B`.

    cdef char jobz = b'S'
    cdef int info, m = B.shape[0], n = B.shape[1], lwork = -1
    cdef float workspace_size

    sgesdd(&jobz, &m, &n, &B[0, 0], &m, &S[0], &U[0, 0], &m, &Vt[0, 0], &m,
           &workspace_size, &lwork, iwork, &info)
    return <int> workspace_size


cdef inline void svd(float[::1, :] B,
                     float[::1, :] U,
                     float[::1] S,
                     float[::1, :] Vt,
                     float* work,
                     int* iwork,
                     const int lwork) noexcept nogil:
    # SVD for the case where `m < n`, which it is for `B`. Uses the optimal
    # workspace size computed by `svd_workspace_size()`.

    cdef char jobz = b'S'
    cdef int info, m = B.shape[0], n = B.shape[1]

    sgesdd(&jobz, &m, &n, &B[0, 0], &m, &S[0], &U[0, 0], &m, &Vt[0, 0], &m,
           work, <int*> &lwork, iwork, &info)


ctypedef void (*matvec_function)(const float[::1] data,
                                 const signed_integer[::1] indices,
                                 const signed_integer[::1] indptr,
                                 const float[::1] V,
                                 const float[::1] clipped_stddev,
                                 float* num_cells_buffer,
                                 float* num_genes_buffer,
                                 const unsigned long long num_cells,
                                 const unsigned num_genes,
                                 const unsigned num_threads,
                                 const unsigned* thread_offsets,
                                 const unsigned long long chunk_size,
                                 float* block_sums) noexcept nogil


ctypedef void (*rmatvec_function)(const float[::1] data,
                                  const signed_integer[::1] indices,
                                  const signed_integer[::1] indptr,
                                  const float[::1] V,
                                  const float[::1] clipped_stddev,
                                  float* num_cells_buffer,
                                  float* num_genes_buffer,
                                  const unsigned long long num_cells,
                                  const unsigned num_genes,
                                  const unsigned num_threads,
                                  const unsigned* thread_offsets,
                                  const unsigned long long chunk_size,
                                  float* block_sums) noexcept nogil


cdef inline void matmul_nn_f_out(const int m, const int n, const int k,
                                 const float[::1, :] A,
                                 const float[::1, :] B,
                                 float[::1, :] C,
                                 const unsigned num_threads) noexcept nogil:
    # Computes `C = A @ B` where all matrices are Fortran-contiguous

    cdef int i, j, row
    cdef float dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7

    for row in prange(m, num_threads=num_threads):
        i = 0
        while i < (n & ~7):
            dp0 = 0; dp1 = 0; dp2 = 0; dp3 = 0
            dp4 = 0; dp5 = 0; dp6 = 0; dp7 = 0
            for j in range(k):
                dp0 = dp0 + A[row, j] * B[j, i + 0]
                dp1 = dp1 + A[row, j] * B[j, i + 1]
                dp2 = dp2 + A[row, j] * B[j, i + 2]
                dp3 = dp3 + A[row, j] * B[j, i + 3]
                dp4 = dp4 + A[row, j] * B[j, i + 4]
                dp5 = dp5 + A[row, j] * B[j, i + 5]
                dp6 = dp6 + A[row, j] * B[j, i + 6]
                dp7 = dp7 + A[row, j] * B[j, i + 7]
            C[row, i + 0] = dp0
            C[row, i + 1] = dp1
            C[row, i + 2] = dp2
            C[row, i + 3] = dp3
            C[row, i + 4] = dp4
            C[row, i + 5] = dp5
            C[row, i + 6] = dp6
            C[row, i + 7] = dp7
            i = i + 8
        while i < n:
            dp0 = 0
            for j in range(k):
                dp0 = dp0 + A[row, j] * B[j, i]
            C[row, i] = dp0
            i = i + 1


cdef inline void matmul_nn_c_out(const int m, const int n, const int k,
                                 const float[::1, :] A,
                                 const float[::1, :] B,
                                 float[:, ::1] C,
                                 const unsigned num_threads) noexcept nogil:
    # Computes `C = A @ B` where A and B are Fortran-contiguous and C
    # is C-contiguous

    cdef int i, j, row
    cdef float dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7

    for row in prange(m, num_threads=num_threads):
        i = 0
        while i < (n & ~7):
            dp0 = 0; dp1 = 0; dp2 = 0; dp3 = 0
            dp4 = 0; dp5 = 0; dp6 = 0; dp7 = 0
            for j in range(k):
                dp0 = dp0 + A[row, j] * B[j, i + 0]
                dp1 = dp1 + A[row, j] * B[j, i + 1]
                dp2 = dp2 + A[row, j] * B[j, i + 2]
                dp3 = dp3 + A[row, j] * B[j, i + 3]
                dp4 = dp4 + A[row, j] * B[j, i + 4]
                dp5 = dp5 + A[row, j] * B[j, i + 5]
                dp6 = dp6 + A[row, j] * B[j, i + 6]
                dp7 = dp7 + A[row, j] * B[j, i + 7]
            C[row, i + 0] = dp0
            C[row, i + 1] = dp1
            C[row, i + 2] = dp2
            C[row, i + 3] = dp3
            C[row, i + 4] = dp4
            C[row, i + 5] = dp5
            C[row, i + 6] = dp6
            C[row, i + 7] = dp7
            i = i + 8
        while i < n:
            dp0 = 0
            for j in range(k):
                dp0 = dp0 + A[row, j] * B[j, i]
            C[row, i] = dp0
            i = i + 1


def irlba(const float[::1] data_matvec,
          const signed_integer[::1] indices_matvec,
          const signed_integer[::1] indptr_matvec,
          const float[::1] data_rmatvec,
          const signed_integer[::1] indices_rmatvec,
          const signed_integer[::1] indptr_rmatvec,
          const unsigned long long num_cells,
          const unsigned num_genes,
          const unsigned k,
          const unsigned subspace_size,
          const float tolerance,
          const unsigned max_iterations,
          const unsigned long long seed,
          const bint match_parallel,
          const bint is_csr,
          const unsigned num_threads,
          const unsigned long long chunk_size,
          float[:, ::1] PCs):

    cdef unsigned num_matvec_threads, num_rmatvec_threads, iteration, i, j, \
        row, col, num_converged, k_current = 0, \
        num_threads_or_cells = min(num_threads, num_cells)
    cdef unsigned long long state = srand(seed), block, block_start, \
        block_end, num_blocks_cells = (num_cells + chunk_size - 1) / chunk_size
    cdef float inverse_norm, dot_product, alpha, inverse_alpha, beta, \
        inverse_beta, remainder_norm, residual, Sbj, block_partial, \
        squared_norm, mean, clip_val = <float> 1e-8, \
        Smax = <float> 2.4221817809573368e-05  # (float32 eps) ** (2 / 3)
    cdef int lwork
    cdef bint converged = False
    cdef uninitialized_vector[float] clipped_stddev_buffer, U_buffer, \
        U_new_buffer, V_buffer, V_new_buffer, B_buffer, Ub_buffer, Sb_buffer, \
        Vbt_buffer, remainder_buffer, work, block_sums_buffer, \
        partial_work_buffer
    cdef uninitialized_vector[int] iwork
    cdef uninitialized_vector[unsigned] matvec_thread_offsets_buffer, \
        rmatvec_thread_offsets_buffer
    cdef float[::1, :] U, U_new, V, V_new, B, Ub, Vbt, temp
    cdef float[::1] clipped_stddev, work2, Sb
    cdef float[:, ::1] partial_work
    cdef unsigned* matvec_thread_offsets
    cdef unsigned* rmatvec_thread_offsets
    cdef float* block_sums
    cdef matvec_function matvec
    cdef rmatvec_function rmatvec

    # Allocate the clipped standard deviation of each gene
    clipped_stddev_buffer.resize(num_genes)
    clipped_stddev = <float[:num_genes]> clipped_stddev_buffer.data()

    # Allocate `U` and `U_new`: the left singular vectors
    U_buffer.resize(num_cells * subspace_size)
    U = <float[:num_cells:1, :subspace_size]> U_buffer.data()
    U_new_buffer.resize(num_cells * subspace_size)
    U_new = <float[:num_cells:1, :subspace_size]> U_new_buffer.data()

    # Allocate `V` and `V_new`: the right singular vectors
    V_buffer.resize(num_genes * subspace_size)
    V = <float[:num_genes:1, :subspace_size]> V_buffer.data()
    V_new_buffer.resize(num_genes * subspace_size)
    V_new = <float[:num_genes:1, :subspace_size]> V_new_buffer.data()

    # Allocate the bidiagonal matrix (`B`) and its left singular vectors
    # (`Ub`), singular values (`Sb`), and transposed right singular vectors
    # (`Vbt`)
    B_buffer.resize(subspace_size * (subspace_size + 1))
    B = <float[:subspace_size:1, :subspace_size + 1]> B_buffer.data()
    Ub_buffer.resize(subspace_size * subspace_size)
    Ub = <float[:subspace_size:1, :subspace_size]> Ub_buffer.data()
    Sb_buffer.resize(subspace_size)
    Sb = <float[:subspace_size]> Sb_buffer.data()
    Vbt_buffer.resize(subspace_size * (subspace_size + 1))
    Vbt = <float[:subspace_size:1, :subspace_size + 1]> Vbt_buffer.data()

    # Allocate the remainder from the Lanczos bidiagonalization (`remainder`)
    remainder_buffer.resize(num_genes)
    cdef float[::1] remainder = <float[:num_genes]> remainder_buffer.data()

    # Allocate the temporary buffers used by `sgesdd()` to perform SVD of `B`
    # (`iwork` and `work`). The optimal size of the latter is determined
    # programmatically via `svd_workspace_size()`.
    iwork.resize(8 * subspace_size)
    lwork = svd_workspace_size(B, Ub, Sb, Vbt, iwork.data())
    work.resize(lwork)

    # Allocate buffers for fixed-block deterministic parallel reductions
    block_sums_buffer.resize(num_blocks_cells)
    block_sums = block_sums_buffer.data()
    partial_work_buffer.resize(num_blocks_cells * subspace_size)
    partial_work = <float[:num_blocks_cells, :subspace_size:1]> \
        partial_work_buffer.data()

    # Get the number of threads for the matvec and rmatvec
    if is_csr:
        num_matvec_threads = min(num_threads, num_cells)
        num_rmatvec_threads = min(num_threads, num_genes)
    else:
        num_matvec_threads = min(num_threads, num_genes)
        num_rmatvec_threads = min(num_threads, num_cells)

    # Get the row offset of the start of each thread when performing the matvec
    # and rmatvec, if running in parallel. This allows for load-balancing: each
    # thread handles about the same number of elements, even if different cells
    # express different numbers of highly variable genes. Also decide which
    # function to use for matvec and rmatvec.
    if num_threads == 1 and not match_parallel:
        if is_csr:
            matvec = matvec_csr_fast
            rmatvec = rmatvec_csr
        else:
            matvec = matvec_csc
            rmatvec = rmatvec_csc
        matvec_thread_offsets = NULL
        rmatvec_thread_offsets = NULL
    else:
        # matvec
        matvec = matvec_csr_parallel
        matvec_thread_offsets_buffer.resize(num_matvec_threads + 1)
        matvec_thread_offsets = matvec_thread_offsets_buffer.data()
        get_thread_offsets(indptr_matvec, matvec_thread_offsets,
                           num_matvec_threads)

        # rmatvec
        rmatvec = rmatvec_csc_parallel
        rmatvec_thread_offsets_buffer.resize(num_rmatvec_threads + 1)
        rmatvec_thread_offsets = rmatvec_thread_offsets_buffer.data()
        get_thread_offsets(indptr_rmatvec, rmatvec_thread_offsets,
                           num_rmatvec_threads)

    # Get the clipped standard deviation of each gene, for variance scaling;
    # use CSC for this if available, for speed

    if num_threads == 1 and is_csr and not match_parallel:
        clipped_stddev_csr(data_matvec, indices_matvec, indptr_matvec,
                           num_cells, num_genes, clip_val, clipped_stddev)
    else:
        clipped_stddev_csc(data_rmatvec, indices_rmatvec, indptr_rmatvec,
                           num_cells, num_genes, clip_val, clipped_stddev,
                           num_threads, rmatvec_thread_offsets)

    # Initialize the first column of `V` with a random normal vector
    for i in range(num_genes):
        V[i, 0] = random_normal(&state)
    inverse_norm = <float> 1 / norm(V[:, 0])
    for i in range(num_genes):
        V[i, 0] *= inverse_norm

    # Initialize `B` with zeros
    B[:] = 0

    # Now that setup is done, run the IRLBA iterations
    if num_threads == 1 and not match_parallel:
        for iteration in range(max_iterations):
            # Perform Lanczos bidiagonalization with reorthogonalization; use
            # for loops instead of `sgemv()` for matrix-vector multiplication
            # to ensure deterministic parallelism
            j = k_current
            while True:
                # `U[:, j] = X_scaled @ V[:, j]`; use `remainder` as a
                # temporary buffer (it happens to have the correct type, float,
                # and length, `num_genes`)
                matvec(data_matvec, indices_matvec, indptr_matvec, V[:, j],
                       clipped_stddev, &U[0, j], &remainder[0], num_cells,
                       num_genes, num_matvec_threads, matvec_thread_offsets,
                       chunk_size, block_sums)

                if j > 0:
                    # U[:, j] -= U[:, :j] @ (U[:, :j].T @ U[:, j])
                    # Use `work` to store the intermediate product
                    # `U[:, :j].T @ U[:, j]`, which has length `j`. This is
                    # guaranteed to fit since `sgesdd()` guarantees that `work`
                    # has at least `4 * subspace_size ** 2 + 7 * subspace_size`
                    # elements, and we only need `j`, which is less than
                    # `subspace_size`.

                    # `work = U[:, :j].T @ U[:, j]`
                    sgemv(b'T', num_cells, j, 1, &U[0, 0], num_cells, &U[0, j],
                          1, 0, work.data(), 1)

                    # `U[:, j] -= U[:, :j] @ work`
                    sgemv(b'N', num_cells, j, -1, &U[0, 0], num_cells,
                       work.data(), 1, 1, &U[0, j], 1)

                alpha = norm(U[:, j])
                if alpha < clip_val:
                    alpha = clip_val
                inverse_alpha = <float> 1 / alpha
                for i in range(num_cells):
                    U[i, j] *= inverse_alpha
                B[j, j] = alpha

                if j < subspace_size - 1:
                    # `V[:, j + 1] = X_scaled.T @ U[:, j]`; use `U_new` as a
                    # temporary buffer (it happens to have the correct type,
                    # float, and length, at least `num_cells`)
                    rmatvec(data_rmatvec, indices_rmatvec, indptr_rmatvec,
                            U[:, j], clipped_stddev, &U_new[0, 0],
                            &V[0, j + 1], num_cells, num_genes,
                            num_rmatvec_threads, rmatvec_thread_offsets,
                            chunk_size, block_sums)

                    # `V[:, j + 1] -= alpha * V[:, j]`
                    for i in range(num_genes):
                        V[i, j + 1] -= alpha * V[i, j]

                    # `V[:, j + 1] -= V[:, :j + 1] @ (V[:, :j + 1].T @
                    # V[:, j + 1])`; use `work` to store the intermediate
                    # product, which has length `j + 1` and is guaranteed to
                    # fit: as mentioned above, `work` has at least
                    # `4 * subspace_size ** 2 + 7 * subspace_size` elements

                    # `work = V[:, :j + 1].T @ V[:, j + 1]`
                    sgemv(b'T', num_genes, j + 1, 1, &V[0, 0], num_genes,
                       &V[0, j + 1], 1, 0, work.data(), 1)

                    # `V[:, j + 1] -= V[:, :j + 1] @ work`
                    sgemv(b'N', num_genes, j + 1, -1, &V[0, 0], num_genes,
                       work.data(), 1, 1, &V[0, j + 1], 1)
                else:
                    # `j == subspace_size - 1` so `V[:, j + 1]` would overwrite
                    # the end of the array. Do the same computation, but store
                    # the results in the `remainder` vector instead of
                    # `V[:, j + 1]`.
                    # `remainder = X_scaled.T @ U[:, j]`
                    rmatvec(data_rmatvec, indices_rmatvec, indptr_rmatvec,
                            U[:, j], clipped_stddev, &U_new[0, 0],
                            &remainder[0], num_cells, num_genes,
                            num_rmatvec_threads, rmatvec_thread_offsets,
                            chunk_size, block_sums)

                    # `remainder -= alpha * V[:, j]`
                    for i in range(num_genes):
                        remainder[i] -= alpha * V[i, j]

                    # `work = V.T @ remainder`
                    sgemv(b'T', num_genes, subspace_size, 1, &V[0, 0],
                          num_genes, &remainder[0], 1, 0, &work[0], 1)

                    # `remainder -= V @ work`
                    sgemv(b'N', num_genes, subspace_size, -1, &V[0, 0],
                          num_genes, &work[0], 1, 1, &remainder[0], 1)

                    break

                beta = norm(V[:, j + 1])
                if beta < clip_val:
                    beta = clip_val
                inverse_beta = <float> 1 / beta
                for i in range(num_genes):
                    V[i, j + 1] *= inverse_beta
                B[j, j + 1] = beta
                j += 1

                PyErr_CheckSignals()

            # Compute the SVD of the bidiagonal matrix `B`, equivalent to
            # `Ub, Sb, Vbt = np.linalg.svd(B, full_matrices=False)`
            svd(B, Ub, Sb, Vbt, work.data(), iwork.data(), lwork)

            # Normalize the `remainder` vector
            remainder_norm = norm(remainder)
            for i in range(num_genes):
                remainder[i] /= remainder_norm

            # Update `Smax`, the largest singular value of `B` we've seen so
            # far
            Smax = max(Sb[0], Smax)

            # Check convergence of singular values via residuals
            num_converged = 0
            for i in range(k):
                residual = remainder_norm * Ub[subspace_size - 1, i]
                num_converged += abs(residual) < tolerance * Smax
            if num_converged == k:
                converged = True
                break

            # For the next iteration, increase `k_current` to `k` + the number
            # of converged singular values, ensuring it stays under the
            # subspace size
            k_current = \
                min(max(k_current, k + num_converged), subspace_size - 1)

            # Restart with new subspace: update `U`, `V` and `B`. The `U` and
            # `V` updates are in-place, but `sgemm()` doesn't support in-place
            # matrix multiplication, so use `U_new` and `V_new` as the output
            # arrays and then swap the pointers so that `U` and `V` point to
            # the output.

            # U[:, :k_current] = U @ Ub[:, :k_current]
            sgemm(b'N', b'N', num_cells, k_current, subspace_size, 1, &U[0, 0],
                  num_cells, &Ub[0, 0], subspace_size, 0, &U_new[0, 0],
                  num_cells)
            temp = U
            U = U_new
            U_new = temp

            # V[:, :k_current] = V @ Vbt.T[:subspace_size, :k_current]
            sgemm(b'N', b'T', num_genes, k_current, subspace_size, 1, &V[0, 0],
                  num_genes, &Vbt[0, 0], subspace_size, 0, &V_new[0, 0],
                  num_genes)
            temp = V
            V = V_new
            V_new = temp

            for i in range(num_genes):
                V[i, k_current] = remainder[i]
            B[:] = 0
            for i in range(k_current):
                B[i, i] = Sb[i]
                residual = remainder_norm * Ub[subspace_size - 1, i]
                B[i, k_current] = residual
    else:
        # Same as the single-threaded version, except that dense matvecs and
        # matmuls use custom, deterministic parallel implementations rather
        # than `sgemm()`/`sgemv()`, and operations on U columns are fused to
        # minimize passes over cell-length vectors. All U columns are zero-mean
        # by construction (see key idea #9 above), which enables the matvec to
        # skip mean-centering (deferring it to the fused reorthogonalization
        # pass) and the rmatvec to skip mean-centering entirely.
        with nogil:
            for iteration in range(max_iterations):
                # Perform Lanczos bidiagonalization with reorthogonalization
                j = k_current
                while True:
                    # `U[:, j] = X_scaled @ V[:, j]` (sparse matmul only;
                    # mean-centering is deferred and fused into the
                    # reorthogonalization pass below); use `remainder` as a
                    # temporary buffer (it happens to have the correct type,
                    # float, and length, `num_genes`)
                    matvec_csr_parallel_raw(
                        data_matvec, indices_matvec, indptr_matvec,
                        V[:, j], clipped_stddev, &U[0, j],
                        &remainder[0], num_cells, num_genes,
                        num_matvec_threads, matvec_thread_offsets)

                    if j > 0:
                        # `U[:, j] -= mean(U[:, j]) + U[:, :j] @ (U[:, :j].T @
                        # U[:, j])`
                        #
                        # Fused blocked pass 1: sum of `U[:, j]` (for mean) and
                        # dot products `U[:, col].T @ U[:, j]`. Since all prior
                        # `U` columns are zero-mean, the dot products are
                        # invariant to whether `U[:, j]` has been
                        # mean-centered.
                        for block in prange(num_blocks_cells,
                                          num_threads=num_threads):
                            block_start = block * chunk_size
                            block_end = min(block_start + chunk_size,
                                            num_cells)
                            block_partial = 0
                            for row in range(block_start, block_end):
                                block_partial = \
                                    block_partial + U[row, j]
                            block_sums[block] = block_partial
                            for col in range(j):
                                block_partial = 0
                                for row in range(block_start,
                                                 block_end):
                                    block_partial = \
                                        block_partial + \
                                        U[row, col] * U[row, j]
                                partial_work[block, col] = \
                                    block_partial
                        mean = 0
                        for block in range(num_blocks_cells):
                            mean += block_sums[block]
                        mean /= num_cells
                        for col in range(j):
                            dot_product = 0
                            for block in range(num_blocks_cells):
                                dot_product = dot_product + \
                                    partial_work[block, col]
                            work[col] = dot_product

                        # Fused blocked pass 2: mean-subtract +
                        # reorthog subtract + squared norm
                        for block in prange(num_blocks_cells,
                                            num_threads=num_threads):
                            block_start = block * chunk_size
                            block_end = min(block_start + chunk_size,
                                            num_cells)
                            block_partial = 0
                            for row in range(block_start, block_end):
                                dot_product = mean
                                for col in range(j):
                                    dot_product = dot_product + \
                                        U[row, col] * work[col]
                                U[row, j] = U[row, j] - dot_product
                                block_partial = block_partial + \
                                    U[row, j] * U[row, j]
                            block_sums[block] = block_partial
                    else:
                        # No reorthogonalization; compute mean and fuse
                        # mean-subtract + squared norm
                        for block in prange(num_blocks_cells,
                                            num_threads=num_threads):
                            block_start = block * chunk_size
                            block_end = min(block_start + chunk_size,
                                            num_cells)
                            block_partial = 0
                            for row in range(block_start, block_end):
                                block_partial = block_partial + U[row, j]
                            block_sums[block] = block_partial
                        mean = 0
                        for block in range(num_blocks_cells):
                            mean += block_sums[block]
                        mean /= num_cells
                        for block in prange(num_blocks_cells,
                                            num_threads=num_threads):
                            block_start = block * chunk_size
                            block_end = min(block_start + chunk_size,
                                            num_cells)
                            block_partial = 0
                            for row in range(block_start, block_end):
                                U[row, j] = U[row, j] - mean
                                block_partial = block_partial + \
                                    U[row, j] * U[row, j]
                            block_sums[block] = block_partial
                    squared_norm = 0
                    for block in range(num_blocks_cells):
                        squared_norm += block_sums[block]
                    alpha = sqrt(squared_norm)
                    if alpha < clip_val:
                        alpha = clip_val
                    inverse_alpha = <float> 1 / alpha
                    for i in prange(num_cells,
                                    num_threads=num_threads_or_cells):
                        U[i, j] *= inverse_alpha
                    B[j, j] = alpha

                    if j < subspace_size - 1:
                        # `V[:, j + 1] = X_scaled.T @ U[:, j]`;
                        # U[:, j] is zero-mean, so skip mean-centering
                        # in the rmatvec
                        rmatvec_csc_parallel_raw(
                            data_rmatvec, indices_rmatvec,
                            indptr_rmatvec, U[:, j], clipped_stddev,
                            &V[0, j + 1], num_cells, num_genes,
                            num_rmatvec_threads,
                            rmatvec_thread_offsets)

                        # `V[:, j + 1] -= alpha * V[:, j]`
                        for i in range(num_genes):
                            V[i, j + 1] -= alpha * V[i, j]

                        # `V[:, j + 1] -= \
                        #      V[:, :j + 1] @ (V[:, :j + 1].T @ V[:, j + 1])`;
                        # use `work` to store the intermediate product, which
                        # has length `j + 1` and is guaranteed to fit: as
                        # mentioned above, `work` has at least
                        # `4 * subspace_size ** 2 + 7 * subspace_size` elements

                        # `work = V[:, :j + 1].T @ V[:, j + 1]`
                        for col in range(j + 1):
                            dot_product = 0
                            for row in range(num_genes):
                                dot_product = \
                                    dot_product + V[row, col] * V[row, j + 1]
                            work[col] = dot_product

                        # `V[:, j + 1] -= V[:, :j + 1] @ work`
                        for row in range(num_genes):
                            dot_product = 0
                            for col in range(j + 1):
                                dot_product = \
                                    dot_product + V[row, col] * work[col]
                            V[row, j + 1] -= dot_product
                    else:
                        # `j == subspace_size - 1` so `V[:, j + 1]` would
                        # overwrite the end of the array. Do the same
                        # computation, but store the results in the `remainder`
                        # vector instead of `V[:, j + 1]`.
                        # `remainder = X_scaled.T @ U[:, j]`; U[:, j] is
                        # zero-mean, so skip mean-centering
                        rmatvec_csc_parallel_raw(
                            data_rmatvec, indices_rmatvec,
                            indptr_rmatvec, U[:, j], clipped_stddev,
                            &remainder[0], num_cells, num_genes,
                            num_rmatvec_threads, rmatvec_thread_offsets)

                        # `remainder -= alpha * V[:, j]`
                        for i in range(num_genes):
                            remainder[i] -= alpha * V[i, j]

                        # `work = V.T @ remainder`
                        for col in range(subspace_size):
                            dot_product = 0
                            for row in range(num_genes):
                                dot_product = \
                                    dot_product + V[row, col] * remainder[row]
                            work[col] = dot_product

                        # `remainder -= V @ work`
                        for row in range(num_genes):
                            dot_product = 0
                            for col in range(subspace_size):
                                dot_product = \
                                    dot_product + V[row, col] * work[col]
                            remainder[row] -= dot_product

                        break

                    beta = norm(V[:, j + 1])
                    if beta < clip_val:
                        beta = clip_val
                    inverse_beta = <float> 1 / beta
                    for i in range(num_genes):
                        V[i, j + 1] *= inverse_beta
                    B[j, j + 1] = beta
                    j += 1

                    with gil:
                        PyErr_CheckSignals()

                # Compute the SVD of the bidiagonal matrix `B`, equivalent to
                # `Ub, Sb, Vbt = np.linalg.svd(B, full_matrices=False)`.
                svd(B, Ub, Sb, Vbt, work.data(), iwork.data(), lwork)

                # Normalize the `remainder` vector
                remainder_norm = norm(remainder)
                for i in range(num_genes):
                    remainder[i] /= remainder_norm

                # Update `Smax`, the largest singular value of `B` we've seen
                # so far
                Smax = max(Sb[0], Smax)

                # Check convergence of singular values via residuals
                num_converged = 0
                for i in range(k):
                    residual = remainder_norm * Ub[subspace_size - 1, i]
                    num_converged += abs(residual) < tolerance * Smax
                if num_converged == k:
                    converged = True
                    break

                # For the next iteration, increase `k_current` to `k` + the
                # number of converged singular values, ensuring it stays under
                # the subspace size
                k_current = min(max(k_current, k + num_converged),
                                subspace_size - 1)

                # Restart with new subspace: update `U`, `V` and `B`. The `U`
                # and `V` updates are in-place, but `sgemm()` doesn't support
                # in-place matrix multiplication, so use `U_new` and `V_new` as
                # the output arrays and then swap the pointers so that `U` and
                # `V` point to the output arrays.

                # U[:, :k_current] = U @ Ub[:, :k_current]
                matmul_nn_f_out(num_cells, k_current, subspace_size, U, Ub,
                                U_new, num_threads_or_cells)
                temp = U
                U = U_new
                U_new = temp

                # V[:, :k_current] = V @ Vbt.T[:subspace_size, :k_current]
                sgemm(b'N', b'T', num_genes, k_current, subspace_size, 1,
                      &V[0, 0], num_genes, &Vbt[0, 0], subspace_size, 0,
                      &V_new[0, 0], num_genes)
                temp = V
                V = V_new
                V_new = temp

                for i in range(num_genes):
                    V[i, k_current] = remainder[i]
                B[:] = 0
                for i in range(k_current):
                    B[i, i] = Sb[i]
                    residual = remainder_norm * Ub[subspace_size - 1, i]
                    B[i, k_current] = residual

    # Construct final PCs from the top `k` components of `U` and `S`:
    # U[:, :k] = U[:, :subspace_size] @ Ub[:, :k]
    # PCs[:] = U[:, :k] * Sb[:k]
    # As an optimization, due to linearity we can switch the order of
    # operations to:
    # Ub[:, :k] *= Sb[:k]
    for j in range(k):
        Sbj = Sb[j]
        for i in range(subspace_size):
            Ub[i, j] *= Sbj
    # followed by:
    # PCs[:] = U[:, :subspace_size] @ Ub[:, :k]
    # However, since `PCs` is C-contiguous whereas `U` and `Ub` are
    # Fortran-contiguous, we actually compute:
    # PCs.T[:] = Ub[:, :k].T @ U[:, :subspace_size].T
    # and then reinterpret PCs as C-contiguous
    if num_threads == 1 and not match_parallel:
        sgemm(b'T', b'T', k, num_cells, subspace_size, 1, &Ub[0, 0],
              subspace_size, &U[0, 0], num_cells, 0, &PCs[0, 0], k)
    else:
        matmul_nn_c_out(num_cells, k, subspace_size,
                        U, Ub, PCs, num_threads_or_cells)

    # Return whether IRLBA converged
    return converged
