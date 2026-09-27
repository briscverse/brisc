# Functions for harmonization and label transfer

from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libcpp.algorithm cimport fill
from libcpp.cmath cimport abs, ceil, exp, log, pow, sqrt
from libcpp.vector cimport vector
from scipy.linalg.cython_blas cimport sgemm, sgemv
from .cyutils cimport bin_count_nogil, integer, randint, srand, \
    uninitialized_vector


def label_transfer(const unsigned[:, ::1] neighbors,
                   const unsigned[::1] original_cell_type_column,
                   const unsigned num_cell_types,
                   unsigned[::1] cell_types,
                   float[::1] confidences,
                   unsigned[::1] next_best_cell_types,
                   float[::1] next_best_confidences,
                   unsigned num_threads):
    cdef unsigned i, j, thread_index, cell_type, count, \
        most_common_cell_type, second_most_common_cell_type, \
        max_count, second_max_count, num_neighbors = neighbors.shape[1]
    cdef unsigned long long num_cells = neighbors.shape[0]
    cdef float inv_num_neighbors = <float> 1 / num_neighbors
    cdef uninitialized_vector[unsigned] counts_buffer
    cdef vector[uninitialized_vector[unsigned]] thread_counts
    cdef unsigned[::1] counts

    num_threads = min(num_threads, num_cells)
    if next_best_cell_types.shape[0] == 0:  # `next_best=False`
        if num_threads <= 1:
            counts_buffer.resize(num_cell_types)
            counts = <unsigned[:num_cell_types]> counts_buffer.data()
            for i in range(num_cells):
                counts[:] = 0
                for j in range(num_neighbors):
                    # Get the cell-type label of this nearest neighbor (using
                    # our integer encoding where the most common cell type is
                    # 0, the next-most common 1, etc.)
                    cell_type = original_cell_type_column[neighbors[i, j]]
                    counts[cell_type] += 1
                if counts[0] >= counts[1]:
                    max_count = counts[0]
                    most_common_cell_type = 0
                else:
                    max_count = counts[1]
                    most_common_cell_type = 1
                for cell_type in range(2, num_cell_types):
                    count = counts[cell_type]
                    if count > max_count:
                        max_count = count
                        most_common_cell_type = cell_type
                cell_types[i] = most_common_cell_type
                confidences[i] = max_count * inv_num_neighbors
        else:
            thread_counts.resize(num_threads)
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_counts[thread_index].resize(num_cell_types)
                for i in prange(num_cells):
                    fill(thread_counts[thread_index].begin(),
                         thread_counts[thread_index].end(), 0)
                    for j in range(num_neighbors):
                        cell_type = original_cell_type_column[neighbors[i, j]]
                        thread_counts[thread_index][cell_type] += 1
                    if thread_counts[thread_index][0] >= \
                            thread_counts[thread_index][1]:
                        max_count = thread_counts[thread_index][0]
                        most_common_cell_type = 0
                    else:
                        max_count = thread_counts[thread_index][1]
                        most_common_cell_type = 1
                    for cell_type in range(2, num_cell_types):
                        count = thread_counts[thread_index][cell_type]
                        if count > max_count:
                            max_count = count
                            most_common_cell_type = cell_type
                    cell_types[i] = most_common_cell_type
                    confidences[i] = max_count * inv_num_neighbors
    else:  # `next_best=True`; also compute the next-best cell type/confidence
        if num_threads <= 1:
            counts_buffer.resize(num_cell_types)
            counts = <unsigned[:num_cell_types]> counts_buffer.data()
            for i in range(num_cells):
                counts[:] = 0
                for j in range(num_neighbors):
                    # Get the cell-type label of this nearest neighbor (using
                    # our integer encoding where the most common cell type is
                    # 0, the next-most common 1, etc.)
                    cell_type = original_cell_type_column[neighbors[i, j]]
                    counts[cell_type] += 1
                if counts[0] >= counts[1]:
                    max_count = counts[0]
                    second_max_count = counts[1]
                    most_common_cell_type = 0
                    second_most_common_cell_type = 1
                else:
                    max_count = counts[1]
                    second_max_count = counts[0]
                    most_common_cell_type = 1
                    second_most_common_cell_type = 0
                for cell_type in range(2, num_cell_types):
                    count = counts[cell_type]
                    if count > max_count:
                        second_max_count = max_count
                        second_most_common_cell_type = \
                            most_common_cell_type
                        max_count = count
                        most_common_cell_type = cell_type
                    elif count > second_max_count:
                        second_max_count = count
                        second_most_common_cell_type = cell_type
                cell_types[i] = most_common_cell_type
                confidences[i] = max_count * inv_num_neighbors
                next_best_cell_types[i] = \
                    second_most_common_cell_type
                next_best_confidences[i] = \
                    second_max_count * inv_num_neighbors
        else:
            thread_counts.resize(num_threads)
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_counts[thread_index].resize(num_cell_types)
                for i in prange(num_cells):
                    fill(thread_counts[thread_index].begin(),
                         thread_counts[thread_index].end(), 0)
                    for j in range(num_neighbors):
                        cell_type = original_cell_type_column[neighbors[i, j]]
                        thread_counts[thread_index][cell_type] += 1
                    if thread_counts[thread_index][0] >= \
                            thread_counts[thread_index][1]:
                        max_count = thread_counts[thread_index][0]
                        second_max_count = thread_counts[thread_index][1]
                        most_common_cell_type = 0
                        second_most_common_cell_type = 1
                    else:
                        max_count = thread_counts[thread_index][1]
                        second_max_count = thread_counts[thread_index][0]
                        most_common_cell_type = 1
                        second_most_common_cell_type = 0
                    for cell_type in range(2, num_cell_types):
                        count = thread_counts[thread_index][cell_type]
                        if count > max_count:
                            second_max_count = max_count
                            second_most_common_cell_type = \
                                most_common_cell_type
                            max_count = count
                            most_common_cell_type = cell_type
                        elif count > second_max_count:
                            second_max_count = count
                            second_most_common_cell_type = cell_type
                    cell_types[i] = most_common_cell_type
                    confidences[i] = max_count * inv_num_neighbors
                    next_best_cell_types[i] = \
                        second_most_common_cell_type
                    next_best_confidences[i] = \
                        second_max_count * inv_num_neighbors


cdef inline void matrix_multiply(const float[:, ::1] A,
                                 const float[:, ::1] B,
                                 float[:, ::1] C,
                                 const bint transpose_A,
                                 const bint transpose_B,
                                 const float alpha,
                                 const float beta) noexcept nogil:
    # A wrapper for `sgemm()` for the case when all matrices are C-major. Flip
    # `A` <-> `B`, `shape[0]` <-> `shape[1]`, and both transpose flags, since
    # BLAS expects Fortran-major. Note that `C`'s dimensions are not required
    # to match `A` and `B`'s (and often don't); `C` is only used for its
    # address.

    cdef int m, n, k, lda, ldb
    cdef char transA, transB
    if transpose_B:
        m = B.shape[0]
        k = B.shape[1]
        lda = k
        transA = b'T'
    else:
        m = B.shape[1]
        k = B.shape[0]
        lda = m
        transA = b'N'
    if transpose_A:
        n = A.shape[1]
        ldb = n
        transB = b'T'
    else:
        n = A.shape[0]
        ldb = k
        transB = b'N'
    sgemm(&transA, &transB, &m, &n, &k, <float*> &alpha,
          <float*> &B[0, 0], &lda, <float*> &A[0, 0], &ldb,
          <float*> &beta, &C[0, 0], &m)


cdef inline void matrix_vector_multiply(
        const float[:, ::1] A,
        const float[::1] X,
        float[::1] Y,
        const bint transpose,
        const float alpha,
        const float beta) noexcept nogil:
    # A wrapper for `sgemv()` for the case when both matrices are C-major. Flip
    # `shape[0]` <-> `shape[1]` and the transpose flag, since BLAS expects
    # Fortran-major.

    cdef int m = A.shape[1], n = A.shape[0], incx = 1, incy = 1
    cdef char trans = b'N' if transpose else b'T'
    sgemv(&trans, &m, &n, <float*> &alpha, <float*> &A[0,0], &m,
          <float*> &X[0], &incx, <float*> &beta, &Y[0], &incy)


cdef inline void normalize_rows_inplace(float[:, ::1] arr):
    cdef unsigned i, j
    cdef float norm
    for i in range(arr.shape[0]):
        norm = 0
        for j in range(arr.shape[1]):
            norm += arr[i, j] * arr[i, j]
        norm = <float> 1 / sqrt(norm)
        for j in range(arr.shape[1]):
            arr[i, j] = arr[i, j] * norm


cdef inline void normalize_rows_inplace_parallel(
        float[:, ::1] arr,
        unsigned num_threads) noexcept nogil:
    cdef unsigned i, j, num_rows = arr.shape[0], num_columns = arr.shape[1]
    cdef float norm

    num_threads = min(num_threads, num_rows)
    for i in prange(num_rows, num_threads=num_threads):
        norm = 0
        for j in range(num_columns):
            norm = norm + arr[i, j] * arr[i, j]
        norm = <float> 1 / sqrt(norm)
        for j in range(num_columns):
            arr[i, j] = arr[i, j] * norm


def normalize_rows(const float[:, ::1] arr,
                   float[:, ::1] out,
                   unsigned num_threads):
    cdef unsigned i, j, num_rows = arr.shape[0], num_columns = arr.shape[1]
    cdef float norm

    num_threads = min(num_threads, num_rows)
    if num_threads <= 1:
        for i in range(num_rows):
            norm = 0
            for j in range(num_columns):
                norm += arr[i, j] * arr[i, j]
            norm = <float> 1 / sqrt(norm)
            for j in range(num_columns):
                out[i, j] = arr[i, j] * norm
    else:
        for i in prange(num_rows, nogil=True, num_threads=num_threads):
            norm = 0
            for j in range(num_columns):
                norm = norm + arr[i, j] * arr[i, j]
            norm = <float> 1 / sqrt(norm)
            for j in range(num_columns):
                out[i, j] = arr[i, j] * norm


def harmony(const float[:, ::1] PCs,
            float[:, ::1] Z,
            float[:, ::1] Y,
            float[:, ::1] R,
            const unsigned[::1] batch_labels,
            const unsigned num_batches,
            const unsigned max_iterations,
            const unsigned max_clustering_iterations,
            const float block_proportion,
            const float tolerance,
            const bint early_stopping,
            const float clustering_tolerance,
            const float theta,
            const float tau,
            const float alpha,
            const float sigma,
            const unsigned chunk_size,
            const unsigned long long seed,
            const bint verbose,
            const unsigned num_threads):

    # The major data structures in the Harmony algorithm are:
    # - `Z` (cells × PCs): the row-normalized principal components
    # - `Y` (clusters × PCs): the centroid of each cluster
    # - `R` (cells × clusters): the soft assignment of each cell to each
    #   cluster, which has values between 0 and 1 and sums to 1 for each cell
    # - `O` (batches × clusters): the cluster-batch co-occurrence matrix, i.e.
    #    the sum of the `R`s across all cells from each batch that are part of
    #    each cluster
    # - `E` (batches × clusters): the expected value of `O` if batches were
    #   randomly distributed across clusters. For a given batch `i` and cluster
    #   `j`, this is just `Pr_b[i]` (the fraction of cells in batch `i`) times
    #   `sum(R[:, j])` (the sum of `R` for that cluster across all batches).
    #    We define `R_sum = sum(R[:, j])`.
    # The Harmony paper also refers to ϕ, the one-hot encoded batch labels. We
    # avoid one-hot encoding for efficiency and just use the raw batch labels.
    #
    # We process cells blockwise and then chunkwise within each block. Why both
    # blocks and chunks? Because `O` and `E` need to be updated by block, so
    # parallelization has to occur within each block.
    #
    # The number of chunks may be slightly less for the last block, so
    # per-chunk arrays are allocated to have
    # `max_chunks_per_block = ceil(block_size / chunk_size)` chunks, of which
    # only the first `num_chunks_per_block` are used for each block.
    #
    # Allocate per-thread storage for `Z_chunk` and `distance = Z_chunk @ Y.T`.
    # Allocate a single buffer for all threads without worrying about false
    # sharing, since the default `chunk_size` (512) is a multiple of the cache
    # line size.

    cdef unsigned long long num_cells = Z.shape[0], state = srand(seed)
    cdef unsigned i, j, k, thread_index, chunk_index, chunk_start, chunk_end, \
        iteration, clustering_iteration, block_index, block_start, block_end, \
        num_chunks_per_block, batch_label, start_row, end_row, \
        num_PCs = Z.shape[1], num_clusters = Y.shape[0], \
        num_chunks = (num_cells + chunk_size - 1) / chunk_size, \
        block_size = <unsigned> ceil(num_cells * block_proportion), \
        num_blocks = (num_cells + block_size - 1) / block_size, \
        max_chunks_per_block = (block_size + chunk_size - 1) / chunk_size, \
        num_cells_per_thread = (num_cells + num_threads - 1) / num_threads, \
        num_threads_or_chunks = min(num_threads, num_chunks)
    cdef float base, kmeans_error, entropy_term, norm, Rij, R_sum, O_sum, \
        diversity_penalty, prev_objective, total, delta_Eij, delta_Oij, \
        Pr_bi, Eij, Oij, Rkj, objective, last_two, old, new, ridge_lambda, \
        factor, batch_total, two_over_sigma = <float> 2 / sigma, \
        exp_neg_two_over_sigma = exp(-two_over_sigma)
    cdef float past_clustering_objectives[3]
    cdef str metrics
    cdef uninitialized_vector[unsigned] N_b_buffer, cell_order_buffer
    cdef uninitialized_vector[float] Pr_b_buffer, theta_batch_buffer, \
        cluster_diversity_penalty_buffer, E_buffer, O_buffer, \
        diversity_buffer, R_sums_buffer, inv_cov_2_buffer, inv_cov_1_buffer, \
        inv_cov_buffer, R_scaled_PCs_buffer, W_buffer, Y_chunk_buffer, \
        Z_chunk_buffer, distance_buffer, delta_O_buffer, delta_E_buffer, \
        ratio_buffer, kmeans_errors, entropy_terms, R_scaled_PCs_chunk_buffer
    N_b_buffer.resize(num_batches)
    cell_order_buffer.resize(num_cells)
    Pr_b_buffer.resize(num_batches)
    theta_batch_buffer.resize(num_batches)
    cluster_diversity_penalty_buffer.resize(num_clusters)
    E_buffer.resize(<unsigned long long> num_batches * num_clusters)
    O_buffer.resize(<unsigned long long> num_batches * num_clusters)
    diversity_buffer.resize(<unsigned long long> num_batches * num_clusters)
    R_sums_buffer.resize(<unsigned long long> num_chunks * num_clusters)
    inv_cov_2_buffer.resize(
        <unsigned long long> (num_batches + 1) * (num_batches + 1))
    inv_cov_1_buffer.resize(
        <unsigned long long> (num_batches + 1) * (num_batches + 1))
    inv_cov_buffer.resize(
        <unsigned long long> (num_batches + 1) * (num_batches + 1))
    R_scaled_PCs_buffer.resize(
        <unsigned long long> num_PCs * (num_batches + 1))
    W_buffer.resize(
        <unsigned long long> num_clusters * (num_batches + 1) * num_PCs)
    Y_chunk_buffer.resize(
        <unsigned long long> num_chunks * num_clusters * num_PCs)
    Z_chunk_buffer.resize(
        <unsigned long long> num_threads * chunk_size * num_PCs)
    distance_buffer.resize(
        <unsigned long long> num_threads * chunk_size * num_clusters)
    delta_O_buffer.resize(
        <unsigned long long> num_chunks * num_batches * num_clusters)
    delta_E_buffer.resize(
        <unsigned long long> num_chunks * num_batches * num_clusters)
    ratio_buffer.resize(
        <unsigned long long> max_chunks_per_block * num_batches * num_clusters)
    kmeans_errors.resize(num_chunks)
    entropy_terms.resize(num_chunks)
    R_scaled_PCs_chunk_buffer.resize(
        <unsigned long long> num_chunks * num_PCs * (num_batches + 1))
    cdef unsigned[::1] \
        N_b = <unsigned[:num_batches]> N_b_buffer.data(), \
        cell_order = <unsigned[:num_cells]> cell_order_buffer.data()
    cdef float[::1] \
        Pr_b = <float[:num_batches]> Pr_b_buffer.data(), \
        theta_batch = <float[:num_batches]> theta_batch_buffer.data(), \
        cluster_diversity_penalty = <float[:num_clusters]> \
            cluster_diversity_penalty_buffer.data()
    cdef float[:, ::1] \
        E = <float[:num_batches, :num_clusters]> E_buffer.data(), \
        O = <float[:num_batches, :num_clusters]> O_buffer.data(), \
        diversity = <float[:num_batches, :num_clusters]> \
            diversity_buffer.data(), \
        R_sums = <float[:num_chunks, :num_clusters]> R_sums_buffer.data(), \
        inv_cov_2 = <float[:num_batches + 1, :num_batches + 1]> \
            inv_cov_2_buffer.data(), \
        inv_cov_1 = <float[:num_batches + 1, :num_batches + 1]> \
            inv_cov_1_buffer.data(), \
        inv_cov = <float[:num_batches + 1, :num_batches + 1]> \
            inv_cov_buffer.data(), \
        R_scaled_PCs = <float[:num_PCs, :num_batches + 1]> \
            R_scaled_PCs_buffer.data()
    cdef float[:, :, ::1] \
        Y_chunk = <float[:num_chunks, :num_clusters, :num_PCs]> \
            Y_chunk_buffer.data(), \
        Z_chunk = <float[:num_threads, :chunk_size, :num_PCs]> \
            Z_chunk_buffer.data(), \
        distances = <float[:num_threads, :chunk_size, :num_clusters]> \
            distance_buffer.data(), \
        delta_O = <float[:num_chunks, :num_batches, :num_clusters]> \
            delta_O_buffer.data(), \
        delta_E = <float[:num_chunks, :num_batches, :num_clusters]> \
            delta_E_buffer.data(), \
        ratio = <float[:max_chunks_per_block, :num_batches, :num_clusters]> \
            ratio_buffer.data(), \
        R_scaled_PCs_chunk = \
            <float[:num_chunks, :num_PCs, :num_batches + 1]> \
            R_scaled_PCs_chunk_buffer.data(), \
        W = <float[:num_clusters, :num_batches + 1, :num_PCs]> W_buffer.data()

    with nogil:
        # Get the number (`N_b`) and fraction (`Pr_b`) of cells in each batch;
        # apply discounting to `theta`, if `tau` is non-zero
        bin_count_nogil(batch_labels, N_b, num_threads)
        if tau:
            for i in range(num_batches):
                Pr_b[i] = <float> N_b[i] / num_cells
                base = exp(-(<float> N_b[i]) / (num_clusters * tau))
                theta_batch[i] = theta * (<float> 1 - base * base)
        else:
            for i in range(num_batches):
                Pr_b[i] = <float> N_b[i] / num_cells
                theta_batch[i] = theta

        # Initialize `R`, `R_sum`, and `O` chunkwise. Compute the initial
        # k-means error and entropy term, the first two components of the
        # objective function, for each chunk.
        with parallel(num_threads=num_threads_or_chunks):
            thread_index = threadid()
            for chunk_index in prange(num_chunks):
                chunk_start = chunk_index * chunk_size
                chunk_end = min(chunk_start + chunk_size, num_cells)
                matrix_multiply(Z[chunk_start:chunk_end], Y,
                                distances[thread_index], transpose_A=False,
                                transpose_B=True, alpha=1, beta=0)
                kmeans_error = 0
                entropy_term = 0
                R_sums[chunk_index, :] = 0
                delta_O[chunk_index, :, :] = 0
                for i in range(chunk_start, chunk_end):
                    batch_label = batch_labels[i]
                    norm = 0
                    for j in range(num_clusters):
                        Rij = exp(two_over_sigma * (distances[
                            thread_index, i - chunk_start, j] - <float> 1))
                        R[i, j] = Rij
                        norm += Rij
                    norm = <float> 1 / norm
                    for j in range(num_clusters):
                        Rij = R[i, j]
                        Rij = Rij * norm
                        R[i, j] = Rij
                        R_sums[chunk_index, j] += Rij
                        delta_O[chunk_index, batch_label, j] += Rij
                        kmeans_error = kmeans_error + \
                            Rij * (<float> 1 - distances[
                                thread_index, i - chunk_start, j])
                        entropy_term = entropy_term + Rij * log(Rij)
                kmeans_errors[chunk_index] = kmeans_error
                entropy_terms[chunk_index] = entropy_term

        # Initialize `E`
        for i in range(num_batches):
            for j in range(num_clusters):
                R_sum = 0
                for chunk_index in range(num_chunks):
                    R_sum += R_sums[chunk_index, j]
                E[i, j] = Pr_b[i] * R_sum

        # Aggregate the initial k-means error and entropy term across chunks
        kmeans_error = 0
        entropy_term = 0
        for chunk_index in range(num_chunks):
            kmeans_error += kmeans_errors[chunk_index]
            entropy_term += entropy_terms[chunk_index]
        kmeans_error *= <float> 2
        entropy_term *= sigma

        # Initialize `O` and compute the initial diversity penalty, the third
        # component of the objective function
        diversity_penalty = 0
        for i in range(num_batches):
            for j in range(num_clusters):
                O_sum = 0
                for chunk_index in range(num_chunks):
                    O_sum += delta_O[chunk_index, i, j]
                O[i, j] = O_sum
                diversity[i, j] = \
                    O_sum * log((O_sum + E[i, j] + <float> 1) /
                                (E[i, j] + <float> 1))
        matrix_vector_multiply(diversity, theta_batch,
                               cluster_diversity_penalty,
                               transpose=True, alpha=1, beta=0)
        for i in range(num_clusters):
            diversity_penalty += cluster_diversity_penalty[i]
        diversity_penalty *= sigma

        # Compute the initial total objective function
        prev_objective = kmeans_error + entropy_term + diversity_penalty

        # Normalize each row of `Y` in-place
        normalize_rows_inplace_parallel(Y, num_threads)

    # Define the random order to iterate over cells in, via the "inside-out"
    # variant of the Fisher-Yates shuffle. The uninitialized variable is
    # intentional!
    for i in range(num_cells):
        j = randint(i + 1, &state)
        cell_order[i] = cell_order[j]
        cell_order[j] = i

    if verbose:
        print(f'Initialization is complete: objective = '
              f'{prev_objective:.2f}')

    # Check for KeyboardInterrupts
    PyErr_CheckSignals()

    # Shrink `R_sums`, `delta_O`, and `delta_E` from `num_chunks` to
    # `max_chunks_per_block` along their first dimension
    R_sums_buffer.resize(
        <unsigned long long> max_chunks_per_block * num_clusters)
    delta_O_buffer.resize(
        <unsigned long long> max_chunks_per_block * num_batches * num_clusters)
    delta_E_buffer.resize(
        <unsigned long long> max_chunks_per_block * num_batches * num_clusters)
    R_sums = <float[:max_chunks_per_block, :num_clusters]> R_sums_buffer.data()
    delta_O = <float[:max_chunks_per_block, :num_batches, :num_clusters]> \
        delta_O_buffer.data()
    delta_E = <float[:max_chunks_per_block, :num_batches, :num_clusters]> \
        delta_E_buffer.data()

    # Now that initialization is done, start the Harmony iterations
    with nogil:
        for iteration in range(1, max_iterations + 1):
            # Perform `max_clustering_iterations` iterations of clustering
            # within each Harmony iteration, stopping early if
            # `early_stopping=True` and convergence is met
            for clustering_iteration in range(
                    1, max_clustering_iterations + 1):
                # Compute `Y`, the normalized cluster centroids, chunkwise
                for chunk_index in prange(num_chunks,
                                          num_threads=num_threads_or_chunks):
                    chunk_start = chunk_index * chunk_size
                    chunk_end = min(chunk_start + chunk_size, num_cells)
                    matrix_multiply(R[chunk_start:chunk_end],
                                    Z[chunk_start:chunk_end],
                                    Y_chunk[chunk_index], transpose_A=True,
                                    transpose_B=False, alpha=1, beta=0)
                for i in prange(num_clusters, num_threads=num_threads):
                    norm = 0
                    for j in range(num_PCs):
                        total = 0
                        for chunk_index in range(num_chunks):
                            total = total + Y_chunk[chunk_index, i, j]
                        Y[i, j] = total
                        norm = norm + total * total
                    norm = <float> 1 / sqrt(norm)
                    for j in range(num_PCs):
                        Y[i, j] *= norm

                # Update `R`, `E`, and `O` by processing cells blockwise and
                # then chunkwise within each block. Note that the full formula
                # for `R` is: `((E + 1) / (O + 1)) ** theta *
                #              exp(-2 / sigma * (1 - Z @ Y.T))`
                # which we can calculate as `ratio * exp(distances)` where
                # `ratio = exp(-2 / sigma) * ((E + 1) / (O + 1)) ** theta` and
                # `distances = 2 / sigma * Z @ Y.T`. Calculating `R` this way
                # saves a few multiplications.
                for block_index in range(num_blocks):
                    block_start = block_index * block_size
                    block_end = min(block_start + block_size, num_cells)
                    num_chunks_per_block = \
                        (block_end - block_start + chunk_size - 1) / chunk_size
                    # The correct number of threads here is actually
                    # `min(num_threads, num_chunks_per_block)`, but that would
                    # spawn a different-sized threadpool from the rest of the
                    # function
                    with parallel(num_threads=num_threads_or_chunks):
                        thread_index = threadid()
                        for chunk_index in prange(num_chunks_per_block):
                            chunk_start = chunk_index * chunk_size
                            chunk_end = min(chunk_start + chunk_size,
                                            block_end - block_start)

                            # Get the chunk of `Z`, the normalized PCs, to
                            # process. Calculate the observed-to-expected ratio
                            # `ratio = exp(-2 / sigma) *
                            #          ((E + 1) / (O + 1)) ** theta` for the
                            # chunk, subtracting the chunk's own contributions
                            # to `E` and `O`, which we store in
                            # `delta_O[chunk_index]` and
                            # `delta_E[chunk_index]`.
                            R_sums[chunk_index, :] = 0
                            delta_O[chunk_index, :, :] = 0
                            for i in range(chunk_end - chunk_start):
                                k = cell_order[block_start + chunk_start + i]
                                batch_label = batch_labels[k]
                                Z_chunk[thread_index, i, :] = Z[k, :]
                                for j in range(num_clusters):
                                    R_sums[chunk_index, j] += R[k, j]
                                    delta_O[chunk_index, batch_label, j] -= \
                                        R[k, j]
                            for i in range(num_batches):
                                Pr_bi = Pr_b[i]
                                for j in range(num_clusters):
                                    delta_Eij = -Pr_bi * R_sums[chunk_index, j]
                                    delta_Oij = delta_O[chunk_index, i, j]
                                    Eij = E[i, j] + delta_Eij
                                    Oij = O[i, j] + delta_Oij
                                    ratio[chunk_index, i, j] = \
                                        exp_neg_two_over_sigma * \
                                        pow((Eij + <float> 1) /
                                            (Oij + Eij + <float> 1),
                                            theta_batch[i])
                                    delta_E[chunk_index, i, j] = delta_Eij

                            # Compute `distances = 2 / sigma * Z @ Y.T`
                            matrix_multiply(
                                Z_chunk[thread_index,
                                       :chunk_end - chunk_start],
                                Y, distances[thread_index], transpose_A=False,
                                transpose_B=True, alpha=two_over_sigma, beta=0)

                            # Update `R`, the fractional soft clustering
                            # assignment of each cell to each cluster, for the
                            # cells in the chunk. Normalize each row of `R` to
                            # ensure it always sums to 1 despite floating-point
                            # error. Also, add back this chunk's contributions
                            # to `O` and `E` to `delta_O[chunk_index]` and
                            # `delta_E[chunk_index]` using the newly-updated
                            # `R`.
                            R_sums[chunk_index, :] = 0
                            for i in range(chunk_end - chunk_start):
                                k = cell_order[block_start + chunk_start + i]
                                batch_label = batch_labels[k]
                                norm = 0
                                for j in range(num_clusters):
                                    R[k, j] = exp(distances[
                                        thread_index, i, j]) * \
                                        ratio[chunk_index, batch_label, j]
                                    norm += R[k, j]
                                norm = <float> 1 / norm
                                for j in range(num_clusters):
                                    Rkj = R[k, j]
                                    Rkj = Rkj * norm
                                    R[k, j] = Rkj
                                    R_sums[chunk_index, j] += Rkj
                                    delta_O[chunk_index, batch_label, j] += Rkj
                            for i in range(num_batches):
                                Pr_bi = Pr_b[i]
                                for j in range(num_clusters):
                                    delta_E[chunk_index, i, j] += \
                                        Pr_bi * R_sums[chunk_index, j]

                    # Update `O` and `E` for this block with the `delta_E` and
                    # `delta_O` for each chunk.
                    for chunk_index in range(num_chunks_per_block):
                        for i in range(num_batches):
                            for j in range(num_clusters):
                                E[i, j] += delta_E[chunk_index, i, j]
                                O[i, j] += delta_O[chunk_index, i, j]

                # Compute the objective function, if we are done clustering or
                # `early_stopping=True`
                if early_stopping or \
                        clustering_iteration == max_clustering_iterations:
                    # Compute its first two components chunkwise: the k-means
                    # error and entropy term
                    with parallel(num_threads=num_threads_or_chunks):
                        thread_index = threadid()
                        for chunk_index in prange(num_chunks):
                            chunk_start = chunk_index * chunk_size
                            chunk_end = \
                                min(chunk_start + chunk_size, num_cells)
                            matrix_multiply(Z[chunk_start:chunk_end], Y,
                                            distances[thread_index],
                                            transpose_A=False,
                                            transpose_B=True, alpha=1, beta=0)
                            kmeans_error = 0
                            entropy_term = 0
                            for i in range(chunk_start, chunk_end):
                                for j in range(num_clusters):
                                    kmeans_error = kmeans_error + \
                                        R[i, j] * (<float> 1 - distances[
                                            thread_index, i - chunk_start, j])
                                    entropy_term = \
                                        entropy_term + R[i, j] * log(R[i, j])
                            kmeans_errors[chunk_index] = kmeans_error
                            entropy_terms[chunk_index] = entropy_term
                    kmeans_error = 0
                    entropy_term = 0
                    for chunk_index in range(num_chunks):
                        kmeans_error += kmeans_errors[chunk_index]
                        entropy_term += entropy_terms[chunk_index]
                    kmeans_error *= <float> 2
                    entropy_term *= sigma

                    # Compute the diversity penalty, the third component of the
                    # objective function
                    diversity_penalty = 0
                    for i in range(num_batches):
                        for j in range(num_clusters):
                            diversity[i, j] = O[i, j] * log(
                                (O[i, j] + E[i, j] + <float> 1) /
                                (E[i, j] + <float> 1))
                    matrix_vector_multiply(diversity, theta_batch,
                                           cluster_diversity_penalty,
                                           transpose=True, alpha=1, beta=0)
                    for i in range(num_clusters):
                        diversity_penalty += cluster_diversity_penalty[i]
                    diversity_penalty *= sigma

                    # Compute the total objective function
                    objective = kmeans_error + entropy_term + diversity_penalty

                    if early_stopping:
                        with gil:
                            # Check for KeyboardInterrupts after each
                            # clustering iteration
                            PyErr_CheckSignals()

                            # If `verbose=True`, print the objective function
                            if verbose:
                                print(f'Clustering iteration '
                                      f'{clustering_iteration:,}: objective = '
                                      f'{objective:.2f} (k-means error = '
                                      f'{kmeans_error:.2f}, entropy term = '
                                      f'{entropy_term:.2f}, diversity '
                                      f'penalty = {diversity_penalty:.2f})')

                        # Exit early if the clustering converged, based on a
                        # sliding window average of the objective over the past
                        # three iterations
                        if clustering_iteration <= 3:
                            past_clustering_objectives[
                                clustering_iteration - 1] = objective
                        else:
                            last_two = past_clustering_objectives[1] + \
                                past_clustering_objectives[2]
                            old = past_clustering_objectives[0] + last_two
                            new = last_two + objective
                            if old - new < clustering_tolerance * abs(old):
                                break
                            else:
                                past_clustering_objectives[0] = \
                                    past_clustering_objectives[1]
                                past_clustering_objectives[1] = \
                                    past_clustering_objectives[2]
                                past_clustering_objectives[2] = objective
                    else:
                        # Check for KeyboardInterrupts after each clustering
                        # iteration
                        with gil:
                            PyErr_CheckSignals()
                else:
                    # Check for KeyboardInterrupts after each clustering
                    # iteration
                    with gil:
                        PyErr_CheckSignals()

            # Apply the Harmony correction to the PCs to get the new Harmony
            # embeddings, `Z`. Precompute all correction matrices, `W`, then
            # apply all corrections in a single pass over `Z`, `PCs`, and `R`.

            # Initialize `inv_cov_2` to the identity matrix
            inv_cov_2[:] = 0
            for i in range(num_batches + 1):
                inv_cov_2[i, i] = 1

            for k in range(num_clusters):
                # Compute `inv_cov_1` and `inv_cov_2`, which will be multiplied
                # together to get the inverse covariance
                inv_cov_1[:] = 0
                norm = 0
                for i in range(num_batches):
                    if O[i, k] < <float> 1e-5 * N_b[i]:
                        inv_cov_2[num_batches, i] = 0
                    else:
                        ridge_lambda = E[i, k] * alpha
                        factor = <float> 1 / (O[i, k] + ridge_lambda)
                        inv_cov_1[i, i] = factor
                        factor *= -O[i, k]
                        inv_cov_2[num_batches, i] = factor
                        norm += O[i, k] * (<float> 1 + factor)
                norm = <float> 1 / norm
                inv_cov_1[num_batches, num_batches] = norm
                for i in range(num_batches):
                    inv_cov_1[num_batches, i] = \
                        inv_cov_2[num_batches, i] * norm

                # Compute the inverse covariance matrix
                # `inv_cov = inv_cov_1.T @ inv_cov_2`
                matrix_multiply(inv_cov_1, inv_cov_2, inv_cov,
                                transpose_A=True, transpose_B=False, alpha=1,
                                beta=0)

                # Compute `R`-scaled PCs in chunks; the last column
                # (`R_scaled_PCs[:, num_batches]`) stores the sum of the other
                # columns
                for chunk_index in prange(num_chunks,
                                          num_threads=num_threads_or_chunks):
                    start_row = chunk_index * chunk_size
                    end_row = min(start_row + chunk_size, num_cells)
                    R_scaled_PCs_chunk[chunk_index, :, :] = 0
                    for i in range(start_row, end_row):
                        batch_label = batch_labels[i]
                        for j in range(num_PCs):
                            R_scaled_PCs_chunk[
                                    chunk_index, j, batch_label] += \
                                PCs[i, j] * R[i, k]
                for j in prange(num_PCs, num_threads=num_threads):
                    total = 0
                    for batch_label in range(num_batches):
                        if O[batch_label, k] < <float> 1e-5 * N_b[batch_label]:
                            R_scaled_PCs[j, batch_label] = 0
                        else:
                            batch_total = 0
                            for chunk_index in range(num_chunks):
                                batch_total = batch_total + R_scaled_PCs_chunk[
                                    chunk_index, j, batch_label]
                            R_scaled_PCs[j, batch_label] = batch_total
                            total = total + batch_total
                    R_scaled_PCs[j, num_batches] = total

                # Compute `W[k] = inv_cov @ R_scaled_PCs.T`
                matrix_multiply(inv_cov, R_scaled_PCs, W[k],
                                transpose_A=False, transpose_B=True,
                                alpha=1, beta=0)

            # Apply all cluster corrections to `Z` in a single pass. For each
            # cell, initialize `Z` from `PCs` and subtract each cluster's
            # correction (`W[k]`) in order.
            for i in prange(num_cells, num_threads=num_threads):
                batch_label = batch_labels[i]
                for j in range(num_PCs):
                    Z[i, j] = PCs[i, j]
                for k in range(num_clusters):
                    for j in range(num_PCs):
                        Z[i, j] -= W[k, batch_label, j] * R[i, k]

            with gil:
                # If `verbose=True`, print the objective function
                if verbose:
                    metrics = (
                        f'objective = {objective:.2f} (k-means error = '
                        f'{kmeans_error:.2f}, entropy term = '
                        f'{entropy_term:.2f}, diversity penalty = '
                        f'{diversity_penalty:.2f})')
                    if max_iterations == 2_147_483_647:
                        print(f'Completed {iteration:,} '
                              f'iteration{"" if iteration == 1 else "s"}: '
                              f'{metrics}')
                    else:
                        print(f'Completed {iteration:,} of {max_iterations:,} '
                              f'iteration'
                              f'{"" if max_iterations == 1 else "s"}: '
                              f'{metrics}')

                # If Harmony converged, return
                if prev_objective - objective < \
                        tolerance * abs(prev_objective):
                    if verbose:
                        print(f'Reached convergence after {iteration:,} '
                              f'iteration{"" if iteration == 1 else "s"}')
                    return
                prev_objective = objective

                # Check for KeyboardInterrupts
                PyErr_CheckSignals()

            # Normalize each row of `Z` in-place
            normalize_rows_inplace_parallel(Z, num_threads)

    if verbose:
        print(f'Failed to converge after {max_iterations:,} '
              f'iteration{"" if max_iterations == 1 else "s"}')


def harmony_original(const float[:, ::1] PCs,
                     float[:, ::1] Z,
                     float[:, ::1] Y,
                     float[:, ::1] R,
                     const unsigned[::1] batch_labels,
                     const unsigned num_batches,
                     const unsigned max_iterations,
                     const unsigned max_clustering_iterations,
                     const float block_proportion,
                     const float tolerance,
                     const bint early_stopping,
                     const float clustering_tolerance,
                     const float theta,
                     const float tau,
                     const float alpha,
                     const float sigma,
                     const unsigned chunk_size,
                     const unsigned long long seed,
                     const bint verbose):
    # A version of `harmony()` that matches the original in its strategy for
    # updating `R`, `O`, and `E`: updates occur blockwise instead of chunkwise,
    # and the contribution of the entire block to `O` and `E` is subtracted
    # before updating `R`, then added back afterwards with the new `R`. Chunks
    # are still used for certain steps, but only to reduce memory usage and
    # cache misses, and never nested inside blocks.

    cdef unsigned long long num_cells = Z.shape[0], state = srand(seed)
    cdef unsigned i, j, k, chunk_index, chunk_start, chunk_end, batch_label, \
        iteration, clustering_iteration, block_index, block_start, block_end, \
        num_PCs = Z.shape[1], num_clusters = Y.shape[0], \
        num_chunks = (num_cells + chunk_size - 1) / chunk_size, \
        block_size = <unsigned> ceil(num_cells * block_proportion), \
        num_blocks = (num_cells + block_size - 1) / block_size
    cdef float base, kmeans_error, entropy_term, norm, Rij, \
        diversity_penalty, prev_objective, Pr_bi, Eij, Oij, Rkj, objective, \
        last_two, old, new, ridge_lambda, factor, total, \
        two_over_sigma = <float> 2 / sigma, \
        exp_neg_two_over_sigma = exp(-two_over_sigma)
    cdef float past_clustering_objectives[3]
    cdef str metrics
    cdef uninitialized_vector[unsigned] N_b_buffer, cell_order_buffer
    cdef uninitialized_vector[float] Pr_b_buffer, theta_batch_buffer, \
        cluster_diversity_penalty_buffer, R_sums_buffer, distance_buffer, \
        E_buffer, O_buffer, diversity_buffer, Z_block_buffer, \
        inv_cov_2_buffer, inv_cov_1_buffer, inv_cov_buffer, \
        R_scaled_PCs_buffer, W_buffer, ratio_buffer
    N_b_buffer.resize(num_batches)
    cell_order_buffer.resize(num_cells)
    Pr_b_buffer.resize(num_batches)
    theta_batch_buffer.resize(num_batches)
    cluster_diversity_penalty_buffer.resize(num_clusters)
    R_sums_buffer.resize(num_clusters)
    distance_buffer.resize(<unsigned long long> num_cells * num_clusters)
    E_buffer.resize(<unsigned long long> num_batches * num_clusters)
    O_buffer.resize(<unsigned long long> num_batches * num_clusters)
    diversity_buffer.resize(<unsigned long long> num_batches * num_clusters)
    Z_block_buffer.resize(<unsigned long long> block_size * num_PCs)
    ratio_buffer.resize(<unsigned long long> num_batches * num_clusters)
    inv_cov_2_buffer.resize(
        <unsigned long long> (num_batches + 1) * (num_batches + 1))
    inv_cov_1_buffer.resize(
        <unsigned long long> (num_batches + 1) * (num_batches + 1))
    inv_cov_buffer.resize(
        <unsigned long long> (num_batches + 1) * (num_batches + 1))
    R_scaled_PCs_buffer.resize(
        <unsigned long long> num_PCs * (num_batches + 1))
    W_buffer.resize(
        <unsigned long long> num_clusters * (num_batches + 1) * num_PCs)
    cdef unsigned[::1] \
        N_b = <unsigned[:num_batches]> N_b_buffer.data(), \
        cell_order = <unsigned[:num_cells]> cell_order_buffer.data()
    cdef float[::1] \
        Pr_b = <float[:num_batches]> Pr_b_buffer.data(), \
        theta_batch = <float[:num_batches]> theta_batch_buffer.data(), \
        cluster_diversity_penalty = <float[:num_clusters]> \
            cluster_diversity_penalty_buffer.data(), \
        R_sums = <float[:num_clusters]> R_sums_buffer.data()
    cdef float[:, ::1] \
        distances = <float[:chunk_size, :num_clusters]> \
            distance_buffer.data(), \
        E = <float[:num_batches, :num_clusters]> E_buffer.data(), \
        O = <float[:num_batches, :num_clusters]> O_buffer.data(), \
        diversity = <float[:num_batches, :num_clusters]> \
            diversity_buffer.data(), \
        Z_block = <float[:block_size, :num_PCs]> Z_block_buffer.data(), \
        ratio = <float[:num_batches, :num_clusters]> ratio_buffer.data(), \
        inv_cov_2 = <float[:num_batches + 1, :num_batches + 1]> \
            inv_cov_2_buffer.data(), \
        inv_cov_1 = <float[:num_batches + 1, :num_batches + 1]> \
            inv_cov_1_buffer.data(), \
        inv_cov = <float[:num_batches + 1, :num_batches + 1]> \
            inv_cov_buffer.data(), \
        R_scaled_PCs = <float[:num_PCs, :num_batches + 1]> \
            R_scaled_PCs_buffer.data()
    cdef float[:, :, ::1] \
        W = <float[:num_clusters, :num_batches + 1, :num_PCs]> W_buffer.data()

    # Get the number (`N_b`) and fraction (`Pr_b`) of cells in each batch;
    # apply discounting to `theta`, if `tau` is non-zero
    bin_count_nogil(batch_labels, N_b, num_threads=1)
    if tau:
        for i in range(num_batches):
            Pr_b[i] = <float> N_b[i] / num_cells
            base = exp(-(<float> N_b[i]) / (num_clusters * tau))
            theta_batch[i] = theta * (<float> 1 - base * base)
    else:
        for i in range(num_batches):
            Pr_b[i] = <float> N_b[i] / num_cells
            theta_batch[i] = theta
    N_b_buffer.clear()

    # Initialize `R`, `R_sums`, and `O` chunkwise. Compute the initial k-means
    # error and entropy term, the first two components of the objective
    # function.
    R_sums[:] = 0
    O[:] = 0
    kmeans_error = 0
    entropy_term = 0
    for chunk_index in range(num_chunks):
        chunk_start = chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_cells)
        matrix_multiply(Z[chunk_start:chunk_end], Y, distances,
                        transpose_A=False, transpose_B=True, alpha=1, beta=0)
        for i in range(chunk_start, chunk_end):
            norm = 0
            for j in range(num_clusters):
                Rij = exp(two_over_sigma *
                          (distances[i - chunk_start, j] - <float> 1))
                R[i, j] = Rij
                norm += Rij
            norm = <float> 1 / norm
            for j in range(num_clusters):
                batch_label = batch_labels[i]
                Rij = R[i, j]
                Rij *= norm
                R[i, j] = Rij
                R_sums[j] += Rij
                O[batch_label, j] += Rij
                kmeans_error += \
                    Rij * (<float> 1 - distances[i - chunk_start, j])
                entropy_term += Rij * log(Rij)
    kmeans_error *= <float> 2
    entropy_term *= sigma

    # Initialize `E`
    for i in range(num_batches):
        for j in range(num_clusters):
            E[i, j] = Pr_b[i] * R_sums[j]

    # Compute the initial diversity penalty, the third component of the
    # objective function
    diversity_penalty = 0
    for i in range(num_batches):
        for j in range(num_clusters):
            diversity[i, j] = \
                O[i, j] * log((O[i, j] + E[i, j] + <float> 1) /
                              (E[i, j] + <float> 1))
    matrix_vector_multiply(diversity, theta_batch, cluster_diversity_penalty,
                           transpose=True, alpha=1, beta=0)
    for i in range(num_clusters):
        diversity_penalty += cluster_diversity_penalty[i]
    diversity_penalty *= sigma

    # Compute the initial total objective function
    prev_objective = kmeans_error + entropy_term + diversity_penalty

    # Normalize each row of `Y` in-place
    normalize_rows_inplace(Y)

    # Define the random order to iterate over cells in, via the "inside-out"
    # variant of the Fisher-Yates shuffle. The uninitialized variable is
    # intentional!
    for i in range(num_cells):
        j = randint(i + 1, &state)
        cell_order[i] = cell_order[j]
        cell_order[j] = i

    if verbose:
        print(f'Initialization is complete: objective = {prev_objective:.2f}')

    # Check for KeyboardInterrupts
    PyErr_CheckSignals()

    # Now that initialization is done, start the Harmony iterations
    for iteration in range(1, max_iterations + 1):
        # Perform `max_clustering_iterations` iterations of clustering
        # within each Harmony iteration, stopping early if
        # `early_stopping=True` and convergence is met
        for clustering_iteration in range(1, max_clustering_iterations + 1):
            # Compute `Y`, the normalized cluster centroids
            matrix_multiply(R, Z, Y, transpose_A=True, transpose_B=False,
                            alpha=1, beta=0)
            for i in range(num_clusters):
                norm = 0
                for j in range(num_PCs):
                    norm += Y[i, j] * Y[i, j]
                norm = <float> 1 / sqrt(norm)
                for j in range(num_PCs):
                    Y[i, j] *= norm

            # Update `R`, `E`, and `O` by processing cells blockwise. Note that
            # the full formula for `R` is:
            # `((E + 1) / (O + 1)) ** theta * exp(-2 / sigma * (1 - Z @ Y.T))`
            # which we can calculate as `ratio * exp(distances)` where
            # `ratio = exp(-2 / sigma) * ((E + 1) / (O + 1)) ** theta` and
            # `distances = 2 / sigma * Z @ Y.T`. Calculating `R` this way
            # saves a few multiplications.
            for block_index in range(num_blocks):
                block_start = block_index * block_size
                block_end = min(block_start + block_size, num_cells)

                # Remove the cells in this block from `E` and `O`. Copy `Z` for
                # these cells into a contiguous array, `Z_block`.
                R_sums[:] = 0
                for i in range(block_end - block_start):
                    k = cell_order[block_start + i]
                    batch_label = batch_labels[k]
                    for j in range(num_clusters):
                        O[batch_label, j] -= R[k, j]
                        R_sums[j] += R[k, j]
                    for j in range(num_PCs):
                        Z_block[i, j] = Z[k, j]
                for i in range(num_batches):
                    Pr_bi = Pr_b[i]
                    for j in range(num_clusters):
                        E[i, j] -= Pr_bi * R_sums[j]

                # Update `R`, the fractional soft clustering assignment of each
                # cell to each cluster, for the cells in the block. Normalize
                # each row of `R` to ensure it always sums to 1 despite
                # floating-point error. Add the removed cells back into `O` and
                # `E`.
                matrix_multiply(Z_block[:block_end - block_start], Y,
                                distances, transpose_A=False, transpose_B=True,
                                alpha=two_over_sigma, beta=0)
                for i in range(num_batches):
                    for j in range(num_clusters):
                        Eij = E[i, j]
                        Oij = O[i, j]
                        ratio[i, j] = exp_neg_two_over_sigma * \
                            pow((Eij + <float> 1) / (Oij + Eij + <float> 1),
                                theta_batch[i])
                R_sums[:] = 0
                for i in range(block_end - block_start):
                    k = cell_order[block_start + i]
                    batch_label = batch_labels[k]
                    norm = 0
                    for j in range(num_clusters):
                        Rkj = exp(distances[i, j]) * ratio[batch_label, j]
                        R[k, j] = Rkj
                        norm += Rkj
                    norm = <float> 1 / norm
                    for j in range(num_clusters):
                        Rkj = R[k, j]
                        Rkj *= norm
                        R[k, j] = Rkj
                        R_sums[j] += Rkj
                        O[batch_label, j] += Rkj
                for i in range(num_batches):
                    Pr_bi = Pr_b[i]
                    for j in range(num_clusters):
                        E[i, j] += Pr_bi * R_sums[j]

            # Check for KeyboardInterrupts after each clustering
            # iteration
            PyErr_CheckSignals()

            # Compute the objective function, if we are done clustering or
            # `early_stopping=True`
            if early_stopping or \
                    clustering_iteration == max_clustering_iterations:
                # Compute its first two components chunkwise: the k-means
                # error and entropy term
                kmeans_error = 0
                entropy_term = 0
                for chunk_index in range(num_chunks):
                    chunk_start = chunk_index * chunk_size
                    chunk_end = min(chunk_start + chunk_size, num_cells)
                    matrix_multiply(Z[chunk_start:chunk_end], Y, distances,
                                    transpose_A=False, transpose_B=True,
                                    alpha=1, beta=0)
                    for i in range(chunk_start, chunk_end):
                        for j in range(num_clusters):
                            kmeans_error += R[i, j] * \
                                (<float> 1 - distances[i - chunk_start, j])
                            entropy_term += R[i, j] * log(R[i, j])
                kmeans_error *= <float> 2
                entropy_term *= sigma

                # Compute the diversity penalty, the third component of the
                # objective function
                diversity_penalty = 0
                for i in range(num_batches):
                    for j in range(num_clusters):
                        diversity[i, j] = O[i, j] * log(
                            (O[i, j] + E[i, j] + <float> 1) /
                            (E[i, j] + <float> 1))
                matrix_vector_multiply(diversity, theta_batch,
                                       cluster_diversity_penalty,
                                       transpose=True, alpha=1, beta=0)
                for i in range(num_clusters):
                    diversity_penalty += cluster_diversity_penalty[i]
                diversity_penalty *= sigma

                # Compute the total objective function
                objective = kmeans_error + entropy_term + diversity_penalty

                if early_stopping:
                    # If `verbose=True`, print the objective function
                    if verbose:
                        print(f'Clustering iteration '
                              f'{clustering_iteration:,}: objective = '
                              f'{objective:.2f} (k-means error = '
                              f'{kmeans_error:.2f}, entropy term = '
                              f'{entropy_term:.2f}, diversity '
                              f'penalty = {diversity_penalty:.2f})')

                    # Exit early if the clustering converged, based on a
                    # sliding window average of the objective over the past
                    # three iterations
                    if clustering_iteration <= 3:
                        past_clustering_objectives[
                            clustering_iteration - 1] = objective
                    else:
                        last_two = past_clustering_objectives[1] + \
                            past_clustering_objectives[2]
                        old = past_clustering_objectives[0] + last_two
                        new = last_two + objective
                        if old - new < clustering_tolerance * abs(old):
                            break
                        else:
                            past_clustering_objectives[0] = \
                                past_clustering_objectives[1]
                            past_clustering_objectives[1] = \
                                past_clustering_objectives[2]
                            past_clustering_objectives[2] = objective

        # Apply the Harmony correction to the PCs to get the new Harmony
        # embeddings, `Z`. Precompute all correction matrices, `W`, then apply
        # all corrections in a single pass over `Z`, `PCs`, and `R`.

        # Initialize `inv_cov_2` to the identity matrix
        inv_cov_2[:] = 0
        for i in range(num_batches + 1):
            inv_cov_2[i, i] = 1

        with nogil:
            for k in range(num_clusters):
                # Compute `inv_cov_1` and `inv_cov_2`, which will be multiplied
                # together to get the inverse covariance
                inv_cov_1[:] = 0
                norm = 0
                for i in range(num_batches):
                    if O[i, k] < <float> 1e-5 * N_b[i]:
                        inv_cov_2[num_batches, i] = 0
                    else:
                        ridge_lambda = E[i, k] * alpha
                        factor = <float> 1 / (O[i, k] + ridge_lambda)
                        inv_cov_1[i, i] = factor
                        factor *= -O[i, k]
                        inv_cov_2[num_batches, i] = factor
                        norm += O[i, k] * (<float> 1 + factor)
                norm = <float> 1 / norm
                inv_cov_1[num_batches, num_batches] = norm
                for i in range(num_batches):
                    inv_cov_1[num_batches, i] = \
                        inv_cov_2[num_batches, i] * norm

                # Compute the inverse covariance matrix
                # `inv_cov = inv_cov_1.T @ inv_cov_2`
                matrix_multiply(inv_cov_1, inv_cov_2, inv_cov,
                                transpose_A=True, transpose_B=False, alpha=1,
                                beta=0)

                # Compute `R`-scaled PCs; the last column
                # (`R_scaled_PCs[:, num_batches]`) stores the sum of the other
                # columns
                R_scaled_PCs[:] = 0
                for i in range(num_cells):
                    batch_label = batch_labels[i]
                    for j in range(num_PCs):
                        R_scaled_PCs[j, batch_label] += R[i, k] * PCs[i, j]
                for j in range(num_PCs):
                    total = 0
                    for batch_label in range(num_batches):
                        if O[batch_label, k] < <float> 1e-5 * N_b[batch_label]:
                            R_scaled_PCs[j, batch_label] = 0
                        else:
                            total += R_scaled_PCs[j, batch_label]
                    R_scaled_PCs[j, num_batches] = total

                # Compute `W[k] = inv_cov @ R_scaled_PCs.T`
                matrix_multiply(inv_cov, R_scaled_PCs, W[k],
                                transpose_A=False, transpose_B=True,
                                alpha=1, beta=0)

            # Apply all cluster corrections to `Z` in a single pass. For each
            # cell, initialize `Z` from `PCs` and subtract each cluster's
            # correction (`W[k]`) in order.
            for i in range(num_cells):
                batch_label = batch_labels[i]
                for j in range(num_PCs):
                    Z[i, j] = PCs[i, j]
                for k in range(num_clusters):
                    for j in range(num_PCs):
                        Z[i, j] -= W[k, batch_label, j] * R[i, k]

        # If `verbose=True`, print the objective function
        if verbose:
            metrics = (
                f'objective = {objective:.2f} (k-means error = '
                f'{kmeans_error:.2f}, entropy term = {entropy_term:.2f}, '
                f'diversity penalty = {diversity_penalty:.2f})')
            if max_iterations == 2_147_483_647:
                print(f'Completed {iteration:,} '
                      f'iteration{"" if iteration == 1 else "s"}: {metrics}')
            else:
                print(f'Completed {iteration:,} of {max_iterations:,} '
                      f'iteration{"" if max_iterations == 1 else "s"}: '
                      f'{metrics}')

        # If Harmony converged, return
        if prev_objective - objective < tolerance * abs(prev_objective):
            if verbose:
                print(f'Reached convergence after {iteration:,} iteration'
                      f'{"" if iteration == 1 else "s"}')
            return
        prev_objective = objective

        # Check for KeyboardInterrupts
        PyErr_CheckSignals()

        # Normalize each row of `Z` in-place
        normalize_rows_inplace(Z)
    if verbose:
        print(f'Failed to converge after {max_iterations:,} '
              f'iteration{"" if max_iterations == 1 else "s"}')