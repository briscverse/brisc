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
                   unsigned num_threads,
                   const bint is_mac):
    cdef unsigned i, j, num_rows = arr.shape[0], num_columns = arr.shape[1]
    cdef float norm

    # Force the parallel path for Mac for floating-point consistency between 1
    # and multiple threads

    num_threads = min(num_threads, num_rows)
    if num_threads <= 1 and not is_mac:
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
    # sharing, since `chunk_size` is a multiple of 16 whenever there is more
    # than one chunk, so each thread's slice spans whole 64-byte cache lines.

    cdef unsigned long long num_cells = Z.shape[0], state = srand(seed)
    cdef unsigned i, j, k, thread_index, chunk_index, chunk_start, chunk_end, \
        iteration, clustering_iteration, block_index, block_start, block_end, \
        num_chunks_per_block, batch_label, start_row, end_row, slot, slab, \
        piece, num_pieces, T_start, num_Y_partials, \
        num_PCs = Z.shape[1], num_clusters = Y.shape[0], \
        num_chunks = (num_cells + chunk_size - 1) / chunk_size, \
        block_size = <unsigned> ceil(num_cells * block_proportion), \
        num_blocks = (num_cells + block_size - 1) / block_size, \
        max_chunks_per_block = (block_size + chunk_size - 1) / chunk_size, \
        num_block_chunks = (num_blocks - 1) * max_chunks_per_block + \
            (num_cells - (num_blocks - 1) * block_size + chunk_size - 1) / \
            chunk_size, \
        num_threads_or_chunks = min(num_threads, num_chunks), \
        num_slabs = min(num_chunks, max(<unsigned long long> 1, num_cells / (
            <unsigned long long> num_batches * num_clusters))), \
        max_pieces = num_chunks + min(
            <unsigned long long> num_slabs * num_batches, num_cells)
    cdef float base, kmeans_error, entropy_term, norm, Rij, \
        diversity_penalty, prev_objective, total, Pr_bi, Eij, Oij, Rkj, \
        objective, last_two, old, new, ridge_lambda, factor, beta, \
        two_over_sigma = <float> 2 / sigma, half_sigma = sigma / <float> 2, \
        exp_neg_two_over_sigma = exp(-two_over_sigma)
    cdef float past_clustering_objectives[3]
    cdef bint compute_objective, compute_Y, converged, cold_start
    cdef str metrics
    cdef uninitialized_vector[unsigned] N_b_buffer, cell_order_buffer, \
        slab_starts_buffer, slab_counts_buffer, batch_cursors_buffer, \
        slab_piece_offsets_buffer, piece_batches_buffer, piece_starts_buffer, \
        batch_order_buffer
    cdef uninitialized_vector[float] Pr_b_buffer, theta_batch_buffer, \
        cluster_diversity_penalty_buffer, E_buffer, O_buffer, \
        diversity_buffer, R_sums_buffer, intercepts_buffer, W_buffer, \
        O_slab_buffer, Y_chunk_buffer, Z_chunk_buffer, distance_buffer, \
        delta_O_buffer, \
        ratio_buffer, kmeans_errors, entropy_terms
    N_b_buffer.resize(num_batches)
    cell_order_buffer.resize(num_cells)
    slab_starts_buffer.resize(num_slabs + 1)
    slab_counts_buffer.resize(<unsigned long long> num_slabs * num_batches)
    batch_cursors_buffer.resize(<unsigned long long> num_threads * num_batches)
    slab_piece_offsets_buffer.resize(num_slabs + 1)
    piece_batches_buffer.resize(max_pieces)
    piece_starts_buffer.resize(max_pieces + 1)
    batch_order_buffer.resize(num_cells)
    Pr_b_buffer.resize(num_batches)
    theta_batch_buffer.resize(num_batches)
    cluster_diversity_penalty_buffer.resize(num_clusters)
    E_buffer.resize(<unsigned long long> num_batches * num_clusters)
    O_buffer.resize(<unsigned long long> num_batches * num_clusters)
    diversity_buffer.resize(<unsigned long long> num_batches * num_clusters)
    R_sums_buffer.resize(
        <unsigned long long> max_chunks_per_block * num_clusters)
    intercepts_buffer.resize(<unsigned long long> num_clusters * num_PCs)
    W_buffer.resize(
        <unsigned long long> num_batches * num_clusters * num_PCs)
    Y_chunk_buffer.resize(
        <unsigned long long> num_block_chunks * num_clusters * num_PCs)
    O_slab_buffer.resize(
        <unsigned long long> num_slabs * num_batches * num_clusters)
    Z_chunk_buffer.resize(
        <unsigned long long> num_threads * chunk_size * num_PCs)
    distance_buffer.resize(
        <unsigned long long> num_threads * chunk_size * num_clusters)
    delta_O_buffer.resize(
        <unsigned long long> max_chunks_per_block * num_batches * num_clusters)
    ratio_buffer.resize(
        <unsigned long long> max_chunks_per_block * num_batches * num_clusters)
    kmeans_errors.resize(num_block_chunks)
    entropy_terms.resize(num_block_chunks)
    cdef unsigned[::1] \
        N_b = <unsigned[:num_batches]> N_b_buffer.data(), \
        cell_order = <unsigned[:num_cells]> cell_order_buffer.data(), \
        slab_starts = <unsigned[:num_slabs + 1]> slab_starts_buffer.data(), \
        slab_piece_offsets = <unsigned[:num_slabs + 1]> \
            slab_piece_offsets_buffer.data(), \
        piece_batches = <unsigned[:max_pieces]> piece_batches_buffer.data(), \
        piece_starts = <unsigned[:max_pieces + 1]> \
            piece_starts_buffer.data(), \
        batch_order = <unsigned[:num_cells]> batch_order_buffer.data()
    cdef unsigned[:, ::1] \
        slab_counts = <unsigned[:num_slabs, :num_batches]> \
            slab_counts_buffer.data(), \
        batch_cursors = <unsigned[:num_threads, :num_batches]> \
            batch_cursors_buffer.data()
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
        R_sums = <float[:max_chunks_per_block, :num_clusters]> \
            R_sums_buffer.data(), \
        intercepts = <float[:num_clusters, :num_PCs]> intercepts_buffer.data()
    cdef float[:, :, ::1] \
        O_slab = <float[:num_slabs, :num_batches, :num_clusters]> \
            O_slab_buffer.data(), \
        Y_chunk = <float[:num_block_chunks, :num_clusters, :num_PCs]> \
            Y_chunk_buffer.data(), \
        Z_chunk = <float[:num_threads, :chunk_size, :num_PCs]> \
            Z_chunk_buffer.data(), \
        distances = <float[:num_threads, :chunk_size, :num_clusters]> \
            distance_buffer.data(), \
        delta_O = <float[:max_chunks_per_block, :num_batches, :num_clusters]> \
            delta_O_buffer.data(), \
        ratio = <float[:max_chunks_per_block, :num_batches, :num_clusters]> \
            ratio_buffer.data(), \
        W = <float[:num_batches, :num_clusters, :num_PCs]> W_buffer.data()

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

        # For the Harmony correction, divide the cells into `num_slabs` slabs
        # of consecutive cells, a partition separate from the blocks and
        # chunks used for clustering. Unless there is only one slab, each
        # slab has at least `num_batches * num_clusters` cells, so it can
        # store its partial sums in its own rows of `Z` (see below). Sort each
        # slab's cells by batch (`batch_order`) and divide each batch's cells
        # within each slab into pieces of at most `chunk_size` cells. Like the
        # chunks when `R` is first touched, slabs are divided among threads in
        # contiguous runs, so each slab's memory accesses are NUMA-local.
        # `num_slabs` does not depend on `num_threads`, so neither do the
        # results.
        for slab in range(num_slabs + 1):
            slab_starts[slab] = slab * num_cells / num_slabs
        for slab in prange(num_slabs, num_threads=num_threads_or_chunks):
            for i in range(num_batches):
                slab_counts[slab, i] = 0
            for i in range(slab_starts[slab], slab_starts[slab + 1]):
                slab_counts[slab, batch_labels[i]] += 1
        num_pieces = 0
        for slab in range(num_slabs):
            slab_piece_offsets[slab] = num_pieces
            start_row = slab_starts[slab]
            for i in range(num_batches):
                for k in range((slab_counts[slab, i] + chunk_size - 1) /
                               chunk_size):
                    piece_batches[num_pieces] = i
                    piece_starts[num_pieces] = start_row + k * chunk_size
                    num_pieces += 1
                start_row += slab_counts[slab, i]
        slab_piece_offsets[num_slabs] = num_pieces
        piece_starts[num_pieces] = num_cells
        with parallel(num_threads=num_threads_or_chunks):
            thread_index = threadid()
            for slab in prange(num_slabs):
                start_row = slab_starts[slab]
                for i in range(num_batches):
                    batch_cursors[thread_index, i] = start_row
                    start_row = start_row + slab_counts[slab, i]
                for i in range(slab_starts[slab], slab_starts[slab + 1]):
                    batch_label = batch_labels[i]
                    batch_order[batch_cursors[thread_index, batch_label]] = i
                    batch_cursors[thread_index, batch_label] += 1

        # Normalize each row of `Y` in-place
        normalize_rows_inplace_parallel(Y, num_threads)

        # Initialize `R`, `O`, and `E` from `Z` and `Y` slab by slab, the same
        # way as the cold start before each later round of clustering. Also
        # compute the initial k-means error and entropy term, the first two
        # components of the objective function, and the partial sum of
        # `R.T @ Z` from which the first round of clustering computes `Y`, for
        # each slab.
        with parallel(num_threads=num_threads_or_chunks):
            thread_index = threadid()
            for slab in prange(num_slabs):
                kmeans_error = 0
                entropy_term = 0
                for piece in range(slab_piece_offsets[slab],
                                   slab_piece_offsets[slab + 1]):
                    batch_label = piece_batches[piece]
                    start_row = piece_starts[piece]
                    end_row = piece_starts[piece + 1]
                    for i in range(end_row - start_row):
                        k = batch_order[start_row + i]
                        for j in range(num_PCs):
                            Z_chunk[thread_index, i, j] = Z[k, j]
                    matrix_multiply(
                        Z_chunk[thread_index, :end_row - start_row], Y,
                        distances[thread_index], transpose_A=False,
                        transpose_B=True, alpha=two_over_sigma, beta=0)
                    if piece == slab_piece_offsets[slab] or \
                            piece_batches[piece - 1] != batch_label:
                        for j in range(num_clusters):
                            O_slab[slab, batch_label, j] = 0
                    for i in range(end_row - start_row):
                        k = batch_order[start_row + i]
                        norm = 0
                        for j in range(num_clusters):
                            Rij = exp(distances[thread_index, i, j] -
                                      two_over_sigma)
                            R[k, j] = Rij
                            norm = norm + Rij
                        norm = <float> 1 / norm
                        for j in range(num_clusters):
                            Rij = R[k, j] * norm
                            R[k, j] = Rij
                            O_slab[slab, batch_label, j] += Rij
                            kmeans_error = kmeans_error + Rij * (
                                <float> 1 - half_sigma *
                                distances[thread_index, i, j])
                            entropy_term = entropy_term + Rij * log(Rij)
                            distances[thread_index, i, j] = Rij
                    beta = piece > slab_piece_offsets[slab]
                    matrix_multiply(
                        distances[thread_index, :end_row - start_row],
                        Z_chunk[thread_index, :end_row - start_row],
                        Y_chunk[slab], transpose_A=True, transpose_B=False,
                        alpha=1, beta=beta)
                kmeans_errors[slab] = kmeans_error
                entropy_terms[slab] = entropy_term
        num_Y_partials = num_slabs

        # Sum each batch's contributions to `O` across slabs, skipping slabs
        # without any of the batch's cells, then compute `E` from `O`'s column
        # sums
        for i in prange(num_batches * num_clusters,
                        num_threads=num_threads_or_chunks):
            batch_label = i / num_clusters
            k = i % num_clusters
            total = 0
            for slab in range(num_slabs):
                if slab_counts[slab, batch_label] > 0:
                    total = total + O_slab[slab, batch_label, k]
            O[batch_label, k] = total
        for k in prange(num_clusters, num_threads=num_threads_or_chunks):
            total = 0
            for i in range(num_batches):
                total = total + O[i, k]
            for i in range(num_batches):
                E[i, k] = Pr_b[i] * total

        # Aggregate the initial k-means error and entropy term across slabs
        kmeans_error = 0
        entropy_term = 0
        for slab in range(num_slabs):
            kmeans_error += kmeans_errors[slab]
            entropy_term += entropy_terms[slab]
        kmeans_error *= <float> 2
        entropy_term *= sigma

        # Compute the initial diversity penalty, the third component of the
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

        # Compute the initial total objective function
        prev_objective = kmeans_error + entropy_term + diversity_penalty

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

    # Now that initialization is done, start the Harmony iterations
    with nogil:
        for iteration in range(1, max_iterations + 1):
            # Perform `max_clustering_iterations` iterations of clustering
            # within each Harmony iteration, stopping early if
            # `early_stopping=True` and convergence is met
            for clustering_iteration in range(
                    1, max_clustering_iterations + 1):
                # Compute `Y`, the normalized cluster centroids, by summing the
                # partial sums of `R.T @ Z` from the last pass over the cells
                for i in prange(num_clusters, num_threads=num_threads):
                    norm = 0
                    for j in range(num_PCs):
                        total = 0
                        for chunk_index in range(num_Y_partials):
                            total = total + Y_chunk[chunk_index, i, j]
                        Y[i, j] = total
                        norm = norm + total * total
                    norm = <float> 1 / sqrt(norm)
                    for j in range(num_PCs):
                        Y[i, j] *= norm

                # Update `R`, `E`, and `O` by processing cells blockwise and
                # then chunkwise within each block. Note that the full formula
                # for `R` is: `((E + 1) / (O + E + 1)) ** theta *
                #              exp(-2 / sigma * (1 - Z @ Y.T))`
                # which we can calculate as `ratio * exp(distances)` where
                # `ratio = exp(-2 / sigma) *
                #          ((E + 1) / (O + E + 1)) ** theta` and
                # `distances = 2 / sigma * Z @ Y.T`. Calculating `R` this way
                # saves a few multiplications.
                #
                # If we are done clustering or `early_stopping=True`, also
                # compute the first two components of the objective function,
                # the k-means error and entropy term, for each chunk. Unless
                # this is the last clustering iteration, also compute each
                # chunk's partial sum of `R.T @ Z` for the next `Y`.
                compute_objective = early_stopping or \
                    clustering_iteration == max_clustering_iterations
                compute_Y = clustering_iteration < max_clustering_iterations
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
                            slot = block_index * max_chunks_per_block + \
                                chunk_index

                            # Get the chunk of `Z`, the normalized PCs, to
                            # process. Calculate the observed-to-expected ratio
                            # `ratio = exp(-2 / sigma) *
                            #          ((E + 1) / (O + E + 1)) ** theta`
                            # for the chunk, subtracting the chunk's own
                            # contributions to `E` and `O`: `Pr_b` times
                            # `R_sums[chunk_index]` and `delta_O[chunk_index]`,
                            # respectively.
                            for j in range(num_clusters):
                                R_sums[chunk_index, j] = 0
                            for i in range(num_batches):
                                for j in range(num_clusters):
                                    delta_O[chunk_index, i, j] = 0
                            for i in range(chunk_end - chunk_start):
                                k = cell_order[block_start + chunk_start + i]
                                batch_label = batch_labels[k]
                                for j in range(num_PCs):
                                    Z_chunk[thread_index, i, j] = Z[k, j]
                                for j in range(num_clusters):
                                    R_sums[chunk_index, j] += R[k, j]
                                    delta_O[chunk_index, batch_label, j] -= \
                                        R[k, j]
                            for i in range(num_batches):
                                Pr_bi = Pr_b[i]
                                for j in range(num_clusters):
                                    Eij = E[i, j] - \
                                        Pr_bi * R_sums[chunk_index, j]
                                    Oij = O[i, j] + delta_O[chunk_index, i, j]
                                    ratio[chunk_index, i, j] = \
                                        exp_neg_two_over_sigma * \
                                        pow((Eij + <float> 1) /
                                            (Oij + Eij + <float> 1),
                                            theta_batch[i])

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
                            # to `O` to `delta_O[chunk_index]` using the
                            # newly-updated `R`, and replace
                            # `R_sums[chunk_index]` with the change in the
                            # chunk's contribution to `R`'s column sums.
                            for j in range(num_clusters):
                                R_sums[chunk_index, j] = \
                                    -R_sums[chunk_index, j]
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
                            if compute_objective:
                                kmeans_error = 0
                                entropy_term = 0
                                for i in range(chunk_end - chunk_start):
                                    k = cell_order[
                                        block_start + chunk_start + i]
                                    for j in range(num_clusters):
                                        Rkj = R[k, j]
                                        kmeans_error = kmeans_error + Rkj * (
                                            <float> 1 - half_sigma *
                                            distances[thread_index, i, j])
                                        entropy_term = \
                                            entropy_term + Rkj * log(Rkj)
                                kmeans_errors[slot] = kmeans_error
                                entropy_terms[slot] = entropy_term
                            if compute_Y:
                                for i in range(chunk_end - chunk_start):
                                    k = cell_order[
                                        block_start + chunk_start + i]
                                    for j in range(num_clusters):
                                        distances[thread_index, i, j] = R[k, j]
                                matrix_multiply(
                                    distances[thread_index,
                                              :chunk_end - chunk_start],
                                    Z_chunk[thread_index,
                                            :chunk_end - chunk_start],
                                    Y_chunk[slot], transpose_A=True,
                                    transpose_B=False, alpha=1, beta=0)

                    # Update `O` and `E` for this block. `E` changes by `Pr_b`
                    # times the change in `R`'s column sums, which we total in
                    # `R_sums[0]`.
                    for j in range(num_clusters):
                        for chunk_index in range(1, num_chunks_per_block):
                            R_sums[0, j] += R_sums[chunk_index, j]
                    for i in prange(num_batches,
                                    num_threads=num_threads_or_chunks):
                        for chunk_index in range(num_chunks_per_block):
                            for j in range(num_clusters):
                                O[i, j] += delta_O[chunk_index, i, j]
                        for j in range(num_clusters):
                            E[i, j] += Pr_b[i] * R_sums[0, j]
                if compute_Y:
                    num_Y_partials = num_block_chunks

                # Compute the objective function, if we are done clustering or
                # `early_stopping=True`
                if compute_objective:
                    # Aggregate the k-means error and entropy term across
                    # chunks
                    kmeans_error = 0
                    entropy_term = 0
                    for chunk_index in range(num_block_chunks):
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

            # If Harmony has converged, this is the last iteration, so the
            # embeddings are not normalized. Unless this is the last iteration,
            # recompute `R`, `O`, and `E` from the corrected embeddings before
            # the next round of clustering (a "cold start"), as Harmony2 does.
            converged = prev_objective - objective < \
                tolerance * abs(prev_objective)
            cold_start = not converged and iteration < max_iterations

            # Apply the Harmony correction to the PCs to get the new Harmony
            # embeddings, `Z`. For each cluster `k`, the correction is a ridge
            # regression of the PCs on an intercept plus one indicator per
            # batch, with each cell weighted by `R[:, k]` and each batch `b`'s
            # coefficient penalized by `lambda = alpha * E[b, k]`. Batches with
            # `O[b, k] < 1e-5 * N_b[b]` are excluded. Rather than inverting the
            # arrowhead-structured regression matrix like Harmony2, we compute
            # the solution directly. With `f = 1 / (O[b, k] + lambda)` and
            # `T[b, k] = R[:, k] @ PCs` summed over the cells in batch `b`:
            #   `intercepts[k] = sum_b(lambda * f * T[b, k]) /
            #                    sum_b(lambda * f * O[b, k])`
            #   `W[b, k] = f * (T[b, k] - O[b, k] * intercepts[k])`
            # and each cell `i` in batch `b` is corrected to
            # `Z[i] = PCs[i] - R[i] @ W[b]`.

            # Compute each slab's partial sums of `T`. The first slab stores
            # them in `W`; the others store them in their own first
            # `num_batches * num_clusters` rows of `Z`, which are free until
            # `Z` is recomputed below.
            with parallel(num_threads=num_threads_or_chunks):
                thread_index = threadid()
                for slab in prange(num_slabs):
                    for piece in range(slab_piece_offsets[slab],
                                       slab_piece_offsets[slab + 1]):
                        batch_label = piece_batches[piece]
                        start_row = piece_starts[piece]
                        end_row = piece_starts[piece + 1]
                        for i in range(end_row - start_row):
                            k = batch_order[start_row + i]
                            for j in range(num_PCs):
                                Z_chunk[thread_index, i, j] = PCs[k, j]
                            for j in range(num_clusters):
                                distances[thread_index, i, j] = R[k, j]
                        beta = piece > slab_piece_offsets[slab] and \
                            piece_batches[piece - 1] == batch_label
                        if slab == 0:
                            matrix_multiply(
                                distances[thread_index, :end_row - start_row],
                                Z_chunk[thread_index, :end_row - start_row],
                                W[batch_label], transpose_A=True,
                                transpose_B=False, alpha=1, beta=beta)
                        else:
                            T_start = slab_starts[slab] + \
                                batch_label * num_clusters
                            matrix_multiply(
                                distances[thread_index, :end_row - start_row],
                                Z_chunk[thread_index, :end_row - start_row],
                                Z[T_start:T_start + num_clusters],
                                transpose_A=True, transpose_B=False, alpha=1,
                                beta=beta)

            # Sum each batch's partial sums of `T` across slabs, skipping slabs
            # without any of the batch's cells
            for i in prange(num_batches * num_clusters,
                            num_threads=num_threads_or_chunks):
                batch_label = i / num_clusters
                k = i % num_clusters
                if slab_counts[0, batch_label] == 0:
                    for j in range(num_PCs):
                        W[batch_label, k, j] = 0
                for slab in range(1, num_slabs):
                    if slab_counts[slab, batch_label] > 0:
                        for j in range(num_PCs):
                            W[batch_label, k, j] += Z[slab_starts[slab] + i, j]

            # Compute `intercepts`
            for k in prange(num_clusters, num_threads=num_threads_or_chunks):
                for j in range(num_PCs):
                    intercepts[k, j] = 0
                norm = 0
                for i in range(num_batches):
                    if O[i, k] >= <float> 1e-5 * N_b[i]:
                        ridge_lambda = E[i, k] * alpha
                        factor = ridge_lambda / (O[i, k] + ridge_lambda)
                        norm = norm + factor * O[i, k]
                        for j in range(num_PCs):
                            intercepts[k, j] += factor * W[i, k, j]
                if norm > <float> 0:
                    norm = <float> 1 / norm
                    for j in range(num_PCs):
                        intercepts[k, j] *= norm

            # Compute `W` from `T`
            for i in prange(num_batches, num_threads=num_threads_or_chunks):
                for k in range(num_clusters):
                    if O[i, k] < <float> 1e-5 * N_b[i]:
                        for j in range(num_PCs):
                            W[i, k, j] = 0
                    else:
                        factor = <float> 1 / (O[i, k] + E[i, k] * alpha)
                        for j in range(num_PCs):
                            W[i, k, j] = factor * (
                                W[i, k, j] - O[i, k] * intercepts[k, j])

            # Correct `Z` slab by slab. Unless Harmony has converged, also
            # normalize each row of `Z`. For a cold start, also recompute `R`,
            # each slab's contributions to `O`, and each slab's partial sum of
            # `R.T @ Z` for the next `Y`.
            with parallel(num_threads=num_threads_or_chunks):
                thread_index = threadid()
                for slab in prange(num_slabs):
                    for piece in range(slab_piece_offsets[slab],
                                       slab_piece_offsets[slab + 1]):
                        batch_label = piece_batches[piece]
                        start_row = piece_starts[piece]
                        end_row = piece_starts[piece + 1]
                        for i in range(end_row - start_row):
                            k = batch_order[start_row + i]
                            for j in range(num_PCs):
                                Z_chunk[thread_index, i, j] = PCs[k, j]
                            for j in range(num_clusters):
                                distances[thread_index, i, j] = R[k, j]
                        matrix_multiply(
                            distances[thread_index, :end_row - start_row],
                            W[batch_label], Z_chunk[thread_index],
                            transpose_A=False, transpose_B=False, alpha=-1,
                            beta=1)
                        if not converged:
                            for i in range(end_row - start_row):
                                norm = 0
                                for j in range(num_PCs):
                                    norm = norm + \
                                        Z_chunk[thread_index, i, j] * \
                                        Z_chunk[thread_index, i, j]
                                norm = <float> 1 / sqrt(norm)
                                for j in range(num_PCs):
                                    Z_chunk[thread_index, i, j] = \
                                        Z_chunk[thread_index, i, j] * norm
                        if cold_start:
                            matrix_multiply(
                                Z_chunk[thread_index, :end_row - start_row],
                                Y, distances[thread_index], transpose_A=False,
                                transpose_B=True, alpha=two_over_sigma, beta=0)
                            if piece == slab_piece_offsets[slab] or \
                                    piece_batches[piece - 1] != batch_label:
                                for j in range(num_clusters):
                                    O_slab[slab, batch_label, j] = 0
                            for i in range(end_row - start_row):
                                k = batch_order[start_row + i]
                                norm = 0
                                for j in range(num_clusters):
                                    Rij = exp(distances[thread_index, i, j] -
                                              two_over_sigma)
                                    R[k, j] = Rij
                                    norm = norm + Rij
                                norm = <float> 1 / norm
                                for j in range(num_clusters):
                                    Rij = R[k, j] * norm
                                    R[k, j] = Rij
                                    O_slab[slab, batch_label, j] += Rij
                                    distances[thread_index, i, j] = Rij
                            beta = piece > slab_piece_offsets[slab]
                            matrix_multiply(
                                distances[thread_index, :end_row - start_row],
                                Z_chunk[thread_index, :end_row - start_row],
                                Y_chunk[slab], transpose_A=True,
                                transpose_B=False, alpha=1, beta=beta)
                        for i in range(end_row - start_row):
                            k = batch_order[start_row + i]
                            for j in range(num_PCs):
                                Z[k, j] = Z_chunk[thread_index, i, j]

            if cold_start:
                # Sum each batch's contributions to `O` across slabs,
                # skipping slabs without any of the batch's cells, then
                # compute `E` from `O`'s column sums
                for i in prange(num_batches * num_clusters,
                                num_threads=num_threads_or_chunks):
                    batch_label = i / num_clusters
                    k = i % num_clusters
                    total = 0
                    for slab in range(num_slabs):
                        if slab_counts[slab, batch_label] > 0:
                            total = total + O_slab[slab, batch_label, k]
                    O[batch_label, k] = total
                for k in prange(num_clusters,
                                num_threads=num_threads_or_chunks):
                    total = 0
                    for i in range(num_batches):
                        total = total + O[i, k]
                    for i in range(num_batches):
                        E[i, k] = Pr_b[i] * total
                num_Y_partials = num_slabs

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
                if converged:
                    if verbose:
                        print(f'Reached convergence after {iteration:,} '
                              f'iteration{"" if iteration == 1 else "s"}')
                    return
                prev_objective = objective

                # Check for KeyboardInterrupts
                PyErr_CheckSignals()

    if verbose:
        print(f'Failed to converge after {max_iterations:,} '
              f'iteration{"" if max_iterations == 1 else "s"}')
