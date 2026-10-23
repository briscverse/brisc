# Functionality for highly-variable gene identification

from cython.parallel cimport parallel, prange, threadid
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from .cyutils cimport get_thread_offset, numeric, signed_integer


def gene_mean_and_variance_csr(const numeric[::1] data,
                               const signed_integer[::1] indices,
                               const signed_integer[::1] indptr,
                               const long long[::1] cell_indices,
                               const unsigned long long num_cells,
                               const unsigned num_dataset_genes,
                               unsigned long long[:, :, ::1] thread_sums,
                               unsigned long long[::1] counts_sum,
                               unsigned long long[::1] squared_counts_sum,
                               float[::1] mean,
                               float[::1] var,
                               unsigned[::1] nonzero_count,
                               unsigned num_threads):
    # `thread_sums[thread_index, gene]` holds the sum, squared sum and number
    # of non-zero counts of each gene across the elements processed by each
    # thread, interleaved so that each element updates a single cache line.
    # It is allocated once by the caller and reused across batches.

    cdef unsigned gene, cell, thread_index
    cdef unsigned long long num_elements, i, j, value, chunk_size, start, \
        end, total_sum, total_sum_of_squares, total_nonzero_count
    cdef float inv_num_cells = <float> 1 / num_cells
    cdef double inv_num_pairs_of_cells = 1.0 / (num_cells * (num_cells - 1))
    cdef unsigned long long[:, ::1] sums

    if cell_indices.shape[0] == 0:
        num_elements = indices.shape[0]
        num_threads = min(num_threads, num_elements)
    else:
        num_threads = min(num_threads, num_cells)

    if num_threads <= 1:
        sums = thread_sums[0]
        for gene in range(num_dataset_genes):
            sums[gene, 0] = 0
            sums[gene, 1] = 0
            sums[gene, 2] = 0
        if cell_indices.shape[0] == 0:
            # Iterate over all elements of the count matrix, ignoring which
            # cell they're from
            for i in range(num_elements):
                gene = indices[i]
                value = <unsigned long long> data[i]
                sums[gene, 0] += value
                sums[gene, 1] += value * value
                sums[gene, 2] += 1
        else:
            # Only iterate over the elements from cells in `cell_indices` (i.e.
            # cells in this batch, and/or passing QC)
            for j in range(num_cells):
                cell = cell_indices[j]
                for i in range(<unsigned long long> indptr[cell],
                               <unsigned long long> indptr[cell + 1]):
                    gene = indices[i]
                    value = <unsigned long long> data[i]
                    sums[gene, 0] += value
                    sums[gene, 1] += value * value
                    sums[gene, 2] += 1

        # Calculate means and variances from the sums and squared sums
        for gene in range(num_dataset_genes):
            counts_sum[gene] = sums[gene, 0]
            squared_counts_sum[gene] = sums[gene, 1]
            nonzero_count[gene] = <unsigned> sums[gene, 2]
            mean[gene] = counts_sum[gene] * inv_num_cells
            var[gene] = <float> (inv_num_pairs_of_cells * (
                <double> num_cells * <double> squared_counts_sum[gene] -
                <double> counts_sum[gene] * <double> counts_sum[gene]))
    else:
        with nogil:
            if cell_indices.shape[0] == 0:
                # Partition the work by elements, not cells, for better
                # load-balancing in case cells have substantially different
                # library sizes
                chunk_size = (num_elements + num_threads - 1) / num_threads
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    for gene in range(num_dataset_genes):
                        thread_sums[thread_index, gene, 0] = 0
                        thread_sums[thread_index, gene, 1] = 0
                        thread_sums[thread_index, gene, 2] = 0

                    start = thread_index * chunk_size
                    end = min(start + chunk_size, num_elements)
                    for i in range(start, end):
                        gene = indices[i]
                        value = <unsigned long long> data[i]
                        thread_sums[thread_index, gene, 0] += value
                        thread_sums[thread_index, gene, 1] += value * value
                        thread_sums[thread_index, gene, 2] += 1
            else:
                # Partition the work by cells
                chunk_size = (num_cells + num_threads - 1) / num_threads
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    for gene in range(num_dataset_genes):
                        thread_sums[thread_index, gene, 0] = 0
                        thread_sums[thread_index, gene, 1] = 0
                        thread_sums[thread_index, gene, 2] = 0

                    start = thread_index * chunk_size
                    end = min(start + chunk_size, num_cells)
                    for j in range(start, end):
                        cell = cell_indices[j]
                        for i in range(<unsigned long long> indptr[cell],
                                       <unsigned long long> indptr[cell + 1]):
                            gene = indices[i]
                            value = <unsigned long long> data[i]
                            thread_sums[thread_index, gene, 0] += value
                            thread_sums[thread_index, gene, 1] += value * value
                            thread_sums[thread_index, gene, 2] += 1

            # Calculate means and variances by aggregating the sums and squared
            # sums across threads
            for gene in prange(num_dataset_genes, num_threads=num_threads):
                total_sum = 0
                total_sum_of_squares = 0
                total_nonzero_count = 0
                for thread_index in range(num_threads):
                    total_sum = total_sum + thread_sums[thread_index, gene, 0]
                    total_sum_of_squares = total_sum_of_squares + \
                        thread_sums[thread_index, gene, 1]
                    total_nonzero_count = total_nonzero_count + \
                        thread_sums[thread_index, gene, 2]
                counts_sum[gene] = total_sum
                squared_counts_sum[gene] = total_sum_of_squares
                nonzero_count[gene] = <unsigned> total_nonzero_count
                mean[gene] = total_sum * inv_num_cells
                var[gene] = <float> (inv_num_pairs_of_cells * (
                    <double> num_cells * <double> total_sum_of_squares -
                    <double> total_sum * <double> total_sum))


def gene_mean_and_variance_csc(const numeric[::1] data,
                               const signed_integer[::1] indices,
                               const signed_integer[::1] indptr,
                               char[::1] cell_mask,
                               const unsigned long long num_cells,
                               const unsigned num_dataset_genes,
                               float[::1] mean,
                               float[::1] var,
                               unsigned[::1] nonzero_count,
                               unsigned num_threads):

    cdef unsigned thread_index
    cdef unsigned long long i, value, sum, sum_of_squares, nnz, gene
    cdef float inv_num_cells = <float> 1 / num_cells
    cdef double inv_num_pairs_of_cells = 1.0 / (num_cells * (num_cells - 1))
    cdef pair[unsigned, unsigned] col_range
    cdef vector[unsigned long long] sum_buffer, sum_of_squares_buffer
    sum_buffer.resize(num_dataset_genes)
    sum_of_squares_buffer.resize(num_dataset_genes)
    cdef unsigned long long[::1] \
        sum_arr = <unsigned long long[:num_dataset_genes]> sum_buffer.data(), \
        sum_of_squares_arr = <unsigned long long[:num_dataset_genes]> \
            sum_of_squares_buffer.data()

    num_threads = min(num_threads, num_dataset_genes)
    if num_threads <= 1:
        if cell_mask.shape[0] == 0:
            for gene in range(num_dataset_genes):
                # Calculate the sum and squared sum for this gene, across cells
                # with non-zero counts for the gene
                sum = 0
                sum_of_squares = 0
                nnz = 0
                for i in range(<unsigned long long> indptr[gene],
                               <unsigned long long> indptr[gene + 1]):
                    value = <unsigned long long> data[i]
                    sum += value
                    sum_of_squares += value * value
                    nnz += 1
                sum_arr[gene] = sum
                sum_of_squares_arr[gene] = sum_of_squares
                nonzero_count[gene] = nnz
        else:
            # Same as the version without a cell mask, but only include cells
            # where `cell_mask` is `True`
            for gene in range(num_dataset_genes):
                sum = 0
                sum_of_squares = 0
                nnz = 0
                for i in range(<unsigned long long> indptr[gene],
                               <unsigned long long> indptr[gene + 1]):
                    if cell_mask[indices[i]]:
                        value = <unsigned long long> data[i]
                        sum += value
                        sum_of_squares += value * value
                        nnz += 1
                sum_arr[gene] = sum
                sum_of_squares_arr[gene] = sum_of_squares
                nonzero_count[gene] = nnz
    else:
        if cell_mask.shape[0] == 0:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                col_range = \
                    get_thread_offset(indptr, thread_index, num_threads)
                for gene in range(col_range.first, col_range.second):
                    sum = 0
                    sum_of_squares = 0
                    nnz = 0
                    for i in range(<unsigned long long> indptr[gene],
                                   <unsigned long long> indptr[gene + 1]):
                        value = <unsigned long long> data[i]
                        sum = sum + value
                        sum_of_squares = \
                            sum_of_squares + value * value
                        nnz = nnz + 1
                    sum_arr[gene] = sum
                    sum_of_squares_arr[gene] = sum_of_squares
                    nonzero_count[gene] = nnz
        else:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                col_range = \
                    get_thread_offset(indptr, thread_index, num_threads)
                for gene in range(col_range.first, col_range.second):
                    sum = 0
                    sum_of_squares = 0
                    nnz = 0
                    for i in range(<unsigned long long> indptr[gene],
                                   <unsigned long long> indptr[gene + 1]):
                        if cell_mask[indices[i]]:
                            value = <unsigned long long> data[i]
                            sum = sum + value
                            sum_of_squares = \
                                sum_of_squares + value * value
                            nnz = nnz + 1
                    sum_arr[gene] = sum
                    sum_of_squares_arr[gene] = sum_of_squares
                    nonzero_count[gene] = nnz

    # Calculate the mean and variance from the sum and squared sum. This is
    # done in a separate loop because it is the only part of the function that
    # does floating-point calculations, and this allows it to be branchless
    # (and therefore consistently SIMDized) regardless of whether there is a QC
    # column.
    for gene in range(num_dataset_genes):
        mean[gene] = sum_arr[gene] * inv_num_cells
        var[gene] = <float> (inv_num_pairs_of_cells * (
            <double> num_cells * <double> sum_of_squares_arr[gene] -
            <double> sum_arr[gene] * <double> sum_arr[gene]))


def clipped_sum_csr(const numeric[::1] data,
                    const signed_integer[::1] indices,
                    const signed_integer[::1] indptr,
                    const unsigned long long num_cells,
                    const unsigned num_dataset_genes,
                    const long long[::1] cell_indices,
                    const float[::1] clip_val,
                    unsigned long long[:, :, ::1] thread_sums,
                    const unsigned long long[::1] counts_sum,
                    const unsigned long long[::1] squared_counts_sum,
                    float[::1] batch_counts_sum,
                    float[::1] squared_batch_counts_sum,
                    unsigned num_threads):
    cdef unsigned cell, gene, thread_index
    cdef unsigned long long i, j, value, chunk_size, start, end, \
        out_of_range_sum, squared_out_of_range_sum, num_out_of_range
    cdef pair[unsigned, unsigned] row_range
    cdef unsigned long long[:, ::1] sums

    # Key insight: the things we're summing are integers except when
    # `value > clip_val[gene]`, in which case we are adding a (floating-point)
    # constant, so keep track of this case separately and add it at the end, to
    # minimize floating-point error. This case is rare, so only update
    # `thread_sums` for these elements, and subtract them from the unclipped
    # sums calculated by `gene_mean_and_variance_csr()`. Also, use an integer
    # version of `clip_val` for these comparisons, to avoid any potential
    # inconsistency in floating-point roundoff between code paths.
    num_threads = min(num_threads, num_cells)
    if num_threads <= 1:
        sums = thread_sums[0]
        for gene in range(num_dataset_genes):
            sums[gene, 0] = 0
            sums[gene, 1] = 0
            sums[gene, 2] = 0
        if cell_indices.shape[0] == 0:
            for cell in range(num_cells):
                for i in range(<unsigned long long> indptr[cell],
                               <unsigned long long> indptr[cell + 1]):
                    gene = indices[i]
                    value = <unsigned long long> data[i]
                    if value > clip_val[gene]:
                        sums[gene, 0] += value
                        sums[gene, 1] += value * value
                        sums[gene, 2] += 1
        else:
            for j in range(<unsigned long long> cell_indices.shape[0]):
                cell = cell_indices[j]
                for i in range(<unsigned long long> indptr[cell],
                               <unsigned long long> indptr[cell + 1]):
                    gene = indices[i]
                    value = <unsigned long long> data[i]
                    if value > clip_val[gene]:
                        sums[gene, 0] += value
                        sums[gene, 1] += value * value
                        sums[gene, 2] += 1
        for gene in range(num_dataset_genes):
            batch_counts_sum[gene] = \
                (counts_sum[gene] - sums[gene, 0]) + \
                sums[gene, 2] * clip_val[gene]
            squared_batch_counts_sum[gene] = \
                (squared_counts_sum[gene] - sums[gene, 1]) + \
                sums[gene, 2] * clip_val[gene] * clip_val[gene]
    else:
        with nogil:
            if cell_indices.shape[0] == 0:
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    for gene in range(num_dataset_genes):
                        thread_sums[thread_index, gene, 0] = 0
                        thread_sums[thread_index, gene, 1] = 0
                        thread_sums[thread_index, gene, 2] = 0

                    row_range = get_thread_offset(indptr, thread_index,
                                                  num_threads)
                    for cell in range(row_range.first, row_range.second):
                        for i in range(<unsigned long long> indptr[cell],
                                       <unsigned long long> indptr[cell + 1]):
                            gene = indices[i]
                            value = <unsigned long long> data[i]
                            if value > clip_val[gene]:
                                thread_sums[thread_index, gene, 0] += value
                                thread_sums[thread_index, gene, 1] += \
                                    value * value
                                thread_sums[thread_index, gene, 2] += 1
            else:
                chunk_size = (num_cells + num_threads - 1) / num_threads
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    for gene in range(num_dataset_genes):
                        thread_sums[thread_index, gene, 0] = 0
                        thread_sums[thread_index, gene, 1] = 0
                        thread_sums[thread_index, gene, 2] = 0

                    start = thread_index * chunk_size
                    end = min(start + chunk_size, num_cells)
                    for j in range(start, end):
                        cell = cell_indices[j]
                        for i in range(<unsigned long long> indptr[cell],
                                       <unsigned long long> indptr[cell + 1]):
                            gene = indices[i]
                            value = <unsigned long long> data[i]
                            if value > clip_val[gene]:
                                thread_sums[thread_index, gene, 0] += value
                                thread_sums[thread_index, gene, 1] += \
                                    value * value
                                thread_sums[thread_index, gene, 2] += 1

            for gene in prange(num_dataset_genes,
                               num_threads=num_threads):
                out_of_range_sum = 0
                squared_out_of_range_sum = 0
                num_out_of_range = 0
                for thread_index in range(num_threads):
                    out_of_range_sum = out_of_range_sum + \
                        thread_sums[thread_index, gene, 0]
                    squared_out_of_range_sum = squared_out_of_range_sum + \
                        thread_sums[thread_index, gene, 1]
                    num_out_of_range = num_out_of_range + \
                        thread_sums[thread_index, gene, 2]
                batch_counts_sum[gene] = \
                    (counts_sum[gene] - out_of_range_sum) + \
                    num_out_of_range * clip_val[gene]
                squared_batch_counts_sum[gene] = \
                    (squared_counts_sum[gene] - squared_out_of_range_sum) + \
                    num_out_of_range * clip_val[gene] * clip_val[gene]


def clipped_sum_csc(const numeric[::1] data,
                    const signed_integer[::1] indices,
                    const signed_integer[::1] indptr,
                    char[::1] cell_mask,
                    const float[::1] clip_val,
                    float[::1] batch_counts_sum,
                    float[::1] squared_batch_counts_sum,
                    unsigned num_threads):
    cdef unsigned cell, gene, thread_index, num_genes = indptr.shape[0] - 1
    cdef unsigned long long i, value, batch_counts_sum_int, \
        squared_batch_counts_sum_int, num_out_of_range
    cdef pair[unsigned, unsigned] col_range

    cdef vector[unsigned long long] batch_counts_sum_int_buffer, \
        squared_batch_counts_sum_int_buffer, num_out_of_range_buffer
    batch_counts_sum_int_buffer.resize(num_genes)
    squared_batch_counts_sum_int_buffer.resize(num_genes)
    num_out_of_range_buffer.resize(num_genes)

    cdef unsigned long long[::1] \
        batch_counts_sum_int_arr = <unsigned long long[:num_genes]> \
            batch_counts_sum_int_buffer.data(), \
        squared_batch_counts_sum_int_arr = <unsigned long long[:num_genes]> \
            squared_batch_counts_sum_int_buffer.data(), \
        num_out_of_range_arr = <unsigned long long[:num_genes]> \
            num_out_of_range_buffer.data()

    num_threads = min(num_threads, num_genes)
    if num_threads <= 1:
        if cell_mask.shape[0] == 0:
            for gene in range(num_genes):
                batch_counts_sum_int = 0
                squared_batch_counts_sum_int = 0
                num_out_of_range = 0
                for i in range(<unsigned long long> indptr[gene],
                               <unsigned long long> indptr[gene + 1]):
                    value = <unsigned long long> data[i]
                    if value > clip_val[gene]:
                        num_out_of_range += 1
                    else:
                        batch_counts_sum_int += value
                        squared_batch_counts_sum_int += value * value
                batch_counts_sum_int_arr[gene] = batch_counts_sum_int
                squared_batch_counts_sum_int_arr[gene] = \
                    squared_batch_counts_sum_int
                num_out_of_range_arr[gene] = num_out_of_range
        else:
            for gene in range(num_genes):
                batch_counts_sum_int = 0
                squared_batch_counts_sum_int = 0
                num_out_of_range = 0
                for i in range(<unsigned long long> indptr[gene],
                               <unsigned long long> indptr[gene + 1]):
                    cell = indices[i]
                    if cell_mask[cell]:
                        value = <unsigned long long> data[i]
                        if value > clip_val[gene]:
                            num_out_of_range += 1
                        else:
                            batch_counts_sum_int += value
                            squared_batch_counts_sum_int += value * value
                batch_counts_sum_int_arr[gene] = batch_counts_sum_int
                squared_batch_counts_sum_int_arr[gene] = \
                    squared_batch_counts_sum_int
                num_out_of_range_arr[gene] = num_out_of_range
    else:
        if cell_mask.shape[0] == 0:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                col_range = get_thread_offset(indptr, thread_index,
                                              num_threads)
                for gene in range(col_range.first, col_range.second):
                    batch_counts_sum_int = 0
                    squared_batch_counts_sum_int = 0
                    num_out_of_range = 0
                    for i in range(<unsigned long long> indptr[gene],
                                   <unsigned long long> indptr[gene + 1]):
                        value = <unsigned long long> data[i]
                        if value > clip_val[gene]:
                            num_out_of_range = num_out_of_range + 1
                        else:
                            batch_counts_sum_int = batch_counts_sum_int + value
                            squared_batch_counts_sum_int = \
                                squared_batch_counts_sum_int + value * value
                    batch_counts_sum_int_arr[gene] = batch_counts_sum_int
                    squared_batch_counts_sum_int_arr[gene] = \
                        squared_batch_counts_sum_int
                    num_out_of_range_arr[gene] = num_out_of_range
        else:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                col_range = get_thread_offset(indptr, thread_index,
                                              num_threads)
                for gene in range(col_range.first, col_range.second):
                    batch_counts_sum_int = 0
                    squared_batch_counts_sum_int = 0
                    num_out_of_range = 0
                    for i in range(<unsigned long long> indptr[gene],
                                   <unsigned long long> indptr[gene + 1]):
                        cell = indices[i]
                        if cell_mask[cell]:
                            value = <unsigned long long> data[i]
                            if value > clip_val[gene]:
                                num_out_of_range = num_out_of_range + 1
                            else:
                                batch_counts_sum_int = \
                                    batch_counts_sum_int + value
                                squared_batch_counts_sum_int = \
                                    squared_batch_counts_sum_int + \
                                    value * value
                    batch_counts_sum_int_arr[gene] = batch_counts_sum_int
                    squared_batch_counts_sum_int_arr[gene] = \
                        squared_batch_counts_sum_int
                    num_out_of_range_arr[gene] = num_out_of_range

    # Calculate the clipped sum and squared sum. This is done in a separate
    # loop because it is the only part of the function that does floating-point
    # calculations, and this allows it to be branchless (and therefore
    # consistently SIMDized) regardless of whether there is a QC column.
    for gene in range(num_genes):
        batch_counts_sum[gene] = \
            batch_counts_sum_int_arr[gene] + \
            num_out_of_range_arr[gene] * clip_val[gene]
        squared_batch_counts_sum[gene] = \
            squared_batch_counts_sum_int_arr[gene] + \
            num_out_of_range_arr[gene] * clip_val[gene] * clip_val[gene]