# Functionality for pseudobulking and marker-gene finding (which uses a
# pseudobulk-style approach)

from cython.parallel cimport parallel, prange, threadid
from libcpp.pair cimport pair
from .cyutils cimport get_thread_offset, numeric, signed_integer, \
    uninitialized_vector


def get_detection_rate(const unsigned[:, ::1] detection_count,
                       const unsigned[::1] total_detection_count,
                       const unsigned[::1] num_cells_per_cell_type,
                       const unsigned total_num_cells,
                       float[:, ::1] detection_rate,
                       unsigned num_threads):
    cdef unsigned cell_type, gene, \
        num_cell_types = detection_count.shape[0], \
        num_genes = detection_count.shape[1]
    cdef unsigned long long num_cells

    num_threads = min(num_threads, num_cell_types)
    if num_threads <= 1:
        for cell_type in range(num_cell_types):
            num_cells = num_cells_per_cell_type[cell_type]
            for gene in range(num_genes):
                detection_rate[cell_type, gene] = \
                    <float> detection_count[cell_type, gene] / num_cells
    else:
        for cell_type in prange(num_cell_types, nogil=True,
                                num_threads=num_threads):
            num_cells = num_cells_per_cell_type[cell_type]
            for gene in range(num_genes):
                detection_rate[cell_type, gene] = \
                    <float> detection_count[cell_type, gene] / num_cells


def get_detection_rate_and_fold_change(
        const unsigned[:, ::1] detection_count,
        const unsigned[::1] total_detection_count,
        const unsigned[::1] num_cells_per_cell_type,
        const unsigned total_num_cells,
        float[:, ::1] detection_rate,
        float[:, ::1] fold_change,
        unsigned num_threads):
    cdef unsigned cell_type, gene, count, background_count, \
        background_num_cells, num_cell_types = detection_count.shape[0], \
        num_genes = detection_count.shape[1]
    cdef unsigned long long num_cells
    cdef float pair_detection_rate, pair_fold_change

    # There's no separate single-threaded branch here because Mac x86 with
    # -ffast-math leads to divergent floating-point error otherwise, even with
    # identical code inside the loop
    num_threads = min(num_threads, num_cell_types)
    for cell_type in prange(num_cell_types, nogil=True,
                            num_threads=num_threads):
        num_cells = num_cells_per_cell_type[cell_type]
        background_num_cells = total_num_cells - num_cells
        for gene in range(num_genes):
            count = detection_count[cell_type, gene]
            pair_detection_rate = <float> count / num_cells
            background_count = total_detection_count[gene] - count
            pair_fold_change = pair_detection_rate * \
                background_num_cells / background_count
            detection_rate[cell_type, gene] = pair_detection_rate
            fold_change[cell_type, gene] = pair_fold_change


def get_detection_rate_and_fold_change_and_pareto_candidates(
        const unsigned[:, ::1] detection_count,
        const unsigned[::1] total_detection_count,
        const unsigned[::1] num_cells_per_cell_type,
        const unsigned total_num_cells,
        const float min_detection_rate,
        const float min_fold_change,
        float[:, ::1] detection_rate,
        float[:, ::1] fold_change,
        char[:, ::1] is_pareto,
        unsigned num_threads):
    cdef unsigned cell_type, gene, count, background_count, \
        background_num_cells, num_cell_types = detection_count.shape[0], \
        num_genes = detection_count.shape[1]
    cdef unsigned long long num_cells
    cdef float pair_detection_rate, pair_fold_change

    # There's no separate single-threaded branch here because Mac x86 with
    # -ffast-math leads to divergent floating-point error otherwise, even with
    # identical code inside the loop
    num_threads = min(num_threads, num_cell_types)
    for cell_type in prange(num_cell_types, nogil=True,
                            num_threads=num_threads):
        num_cells = num_cells_per_cell_type[cell_type]
        background_num_cells = total_num_cells - num_cells
        for gene in range(num_genes):
            count = detection_count[cell_type, gene]
            pair_detection_rate = <float> count / num_cells
            background_count = total_detection_count[gene] - count
            pair_fold_change = pair_detection_rate * \
                background_num_cells / background_count
            detection_rate[cell_type, gene] = pair_detection_rate
            fold_change[cell_type, gene] = pair_fold_change
            is_pareto[cell_type, gene] = \
                (pair_detection_rate >= min_detection_rate) & \
                (pair_fold_change >= min_fold_change)


def groupby_getnnz_csc(
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const int[::1] group_map,
        const bint has_missing,
        unsigned[:, ::1] nnz,
        unsigned num_threads):
    cdef unsigned column, thread_index, num_columns = nnz.shape[1]
    cdef int group
    cdef unsigned long long row
    cdef pair[unsigned, unsigned] col_range

    nnz[:] = 0
    num_threads = min(num_threads, num_columns)
    if num_threads <= 1:
        if has_missing:
            # For each gene (column of the sparse array)...
            for column in range(num_columns):
                # For each cell (row) that's non-zero for this gene...
                for row in range(<unsigned long long> indptr[column],
                                 <unsigned long long> indptr[column + 1]):
                    # Get the group index for this row (-1 if it failed QC)
                    group = group_map[indices[row]]
                    if group == -1:
                        continue

                    # Add 1 to the total for this group and column
                    nnz[group, column] += 1
        else:
            for column in range(num_columns):
                for row in range(<unsigned long long> indptr[column],
                                 <unsigned long long> indptr[column + 1]):
                    group = group_map[indices[row]]
                    nnz[group, column] += 1
    else:
        if has_missing:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                col_range = \
                    get_thread_offset(indptr, thread_index, num_threads)
                for column in range(col_range.first, col_range.second):
                    for row in range(<unsigned long long> indptr[column],
                                     <unsigned long long> indptr[column + 1]):
                        group = group_map[indices[row]]
                        if group == -1:
                            continue
                        nnz[group, column] += 1
        else:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                col_range = \
                    get_thread_offset(indptr, thread_index, num_threads)
                for column in range(col_range.first, col_range.second):
                    for row in range(<unsigned long long> indptr[column],
                                     <unsigned long long> indptr[column + 1]):
                        group = group_map[indices[row]]
                        nnz[group, column] += 1


def groupby_getnnz_csc_for_gene_subset(const signed_integer[::1] indices,
                                       const signed_integer[::1] indptr,
                                       const int[::1] group_map,
                                       const unsigned[::1] gene_map,
                                       const bint has_missing,
                                       unsigned[:, ::1] nnz,
                                       unsigned num_threads):
    cdef unsigned column, gene, thread_index, chunk_size, start_col, end_col, \
        num_columns = nnz.shape[1]
    cdef int group
    cdef unsigned long long cell

    nnz[:] = 0
    num_threads = min(num_threads, num_columns)
    if num_threads <= 1:
        if has_missing:
            # For each gene...
            for column in range(num_columns):
                # Get the index of this gene in the count matrix
                gene = gene_map[column]
                # For each cell (row) that's non-zero for this gene...
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    # Get the group index for this cell (-1 if it failed QC)
                    group = group_map[indices[cell]]
                    if group == -1:
                        continue

                    # Add 1 to the nnz for this group and gene
                    nnz[group, column] += 1
        else:
            for column in range(num_columns):
                gene = gene_map[column]
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    group = group_map[indices[cell]]
                    nnz[group, column] += 1
    else:
        if has_missing:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                chunk_size = (num_columns + num_threads - 1) / num_threads
                start_col = chunk_size * thread_index
                end_col = min(start_col + chunk_size, num_columns)
                for column in range(start_col, end_col):
                    gene = gene_map[column]
                    for cell in range(<unsigned long long> indptr[gene],
                                      <unsigned long long> indptr[gene + 1]):
                        group = group_map[indices[cell]]
                        if group == -1:
                            continue
                        nnz[group, column] += 1
        else:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                chunk_size = (num_columns + num_threads - 1) / num_threads
                start_col = chunk_size * thread_index
                end_col = min(start_col + chunk_size, num_columns)
                for column in range(start_col, end_col):
                    gene = gene_map[column]
                    for cell in range(<unsigned long long> indptr[gene],
                                      <unsigned long long> indptr[gene + 1]):
                        group = group_map[indices[cell]]
                        nnz[group, column] += 1


def groupby_getnnz_and_total_csc_for_gene_subset(
        const numeric[::1] data,
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const int[::1] group_map,
        const unsigned[::1] gene_map,
        const bint has_missing,
        unsigned[:, ::1] nnz,
        double[:, ::1] total,
        unsigned num_threads):
    cdef unsigned column, gene, thread_index, chunk_size, start_col, end_col, \
        num_columns = nnz.shape[1]
    cdef int group
    cdef unsigned long long cell

    nnz[:] = 0
    total[:] = 0
    num_threads = min(num_threads, num_columns)
    if num_threads <= 1:
        if has_missing:
            # For each gene...
            for column in range(num_columns):
                # Get the index of this gene in the count matrix
                gene = gene_map[column]
                # For each cell (row) that's non-zero for this gene...
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    # Get the group index for this cell (-1 if it failed QC)
                    group = group_map[indices[cell]]
                    if group == -1:
                        continue

                    # Add 1 to the nnz for this group and gene
                    nnz[group, column] += 1

                    # Add the data value to the total for this group and gene
                    total[group, column] += <double> data[cell]
        else:
            for column in range(num_columns):
                gene = gene_map[column]
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    group = group_map[indices[cell]]
                    nnz[group, column] += 1
                    total[group, column] += <double> data[cell]
    else:
        if has_missing:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                chunk_size = (num_columns + num_threads - 1) / num_threads
                start_col = chunk_size * thread_index
                end_col = min(start_col + chunk_size, num_columns)
                for column in range(start_col, end_col):
                    gene = gene_map[column]
                    for cell in range(<unsigned long long> indptr[gene],
                                      <unsigned long long> indptr[gene + 1]):
                        group = group_map[indices[cell]]
                        if group == -1:
                            continue
                        nnz[group, column] += 1
                        total[group, column] += <double> data[cell]
        else:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                chunk_size = (num_columns + num_threads - 1) / num_threads
                start_col = chunk_size * thread_index
                end_col = min(start_col + chunk_size, num_columns)
                for column in range(start_col, end_col):
                    gene = gene_map[column]
                    for cell in range(<unsigned long long> indptr[gene],
                                      <unsigned long long> indptr[gene + 1]):
                        group = group_map[indices[cell]]
                        nnz[group, column] += 1
                        total[group, column] += <double> data[cell]


def groupby_getnnz_csr(const signed_integer[::1] indices,
                       const signed_integer[::1] indptr,
                       const unsigned[::1] group_indices,
                       const unsigned[::1] group_ends,
                       unsigned[:, ::1] nnz,
                       unsigned num_threads):
    cdef unsigned group, cell, row, thread_index, group_start, group_end, \
        chunk_size, start_cell, end_cell, gene_index, \
        num_groups = group_ends.shape[0], num_genes = nnz.shape[1]
    cdef unsigned long long gene, total
    cdef uninitialized_vector[unsigned] thread_nnz_buffer
    cdef unsigned[:, ::1] thread_nnz

    if num_threads == 1:
        # For each group (cell type)...
        for group in range(num_groups):
            # Initialize all elements of the group to 0
            nnz[group, :] = 0
            # For each cell within this group...
            for cell in range(0 if group == 0 else group_ends[group - 1],
                              group_ends[group]):
                # Get this cell's row index in the sparse array
                row = group_indices[cell]

                # For each gene (column) that's non-zero for this
                # cell...
                for gene in range(<unsigned long long> indptr[row],
                                  <unsigned long long> indptr[row + 1]):
                    # Add 1 to the nnz for this group and gene
                    nnz[group, indices[gene]] += 1
    elif num_threads <= num_groups:
        # Enough groups to keep all threads busy: parallelize over groups
        for group in prange(num_groups, nogil=True, num_threads=num_threads):
            nnz[group, :] = 0
            for cell in range(0 if group == 0 else group_ends[group - 1],
                              group_ends[group]):
                row = group_indices[cell]
                for gene in range(<unsigned long long> indptr[row],
                                  <unsigned long long> indptr[row + 1]):
                    nnz[group, indices[gene]] += 1
    else:
        # Fewer groups than threads: process each group with all threads, using
        # a per-thread scratch buffer to avoid write races, then reduce across
        # threads
        thread_nnz_buffer.resize(num_threads * num_genes)
        thread_nnz = \
            <unsigned[:num_threads, :num_genes]> thread_nnz_buffer.data()
        for group in range(num_groups):
            group_start = 0 if group == 0 else group_ends[group - 1]
            group_end = group_ends[group]

            # Phase 1: zero this thread's scratch row and accumulate counts
            # for this thread's cell chunk
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                for gene_index in range(num_genes):
                    thread_nnz[thread_index, gene_index] = 0

                chunk_size = \
                    (group_end - group_start + num_threads - 1) / num_threads
                start_cell = group_start + thread_index * chunk_size
                end_cell = start_cell + chunk_size
                if end_cell > group_end:
                    end_cell = group_end
                if start_cell > group_end:
                    start_cell = group_end

                for cell in range(start_cell, end_cell):
                    row = group_indices[cell]
                    for gene in range(<unsigned long long> indptr[row],
                                      <unsigned long long> indptr[row + 1]):
                        thread_nnz[thread_index, indices[gene]] += 1

            # Phase 2: parallel reduction across threads, by gene
            for gene_index in prange(num_genes, nogil=True,
                                     num_threads=num_threads):
                total = 0
                for thread_index in range(num_threads):
                    total = total + thread_nnz[thread_index, gene_index]
                nnz[group, gene_index] = <unsigned> total


def groupby_getnnz_csr_for_gene_subset(const signed_integer[::1] indices,
                                       const signed_integer[::1] indptr,
                                       const unsigned[::1] group_indices,
                                       const unsigned[::1] group_ends,
                                       const int[::1] gene_map,
                                       unsigned[:, ::1] nnz,
                                       unsigned num_threads):
    cdef unsigned group, cell, row, thread_index, group_start, group_end, \
        chunk_size, start_cell, end_cell, gene_index, \
        num_groups = group_ends.shape[0], num_genes = nnz.shape[1]
    cdef int column
    cdef unsigned long long gene, total
    cdef uninitialized_vector[unsigned] thread_nnz_buffer
    cdef unsigned[:, ::1] thread_nnz

    if num_threads == 1:
        # For each group (cell type)...
        for group in range(num_groups):
            # Initialize all elements of the group to 0
            nnz[group, :] = 0
            # For each cell within this group...
            for cell in range(0 if group == 0 else group_ends[group - 1],
                              group_ends[group]):
                # Get this cell's row index in the sparse array
                row = group_indices[cell]
                # For each gene (column) that's non-zero for this cell...
                for gene in range(<unsigned long long> indptr[row],
                                  <unsigned long long> indptr[row + 1]):
                    # Get this gene's column index in `nnz` (-1 if the gene is
                    # not in `genes`)
                    column = gene_map[indices[gene]]
                    if column == -1:
                        continue

                    # Add 1 to the nnz for this group and gene
                    nnz[group, column] += 1
    elif num_threads <= num_groups:
        # Enough groups to keep all threads busy: parallelize over groups
        for group in prange(num_groups, nogil=True, num_threads=num_threads):
            nnz[group, :] = 0
            for cell in range(0 if group == 0 else group_ends[group - 1],
                              group_ends[group]):
                row = group_indices[cell]
                for gene in range(<unsigned long long> indptr[row],
                                  <unsigned long long> indptr[row + 1]):
                    column = gene_map[indices[gene]]
                    if column == -1:
                        continue
                    nnz[group, column] += 1
    else:
        # Fewer groups than threads: process each group with all threads, using
        # a per-thread scratch buffer to avoid write races, then reduce across
        # threads
        thread_nnz_buffer.resize(num_threads * num_genes)
        thread_nnz = \
            <unsigned[:num_threads, :num_genes]> thread_nnz_buffer.data()
        for group in range(num_groups):
            group_start = 0 if group == 0 else group_ends[group - 1]
            group_end = group_ends[group]

            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                for gene_index in range(num_genes):
                    thread_nnz[thread_index, gene_index] = 0

                # Phase 1: zero this thread's scratch row and accumulate counts
                # for this thread's cell chunk
                chunk_size = \
                    (group_end - group_start + num_threads - 1) / num_threads
                start_cell = group_start + thread_index * chunk_size
                end_cell = start_cell + chunk_size
                if end_cell > group_end:
                    end_cell = group_end
                if start_cell > group_end:
                    start_cell = group_end

                for cell in range(start_cell, end_cell):
                    row = group_indices[cell]
                    for gene in range(<unsigned long long> indptr[row],
                                      <unsigned long long> indptr[row + 1]):
                        column = gene_map[indices[gene]]
                        if column == -1:
                            continue
                        thread_nnz[thread_index, column] += 1

            # Phase 2: parallel reduction across threads, by gene
            for gene_index in prange(num_genes, nogil=True,
                                     num_threads=num_threads):
                total = 0
                for thread_index in range(num_threads):
                    total = total + thread_nnz[thread_index, gene_index]
                nnz[group, gene_index] = <unsigned> total


def groupby_getnnz_and_total_csr_for_gene_subset(
        const numeric[::1] data,
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const unsigned[::1] group_indices,
        const unsigned[::1] group_ends,
        const int[::1] gene_map,
        unsigned[:, ::1] nnz,
        double[:, ::1] total,
        unsigned num_threads):
    cdef unsigned group, cell, row, thread_index, group_start, group_end, \
        chunk_size, start_cell, end_cell, gene_index, nnz_total, \
        num_groups = group_ends.shape[0], num_genes = nnz.shape[1]

    cdef int column
    cdef unsigned long long gene
    cdef double gene_total
    cdef uninitialized_vector[unsigned] thread_nnz_buffer
    cdef unsigned[:, ::1] thread_nnz
    cdef uninitialized_vector[double] thread_total_buffer
    cdef double[:, ::1] thread_total

    if num_threads == 1:
        # For each group (cell type)...
        for group in range(num_groups):
            # Initialize all elements of the group to 0
            nnz[group, :] = 0
            total[group, :] = 0
            # For each cell within this group...
            for cell in range(0 if group == 0 else group_ends[group - 1],
                              group_ends[group]):
                # Get this cell's row index in the sparse array
                row = group_indices[cell]
                # For each gene (column) that's non-zero for this cell...
                for gene in range(<unsigned long long> indptr[row],
                                  <unsigned long long> indptr[row + 1]):
                    # Get this gene's column index in `nnz` and `total` (-1 if
                    # the gene is not in `genes`)
                    column = gene_map[indices[gene]]
                    if column == -1:
                        continue

                    # Add 1 to the nnz for this group and gene
                    nnz[group, column] += 1

                    # Add the data value to the total for this group and gene
                    total[group, column] += <double> data[gene]
    elif num_threads <= num_groups:
        # Enough groups to keep all threads busy: parallelize over groups
        for group in prange(num_groups, nogil=True, num_threads=num_threads):
            nnz[group, :] = 0
            total[group, :] = 0
            for cell in range(0 if group == 0 else group_ends[group - 1],
                              group_ends[group]):
                row = group_indices[cell]
                for gene in range(<unsigned long long> indptr[row],
                                  <unsigned long long> indptr[row + 1]):
                    column = gene_map[indices[gene]]
                    if column == -1:
                        continue
                    nnz[group, column] += 1
                    total[group, column] += <double> data[gene]
    else:
        # Fewer groups than threads: process each group with all threads, using
        # per-thread scratch buffers to avoid write races, then reduce across
        # threads
        thread_nnz_buffer.resize(num_threads * num_genes)
        thread_nnz = \
            <unsigned[:num_threads, :num_genes]> thread_nnz_buffer.data()
        thread_total_buffer.resize(num_threads * num_genes)
        thread_total = \
            <double[:num_threads, :num_genes]> thread_total_buffer.data()
        for group in range(num_groups):
            group_start = 0 if group == 0 else group_ends[group - 1]
            group_end = group_ends[group]

            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                for gene_index in range(num_genes):
                    thread_nnz[thread_index, gene_index] = 0
                    thread_total[thread_index, gene_index] = 0

                chunk_size = \
                    (group_end - group_start + num_threads - 1) / num_threads
                start_cell = group_start + thread_index * chunk_size
                end_cell = start_cell + chunk_size
                if end_cell > group_end:
                    end_cell = group_end
                if start_cell > group_end:
                    start_cell = group_end

                for cell in range(start_cell, end_cell):
                    row = group_indices[cell]
                    for gene in range(<unsigned long long> indptr[row],
                                      <unsigned long long> indptr[row + 1]):
                        column = gene_map[indices[gene]]
                        if column == -1:
                            continue
                        thread_nnz[thread_index, column] += 1
                        thread_total[thread_index, column] += \
                            <double> data[gene]

            for gene_index in prange(num_genes, nogil=True,
                                     num_threads=num_threads):
                nnz_total = 0
                gene_total = 0
                for thread_index in range(num_threads):
                    nnz_total = \
                        nnz_total + thread_nnz[thread_index, gene_index]
                    gene_total = \
                        gene_total + thread_total[thread_index, gene_index]
                nnz[group, gene_index] = nnz_total
                total[group, gene_index] = gene_total


def groupby_sum_csr(const numeric[::1] data,
                    const signed_integer[::1] indices,
                    const signed_integer[::1] indptr,
                    const unsigned[::1] group_indices,
                    const unsigned[::1] group_ends,
                    unsigned[:, ::1] result,
                    unsigned num_threads):
    cdef unsigned group, cell, row, num_groups = group_ends.shape[0], \
        num_genes = result.shape[1]
    cdef unsigned long long gene

    num_threads = min(num_threads, num_groups)
    if num_threads <= 1:
        # For each group (cell type-sample pair)...
        for group in range(num_groups):
            # Initialize all elements of the group to 0
            result[group, :] = 0
            # For each cell within this group...
            for cell in range(
                    0 if group == 0 else group_ends[group - 1],
                    group_ends[group]):
                # Get this cell's row index in the sparse
                # matrix
                row = group_indices[cell]

                # For each gene (column) that's non-zero for
                # this cell...
                for gene in range(<unsigned long long> indptr[row],
                                  <unsigned long long> indptr[row + 1]):
                    # Add the value at this cell and gene to
                    # the total for this group and gene
                    result[group, indices[gene]] += <unsigned> data[gene]
    else:
        for group in prange(num_groups, nogil=True,
                            num_threads=num_threads):
            result[group, :] = 0
            for cell in range(
                    0 if group == 0 else group_ends[group - 1],
                    group_ends[group]):
                row = group_indices[cell]
                for gene in range(<unsigned long long> indptr[row],
                                  <unsigned long long> indptr[row + 1]):
                    result[group, indices[gene]] += <unsigned> data[gene]


def groupby_sum_csc(const numeric[::1] data,
                    const signed_integer[::1] indices,
                    const signed_integer[::1] indptr,
                    const int[::1] group_map,
                    const bint has_missing,
                    unsigned[:, ::1] result,
                    unsigned num_threads):
    cdef unsigned gene, thread_index, num_genes = result.shape[1]
    cdef int group
    cdef unsigned long long cell
    cdef pair[unsigned, unsigned] col_range

    result[:] = 0
    num_threads = min(num_threads, num_genes)
    if num_threads <= 1:
        if has_missing:
            # For each gene (column of the sparse array)...
            for gene in range(num_genes):
                # For each cell (row) that's non-zero for this gene...
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    # Get the group index for this cell (-1 if it failed QC)
                    group = group_map[indices[cell]]
                    if group == -1:
                        continue

                    # Add the value at this cell and gene to the total for this
                    # group and gene
                    result[group, gene] += <unsigned> data[cell]
        else:
            for gene in range(num_genes):
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    group = group_map[indices[cell]]
                    result[group, gene] += <unsigned> data[cell]
    else:
        if has_missing:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                col_range = \
                    get_thread_offset(indptr, thread_index, num_threads)
                for gene in range(col_range.first, col_range.second):
                    for cell in range(<unsigned long long> indptr[gene],
                                      <unsigned long long> indptr[gene + 1]):
                        group = group_map[indices[cell]]
                        if group == -1:
                            continue
                        result[group, gene] += <unsigned> data[cell]
        else:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                col_range = \
                    get_thread_offset(indptr, thread_index, num_threads)
                for gene in range(col_range.first, col_range.second):
                    for cell in range(<unsigned long long> indptr[gene],
                                      <unsigned long long> indptr[gene + 1]):
                        group = group_map[indices[cell]]
                        result[group, gene] += <unsigned> data[cell]


def pareto_front(float[:, ::1] detection_rate,
                 float[:, ::1] fold_change,
                 char[:, ::1] is_pareto,
                 unsigned num_threads):
    cdef unsigned gene, other_gene, cell_type, \
        num_cell_types = detection_rate.shape[0], \
        num_genes = detection_rate.shape[1]
    cdef float gene_detection_rate, gene_fold_change, other_detection_rate, \
        other_fold_change

    if num_threads == 1:
        for cell_type in range(num_cell_types):
            for gene in range(num_genes):
                if not is_pareto[cell_type, gene]:
                    continue
                gene_detection_rate = detection_rate[cell_type, gene]
                gene_fold_change = fold_change[cell_type, gene]
                for other_gene in range(num_genes):
                    if gene == other_gene or \
                            not is_pareto[cell_type, other_gene]:
                        continue
                    if gene_detection_rate <= \
                            detection_rate[cell_type, other_gene] \
                            and gene_fold_change <= \
                            fold_change[cell_type, other_gene]:
                        is_pareto[cell_type, gene] = 0
                        break
    else:
        for cell_type in prange(num_cell_types, nogil=True,
                                num_threads=num_threads):
            for gene in range(num_genes):
                if not is_pareto[cell_type, gene]:
                    continue
                gene_detection_rate = detection_rate[cell_type, gene]
                gene_fold_change = fold_change[cell_type, gene]
                for other_gene in range(num_genes):
                    # Skip the `or not is_pareto[cell_type, other_gene]` check
                    # from the serial version, which would cause race
                    # conditions. Basically, compare the gene against all
                    # genes, not just candidate Pareto-optimal genes. This is
                    # okay because non-candidates are definitely not going to
                    # Pareto-dominate this gene, and because we add an
                    # `other_gene < gene` check below to reproduce the
                    # highest-wins tiebreaking of the serial version.
                    if gene == other_gene:
                        continue
                    other_detection_rate = \
                        detection_rate[cell_type, other_gene]
                    other_fold_change = fold_change[cell_type, other_gene]
                    if gene_detection_rate <= other_detection_rate and \
                            gene_fold_change <= other_fold_change:
                        if gene_detection_rate == other_detection_rate and \
                                gene_fold_change == other_fold_change and \
                                other_gene < gene:
                            continue
                        is_pareto[cell_type, gene] = 0
                        break