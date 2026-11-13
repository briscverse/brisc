# Functionality for quality control

from cython.parallel cimport parallel, prange, threadid
from libcpp.algorithm cimport lower_bound
from libcpp.vector cimport vector
from .cyutils cimport atomic_and, get_thread_offset, numeric, signed_integer


def malat1_mask_csr(
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const signed_integer MALAT1_index,
        char[::1] MALAT1_mask,
        unsigned num_threads):

    cdef unsigned row, thread_index
    cdef unsigned long long num_cells = indptr.shape[0] - 1
    cdef signed_integer* start
    cdef signed_integer* end
    cdef signed_integer* found
    cdef unsigned start_row, end_row

    num_threads = min(num_threads, num_cells)
    if num_threads <= 1:
        for row in range(num_cells):
            start = <signed_integer*> &indices[indptr[row]]
            end = start + indptr[row + 1] - indptr[row]
            found = lower_bound(start, end, MALAT1_index)
            MALAT1_mask[row] = found != end and found[0] == MALAT1_index
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            start_row, end_row = \
                get_thread_offset(indptr, thread_index, num_threads)
            for row in range(start_row, end_row):
                start = <signed_integer*> &indices[indptr[row]]
                end = start + indptr[row + 1] - indptr[row]
                found = lower_bound(start, end, MALAT1_index)
                MALAT1_mask[row] = found != end and found[0] == MALAT1_index


def malat1_mask_csr_check(
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const signed_integer MALAT1_index,
        char[::1] MALAT1_mask,
        unsigned num_threads):

    cdef unsigned row, thread_index
    cdef unsigned long long col, row_start, row_end, \
        num_cells = indptr.shape[0] - 1
    cdef unsigned start_row, end_row
    cdef bint has_canonical_format = True, has_sorted_indices = True

    num_threads = min(num_threads, num_cells)
    if num_threads <= 1:
        for row in range(num_cells):
            MALAT1_mask[row] = False
            row_start = <unsigned long long> indptr[row]
            row_end = <unsigned long long> indptr[row + 1]
            if row_start < row_end:
                if indices[row_start] == MALAT1_index:
                    MALAT1_mask[row] = True
                for col in range(row_start + 1, row_end):
                    if indices[col - 1] >= indices[col]:
                        has_canonical_format = False
                        if indices[col - 1] > indices[col]:
                            has_sorted_indices = False
                    if indices[col] == MALAT1_index:
                        MALAT1_mask[row] = True
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            start_row, end_row = \
                get_thread_offset(indptr, thread_index, num_threads)
            for row in range(start_row, end_row):
                MALAT1_mask[row] = False
                row_start = <unsigned long long> indptr[row]
                row_end = <unsigned long long> indptr[row + 1]
                if row_start < row_end:
                    if indices[row_start] == MALAT1_index:
                        MALAT1_mask[row] = True
                    for col in range(row_start + 1, row_end):
                        if indices[col - 1] >= indices[col]:
                            atomic_and(has_canonical_format, False)
                            if indices[col - 1] > indices[col]:
                                atomic_and(has_sorted_indices, False)
                        if indices[col] == MALAT1_index:
                            MALAT1_mask[row] = True
    return has_canonical_format, has_sorted_indices


def malat1_mask_csr_scan(
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const signed_integer MALAT1_index,
        char[::1] MALAT1_mask,
        unsigned num_threads):

    cdef unsigned row, thread_index
    cdef unsigned long long col, row_start, row_end, \
        num_cells = indptr.shape[0] - 1
    cdef unsigned start_row, end_row

    num_threads = min(num_threads, num_cells)
    if num_threads <= 1:
        for row in range(num_cells):
            MALAT1_mask[row] = False
            row_start = <unsigned long long> indptr[row]
            row_end = <unsigned long long> indptr[row + 1]
            for col in range(row_start, row_end):
                if indices[col] == MALAT1_index:
                    MALAT1_mask[row] = True
                    break
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            start_row, end_row = \
                get_thread_offset(indptr, thread_index, num_threads)
            for row in range(start_row, end_row):
                MALAT1_mask[row] = False
                row_start = <unsigned long long> indptr[row]
                row_end = <unsigned long long> indptr[row + 1]
                for col in range(row_start, row_end):
                    if indices[col] == MALAT1_index:
                        MALAT1_mask[row] = True
                        break


def mito_mask_csr(const numeric[::1] data,
                  const signed_integer[::1] indices,
                  const signed_integer[::1] indptr,
                  char[::1] mt_genes,
                  const float max_mito_fraction,
                  char[::1] mito_mask,
                  unsigned num_threads):

    cdef unsigned row, row_sum, mt_sum, thread_index, \
        num_genes = indptr.shape[0] - 1
    cdef unsigned long long col
    cdef unsigned start_row, end_row

    num_threads = min(num_threads, num_genes)
    if num_threads <= 1:
        for row in range(num_genes):
            row_sum = 0
            mt_sum = 0
            for col in range(<unsigned long long> indptr[row],
                             <unsigned long long> indptr[row + 1]):
                row_sum = row_sum + <unsigned> data[col]
                if mt_genes[indices[col]]:
                    mt_sum = mt_sum + <unsigned> data[col]
            mito_mask[row] = row_sum > 0 and \
                (<float> mt_sum / row_sum) <= max_mito_fraction
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            start_row, end_row = \
                get_thread_offset(indptr, thread_index, num_threads)
            for row in range(start_row, end_row):
                row_sum = 0
                mt_sum = 0
                for col in range(<unsigned long long> indptr[row],
                                 <unsigned long long> indptr[row + 1]):
                    row_sum = row_sum + <unsigned> data[col]
                    if mt_genes[indices[col]]:
                        mt_sum = mt_sum + <unsigned> data[col]
                mito_mask[row] = row_sum > 0 and (
                    <float> mt_sum / row_sum) <= max_mito_fraction


def mito_mask_csc(const numeric[::1] data,
                  const signed_integer[::1] indices,
                  const signed_integer[::1] indptr,
                  char[::1] mt_genes,
                  const float max_mito_fraction,
                  char[::1] mito_mask,
                  unsigned num_threads):

    cdef unsigned thread_index, gene, row_sum, mt_sum, \
        num_genes = indptr.shape[0] - 1
    cdef unsigned long long cell, num_cells = mito_mask.shape[0]
    cdef unsigned start_col, end_col
    cdef vector[unsigned] row_sums_buffer, mt_sums_buffer
    cdef vector[vector[unsigned]] thread_row_sums, thread_mt_sums
    cdef unsigned[::1] row_sums, mt_sums

    num_threads = min(num_threads, num_genes)
    if num_threads <= 1:
        row_sums_buffer.resize(num_cells)
        row_sums = <unsigned[:num_cells]> row_sums_buffer.data()
        mt_sums_buffer.resize(num_cells)
        mt_sums = <unsigned[:num_cells]> mt_sums_buffer.data()
        for gene in range(num_genes):
            if mt_genes[gene]:
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    row_sums[indices[cell]] += <unsigned> data[cell]
                    mt_sums[indices[cell]] += <unsigned> data[cell]
            else:
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    row_sums[indices[cell]] += <unsigned> data[cell]
        for cell in range(num_cells):
            mito_mask[cell] = row_sums[cell] > 0 and \
                <float> mt_sums[cell] / row_sums[cell] <= max_mito_fraction
    else:
        # Store total counts per cell and total mitochondrial counts per cell
        # for each thread in temporary buffers, then aggregate at the end.
        thread_row_sums.resize(num_threads)
        thread_mt_sums.resize(num_threads)
        with nogil:
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_row_sums[thread_index].resize(num_cells)
                thread_mt_sums[thread_index].resize(num_cells)
                start_col, end_col = \
                    get_thread_offset(indptr, thread_index, num_threads)
                for gene in range(start_col, end_col):
                    if mt_genes[gene]:
                        for cell in range(
                                <unsigned long long> indptr[gene],
                                <unsigned long long> indptr[gene + 1]):
                            thread_row_sums[thread_index][indices[cell]] += \
                                <unsigned> data[cell]
                            thread_mt_sums[thread_index][indices[cell]] += \
                                <unsigned> data[cell]
                    else:
                        for cell in range(
                                <unsigned long long> indptr[gene],
                                <unsigned long long> indptr[gene + 1]):
                            thread_row_sums[thread_index][indices[cell]] += \
                                <unsigned> data[cell]

            # Populate the mask
            for cell in prange(num_cells, num_threads=num_threads):
                row_sum = 0
                mt_sum = 0
                for thread_index in range(num_threads):
                    row_sum = row_sum + thread_row_sums[thread_index][cell]
                    mt_sum = mt_sum + thread_mt_sums[thread_index][cell]
                mito_mask[cell] = row_sum > 0 and \
                    <float> mt_sum / row_sum <= max_mito_fraction


def qc_metrics_csr(const numeric[::1] data,
                   const signed_integer[::1] indices,
                   const signed_integer[::1] indptr,
                   char[::1] mt_genes,
                   unsigned[::1] num_counts,
                   unsigned[::1] num_genes_per_cell,
                   float[::1] mito_fraction,
                   unsigned num_threads):

    cdef unsigned row, row_sum, mt_sum, thread_index, \
        num_genes = indptr.shape[0] - 1
    cdef unsigned long long col
    cdef unsigned start_row, end_row

    num_threads = min(num_threads, num_genes)
    if num_threads <= 1:
        for row in range(num_genes):
            row_sum = 0
            mt_sum = 0
            for col in range(<unsigned long long> indptr[row],
                             <unsigned long long> indptr[row + 1]):
                row_sum = row_sum + <unsigned> data[col]
                if mt_genes[indices[col]]:
                    mt_sum = mt_sum + <unsigned> data[col]
            num_counts[row] = row_sum
            num_genes_per_cell[row] = indptr[row + 1] - indptr[row]
            mito_fraction[row] = <float> mt_sum / row_sum
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            start_row, end_row = \
                get_thread_offset(indptr, thread_index, num_threads)
            for row in range(start_row, end_row):
                row_sum = 0
                mt_sum = 0
                for col in range(<unsigned long long> indptr[row],
                                 <unsigned long long> indptr[row + 1]):
                    row_sum = row_sum + <unsigned> data[col]
                    if mt_genes[indices[col]]:
                        mt_sum = mt_sum + <unsigned> data[col]
                num_counts[row] = row_sum
                num_genes_per_cell[row] = indptr[row + 1] - indptr[row]
                mito_fraction[row] = <float> mt_sum / row_sum


def qc_metrics_csc(const numeric[::1] data,
                   const signed_integer[::1] indices,
                   const signed_integer[::1] indptr,
                   char[::1] mt_genes,
                   unsigned[::1] num_counts,
                   unsigned[::1] num_genes_per_cell,
                   float[::1] mito_fraction,
                   unsigned num_threads):

    cdef unsigned thread_index, gene, row_sum, mt_sum, \
        num_genes = indptr.shape[0] - 1
    cdef volatile float mt_sum_volatile
    cdef unsigned long long cell, num_cells = num_counts.shape[0]
    cdef unsigned start_col, end_col
    cdef vector[unsigned] mt_sums_buffer
    cdef vector[vector[unsigned]] thread_row_sums, thread_mt_sums, \
        thread_num_genes_per_cell
    cdef unsigned[::1] mt_sums

    num_threads = min(num_threads, num_genes)
    if num_threads <= 1:
        mt_sums_buffer.resize(num_cells)
        mt_sums = <unsigned[:num_cells]> mt_sums_buffer.data()
        num_counts[:] = 0
        num_genes_per_cell[:] = 0
        for gene in range(num_genes):
            if mt_genes[gene]:
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    num_counts[indices[cell]] += <unsigned> data[cell]
                    mt_sums[indices[cell]] += <unsigned> data[cell]
                    num_genes_per_cell[indices[cell]] += 1
            else:
                for cell in range(<unsigned long long> indptr[gene],
                                  <unsigned long long> indptr[gene + 1]):
                    num_counts[indices[cell]] += <unsigned> data[cell]
                    num_genes_per_cell[indices[cell]] += 1
        for cell in range(num_cells):
            # Forcing the numerator to be volatile is necessary to ensure
            # exactly the same floating-point behavior as the parallel CSC
            # version and the CSR versions under -ffast-math
            mt_sum_volatile = mt_sums[cell]
            row_sum = num_counts[cell]
            mito_fraction[cell] = mt_sum_volatile / row_sum
    else:
        # Store total counts per cell, genes with non-zero expression per cell,
        # and total mitochondrial counts per cell for each thread in temporary
        # buffers, then aggregate at the end.
        thread_row_sums.resize(num_threads)
        thread_mt_sums.resize(num_threads)
        thread_num_genes_per_cell.resize(num_threads)
        with nogil:
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_row_sums[thread_index].resize(num_cells)
                thread_mt_sums[thread_index].resize(num_cells)
                thread_num_genes_per_cell[thread_index].resize(num_cells)
                start_col, end_col = \
                    get_thread_offset(indptr, thread_index, num_threads)
                for gene in range(start_col, end_col):
                    if mt_genes[gene]:
                        for cell in range(
                                <unsigned long long> indptr[gene],
                                <unsigned long long> indptr[gene + 1]):
                            thread_row_sums[thread_index][indices[cell]] += \
                                <unsigned> data[cell]
                            thread_mt_sums[thread_index][indices[cell]] += \
                                <unsigned> data[cell]
                            thread_num_genes_per_cell[thread_index][
                                indices[cell]] += 1
                    else:
                        for cell in range(
                                <unsigned long long> indptr[gene],
                                <unsigned long long> indptr[gene + 1]):
                            thread_row_sums[thread_index][indices[cell]] += \
                                <unsigned> data[cell]
                            thread_num_genes_per_cell[thread_index][
                                indices[cell]] += 1

            # Populate the output arrays
            for cell in prange(num_cells, num_threads=num_threads):
                row_sum = 0
                mt_sum = 0
                num_genes = 0
                for thread_index in range(num_threads):
                    row_sum = row_sum + thread_row_sums[thread_index][cell]
                    mt_sum = mt_sum + thread_mt_sums[thread_index][cell]
                    num_genes = num_genes + \
                        thread_num_genes_per_cell[thread_index][cell]
                num_counts[cell] = row_sum
                num_genes_per_cell[cell] = num_genes
                mito_fraction[cell] = <float> mt_sum / row_sum