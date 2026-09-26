from cython.parallel cimport parallel, prange, threadid
from libc.string cimport memcpy
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from .cyutils cimport get_thread_offset, numeric, signed_integer, \
    uninitialized_vector


cdef inline float log1p(const float x) noexcept nogil:
    # log1p(x) for finite x >= 0, adapted from fdlibm's `log1pf()`. This
    # version is vectorized by the compiler, unlike the C++ stdlib's `log1p()`,
    # which is only vectorized on GCC with glibc >= 2.35. Not compatible with
    # fast-math semantics.
    #
    # log1p(x) is calculated as log(u) + c / u, where u is 1 + x rounded to a
    # float and c = x - (u - 1) is the part of x lost in that rounding, which
    # matters when x is small. To calculate log(u), u is split into m * 2^k
    # with m in [sqrt(1/2), sqrt(2)), so that log(u) = k * log(2) + log(m), and
    # log(m) is approximated with fdlibm's polynomial in s = (m - 1) / (m + 1).
    cdef float coefficient_1 = <float> 0.66666662693, \
        coefficient_2 = <float> 0.40000972152, \
        coefficient_3 = <float> 0.28498786688, \
        coefficient_4 = <float> 0.24279078841
    # log(2), split into a high part with trailing zero bits, so that
    # k * log2_high is exact, and a low part with the remainder
    cdef float log2_high = <float> 0.69313812256, \
        log2_low = <float> 9.0580006145e-06
    cdef float u, c, m, f, s, z, w, polynomial, half_f_squared
    cdef unsigned bits
    cdef int k  # int since x86 can't vectorize unsigned-to-float pre-AVX-512
    u = x + <float> 1
    c = (x - (u - <float> 1)) / u
    # Shift u's bit pattern by the difference between the bit patterns of 1
    # and sqrt(1/2), so that values of m in [sqrt(1/2), sqrt(2)) map to an
    # exponent of 0; k is then the exponent (u >= 1, so this can't wrap).
    # memcpy() is the portable way to read a float's bits, and compilers turn
    # it into a register move.
    memcpy(&bits, &u, sizeof(float))
    bits += 0x3f800000u - 0x3f3504f3u
    k = (bits >> 23) - 127
    # Reconstruct m from the mantissa bits, undoing the shift
    bits = (bits & 0x007fffffu) + 0x3f3504f3u
    memcpy(&m, &bits, sizeof(float))
    f = m - <float> 1
    s = f / (f + <float> 2)
    z = s * s
    w = z * z
    polynomial = z * (coefficient_1 + w * coefficient_3) + \
        w * (coefficient_2 + w * coefficient_4)
    half_f_squared = f * f / <float> 2
    # Add the terms from smallest to largest, so that the rounding errors of
    # the small terms are absorbed before the large terms are added
    return s * (half_f_squared + polynomial) + (k * log2_low + c) - \
        half_f_squared + f + k * log2_high


def normalize_csr(const numeric[::1] data,
                  const signed_integer[::1] indices,
                  const signed_integer[::1] indptr,
                  char[::1] QC_column,
                  float[::1] normalized_data,
                  unsigned long long[::1] row_sums,
                  const unsigned long long num_cells,
                  const unsigned method_number,
                  unsigned num_threads):

    cdef bint has_QC_column = QC_column.shape[0] != 0
    cdef unsigned i, thread_index
    cdef unsigned long long row_sum, j, num_QCed_cells, thread_total_sum, \
        thread_num_QCed_cells, total_sum = 0
    cdef float normalization_factor, inverse_size_factor, \
        new_normalization_factor, new_inverse_size_factor, new_row_sum, \
        new_total_sum = 0
    cdef pair[unsigned, unsigned] row_range
    cdef uninitialized_vector[unsigned] thread_nums_QCed_cells
    cdef uninitialized_vector[unsigned long long] thread_total_sums
    cdef uninitialized_vector[float] new_row_sums

    num_threads = min(num_threads, num_cells)
    if num_threads <= 1:
        if method_number == 0:  # 'logCP10k'
            # Step 1 and 2: since the normalization factor is a constant
            # 10,000, calculate each cell's row sum and inverse size factor,
            # then multiply its counts by it and log1p-transform them, while
            # the row is still in cache
            normalization_factor = 10000
            for i in range(num_cells):
                row_sum = 0
                for j in range(<unsigned long long> indptr[i],
                               <unsigned long long> indptr[i + 1]):
                    row_sum += <unsigned long long> data[j]
                row_sums[i] = row_sum
                inverse_size_factor = normalization_factor / row_sum
                for j in range(<unsigned long long> indptr[i],
                               <unsigned long long> indptr[i + 1]):
                    normalized_data[j] = \
                        log1p(<float> data[j] * inverse_size_factor)
        else:
            # Step 1a and 1b: calculate row sums, and take their mean (across
            # cells passing QC, if `QC_column` was specified) as the
            # normalization factor for the size factor calculation
            if not has_QC_column:
                for i in range(num_cells):
                    row_sum = 0
                    for j in range(<unsigned long long> indptr[i],
                                   <unsigned long long> indptr[i + 1]):
                        row_sum += <unsigned long long> data[j]
                    row_sums[i] = row_sum
                    total_sum += row_sum
                num_QCed_cells = num_cells
            else:
                num_QCed_cells = 0
                for i in range(num_cells):
                    row_sum = 0
                    for j in range(<unsigned long long> indptr[i],
                                   <unsigned long long> indptr[i + 1]):
                        row_sum += <unsigned long long> data[j]
                    row_sums[i] = row_sum
                    if QC_column[i]:
                        total_sum += row_sum
                        num_QCed_cells += 1
            normalization_factor = <float> total_sum / num_QCed_cells

        if method_number == 1:  # 'log1pPF'
            # Step 1c and 2: calculate each cell's inverse size factor and
            # multiply all counts for that cell by it, then log1p-transform
            for i in range(num_cells):
                inverse_size_factor = normalization_factor / row_sums[i]
                for j in range(<unsigned long long> indptr[i],
                               <unsigned long long> indptr[i + 1]):
                    normalized_data[j] = \
                        log1p(<float> data[j] * inverse_size_factor)
        elif method_number == 2:  # 'PFlog1pPF'
            # Step 1c, 2, and 3a: in addition to calculating each cell's
            # size factor and multiplying the counts by it, also calculate
            # the new row sums and total sum for a second round of proportional
            # fitting. Sum each cell's new row sum in a separate loop from the
            # log1p, since under clang, the floating-point sum would stop the
            # log1p from being vectorized. Each row is summed in order, which
            # (given sorted indices) is the order in which `normalize_csc()`
            # accumulates it.
            new_row_sums.resize(num_cells)
            for i in range(num_cells):
                new_row_sum = 0
                inverse_size_factor = normalization_factor / row_sums[i]
                for j in range(<unsigned long long> indptr[i],
                               <unsigned long long> indptr[i + 1]):
                    normalized_data[j] = \
                        log1p(<float> data[j] * inverse_size_factor)
                for j in range(<unsigned long long> indptr[i],
                               <unsigned long long> indptr[i + 1]):
                    new_row_sum += normalized_data[j]
                new_row_sums[i] = new_row_sum
            if not has_QC_column:
                for i in range(num_cells):
                    new_total_sum += new_row_sums[i]
            else:
                for i in range(num_cells):
                    if QC_column[i]:
                        new_total_sum += new_row_sums[i]
            new_normalization_factor = new_total_sum / num_QCed_cells

            # Step 3b: calculate each cell's new inverse size factor and
            # multiply all normalized counts for that cell by it
            for i in range(num_cells):
                new_inverse_size_factor = \
                    new_normalization_factor / new_row_sums[i]
                for j in range(<unsigned long long> indptr[i],
                               <unsigned long long> indptr[i + 1]):
                    normalized_data[j] *= new_inverse_size_factor
    else:
        with nogil:
            if method_number == 0:  # 'logCP10k'
                # Step 1 and 2: as in the single-threaded version, in a single
                # pass over each row
                normalization_factor = 10000
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    row_range = \
                        get_thread_offset(indptr, thread_index, num_threads)
                    for i in range(row_range.first, row_range.second):
                        row_sum = 0
                        for j in range(<unsigned long long> indptr[i],
                                       <unsigned long long> indptr[i + 1]):
                            row_sum = row_sum + <unsigned long long> data[j]
                        row_sums[i] = row_sum
                        inverse_size_factor = normalization_factor / row_sum
                        for j in range(<unsigned long long> indptr[i],
                                       <unsigned long long> indptr[i + 1]):
                            normalized_data[j] = \
                                log1p(<float> data[j] * inverse_size_factor)
            else:
                # Step 1a and 1b: calculate row sums, and take their mean
                # (across cells passing QC, if `QC_column` was specified) as
                # the normalization factor for the size factor calculation
                thread_total_sums.resize(num_threads)
                if not has_QC_column:
                    with parallel(num_threads=num_threads):
                        thread_index = threadid()
                        row_range = get_thread_offset(
                            indptr, thread_index, num_threads)
                        thread_total_sum = 0
                        for i in range(row_range.first, row_range.second):
                            row_sum = 0
                            for j in range(<unsigned long long> indptr[i],
                                           <unsigned long long> indptr[i + 1]):
                                row_sum = \
                                    row_sum + <unsigned long long> data[j]
                            row_sums[i] = row_sum
                            thread_total_sum = thread_total_sum + row_sum
                        thread_total_sums[thread_index] = thread_total_sum
                    for thread_index in range(num_threads):
                        total_sum += thread_total_sums[thread_index]
                    num_QCed_cells = num_cells
                else:
                    num_QCed_cells = 0
                    thread_nums_QCed_cells.resize(num_threads)
                    with parallel(num_threads=num_threads):
                        thread_index = threadid()
                        row_range = get_thread_offset(
                            indptr, thread_index, num_threads)
                        thread_total_sum = 0
                        thread_num_QCed_cells = 0
                        for i in range(row_range.first, row_range.second):
                            row_sum = 0
                            for j in range(<unsigned long long> indptr[i],
                                           <unsigned long long> indptr[i + 1]):
                                row_sum = \
                                    row_sum + <unsigned long long> data[j]
                            row_sums[i] = row_sum
                            if QC_column[i]:
                                thread_total_sum = thread_total_sum + row_sum
                                thread_num_QCed_cells = \
                                    thread_num_QCed_cells + 1
                        thread_total_sums[thread_index] = thread_total_sum
                        thread_nums_QCed_cells[thread_index] = \
                            thread_num_QCed_cells
                    for thread_index in range(num_threads):
                        total_sum += thread_total_sums[thread_index]
                        num_QCed_cells += thread_nums_QCed_cells[thread_index]
                normalization_factor = <float> total_sum / num_QCed_cells
            if method_number == 1:  # 'log1pPF'
                # Step 1c and 2: calculate each cell's inverse size factor and
                # multiply all counts for that cell by it, then log1p-transform
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    row_range = \
                        get_thread_offset(indptr, thread_index, num_threads)
                    for i in range(row_range.first, row_range.second):
                        inverse_size_factor = \
                            normalization_factor / row_sums[i]
                        for j in range(<unsigned long long> indptr[i],
                                       <unsigned long long> indptr[i + 1]):
                            normalized_data[j] = \
                                log1p(<float> data[j] * inverse_size_factor)
            elif method_number == 2:  # 'PFlog1pPF'
                # Step 1c, 2, and 3a: in addition to calculating each cell's
                # size factor and multiplying the counts by it, also calculate
                # the new row sums and total sum for a second round of
                # proportional fitting. As in the single-threaded version, sum
                # each cell's new row sum in a separate loop from the log1p
                # (since under clang, the floating-point sum would stop the
                # log1p from being vectorized), and sum `new_row_sums`
                # single-threaded afterwards to get the same order of
                # operations as the CSC version.
                new_row_sums.resize(num_cells)
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    row_range = \
                        get_thread_offset(indptr, thread_index, num_threads)
                    for i in range(row_range.first, row_range.second):
                        new_row_sum = 0
                        inverse_size_factor = \
                            normalization_factor / row_sums[i]
                        for j in range(<unsigned long long> indptr[i],
                                       <unsigned long long> indptr[i + 1]):
                            normalized_data[j] = \
                                log1p(<float> data[j] * inverse_size_factor)
                        for j in range(<unsigned long long> indptr[i],
                                       <unsigned long long> indptr[i + 1]):
                            new_row_sum = new_row_sum + normalized_data[j]
                        new_row_sums[i] = new_row_sum
                if not has_QC_column:
                    for i in range(num_cells):
                        new_total_sum += new_row_sums[i]
                else:
                    for i in range(num_cells):
                        if QC_column[i]:
                            new_total_sum += new_row_sums[i]
                new_normalization_factor = new_total_sum / num_QCed_cells

                # Step 3b: calculate each cell's new inverse size factor and
                # multiply all normalized counts for that cell by it
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    row_range = \
                        get_thread_offset(indptr, thread_index, num_threads)
                    for i in range(row_range.first, row_range.second):
                        new_inverse_size_factor = \
                            new_normalization_factor / new_row_sums[i]
                        for j in range(<unsigned long long> indptr[i],
                                       <unsigned long long> indptr[i + 1]):
                            normalized_data[j] *= new_inverse_size_factor


def normalize_csc(const numeric[::1] data,
                  const signed_integer[::1] indices,
                  const signed_integer[::1] indptr,
                  char[::1] QC_column,
                  float[::1] normalized_data,
                  unsigned long long[::1] row_sums,
                  const unsigned long long num_cells,
                  const unsigned method_number,
                  unsigned num_threads):

    cdef bint has_QC_column = QC_column.shape[0] != 0
    cdef unsigned i, thread_index
    cdef unsigned long long j, num_QCed_cells, start, end, chunk_size, \
        num_elements = data.shape[0], total_sum = 0
    cdef float normalization_factor, inverse_size_factor, \
        new_normalization_factor, new_inverse_size_factor, new_total_sum = 0
    cdef vector[vector[unsigned long long]] thread_row_sums
    cdef vector[float] new_row_sums

    num_threads = min(num_threads, min(num_cells, num_elements))
    if num_threads <= 1:
        # Step 1a: calculate row sums
        row_sums[:] = 0
        for j in range(num_elements):
            row_sums[indices[j]] += <unsigned long long> data[j]

        # Step 1b: calculate the normalization factor for the size factor
        # calculation
        if method_number == 0:  # 'logCP10k'
            normalization_factor = 10000
        else:
            # Take the mean of the row sums (across cells passing QC, if
            # `QC_column` was specified) as the size factor
            if not has_QC_column:
                for i in range(num_cells):
                    total_sum += row_sums[i]
                num_QCed_cells = num_cells
            else:
                num_QCed_cells = 0
                for i in range(num_cells):
                    if QC_column[i]:
                        total_sum += row_sums[i]
                        num_QCed_cells += 1
            normalization_factor = <float> total_sum / num_QCed_cells

        # Step 1c and 2: multiply each count by its cell's inverse size
        # factor, then log1p-transform
        for j in range(num_elements):
            inverse_size_factor = normalization_factor / row_sums[indices[j]]
            normalized_data[j] = log1p(<float> data[j] * inverse_size_factor)

        if method_number == 2:  # 'PFlog1pPF'
            # Step 3a: calculate the new row sums and total sum for a second
            # round of proportional fitting
            new_row_sums.resize(num_cells)
            for j in range(num_elements):
                new_row_sums[indices[j]] += normalized_data[j]
            if not has_QC_column:
                for i in range(num_cells):
                    new_total_sum += new_row_sums[i]
            else:
                for i in range(num_cells):
                    if QC_column[i]:
                        new_total_sum += new_row_sums[i]
            new_normalization_factor = new_total_sum / num_QCed_cells

            # Step 3b: calculate each cell's new inverse size factor and
            # multiply all normalized counts for that cell by it
            for j in range(num_elements):
                new_inverse_size_factor = \
                    new_normalization_factor / new_row_sums[indices[j]]
                normalized_data[j] *= new_inverse_size_factor
    else:
        with nogil:
            # Step 1a: calculate row sums. Store row sums for each thread in a
            # temporary buffer, then aggregate at the end. As an optimization,
            # put the row sums for the last thread
            # (`thread_index == num_threads - 1`) directly into the final
            # `row_sums` vector.
            thread_row_sums.resize(num_threads - 1)
            chunk_size = (num_elements + num_threads - 1) / num_threads
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                start = thread_index * chunk_size
                if thread_index == num_threads - 1:
                    end = num_elements
                    for j in range(start, end):
                        row_sums[indices[j]] += <unsigned long long> data[j]
                else:
                    thread_row_sums[thread_index].resize(num_cells)
                    end = min(start + chunk_size, num_elements)
                    for j in range(start, end):
                        thread_row_sums[thread_index][indices[j]] += \
                            <unsigned long long> data[j]
            for thread_index in range(num_threads - 1):
                for i in range(num_cells):
                    row_sums[i] += thread_row_sums[thread_index][i]

            # Step 1b: calculate the normalization factor for the size factor
            # calculation
            if method_number == 0:  # 'logCP10k'
                normalization_factor = 10000
            else:
                # Take the mean of the row sums (across cells passing QC, if
                # `QC_column` was specified) as the size factor
                if not has_QC_column:
                    for i in prange(num_cells, num_threads=num_threads):
                        total_sum += row_sums[i]
                    num_QCed_cells = num_cells
                else:
                    num_QCed_cells = 0
                    for i in prange(num_cells, num_threads=num_threads):
                        if QC_column[i]:
                            total_sum += row_sums[i]
                            num_QCed_cells += 1
                normalization_factor = <float> total_sum / num_QCed_cells

            # Step 1c and 2: multiply each count by its cell's inverse size
            # factor, then log1p-transform
            for j in prange(num_elements, num_threads=num_threads):
                inverse_size_factor = \
                    normalization_factor / row_sums[indices[j]]
                normalized_data[j] = \
                    log1p(<float> data[j] * inverse_size_factor)

            if method_number == 2:  # 'PFlog1pPF'
                # Step 3a: calculate the new row sums and total sum for a
                # second round of proportional fitting. This must be done
                # single-threaded to maintain a consistent order of operations
                # and avoid differences due to floating-point error between the
                # single-threaded and parallel versions.
                new_row_sums.resize(num_cells)
                for j in range(num_elements):
                    new_row_sums[indices[j]] += normalized_data[j]
                if not has_QC_column:
                    for i in range(num_cells):
                        new_total_sum += new_row_sums[i]
                else:
                    for i in range(num_cells):
                        if QC_column[i]:
                            new_total_sum += new_row_sums[i]
                new_normalization_factor = new_total_sum / num_QCed_cells

                # Step 3b: calculate each cell's new inverse size factor and
                # multiply all normalized counts for that cell by it
                for j in prange(num_elements, num_threads=num_threads):
                    new_inverse_size_factor = \
                        new_normalization_factor / new_row_sums[indices[j]]
                    normalized_data[j] *= new_inverse_size_factor
