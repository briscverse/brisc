# Code for calculating norm factors using the approach of edgeR's
# `calcNormFactors()`

from cython.parallel cimport parallel, threadid
from libcpp.algorithm cimport sort
from libcpp.cmath cimport abs, exp, exp2, log, log2
from libcpp.vector cimport vector
from .cyutils cimport numeric, uninitialized_vector


ctypedef fused sum_type:
    long long
    unsigned long long
    float
    double


cdef extern from * nogil:
    """
    struct Compare {
        const float* data;
        Compare() noexcept {}
        Compare(const float* d) noexcept : data(d) {}
        bool operator()(unsigned a, unsigned b) const noexcept {
            return data[a] < data[b];
        }
    };
    """
    cdef cppclass Compare:
        Compare(const float*) noexcept
        bint operator()(unsigned, unsigned) noexcept


cdef inline void argsort(const float* arr, unsigned* indices,
                         const unsigned n) noexcept nogil:
    cdef unsigned i
    for i in range(n):
        indices[i] = i
    sort(indices, indices + n, Compare(arr))


cdef inline void rankdata(const float* data,
                          unsigned* indices,
                          float* ranks,
                          const unsigned n) noexcept nogil:
    cdef unsigned i = 0, start_pos = 0
    cdef float current_val, rank
    cdef bint end = False

    argsort(&data[0], &indices[0], n)

    while True:
        current_val = data[indices[i]]

        # Count elements equal to current value
        i += 1
        if i == n:
            end = True
        else:
            while data[indices[i]] == current_val:
                i += 1
                if i == n:
                    end = True
                    break

        # Assign average rank to all tied elements
        rank = <float> (start_pos + i + 1) * <float> 0.5
        while start_pos < i:
            ranks[indices[start_pos]] = rank
            start_pos += 1

        if end:
            break


def calc_norm_factors(const numeric[:, ::1] X,
                      const float logratio_trim,
                      const float sum_trim,
                      const float A_cutoff,
                      const unsigned ref_sample,
                      float[::1] norm_factors,
                      sum_type[::1] library_size,
                      unsigned num_threads):
    cdef numeric ref_count, count
    cdef sum_type ref_library_size
    cdef unsigned i, j, n, loL, hiL, loS, hiS, thread_index, start_sample, \
        end_sample, num_samples = X.shape[0], num_genes = X.shape[1]
    cdef float inverse_ref_library_size, inverse_relative_library_size_i, \
        logR_, absE_, inverse_library_size, total_inverse_library_size, \
        norm_factor, total_weight, variance, weight, scale
    cdef bint large_enough_logR
    cdef uninitialized_vector[float] inverse_relative_library_size_buffer, \
        log_normalized_X_ref_buffer, logR_buffer, absE_buffer, \
        logR_rank_buffer, absE_rank_buffer
    cdef uninitialized_vector[numeric] counts_buffer, ref_counts_buffer
    cdef uninitialized_vector[unsigned] indices_buffer
    cdef vector[vector[float]] thread_logR, thread_absE, \
        thread_logR_rank, thread_absE_rank
    cdef vector[vector[numeric]] thread_counts, thread_ref_counts
    cdef vector[vector[unsigned]] thread_indices
    cdef float[::1] inverse_relative_library_size, log_normalized_X_ref, \
        logR, absE, logR_rank, absE_rank
    cdef numeric[::1] counts, ref_counts
    cdef unsigned[::1] indices

    inverse_relative_library_size_buffer.resize(num_samples)
    log_normalized_X_ref_buffer.resize(num_genes)
    inverse_relative_library_size = \
        <float[:num_samples]> inverse_relative_library_size_buffer.data()
    log_normalized_X_ref = \
        <float[:num_genes]> log_normalized_X_ref_buffer.data()

    # Calculate each sample's library size relative to the
    # reference sample's (to use in the `logR` calculation)
    ref_library_size = library_size[ref_sample]
    for i in range(num_samples):
        inverse_relative_library_size[i] = \
            <float> ref_library_size / <float> library_size[i]

    # Calculate each gene's log normalized expression (to use in the `absE`
    # calculation)
    inverse_ref_library_size = <float> 1 / <float> ref_library_size
    for j in range(num_genes):
        count = X[ref_sample, j]
        log_normalized_X_ref[j] = \
            log2(<float> count * inverse_ref_library_size)

    # Calculate the normalization factor for each sample
    num_threads = min(num_threads, num_samples)
    if num_threads <= 1:
        logR_buffer.resize(num_genes)
        absE_buffer.resize(num_genes)
        logR_rank_buffer.resize(num_genes)
        absE_rank_buffer.resize(num_genes)
        counts_buffer.resize(num_genes)
        ref_counts_buffer.resize(num_genes)
        indices_buffer.resize(num_genes)

        logR = <float[:num_genes]> logR_buffer.data()
        absE = <float[:num_genes]> absE_buffer.data()
        logR_rank = <float[:num_genes]> logR_rank_buffer.data()
        absE_rank = <float[:num_genes]> absE_rank_buffer.data()
        counts = <numeric[:num_genes]> counts_buffer.data()
        ref_counts = <numeric[:num_genes]> ref_counts_buffer.data()
        indices = <unsigned[:num_genes]> indices_buffer.data()

        for i in range(num_samples):
            inverse_library_size = <float> 1 / <float> library_size[i]
            inverse_relative_library_size_i = inverse_relative_library_size[i]
            large_enough_logR = False
            n = 0
            for j in range(num_genes):
                # Get the count and reference count for this gene; skip the
                # gene if either are 0
                ref_count = X[ref_sample, j]
                if not ref_count:
                    continue

                count = X[i, j]
                if not count:
                    continue

                # Calculate the log ratio of expression accounting for library
                # size
                logR_ = log2(inverse_relative_library_size_i * (
                    <float> count / <float> ref_count))

                # Calculate "absolute expression": the average log2 expression
                # of this gene between this sample and the reference sample
                absE_ = <float> 0.5 * (
                    log2(<float> count * inverse_library_size) +
                    log_normalized_X_ref[j])

                # Cutoff based on `A_cutoff`
                if absE_ <= A_cutoff:
                    continue

                # Store `logR`, `absE`, and the count for genes passing the
                # infinite value and `A_cutoff` filters above
                logR[n] = logR_
                absE[n] = absE_
                counts[n] = count
                ref_counts[n] = ref_count
                n += 1

                # Keep track of whether any gene's `logR` is above 1e-6 in
                # magnitude for this sample
                large_enough_logR |= abs(logR_) >= <float> 1e-6

            # If every gene's `logR` is below 1e-6 in magnitude for this sample
            # (i.e. expression is extremely low across the board), set the
            # sample's norm factor to 1
            if not large_enough_logR:
                norm_factors[i] = 1
                continue

            # Rank genes by `logR` and `absE`
            loL = <unsigned>(n * logratio_trim)
            hiL = n - loL
            loS = <unsigned>(n * sum_trim)
            hiS = n - loS
            rankdata(&logR[0], &indices[0], &logR_rank[0], n)
            rankdata(&absE[0], &indices[0], &absE_rank[0], n)

            # Calculate the norm factors themselves. Find genes with
            # intermediate ranks of both `logR` and `absE` (this is the
            # "trimmed" part, the "T" in "TMM"). The norm factors are 2 to the
            # power of the weighted average of the logRs for these
            # intermediate-ranked genes, where the weights are the inverse
            # asymptotic variances.
            total_inverse_library_size = \
                inverse_library_size + inverse_ref_library_size
            norm_factor = 0
            total_weight = 0
            for j in range(n):
                if loL + 1 <= logR_rank[j] <= hiL and \
                        loS + 1 <= absE_rank[j] <= hiS:
                    variance = <float> 1 / <float> counts[j] + \
                        <float> 1 / <float> ref_counts[j] - \
                        total_inverse_library_size
                    weight = <float> 1 / variance
                    norm_factor += weight * logR[j]
                    total_weight += weight
            # Results will be missing if the two libraries share no
            # features with positive counts; in this case, set to 1
            norm_factor = \
                exp2(norm_factor / total_weight) if total_weight else 1
            norm_factors[i] = norm_factor
    else:
        # Same as the single-threaded version, but with per-thread buffers
        thread_logR.resize(num_threads)
        thread_absE.resize(num_threads)
        thread_logR_rank.resize(num_threads)
        thread_absE_rank.resize(num_threads)
        thread_counts.resize(num_threads)
        thread_ref_counts.resize(num_threads)
        thread_indices.resize(num_threads)

        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            thread_logR[thread_index].resize(num_genes)
            thread_absE[thread_index].resize(num_genes)
            thread_logR_rank[thread_index].resize(num_genes)
            thread_absE_rank[thread_index].resize(num_genes)
            thread_counts[thread_index].resize(num_genes)
            thread_ref_counts[thread_index].resize(num_genes)
            thread_indices[thread_index].resize(num_genes)

            start_sample = (thread_index * num_samples) / num_threads
            end_sample = ((thread_index + 1) * num_samples) / num_threads \
                if thread_index != num_threads - 1 else num_samples
            for i in range(start_sample, end_sample):
                inverse_library_size = <float> 1 / <float> library_size[i]
                inverse_relative_library_size_i = \
                    inverse_relative_library_size[i]
                large_enough_logR = False
                n = 0
                for j in range(num_genes):
                    ref_count = X[ref_sample, j]
                    if not ref_count:
                        continue
                    count = X[i, j]
                    if not count:
                        continue
                    logR_ = log2(inverse_relative_library_size_i * (
                        <float> count / <float> ref_count))
                    absE_ = <float> 0.5 * (
                        log2(<float> count * inverse_library_size) +
                        log_normalized_X_ref[j])
                    if absE_ <= A_cutoff:
                        continue
                    thread_logR[thread_index][n] = logR_
                    thread_absE[thread_index][n] = absE_
                    thread_counts[thread_index][n] = count
                    thread_ref_counts[thread_index][n] = ref_count
                    n = n + 1
                    large_enough_logR = \
                        large_enough_logR | (abs(logR_) >= <float> 1e-6)
                if not large_enough_logR:
                    norm_factors[i] = 1
                    continue
                loL = <unsigned>(n * logratio_trim)
                hiL = n - loL
                loS = <unsigned>(n * sum_trim)
                hiS = n - loS
                rankdata(&thread_logR[thread_index][0],
                         &thread_indices[thread_index][0],
                         &thread_logR_rank[thread_index][0], n)
                rankdata(&thread_absE[thread_index][0],
                         &thread_indices[thread_index][0],
                         &thread_absE_rank[thread_index][0], n)
                total_inverse_library_size = \
                    inverse_library_size + inverse_ref_library_size
                norm_factor = 0
                total_weight = 0
                for j in range(n):
                    if loL + 1 <= thread_logR_rank[thread_index][j] <= hiL \
                            and loS + 1 <= thread_absE_rank[thread_index][j] \
                            <= hiS:
                        variance = \
                            <float> 1 / \
                                <float> thread_counts[thread_index][j] + \
                            <float> 1 / \
                                <float> thread_ref_counts[thread_index][j] - \
                            total_inverse_library_size
                        weight = <float> 1 / variance
                        norm_factor = \
                            norm_factor + weight * thread_logR[thread_index][j]
                        total_weight = total_weight + weight
                norm_factor = \
                    exp2(norm_factor / total_weight) if total_weight else 1
                norm_factors[i] = norm_factor

    # Normalize factors across samples so that they multiply to 1
    scale = 0
    for i in range(num_samples):
        scale += log(norm_factors[i])
    scale = exp(-scale / num_samples)
    for i in range(num_samples):
        norm_factors[i] *= scale

    # Multiply norm factors by library sizes
    for i in range(num_samples):
        norm_factors[i] *= <float> library_size[i]