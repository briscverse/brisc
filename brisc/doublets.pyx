# Functionality for doublet detection

from cython.parallel cimport parallel, prange, threadid
from libc.float cimport FLT_MAX
from libcpp.algorithm cimport nth_element, sort
from libcpp.cmath cimport abs, erfc, exp, floor, log, log1p, sqrt
from libcpp.pair cimport pair
from .cyutils cimport get_thread_offset, max_heap_pop, max_heap_replace_top, \
    numeric, signed_integer, srand, rand, randint


cdef double log_one_half = -0.6931471805599453
cdef double log_sqrt_2_pi = 0.91893853320467274
cdef double one_over_sqrt_2 = 0.70710678118654752
cdef double one_over_sqrt_pi = 0.5641895835477563
cdef double[5] gamma_A = [
    8.11614167470508450300E-4, -5.95061904284301438324E-4,
    7.93650340457716943945E-4, -2.77777777730099687205E-3,
    8.33333333333331927722E-2]
cdef double[6] gamma_B = [
    -1.37825152569120859100E3, -3.88016315134637840924E4,
    -3.31612992738871184744E5, -1.16237097492762307383E6,
    -1.72173700820839662146E6, -8.53555664245765465627E5]
cdef double[6] gamma_C = [
    -3.51815701436523470549E2, -1.70642106651881159223E4,
    -2.20528590553854454839E5, -1.13933444367982507207E6,
    -2.53252307177582951285E6, -2.01889141433532773231E6]
cdef double[6] P = [
    0.5641895835477550741253201704,
    1.275366644729965952479585264,
    5.019049726784267463450058,
    6.1602098531096305440906,
    7.409740605964741794425,
    2.97886562639399288862]
cdef double[6] Q = [
    2.260528520767326969591866945,
    9.396034016235054150430579648,
    12.0489519278551290360340491,
    17.08144074746600431571095,
    9.608965327192787870698,
    3.3690752069827527677]


cdef inline double p1evl(const double x,
                         const double* coef,
                         const unsigned N) noexcept nogil:
    cdef double ans
    cdef unsigned i

    ans = x + coef[0]
    for i in range(1, N):
        ans = ans * x + coef[i]
    return ans


cdef inline double polevl(const double x,
                          const double* coef,
                          const unsigned N) noexcept nogil:
    cdef double ans
    cdef unsigned i

    ans = coef[0]
    for i in range(1, N):
        ans = ans * x + coef[i]
    return ans


cdef inline double log_erfc(const double x) noexcept nogil:
    # Based on GSL's gsl_sf_log_erfc_e at
    # github.com/ampl/gsl/blob/master/specfunc/erfc.c#L306

    cdef double y, series

    if x * x < 0.02460783300575925:
        y = x * one_over_sqrt_pi
        series = 0.00048204
        series = y * series - 0.00142906
        series = y * series + 0.0013200243174
        series = y * series + 0.0009461589032
        series = y * series - 0.0045563339802
        series = y * series + 0.00556964649138
        series = y * series + 0.00125993961762116
        series = y * series - 0.01621575378835404
        series = y * series + 0.02629651521057465
        series = y * series - 0.001829764677455021
        series = y * series - 0.09439510239319526
        series = y * series + 0.28613578213673563
        series = y * series + 1
        series = y * series + 1
        return -2 * y * series
    elif x > 8:
        return log(polevl(x, &P[0], 6) / p1evl(x, &Q[0], 6)) - \
            x * x
    else:
        return log(erfc(x))


cdef inline double gammaln(double x) noexcept nogil:
    # Simplified from
    # github.com/scipy/scipy/blob/main/scipy/special/xsf/cephes/gamma.h, based
    # on the knowledge that `x` will always be positive and finite when
    # calculating terms in the binomial distribution

    cdef double p, q, u, z

    if x < 13:
        z = 1
        p = 0
        u = x
        while u >= 3:
            p -= 1
            u = x + p
            z *= u
        while u < 2:
            z /= u
            p += 1
            u = x + p
        z = abs(z)
        if u == 2:
            return log(z)
        p -= 2
        x = x + p
        p = x * polevl(x, &gamma_B[0], 6) / \
            p1evl(x, &gamma_C[0], 6)
        return log(z) + p
    elif x >= 1000:
        q = (x - 0.5) * log(x) - x + log_sqrt_2_pi
        if x > 1e8:
            return q
        p = 1.0 / (x * x)
        p = ((7.9365079365079365079365e-4 * p -
              2.7777777777777777777778e-3) *
             p + 0.0833333333333333333333) / x
        return q + p
    else:
        q = (x - 0.5) * log(x) - x + log_sqrt_2_pi
        p = 1.0 / (x * x)
        return q + polevl(p, &gamma_A[0], 5) / x


cdef inline double binom_logsf_term(
        const unsigned j,
        const unsigned n,
        const double log_p,
        const double log1p_q,
        const double gammaln_n_plus_1) noexcept nogil:
    return gammaln_n_plus_1 - gammaln(j + 1) - \
        gammaln(n - j + 1) + j * log_p + (n - j) * log1p_q


cdef inline double binom_logsf(const unsigned k,
                               const unsigned n,
                               const double p) noexcept nogil:
    cdef unsigned j, j_max
    cdef double mu, sigma, z
    cdef double sum_exp, max_term, term, log_p, log1p_q, \
        gammaln_n_plus_1

    # Use the normal approximation when n * p and n * (1 - p) are
    # both greater than 10; add 0.5 for continuity correction
    if n * p > 10 and n * (1 - p) > 10:
        mu = n * p
        sigma = sqrt(mu * (1 - p))
        z = (k + 0.5 - mu) / sigma
        return log_erfc(z * one_over_sqrt_2) + log_one_half

    # Otherwise, compute the exact binomial p-value
    log_p = log(p)
    log1p_q = log1p(-p)
    gammaln_n_plus_1 = gammaln(n + 1)

    # Find `j_max`, the value of `j` with the largest binomial term
    # (the `term` variabel below). For a binomial distribution, the
    # mode (maximum probability) occurs at `floor((n + 1) * p)`.
    # However, since we are summing from `k + 1` to `n`, not 0 to
    # `n`, we need to ensure `j_max` is at least `k + 1`.
    j_max = <unsigned> floor((n + 1) * p)
    if j_max <= k:
        j_max = k + 1

    max_term = binom_logsf_term(j_max, n, log_p, log1p_q,
                                gammaln_n_plus_1)

    # Sum the terms of the binomial via the logsumexp trick. This
    # improves numerical stability by subtracting off the max term
    # from each term before exponentiation, then adding back the
    # max term at the end.
    sum_exp = 0
    for j in range(k + 1, n + 1):
        term = binom_logsf_term(j, n, log_p, log1p_q,
                                gammaln_n_plus_1)
        sum_exp += exp(term - max_term)

    return log(sum_exp) + max_term


cdef extern from * nogil:
    """
    struct CompareGreater {
        const float* data;
        CompareGreater() noexcept {}
        CompareGreater(const float* d) noexcept : data(d) {}
        bool operator()(unsigned a, unsigned b) const noexcept {
            return data[a] > data[b];
        }
    };
    """
    cdef cppclass CompareGreater:
        CompareGreater(const float*) noexcept
        bint operator()(unsigned, unsigned) noexcept


def get_hvgs(unsigned[::1] all_detection_counts,
             unsigned[::1] detection_counts,
             float[::1] ps,
             unsigned[::1] hvgs,
             float[::1] distances,
             const unsigned long long num_cells,
             const unsigned num_total_genes,
             unsigned num_genes):
    cdef unsigned gene, detection_count, i
    cdef float worst_distance, p, distance, \
        inverse_num_cells = <float> 1 / num_cells

    # Normalize `detection_count` by `num_cells` to get the detection rate `p`.
    # Get the `num_genes` most variable genes (those with detection rates
    # closest to 50%), using a min-heap to keep track of the `num_genes`
    # smallest "distances" from 50%.
    for gene in range(num_genes):
        distances[gene] = FLT_MAX
    worst_distance = FLT_MAX
    for gene in range(num_total_genes):
        detection_count = all_detection_counts[gene]
        p = detection_count * inverse_num_cells
        distance = abs(p - <float> 0.5)
        if distance < worst_distance:
            max_heap_replace_top(&hvgs[0], &distances[0], gene, distance,
                                 num_genes)
            worst_distance = distances[0]

    # Exclude genes with detection rates of 0% or 100%, in the rare case that
    # these make it into the top `num_genes` genes
    while True:
        gene = hvgs[0]
        detection_count = all_detection_counts[gene]
        if detection_count != 0 and detection_count != num_cells:
            break
        max_heap_pop(&hvgs[0], &distances[0], num_genes)
        num_genes -= 1
        if num_genes == 0:
            error_message = (
                'all genes are present in either 0% or 100% of cells')
            raise ValueError(error_message)

    # Sort the indices of the highly variable genes
    sort(&hvgs[0], &hvgs[0] + num_genes)

    # Populate `detection_counts` and `ps` for the highly variable genes
    for i in range(num_genes):
        gene = hvgs[i]
        detection_count = all_detection_counts[gene]
        detection_counts[i] = detection_count
        ps[i] = detection_count * inverse_num_cells

    # Return the number of genes (same as the input `num_genes` except in the
    # rare case mentioned above)
    return num_genes


def compute_obs(const unsigned[::1] detection_counts,
                const signed_integer[::1] indices,
                const signed_integer[::1] indptr,
                unsigned[:, ::1] obs,
                unsigned num_threads):

    # obs[i, j] is the number of cells in which exactly one of the two genes i
    # and j is expressed. If we define:
    # - (1) as the number of cells where gene i is expressed
    # - (2) as the number of cells where gene j is expressed
    # - (3) as the number of cells where both genes i and j are expressed
    # then obs[i, j] = (1) + (2) - 2 * (3)
    cdef unsigned detection_counts_i, cell, gene_i, gene_j, thread_index, \
        num_genes = detection_counts.shape[0]
    cdef unsigned long long i, j, i_start, i_end, j_end, \
        num_cells = indptr.shape[0] - 1
    cdef signed_integer indices_i

    num_threads = min(num_threads, min(num_cells, num_genes))
    if num_threads <= 1:
        # Initialize the above-diagonal entries of obs to (1) + (2)
        for i in range(num_genes):
            detection_counts_i = detection_counts[i]
            for j in range(i + 1, num_genes):
                obs[i, j] = detection_counts_i + detection_counts[j]

        # Now subtract off 2 * (3) for the above-diagonal entries: iterate over
        # all pairs of genes i and j within each cell, and subtract 2 from
        # `obs[i, j]` for each pair
        for cell in range(num_cells):
            for i in range(<unsigned long long> indptr[cell],
                           <unsigned long long> indptr[cell + 1]):
                indices_i = indices[i]
                for j in range(i + 1,
                               <unsigned long long> indptr[cell + 1]):
                    obs[indices_i, indices[j]] -= 2
    else:
        with nogil:
            for i in prange(num_genes, num_threads=num_threads):
                detection_counts_i = detection_counts[i]
                for j in range(i + 1, num_genes):
                    obs[i, j] = detection_counts_i + detection_counts[j]

            # Each thread only processes specific genes `i`, for all genes `j`
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                for cell in range(num_cells):
                    for i in range(<unsigned long long> indptr[cell],
                                   <unsigned long long> indptr[cell + 1]):
                        indices_i = indices[i]
                        if indices_i % num_threads == thread_index:
                            for j in range(i + 1,
                                           <unsigned long long>
                                           indptr[cell + 1]):
                                obs[indices_i, indices[j]] -= 2


def compute_S(const unsigned[:, ::1] obs,
              const float[::1] ps,
              const unsigned long long num_cells,
              float[:, ::1] S,
              unsigned num_threads):

    cdef unsigned num_genes = ps.shape[0]
    cdef unsigned i, j
    cdef float ps_i

    num_threads = min(num_threads, num_genes)
    if num_threads <= 1:
        for i in range(num_genes):
            ps_i = ps[i]
            for j in range(i + 1, num_genes):
                S[i, j] = <float> 0 if obs[i, j] == 0 else <float> binom_logsf(
                    k=obs[i, j] - 1, n=num_cells,
                    p=ps_i * (<float> 1 - ps[j]) + (<float> 1 - ps_i) * ps[j])
    else:
        for i in prange(num_genes, nogil=True,
                        num_threads=num_threads):
            ps_i = ps[i]
            for j in range(i + 1, num_genes):
                S[i, j] = <float> 0 if obs[i, j] == 0 else <float> binom_logsf(
                    k=obs[i, j] - 1, n=num_cells,
                    p=ps_i * (<float> 1 - ps[j]) + (<float> 1 - ps_i) * ps[j])


def compute_cxds(
        const signed_integer[::1] indices,
        const signed_integer[::1] indptr,
        const float[:, ::1] S,
        float[::1] cxds_scores,
        unsigned num_threads):

    cdef unsigned cell, gene_i, gene_j, thread_index
    cdef unsigned long long i, j, i_start, i_end, j_end, \
        num_cells = indptr.shape[0] - 1
    cdef float cxds_score
    cdef pair[unsigned, unsigned] row_range

    # Iterate over all pairs of genes i and j within each cell, and
    # subtract `S[i, j]` from the cell's cxds score for each pair
    num_threads = min(num_threads, num_cells)
    if num_threads <= 1:
        for cell in range(num_cells):
            cxds_score = 0
            for gene_i in range(<unsigned long long> indptr[cell],
                                <unsigned long long> indptr[cell + 1]):
                i = indices[gene_i]
                for gene_j in range(gene_i + 1,
                                    <unsigned long long> indptr[cell + 1]):
                    j = indices[gene_j]
                    cxds_score = cxds_score - S[i, j]
            cxds_scores[cell] = cxds_score
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(indptr, thread_index, num_threads)
            for cell in range(row_range.first, row_range.second):
                cxds_score = 0
                for gene_i in range(<unsigned long long> indptr[cell],
                                    <unsigned long long> indptr[cell + 1]):
                    i = indices[gene_i]
                    for gene_j in range(gene_i + 1,
                                        <unsigned long long> indptr[cell + 1]):
                        j = indices[gene_j]
                        cxds_score = cxds_score - S[i, j]
                cxds_scores[cell] = cxds_score


def get_simulated_doublets_max_nnz(const signed_integer[::1] indptr,
                                   const unsigned long long num_cells,
                                   const unsigned long long seed):
    cdef unsigned long long max_nnz = 0, state = srand(seed)
    cdef unsigned sim_index, cell_i, cell_j

    for sim_index in range(num_cells):
        cell_i = randint(num_cells, &state)
        cell_j = randint(num_cells, &state)
        # The absolute maximum nnz occurs if the two cells share zero genes
        max_nnz += (indptr[cell_i + 1] - indptr[cell_i]) + \
                   (indptr[cell_j + 1] - indptr[cell_j])

    return max_nnz


def simulate_doublets(const numeric[::1] data,
                      const signed_integer[::1] indices,
                      const signed_integer[::1] indptr,
                      signed_integer[::1] sim_indices,
                      signed_integer[::1] sim_indptr,
                      const unsigned long long num_cells,
                      const unsigned long long seed):

    cdef unsigned long long i, j, i_end, j_end, nnz = 0, state = srand(seed), \
        coinflip_state = srand(seed + 1)
    cdef unsigned sim_index, cell_i, cell_j

    sim_indptr[0] = 0
    for sim_index in range(1, num_cells + 1):
        cell_i = randint(num_cells, &state)
        cell_j = randint(num_cells, &state)
        i = indptr[cell_i]
        i_end = indptr[cell_i + 1]
        j = indptr[cell_j]
        j_end = indptr[cell_j + 1]
        if i == i_end or j == j_end:
            sim_indptr[sim_index] = nnz
            continue
        while True:
            if indices[i] == indices[j]:
                sim_indices[nnz] = indices[i]
                nnz = nnz + 1
                i = i + 1
                j = j + 1
                if i == i_end or j == j_end:
                    break
            elif indices[i] < indices[j]:
                # `rand(&coinflip_state) & 1` gives a random Boolean; only
                # coin-flip if `data[i] == 1`
                if data[i] > <numeric> 1 or rand(&coinflip_state) & 1:
                    sim_indices[nnz] = indices[i]
                    nnz = nnz + 1
                i = i + 1
                if i == i_end:
                    break
            else:
                if data[j] > <numeric> 1 or rand(&coinflip_state) & 1:
                    sim_indices[nnz] = indices[j]
                    nnz = nnz + 1
                j = j + 1
                if j == j_end:
                    break
        # Process the tails
        while i < i_end:
            if data[i] > <numeric> 1 or rand(&coinflip_state) & 1:
                sim_indices[nnz] = indices[i]
                nnz = nnz + 1
            i = i + 1
        while j < j_end:
            if data[j] > <numeric> 1 or rand(&coinflip_state) & 1:
                sim_indices[nnz] = indices[j]
                nnz = nnz + 1
            j = j + 1
        sim_indptr[sim_index] = nnz


def call_doublets(const float[::1] cxds_scores,
                  const float median_cxds_score_sim,
                  const unsigned[::1] batch_indices,
                  unsigned[::1] doublet_indices,
                  char[::1] singlets,
                  float[::1] doublet_scores,
                  const float doublet_fraction,
                  unsigned num_threads):
    cdef unsigned i, num_doublets, \
        parallel_threshold = max(10_000, num_threads)
    cdef unsigned long long num_cells = cxds_scores.shape[0]

    if doublet_fraction == <float> -1:  # `doublet_fraction is None`
        # Call doublets based on whether their cxds score is above the median
        # cxds score for simulated doublets; `doublet_indices` is unused

        num_threads = min(num_threads, num_cells)
        if batch_indices.shape[0] == 0:  # `batch_column is None`
            if num_threads <= 1 or num_cells < parallel_threshold:
                for i in range(num_cells):
                    singlets[i] = cxds_scores[i] < median_cxds_score_sim
            else:
                for i in prange(num_cells, nogil=True,
                                num_threads=num_threads):
                    singlets[i] = cxds_scores[i] < median_cxds_score_sim
        else:  # `batch_column is not None`
            # Same as when `batch_column is None`, but use `batch_indices[i]`
            # instead of `i` on the left, and also assign `doublet_scores`

            if doublet_scores.shape[0] == 0:  # `not return_scores`
                if num_threads <= 1 or num_cells < parallel_threshold:
                    for i in range(num_cells):
                        singlets[batch_indices[i]] = \
                            cxds_scores[i] < median_cxds_score_sim
                else:
                    for i in prange(num_cells, nogil=True,
                                    num_threads=num_threads):
                        singlets[batch_indices[i]] = \
                            cxds_scores[i] < median_cxds_score_sim
            else:  # `return scores`
                if num_threads <= 1 or num_cells < parallel_threshold:
                    for i in range(num_cells):
                        singlets[batch_indices[i]] = \
                            cxds_scores[i] < median_cxds_score_sim
                        doublet_scores[batch_indices[i]] = cxds_scores[i]
                else:
                    for i in prange(num_cells, nogil=True,
                                    num_threads=num_threads):
                        singlets[batch_indices[i]] = \
                            cxds_scores[i] < median_cxds_score_sim
                        doublet_scores[batch_indices[i]] = cxds_scores[i]
    else:  # `doublet_fraction is not None`
        # Call a fixed fraction of cells as doublets; `median_cxds_score_sim`
        # is unused

        num_doublets = <unsigned>(num_cells * doublet_fraction)
        for i in range(num_cells):
            doublet_indices[i] = i
        nth_element(&doublet_indices[0], &doublet_indices[0] + num_doublets,
                    &doublet_indices[0] + num_cells,
                    CompareGreater(&cxds_scores[0]))

        if batch_indices.shape[0] == 0:  # `batch_column is None`
            num_threads = min(num_threads, num_doublets)
            if num_threads <= 1 or num_doublets < parallel_threshold:
                for i in range(num_doublets):
                    singlets[doublet_indices[i]] = False
            else:
                for i in prange(num_doublets, nogil=True,
                                num_threads=num_threads):
                    singlets[doublet_indices[i]] = False
            if num_threads <= 1 or \
                    num_cells - num_doublets < parallel_threshold:
                for i in range(num_doublets, num_cells):
                    singlets[doublet_indices[i]] = True
            else:
                for i in prange(num_doublets, num_cells, nogil=True,
                                num_threads=num_threads):
                    singlets[doublet_indices[i]] = True
        else:  # `batch_column is not None`
            # Same as when `batch_column is None`, but use
            # `batch_indices[doublet_indices[i]]` instead of
            # `doublet_indices[i]` and `batch_indices[i]` instead of `i` on the
            # left

            if num_threads <= 1 or num_doublets < parallel_threshold:
                for i in range(num_doublets):
                    singlets[batch_indices[doublet_indices[i]]] = False
            else:
                for i in prange(num_doublets, nogil=True,
                                num_threads=min(num_threads, num_doublets)):
                    singlets[batch_indices[doublet_indices[i]]] = False
            if num_threads <= 1 or \
                    num_cells - num_doublets < parallel_threshold:
                for i in range(num_doublets, num_cells):
                    singlets[batch_indices[doublet_indices[i]]] = True
            else:
                for i in prange(num_doublets, num_cells, nogil=True,
                                num_threads=min(num_threads, num_doublets)):
                    singlets[batch_indices[doublet_indices[i]]] = True
            if doublet_scores.shape[0] > 0:  # `return_scores`
                if num_threads <= 1 or num_cells < parallel_threshold:
                    for i in range(num_cells):
                        doublet_scores[batch_indices[i]] = cxds_scores[i]
                else:
                    for i in prange(num_cells, nogil=True,
                                    num_threads=min(num_threads, num_cells)):
                        doublet_scores[batch_indices[i]] = cxds_scores[i]
