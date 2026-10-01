# Functions for embedding with PaCMAP, LocalMAP and UMAP

from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libc.float cimport FLT_MAX
from libc.limits cimport UINT_MAX
from libc.math cimport M_PI
from libc.string cimport memcpy
from libcpp.algorithm cimport sort
from libcpp.cmath cimport cos, exp, log, log2, pow, sqrt
from libcpp.vector cimport vector
from .cyutils cimport atomic_or, max_heap_replace_top, max_heap_sort, \
    randint, random_normal, random_uniform, signed_integer, srand, \
    uninitialized_vector


cdef inline void get_neighbor_pairs(const float[:, ::1] X,
                                    const unsigned[:, ::1] neighbors,
                                    const float[:, ::1] distances,
                                    unsigned[:, ::1] neighbor_pairs,
                                    float[::1] total_distances,
                                    unsigned num_threads,
                                    bint& too_large,
                                    bint& self_neighbors,
                                    const bint match_parallel):
    cdef unsigned i, j, k, neighbor, thread_index, \
        num_total_neighbors = distances.shape[1], \
        num_neighbors = neighbor_pairs.shape[1], num_PCs = X.shape[1]
    cdef unsigned long long num_cells = X.shape[0]
    cdef float worst_distance, scaled_distance
    cdef uninitialized_vector[float] scaled_distances_i_buffer
    cdef vector[uninitialized_vector[float]] thread_scaled_distances_i
    cdef float[::1] scaled_distances_i

    num_threads = min(num_threads, num_cells)
    if num_threads <= 1 and not match_parallel:
        # Calculate the total Euclidean distance from each cell to its 4th-,
        # 5th-, and 6th-nearest neighbors. (The original implementation uses
        # the average, but since we are just using this for ranking, avoiding
        # the division by 3 is fine.)
        for i in range(num_cells):
            total_distances[i] = (
                sqrt(distances[i, 3]) + sqrt(distances[i, 4]) +
                sqrt(distances[i, 5]))
            if total_distances[i] < <float> 3e-10:
                total_distances[i] = <float> 3e-10

        # Select the `num_neighbors` of each cell's `num_total_neighbors`
        # nearest neighbors with the lowest scaled distances. We define the
        # scaled distance between cells `i` and `j` as the squared Euclidean
        # distance from `i` to `j`, divided by
        # `average_distance[i] * average_distance[j]`. However, when ranking
        # cells by their scaled distance to cell `i`, we can ignore the
        # normalization by `average_distance[i]` since it is constant for all
        # neighbors `j`.
        scaled_distances_i_buffer.resize(num_neighbors)
        scaled_distances_i = \
            <float[:num_neighbors]> scaled_distances_i_buffer.data()
        for i in range(num_cells):
            for j in range(num_neighbors):
                scaled_distances_i[j] = FLT_MAX
            worst_distance = FLT_MAX
            for j in range(num_total_neighbors):
                neighbor = neighbors[i, j]
                if neighbor >= num_cells:
                    too_large = True
                    return
                elif neighbor == i:
                    self_neighbors = True
                    return
                scaled_distance = distances[i, j] / total_distances[neighbor]
                if scaled_distance < worst_distance:
                    max_heap_replace_top(&neighbor_pairs[i, 0],
                                         &scaled_distances_i[0],
                                         neighbor, scaled_distance,
                                         num_neighbors)
                    worst_distance = scaled_distances_i[0]

            # Sort the heap to get nearest neighbors in ascending order of
            # scaled distance
            max_heap_sort(&neighbor_pairs[i, 0], &scaled_distances_i[0],
                          num_neighbors)
    else:
        # Same as the single-threaded version, but use
        # `thread_scaled_distances_i` instead of `scaled_distances_i`. Also,
        # use hardware-level atomics to thread-safely flag out-of-bounds
        # neighbors.

        with nogil:
            for i in prange(num_cells, num_threads=num_threads):
                total_distances[i] = (
                    sqrt(distances[i, 3]) + sqrt(distances[i, 4]) +
                    sqrt(distances[i, 5]))
                if total_distances[i] < <float> 3e-10:
                    total_distances[i] = <float> 3e-10

            thread_scaled_distances_i.resize(num_threads)
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_scaled_distances_i[thread_index].resize(num_neighbors)
                for i in prange(num_cells):
                    for j in range(num_neighbors):
                        thread_scaled_distances_i[thread_index][j] = FLT_MAX
                    worst_distance = FLT_MAX
                    for j in range(num_total_neighbors):
                        neighbor = neighbors[i, j]
                        if neighbor >= num_cells:
                            atomic_or(too_large, True)
                            return
                        elif neighbor == i:
                            atomic_or(self_neighbors, True)
                            return
                        scaled_distance = \
                            distances[i, j] / total_distances[neighbor]
                        if scaled_distance < worst_distance:
                            max_heap_replace_top(
                                &neighbor_pairs[i, 0],
                                thread_scaled_distances_i[thread_index].data(),
                                neighbor, scaled_distance, num_neighbors)
                            worst_distance = \
                                thread_scaled_distances_i[thread_index][0]
                    max_heap_sort(
                        &neighbor_pairs[i, 0],
                        thread_scaled_distances_i[thread_index].data(),
                        num_neighbors)


cdef inline void sample_mid_near_pairs(const float[:, ::1] X,
                                       unsigned[:, ::1] mid_near_pairs,
                                       const unsigned long long seed,
                                       const unsigned num_threads,
                                       const bint match_parallel):
    cdef unsigned i, j, k, l, sampled_k, closest_cell, second_closest_cell, \
        thread_index, n = X.shape[0], \
        num_mid_near_pairs = mid_near_pairs.shape[1], num_PCs = X.shape[1]
    cdef float difference, distance, smallest, second_smallest
    cdef unsigned long long state
    cdef uninitialized_vector[unsigned] sampled_buffer
    cdef vector[uninitialized_vector[unsigned]] thread_sampled
    cdef unsigned[::1] sampled

    if num_threads == 1 and not match_parallel:
        sampled_buffer.resize(6)
        sampled = <unsigned[:6]> sampled_buffer.data()
        for i in range(n):
            state = srand(seed + i)
            for j in range(num_mid_near_pairs):
                # Randomly sample 6 cells (which are not the
                # current cell) and select the 2nd-closest
                smallest = FLT_MAX
                second_smallest = FLT_MAX
                for k in range(6):
                    while True:
                        # Sample a random cell...
                        sampled_k = randint(n, &state)

                        # ...that is not this cell...
                        if sampled_k == i:
                            continue

                        # ...nor a previously sampled cell in this batch
                        for l in range(k):
                            if sampled_k == sampled[l]:
                                break
                        else:
                            # ...nor a previously finalized mid-near pair
                            for l in range(j):
                                if sampled_k == mid_near_pairs[i, l]:
                                    break
                            else:
                                sampled[k] = sampled_k
                                break
                for k in range(6):
                    sampled_k = sampled[k]
                    difference = X[i, 0] - X[sampled_k, 0]
                    distance = difference * difference
                    for l in range(1, num_PCs):
                        difference = X[i, l] - X[sampled_k, l]
                        distance += difference * difference
                    if distance < smallest:
                        second_smallest = smallest
                        second_closest_cell = closest_cell
                        smallest = distance
                        closest_cell = sampled_k
                    elif distance < second_smallest:
                        second_smallest = distance
                        second_closest_cell = sampled_k
                mid_near_pairs[i, j] = second_closest_cell
    else:
        thread_sampled.resize(num_threads)
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            thread_sampled[thread_index].resize(6)
            for i in prange(n):
                state = srand(seed + i)
                for j in range(num_mid_near_pairs):
                    smallest = FLT_MAX
                    second_smallest = FLT_MAX
                    for k in range(6):
                        while True:
                            sampled_k = randint(n, &state)
                            if sampled_k == i:
                                continue
                            for l in range(k):
                                if sampled_k == \
                                        thread_sampled[thread_index][l]:
                                    break
                            else:
                                for l in range(j):
                                    if sampled_k == mid_near_pairs[i, l]:
                                        break
                                else:
                                    thread_sampled[thread_index][k] = sampled_k
                                    break
                    for k in range(6):
                        sampled_k = thread_sampled[thread_index][k]
                        difference = X[i, 0] - X[sampled_k, 0]
                        distance = difference * difference
                        for l in range(1, num_PCs):
                            difference = X[i, l] - X[sampled_k, l]
                            distance = distance + difference * difference
                        if distance < smallest:
                            second_smallest = smallest
                            second_closest_cell = closest_cell
                            smallest = distance
                            closest_cell = sampled_k
                        elif distance < second_smallest:
                            second_smallest = distance
                            second_closest_cell = sampled_k
                    mid_near_pairs[i, j] = second_closest_cell


cdef inline void sample_further_pairs(
        const float[:, ::1] X,
        const unsigned[:, ::1] neighbor_pairs,
        unsigned[:, ::1] further_pairs,
        const unsigned long long seed,
        const unsigned num_threads,
        const bint match_parallel) noexcept nogil:
    cdef unsigned i, j, k, further_pair_index, n = X.shape[0], \
        num_further_pairs = further_pairs.shape[1], \
        num_neighbors = neighbor_pairs.shape[1]
    cdef unsigned long long state

    if num_threads == 1 and not match_parallel:
        for i in range(n):
            state = srand(seed + i)
            for j in range(num_further_pairs):
                while True:
                    # Sample a random cell...
                    further_pair_index = randint(n, &state)

                    # ...that is not this cell...
                    if further_pair_index == i:
                        continue

                    # ...nor one of its nearest neighbors...
                    for k in range(num_neighbors):
                        if further_pair_index == neighbor_pairs[i, k]:
                            break
                    else:
                        # ...nor a previously sampled cell
                        for k in range(j):
                            if further_pair_index == further_pairs[i, k]:
                                break
                        else:
                            # Sampling successful - assign the further pair
                            further_pairs[i, j] = further_pair_index
                            break
    else:
        for i in prange(n, nogil=True, num_threads=num_threads):
            state = srand(seed + i)
            for j in range(num_further_pairs):
                while True:
                    further_pair_index = randint(n, &state)
                    if further_pair_index == i:
                        continue
                    for k in range(num_neighbors):
                        if further_pair_index == neighbor_pairs[i, k]:
                            break
                    else:
                        for k in range(j):
                            if further_pair_index == further_pairs[i, k]:
                                break
                        else:
                            further_pairs[i, j] = further_pair_index
                            break


cdef inline void sample_further_pairs_nearby(
        const float[:, ::1] embedding,
        const unsigned[:, ::1] neighbor_pairs,
        unsigned[:, ::1] further_pairs,
        const unsigned long long seed,
        const float max_distance_squared,
        const unsigned num_threads) noexcept nogil:
    cdef unsigned i, j, k, count, further_pair_index, n = embedding.shape[0], \
        num_further_pairs = further_pairs.shape[1], \
        num_neighbors = neighbor_pairs.shape[1]
    cdef unsigned long long state

    # Use a parallel loop even when single-threaded, to ensure identical
    # results regardless of the number of threads
    for i in prange(n, nogil=True, num_threads=num_threads):
        state = srand(seed + i)
        for j in range(num_further_pairs):
            # Give up after 100 trials, and keep the further pair as-is;
            # this corrects the original implementation's logic by counting
            # all types of failures towards the 100 trials, including ones
            # where the candidate further pair is too far away
            for count in range(100):
                # Sample a random cell...
                further_pair_index = randint(n, &state)

                # ...that is not this cell...
                if further_pair_index == i:
                    continue

                # ...nor one of its nearest neighbors...
                for k in range(num_neighbors):
                    if further_pair_index == neighbor_pairs[i, k]:
                        break
                else:
                    # ...nor a previously sampled cell...
                    for k in range(j):
                        if further_pair_index == further_pairs[i, k]:
                            break
                    else:
                        # ...nor a cell farther than `max_distance` away
                        # (in embedding space)
                        if (embedding[i, 0] -
                                embedding[further_pair_index, 0]) * \
                                (embedding[i, 0] -
                                 embedding[further_pair_index, 0]) + \
                                (embedding[i, 1] -
                                 embedding[further_pair_index, 1]) * \
                                (embedding[i, 1] -
                                 embedding[further_pair_index, 1]) > \
                                max_distance_squared:
                            continue
                        else:
                            # Sampling successful - assign the further pair
                            further_pairs[i, j] = further_pair_index
                            break


cdef inline void reformat_for_parallel(
        const unsigned[:, ::1] pairs,
        unsigned[::1] pair_indices,
        unsigned[::1] pair_indptr,
        const unsigned num_threads) noexcept nogil:
    cdef unsigned i, j, k, dest_index, num_pairs_per_cell = pairs.shape[1]
    cdef unsigned long long index, num_cells = pairs.shape[0]
    cdef uninitialized_vector[unsigned] dest_indices
    dest_indices.resize(num_cells)

    # Perform first-touch NUMA page allocation for the outputs
    if num_threads > 1:
        for index in prange(pair_indices.shape[0], num_threads=num_threads):
            pair_indices[index] = 0
        for index in prange(pair_indptr.shape[0], num_threads=num_threads):
            pair_indptr[index] = 0

    # Tabulate how often each cell appears in pairs; at a minimum, it
    # will appear `pairs.shape[1]` times (i.e. the number of
    # neighbors), as the `i` in the pair, but it will also appear a
    # variable number of times as the `j` in the pair.
    pair_indptr[0] = 0
    pair_indptr[1:] = pairs.shape[1]
    for i in range(num_cells):
        for k in range(num_pairs_per_cell):
            j = pairs[i, k]
            pair_indptr[j + 1] += 1

    # Cumsum the values in `pair_indptr`
    for i in range(2, pair_indptr.shape[0]):
        pair_indptr[i] += pair_indptr[i - 1]

    # Now that we know how many pairs each cell is a part of, do a
    # second pass over `pairs` to populate `pair_indices` with the
    # pairs' indices. Use a temporary buffer, `dest_indices`, to keep
    # track of the index within `pair_indptr` to write each cell's next
    # pair to. Note: this logic assumes no self-neighbors (checked
    # earlier in `get_neighbor_pairs()`).
    memcpy(dest_indices.data(), &pair_indptr[0], num_cells * sizeof(unsigned))
    for i in range(num_cells):
        for k in range(num_pairs_per_cell):
            j = pairs[i, k]
            pair_indices[dest_indices[i]] = j
            pair_indices[dest_indices[j]] = i
            dest_indices[i] += 1
            dest_indices[j] += 1


cdef inline void get_gradients_fast(const float[:, ::1] embedding,
                                    const unsigned[:, ::1] neighbor_pairs,
                                    const unsigned[:, ::1] mid_near_pairs,
                                    const unsigned[:, ::1] further_pairs,
                                    const float w_neighbors,
                                    const float w_mid_near,
                                    float[:, ::1] gradients):
    cdef unsigned i, j, k, num_neighbors = neighbor_pairs.shape[1], \
        num_mid_near_pairs = mid_near_pairs.shape[1], \
        num_further_pairs = further_pairs.shape[1]
    cdef unsigned long long num_cells = neighbor_pairs.shape[0]
    cdef float embedding_i0, embedding_i1, gradients_i0, gradients_i1, \
        embedding_ij_0, embedding_ij_1, distance_ij, w

    gradients[:] = 0
    for i in range(num_cells):
        embedding_i0 = embedding[i, 0]
        embedding_i1 = embedding[i, 1]
        gradients_i0 = 0
        gradients_i1 = 0

        # Nearest-neighbor pairs
        for k in range(num_neighbors):
            j = neighbor_pairs[i, k]
            embedding_ij_0 = embedding_i0 - embedding[j, 0]
            embedding_ij_1 = embedding_i1 - embedding[j, 1]
            distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
                embedding_ij_1 * embedding_ij_1
            w = w_neighbors * (<float> 20 /
                ((<float> 10 + distance_ij) * (<float> 10 + distance_ij)))
            gradients_i0 += w * embedding_ij_0
            gradients[j, 0] -= w * embedding_ij_0
            gradients_i1 += w * embedding_ij_1
            gradients[j, 1] -= w * embedding_ij_1

        # Mid-near pairs
        for k in range(num_mid_near_pairs):
            j = mid_near_pairs[i, k]
            embedding_ij_0 = embedding_i0 - embedding[j, 0]
            embedding_ij_1 = embedding_i1 - embedding[j, 1]
            distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
                embedding_ij_1 * embedding_ij_1
            w = w_mid_near * (<float> 20000 / ((<float> 10000 + distance_ij) *
                                               (<float> 10000 + distance_ij)))
            gradients_i0 += w * embedding_ij_0
            gradients[j, 0] -= w * embedding_ij_0
            gradients_i1 += w * embedding_ij_1
            gradients[j, 1] -= w * embedding_ij_1

        # Further pairs
        for k in range(num_further_pairs):
            j = further_pairs[i, k]
            embedding_ij_0 = embedding_i0 - embedding[j, 0]
            embedding_ij_1 = embedding_i1 - embedding[j, 1]
            distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
                embedding_ij_1 * embedding_ij_1
            w = <float> 2 / \
                ((<float> 1 + distance_ij) * (<float> 1 + distance_ij))
            gradients_i0 -= w * embedding_ij_0
            gradients[j, 0] += w * embedding_ij_0
            gradients_i1 -= w * embedding_ij_1
            gradients[j, 1] += w * embedding_ij_1

        gradients[i, 0] += gradients_i0
        gradients[i, 1] += gradients_i1


cdef inline void get_scaled_gradients_fast(
        const float[:, ::1] embedding,
        const unsigned[:, ::1] neighbor_pairs,
        const unsigned[:, ::1] mid_near_pairs,
        const unsigned[:, ::1] further_pairs,
        const float w_neighbors,
        const float w_mid_near,
        const float half_max_distance,
        float[:, ::1] gradients):
    cdef unsigned i, j, k, num_neighbors = neighbor_pairs.shape[1], \
        num_mid_near_pairs = mid_near_pairs.shape[1], \
        num_further_pairs = further_pairs.shape[1]
    cdef unsigned long long num_cells = neighbor_pairs.shape[0]
    cdef float embedding_i0, embedding_i1, gradients_i0, gradients_i1, \
        embedding_ij_0, embedding_ij_1, distance_ij, w

    gradients[:] = 0
    for i in range(num_cells):
        embedding_i0 = embedding[i, 0]
        embedding_i1 = embedding[i, 1]
        gradients_i0 = 0
        gradients_i1 = 0

        # Nearest-neighbor pairs
        for k in range(num_neighbors):
            j = neighbor_pairs[i, k]
            embedding_ij_0 = embedding_i0 - embedding[j, 0]
            embedding_ij_1 = embedding_i1 - embedding[j, 1]
            distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
                embedding_ij_1 * embedding_ij_1
            w = w_neighbors * (<float> 20 /
                ((<float> 10 + distance_ij) * (<float> 10 + distance_ij)))
            # this is the only line that differs from `get_gradients_fast()`
            w *= half_max_distance / sqrt(distance_ij)
            gradients_i0 += w * embedding_ij_0
            gradients[j, 0] -= w * embedding_ij_0
            gradients_i1 += w * embedding_ij_1
            gradients[j, 1] -= w * embedding_ij_1

        # Mid-near pairs
        for k in range(num_mid_near_pairs):
            j = mid_near_pairs[i, k]
            embedding_ij_0 = embedding_i0 - embedding[j, 0]
            embedding_ij_1 = embedding_i1 - embedding[j, 1]
            distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
                embedding_ij_1 * embedding_ij_1
            w = w_mid_near * (<float> 20000 / ((<float> 10000 + distance_ij) *
                                               (<float> 10000 + distance_ij)))
            gradients_i0 += w * embedding_ij_0
            gradients[j, 0] -= w * embedding_ij_0
            gradients_i1 += w * embedding_ij_1
            gradients[j, 1] -= w * embedding_ij_1

        # Further pairs
        for k in range(num_further_pairs):
            j = further_pairs[i, k]
            embedding_ij_0 = embedding_i0 - embedding[j, 0]
            embedding_ij_1 = embedding_i1 - embedding[j, 1]
            distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
                embedding_ij_1 * embedding_ij_1
            w = <float> 2 / \
                ((<float> 1 + distance_ij) * (<float> 1 + distance_ij))
            gradients_i0 -= w * embedding_ij_0
            gradients[j, 0] += w * embedding_ij_0
            gradients_i1 -= w * embedding_ij_1
            gradients[j, 1] += w * embedding_ij_1

        gradients[i, 0] += gradients_i0
        gradients[i, 1] += gradients_i1


cdef inline void get_gradient(const float[:, ::1] embedding,
                              const unsigned[::1] neighbor_pair_indices,
                              const unsigned[::1] neighbor_pair_indptr,
                              const unsigned[::1] mid_near_pair_indices,
                              const unsigned[::1] mid_near_pair_indptr,
                              const unsigned[::1] further_pair_indices,
                              const unsigned[::1] further_pair_indptr,
                              const float w_neighbors,
                              const float w_mid_near,
                              float[:, ::1] gradients,
                              const unsigned i) noexcept nogil:
    cdef unsigned j, k
    cdef unsigned long long num_cells = embedding.shape[0]
    cdef float embedding_ij_0, embedding_ij_1, distance_ij, w, \
        embedding_i0 = embedding[i, 0], embedding_i1 = embedding[i, 1], \
        gradient_i0 = 0, gradient_i1 = 0

    # Nearest-neighbor pairs
    for k in range(neighbor_pair_indptr[i],
                   neighbor_pair_indptr[i + 1]):
        j = neighbor_pair_indices[k]
        embedding_ij_0 = embedding_i0 - embedding[j, 0]
        embedding_ij_1 = embedding_i1 - embedding[j, 1]
        distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
            embedding_ij_1 * embedding_ij_1
        w = w_neighbors * (<float> 20 /
            ((<float> 10 + distance_ij) * (<float> 10 + distance_ij)))
        gradient_i0 = gradient_i0 + w * embedding_ij_0
        gradient_i1 = gradient_i1 + w * embedding_ij_1

    # Mid-near pairs
    for k in range(mid_near_pair_indptr[i],
                   mid_near_pair_indptr[i + 1]):
        j = mid_near_pair_indices[k]
        embedding_ij_0 = embedding_i0 - embedding[j, 0]
        embedding_ij_1 = embedding_i1 - embedding[j, 1]
        distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
            embedding_ij_1 * embedding_ij_1
        w = w_mid_near * (<float> 20000 / ((<float> 10000 + distance_ij) *
                                           (<float> 10000 + distance_ij)))
        gradient_i0 = gradient_i0 + w * embedding_ij_0
        gradient_i1 = gradient_i1 + w * embedding_ij_1

    # Further pairs
    for k in range(further_pair_indptr[i],
                   further_pair_indptr[i + 1]):
        j = further_pair_indices[k]
        embedding_ij_0 = embedding_i0 - embedding[j, 0]
        embedding_ij_1 = embedding_i1 - embedding[j, 1]
        distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
            embedding_ij_1 * embedding_ij_1
        w = <float> 2 / ((<float> 1 + distance_ij) * (<float> 1 + distance_ij))
        gradient_i0 = gradient_i0 - w * embedding_ij_0
        gradient_i1 = gradient_i1 - w * embedding_ij_1

    gradients[i, 0] = gradient_i0
    gradients[i, 1] = gradient_i1


cdef inline void get_gradients_parallel(
        const float[:, ::1] embedding,
        const unsigned[::1] neighbor_pair_indices,
        const unsigned[::1] neighbor_pair_indptr,
        const unsigned[::1] mid_near_pair_indices,
        const unsigned[::1] mid_near_pair_indptr,
        const unsigned[::1] further_pair_indices,
        const unsigned[::1] further_pair_indptr,
        const float w_neighbors,
        const float w_mid_near,
        float[:, ::1] gradients,
        const unsigned num_threads) noexcept nogil:
    cdef unsigned i
    cdef unsigned long long num_cells = embedding.shape[0]

    for i in prange(num_cells, num_threads=num_threads):
        get_gradient(embedding, neighbor_pair_indices, neighbor_pair_indptr,
                     mid_near_pair_indices, mid_near_pair_indptr,
                     further_pair_indices, further_pair_indptr, w_neighbors,
                     w_mid_near, gradients, i)


cdef inline void get_scaled_gradient(const float[:, ::1] embedding,
                                     const unsigned[::1] neighbor_pair_indices,
                                     const unsigned[::1] neighbor_pair_indptr,
                                     const unsigned[::1] mid_near_pair_indices,
                                     const unsigned[::1] mid_near_pair_indptr,
                                     const unsigned[::1] further_pair_indices,
                                     const unsigned[::1] further_pair_indptr,
                                     const float w_neighbors,
                                     const float w_mid_near,
                                     const float half_max_distance,
                                     float[:, ::1] gradients,
                                     const unsigned i) noexcept nogil:
    cdef unsigned j, k
    cdef unsigned long long num_cells = embedding.shape[0]
    cdef float embedding_ij_0, embedding_ij_1, distance_ij, w, \
        embedding_i0 = embedding[i, 0], embedding_i1 = embedding[i, 1], \
        gradient_i0 = 0, gradient_i1 = 0

    # Nearest-neighbor pairs
    for k in range(neighbor_pair_indptr[i],
                   neighbor_pair_indptr[i + 1]):
        j = neighbor_pair_indices[k]
        embedding_ij_0 = embedding_i0 - embedding[j, 0]
        embedding_ij_1 = embedding_i1 - embedding[j, 1]
        distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
            embedding_ij_1 * embedding_ij_1
        w = w_neighbors * (<float> 20 /
            ((<float> 10 + distance_ij) * (<float> 10 + distance_ij)))
        # this is the only line that differs from `get_gradient()`
        w *= half_max_distance / sqrt(distance_ij)
        gradient_i0 = gradient_i0 + w * embedding_ij_0
        gradient_i1 = gradient_i1 + w * embedding_ij_1

    # Mid-near pairs
    for k in range(mid_near_pair_indptr[i],
                   mid_near_pair_indptr[i + 1]):
        j = mid_near_pair_indices[k]
        embedding_ij_0 = embedding_i0 - embedding[j, 0]
        embedding_ij_1 = embedding_i1 - embedding[j, 1]
        distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
            embedding_ij_1 * embedding_ij_1
        w = w_mid_near * (<float> 20000 / ((<float> 10000 + distance_ij) *
                                           (<float> 10000 + distance_ij)))
        gradient_i0 = gradient_i0 + w * embedding_ij_0
        gradient_i1 = gradient_i1 + w * embedding_ij_1

    # Further pairs
    for k in range(further_pair_indptr[i],
                   further_pair_indptr[i + 1]):
        j = further_pair_indices[k]
        embedding_ij_0 = embedding_i0 - embedding[j, 0]
        embedding_ij_1 = embedding_i1 - embedding[j, 1]
        distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
            embedding_ij_1 * embedding_ij_1
        w = <float> 2 / ((<float> 1 + distance_ij) * (<float> 1 + distance_ij))
        gradient_i0 = gradient_i0 - w * embedding_ij_0
        gradient_i1 = gradient_i1 - w * embedding_ij_1

    gradients[i, 0] = gradient_i0
    gradients[i, 1] = gradient_i1


cdef inline void get_scaled_gradients_parallel(
        const float[:, ::1] embedding,
        const unsigned[::1] neighbor_pair_indices,
        const unsigned[::1] neighbor_pair_indptr,
        const unsigned[::1] mid_near_pair_indices,
        const unsigned[::1] mid_near_pair_indptr,
        const unsigned[::1] further_pair_indices,
        const unsigned[::1] further_pair_indptr,
        const float w_neighbors,
        const float w_mid_near,
        const float half_max_distance,
        float[:, ::1] gradients,
        const unsigned num_threads) noexcept nogil:
    cdef unsigned i
    cdef unsigned long long num_cells = embedding.shape[0]

    for i in prange(num_cells, num_threads=num_threads):
        get_scaled_gradient(embedding, neighbor_pair_indices,
                            neighbor_pair_indptr, mid_near_pair_indices,
                            mid_near_pair_indptr, further_pair_indices,
                            further_pair_indptr, w_neighbors,
                            w_mid_near, half_max_distance, gradients, i)


cdef inline void update_cell_embedding_adam(
        float& embedding,
        const float gradient,
        float& momentum,
        float& velocity,
        const unsigned long long num_cells,
        const float beta1,
        const float beta2,
        const float learning_rate) noexcept nogil:
    momentum += (<float> 1 - beta1) * (gradient - momentum)
    velocity += (<float> 1 - beta2) * (gradient * gradient - velocity)
    embedding -= learning_rate * momentum / (sqrt(velocity) + <float> 1e-7)


cdef inline void update_embedding_adam(
        float[:, ::1] embedding,
        const float[:, ::1] gradients,
        float[:, ::1] momentum,
        float[:, ::1] velocity,
        const unsigned long long num_cells,
        const float beta1,
        const float beta2,
        float learning_rate,
        const unsigned iteration):
    cdef unsigned i
    learning_rate *= sqrt(<float> 1 - pow(beta2, <float> (iteration + 1))) / \
        (<float> 1 - pow(beta1, <float> (iteration + 1)))
    for i in range(num_cells):
        update_cell_embedding_adam(embedding[i, 0], gradients[i, 0],
                                   momentum[i, 0], velocity[i, 0], num_cells,
                                   beta1, beta2, learning_rate)
        update_cell_embedding_adam(embedding[i, 1], gradients[i, 1],
                                   momentum[i, 1], velocity[i, 1], num_cells,
                                   beta1, beta2, learning_rate)


cdef inline void update_embedding_adam_parallel(
        float[:, ::1] embedding,
        const float[:, ::1] gradients,
        float[:, ::1] momentum,
        float[:, ::1] velocity,
        const unsigned long long num_cells,
        const float beta1,
        const float beta2,
        float learning_rate,
        const unsigned iteration,
        const unsigned num_threads) noexcept nogil:
    cdef unsigned i
    learning_rate *= sqrt(<float> 1 - pow(beta2, <float> (iteration + 1))) / \
        (<float> 1 - pow(beta1, <float> (iteration + 1)))
    for i in prange(num_cells, num_threads=num_threads):
        update_cell_embedding_adam(embedding[i, 0], gradients[i, 0],
                                   momentum[i, 0], velocity[i, 0], num_cells,
                                   beta1, beta2, learning_rate)
        update_cell_embedding_adam(embedding[i, 1], gradients[i, 1],
                                   momentum[i, 1], velocity[i, 1], num_cells,
                                   beta1, beta2, learning_rate)


cdef inline void pacmap_serial_fast(const float[:, ::1] PCs,
                                    float[:, ::1] embedding,
                                    float[:, ::1] momentum,
                                    float[:, ::1] velocity,
                                    float[:, ::1] gradients,
                                    const unsigned[:, ::1] neighbor_pairs,
                                    const unsigned[:, ::1] mid_near_pairs,
                                    unsigned[:, ::1] further_pairs,
                                    const unsigned num_phase_1_iterations,
                                    const unsigned num_phase_2_iterations,
                                    const unsigned num_phase_3_iterations,
                                    const float learning_rate):
    cdef unsigned i, iteration, num_iterations = num_phase_1_iterations + \
        num_phase_2_iterations + num_phase_3_iterations, \
        w_mid_near_init = 1000
    cdef unsigned long long num_cells = PCs.shape[0]
    cdef float iteration_fraction, w_mid_near, w_neighbors, \
        beta1 = <float> 0.9, beta2 = <float> 0.999

    # Initialize the embedding, momentum and velocity
    for i in range(num_cells):
        embedding[i, 0] = <float> 0.01 * PCs[i, 0]
        embedding[i, 1] = <float> 0.01 * PCs[i, 1]
    momentum[:] = 0
    velocity[:] = 0

    # Optimize the embedding
    for iteration in range(num_iterations):
        if iteration < num_phase_1_iterations:
            iteration_fraction = <float> iteration / num_phase_1_iterations
            w_mid_near = (<float> 1 - iteration_fraction) * w_mid_near_init + \
                iteration_fraction * <float> 3
            w_neighbors = 2
        elif iteration < num_phase_1_iterations + num_phase_2_iterations:
            w_mid_near = 3
            w_neighbors = 3
        else:
            w_mid_near = 0
            w_neighbors = 1

        # Calculate gradients
        get_gradients_fast(embedding, neighbor_pairs, mid_near_pairs,
                           further_pairs, w_neighbors, w_mid_near,
                           gradients)

        # Update the embedding based on the gradients, via the Adam
        # optimizer
        update_embedding_adam(embedding, gradients, momentum, velocity,
                              num_cells, beta1, beta2, learning_rate,
                              iteration)

        # Check for KeyboardInterrupts
        if iteration % 8 == 7:
            PyErr_CheckSignals()


cdef inline void pacmap_parallel(const float[:, ::1] PCs,
                                 float[:, ::1] embedding,
                                 float[:, ::1] momentum,
                                 float[:, ::1] velocity,
                                 float[:, ::1] gradients,
                                 const unsigned[::1] neighbor_pair_indices,
                                 const unsigned[::1] neighbor_pair_indptr,
                                 const unsigned[::1] mid_near_pair_indices,
                                 const unsigned[::1] mid_near_pair_indptr,
                                 unsigned[::1] further_pair_indices,
                                 unsigned[::1] further_pair_indptr,
                                 const unsigned num_phase_1_iterations,
                                 const unsigned num_phase_2_iterations,
                                 const unsigned num_phase_3_iterations,
                                 const float learning_rate,
                                 const unsigned num_threads):
    cdef unsigned i, iteration, num_iterations = num_phase_1_iterations + \
        num_phase_2_iterations + num_phase_3_iterations, \
        w_mid_near_init = 1000
    cdef unsigned long long num_cells = PCs.shape[0]
    cdef float iteration_fraction, w_mid_near, w_neighbors, \
        beta1 = <float> 0.9, beta2 = <float> 0.999

    with nogil:
        # Initialize the embedding, momentum and velocity. Momentum and
        # velocity are initialized in parallel for NUMA first-touch reasons.
        for i in prange(num_cells, num_threads=num_threads):
            embedding[i, 0] = <float> 0.01 * PCs[i, 0]
            embedding[i, 1] = <float> 0.01 * PCs[i, 1]
            momentum[i, 0] = 0
            momentum[i, 1] = 0
            velocity[i, 0] = 0
            velocity[i, 1] = 0

        # Optimize the embedding
        for iteration in range(num_iterations):
            if iteration < num_phase_1_iterations:
                iteration_fraction = <float> iteration / num_phase_1_iterations
                w_mid_near = (<float> 1 - iteration_fraction) * \
                    w_mid_near_init + iteration_fraction * <float> 3
                w_neighbors = 2
            elif iteration < num_phase_1_iterations + num_phase_2_iterations:
                w_mid_near = 3
                w_neighbors = 3
            else:
                w_mid_near = 0
                w_neighbors = 1

            # Calculate gradients
            get_gradients_parallel(
                embedding, neighbor_pair_indices, neighbor_pair_indptr,
                mid_near_pair_indices, mid_near_pair_indptr,
                further_pair_indices, further_pair_indptr, w_neighbors,
                w_mid_near, gradients, num_threads)

            # Update the embedding based on the gradients, via the Adam
            # optimizer
            update_embedding_adam_parallel(embedding, gradients, momentum,
                                           velocity, num_cells, beta1, beta2,
                                           learning_rate, iteration,
                                           num_threads)

            # Check for KeyboardInterrupts
            if iteration % 8 == 7:
                with gil:
                    PyErr_CheckSignals()


def pacmap(const float[:, ::1] PCs,
           float[:, ::1] embedding,
           float[:, ::1] momentum,
           float[:, ::1] velocity,
           float[:, ::1] gradients,
           float[::1] total_distances,
           unsigned[:, ::1] neighbor_pairs,
           unsigned[:, ::1] mid_near_pairs,
           unsigned[:, ::1] further_pairs,
           unsigned[::1] neighbor_pair_indices,
           unsigned[::1] neighbor_pair_indptr,
           unsigned[::1] mid_near_pair_indices,
           unsigned[::1] mid_near_pair_indptr,
           unsigned[::1] further_pair_indices,
           unsigned[::1] further_pair_indptr,
           const unsigned[:, ::1] neighbors,
           const float[:, ::1] distances,
           const unsigned num_neighbors,
           const unsigned num_extra_neighbors,
           const unsigned num_mid_near_pairs,
           const unsigned num_further_pairs,
           const unsigned num_phase_1_iterations,
           const unsigned num_phase_2_iterations,
           const unsigned num_phase_3_iterations,
           const float learning_rate,
           const unsigned long long seed,
           const bint match_parallel,
           str neighbors_key,
           unsigned num_threads):
    cdef unsigned i, num_cells = PCs.shape[0]
    cdef bint too_large = False, self_neighbors = False
    cdef str error_message

    # Select the `num_neighbors` of the `num_total_neighbors`
    # nearest-neighbor pairs with the lowest scaled distances
    get_neighbor_pairs(PCs, neighbors, distances, neighbor_pairs,
                       total_distances, num_threads, too_large,
                       self_neighbors, match_parallel)

    # If any nearest-neighbor indices were out of bounds or equal to the
    # cell's own index, raise an error
    if too_large:
        error_message = (
            f'some nearest-neighbor indices in obsm[{neighbors_key!r}] '
            f'are >= the total number of cells, '
            f'{neighbors.shape[0]:,}. This may happen if '
            f'you subset this SingleCell dataset between neighbors() and '
            f'pacmap(); if so, make sure to run neighbors() after, not '
            f'before, subsetting.')
        raise ValueError(error_message)
    elif self_neighbors:
        error_message = (
            f'some nearest-neighbor indices in obsm[{neighbors_key!r}] '
            f'indicate that a cell is its own neighbor, i.e. '
            f'obsm[{neighbors_key!r}][i, j] == i for some i and j. This '
            f'may happen if you created obsm[{neighbors_key!r}] manually '
            f'rather than following the recommended approach of running '
            f'neighbors().')
        raise ValueError(error_message)

    # Sample mid-near pairs
    sample_mid_near_pairs(PCs, mid_near_pairs, seed, num_threads,
                          match_parallel)

    # Sample further pairs
    sample_further_pairs(PCs, neighbor_pairs, further_pairs,
                         seed + num_cells * num_mid_near_pairs, num_threads,
                         match_parallel)

    # If multithreaded, or single-threaded with `match_parallel=True`,
    # reformat the three lists of pairs to ensure deterministic
    # parallelism. Specifically, transform pairs of cell indices from the
    # original format of a 2D array `pairs` where `pairs[i]` contains all
    # js for which (i, j) is a pair, to a pair of 1D arrays `pair_indices`
    # and `pair_indptr` forming a sparse array, where
    # `pair_indices[pair_indptr[i]:pair_indptr[i + 1]]` contains all js for
    # which (i, j) is a pair or (j, i) is a pair. `pair_indices` must have
    # length `2 * pairs.size`, since each pair will appear twice, once for
    # (i, j) and once for (j, i). `pair_indptr` must have length equal to
    # the number of cells plus one, just like for scipy sparse matrices.
    if num_threads == 1 and not match_parallel:
        pacmap_serial_fast(PCs, embedding, momentum, velocity, gradients,
                           neighbor_pairs, mid_near_pairs, further_pairs,
                           num_phase_1_iterations, num_phase_2_iterations,
                           num_phase_3_iterations, learning_rate)
    else:
        num_threads = min(num_threads, num_cells)
        if num_threads < 3:
            reformat_for_parallel(neighbor_pairs, neighbor_pair_indices,
                                  neighbor_pair_indptr, num_threads)
            reformat_for_parallel(mid_near_pairs, mid_near_pair_indices,
                                  mid_near_pair_indptr, num_threads)
            reformat_for_parallel(further_pairs, further_pair_indices,
                                  further_pair_indptr, num_threads)
        else:
            for i in prange(3, nogil=True, num_threads=3):
                if i == 0:
                    reformat_for_parallel(neighbor_pairs,
                                          neighbor_pair_indices,
                                          neighbor_pair_indptr, num_threads)
                elif i == 1:
                    reformat_for_parallel(mid_near_pairs,
                                          mid_near_pair_indices,
                                          mid_near_pair_indptr, num_threads)
                else:
                    reformat_for_parallel(further_pairs, further_pair_indices,
                                          further_pair_indptr, num_threads)
        PyErr_CheckSignals()

        # Use the parallel implementation even when single-threaded, to
        # ensure identical results regardless of the number of threads
        pacmap_parallel(PCs, embedding, momentum, velocity, gradients,
                        neighbor_pair_indices, neighbor_pair_indptr,
                        mid_near_pair_indices, mid_near_pair_indptr,
                        further_pair_indices, further_pair_indptr,
                        num_phase_1_iterations, num_phase_2_iterations,
                        num_phase_3_iterations, learning_rate, num_threads)


cdef inline void localmap_serial_fast(const float[:, ::1] PCs,
                                      float[:, ::1] embedding,
                                      float[:, ::1] momentum,
                                      float[:, ::1] velocity,
                                      float[:, ::1] gradients,
                                      const unsigned[:, ::1] neighbor_pairs,
                                      const unsigned[:, ::1] mid_near_pairs,
                                      unsigned[:, ::1] further_pairs,
                                      const unsigned num_phase_1_iterations,
                                      const unsigned num_phase_2_iterations,
                                      const unsigned num_phase_3_iterations,
                                      const float learning_rate,
                                      const float max_distance,
                                      const unsigned long long seed):
    cdef unsigned i, iteration, num_iterations = num_phase_1_iterations + \
        num_phase_2_iterations + num_phase_3_iterations, \
        w_mid_near_init = 1000
    cdef unsigned long long num_cells = PCs.shape[0]
    cdef float iteration_fraction, w_mid_near, w_neighbors, \
        half_max_distance = <float> 0.5 * max_distance, \
        max_distance_squared = max_distance * max_distance, \
        beta1 = <float> 0.9, beta2 = <float> 0.999

    # Initialize the embedding, momentum and velocity
    for i in range(num_cells):
        embedding[i, 0] = <float> 0.01 * PCs[i, 0]
        embedding[i, 1] = <float> 0.01 * PCs[i, 1]
    momentum[:] = 0
    velocity[:] = 0

    # Optimize the embedding
    for iteration in range(num_iterations):
        if iteration < num_phase_1_iterations:
            iteration_fraction = <float> iteration / num_phase_1_iterations
            w_mid_near = (<float> 1 - iteration_fraction) * w_mid_near_init + \
                iteration_fraction * <float> 3
            w_neighbors = 2
        elif iteration < num_phase_1_iterations + num_phase_2_iterations:
            w_mid_near = 3
            w_neighbors = 3
        else:
            w_mid_near = 0
            w_neighbors = 1

        # Calculate gradients
        if iteration >= num_phase_1_iterations + num_phase_2_iterations:
            get_scaled_gradients_fast(embedding, neighbor_pairs,
                                      mid_near_pairs, further_pairs,
                                      w_neighbors, w_mid_near,
                                      half_max_distance, gradients)
        else:
            get_gradients_fast(embedding, neighbor_pairs, mid_near_pairs,
                               further_pairs, w_neighbors, w_mid_near,
                               gradients)

        # Update the embedding based on the gradients, via the Adam
        # optimizer
        update_embedding_adam(embedding, gradients, momentum, velocity,
                              num_cells, beta1, beta2, learning_rate,
                              iteration)

        # Re-sample further pairs every 10 iterations in phase 3
        if iteration >= num_phase_1_iterations + num_phase_2_iterations \
                and iteration % 10 == 0:
            sample_further_pairs_nearby(embedding, neighbor_pairs,
                                        further_pairs, seed + iteration,
                                        max_distance_squared, num_threads=1)

        # Check for KeyboardInterrupts
        elif iteration % 8 == 7:
            PyErr_CheckSignals()


cdef inline void localmap_parallel(const float[:, ::1] PCs,
                                   float[:, ::1] embedding,
                                   float[:, ::1] momentum,
                                   float[:, ::1] velocity,
                                   float[:, ::1] gradients,
                                   const unsigned[:, ::1] neighbor_pairs,
                                   const unsigned[::1] neighbor_pair_indices,
                                   const unsigned[::1] neighbor_pair_indptr,
                                   const unsigned[::1] mid_near_pair_indices,
                                   const unsigned[::1] mid_near_pair_indptr,
                                   unsigned[:, ::1] further_pairs,
                                   unsigned[::1] further_pair_indices,
                                   unsigned[::1] further_pair_indptr,
                                   const unsigned num_phase_1_iterations,
                                   const unsigned num_phase_2_iterations,
                                   const unsigned num_phase_3_iterations,
                                   const float learning_rate,
                                   const float max_distance,
                                   const unsigned long long seed,
                                   const unsigned num_threads):
    cdef unsigned i, iteration, num_iterations = num_phase_1_iterations + \
        num_phase_2_iterations + num_phase_3_iterations, \
        w_mid_near_init = 1000
    cdef unsigned long long num_cells = PCs.shape[0]
    cdef float iteration_fraction, w_mid_near, w_neighbors, \
        half_max_distance = <float> 0.5 * max_distance, \
        max_distance_squared = max_distance * max_distance, \
        beta1 = <float> 0.9, beta2 = <float> 0.999

    with nogil:
        # Initialize the embedding, momentum and velocity. Momentum and
        # velocity are initialized in parallel for NUMA first-touch reasons.
        for i in prange(num_cells, num_threads=num_threads):
            embedding[i, 0] = <float> 0.01 * PCs[i, 0]
            embedding[i, 1] = <float> 0.01 * PCs[i, 1]
            momentum[i, 0] = 0
            momentum[i, 1] = 0
            velocity[i, 0] = 0
            velocity[i, 1] = 0

        # Optimize the embedding
        for iteration in range(num_iterations):
            if iteration < num_phase_1_iterations:
                iteration_fraction = <float> iteration / num_phase_1_iterations
                w_mid_near = (<float> 1 - iteration_fraction) * \
                    w_mid_near_init + iteration_fraction * <float> 3
                w_neighbors = 2
            elif iteration < num_phase_1_iterations + num_phase_2_iterations:
                w_mid_near = 3
                w_neighbors = 3
            else:
                w_mid_near = 0
                w_neighbors = 1

            # Calculate gradients
            if iteration >= num_phase_1_iterations + num_phase_2_iterations:
                get_scaled_gradients_parallel(
                    embedding, neighbor_pair_indices, neighbor_pair_indptr,
                    mid_near_pair_indices, mid_near_pair_indptr,
                    further_pair_indices, further_pair_indptr, w_neighbors,
                    w_mid_near, half_max_distance, gradients, num_threads)
            else:
                get_gradients_parallel(
                    embedding, neighbor_pair_indices, neighbor_pair_indptr,
                    mid_near_pair_indices, mid_near_pair_indptr,
                    further_pair_indices, further_pair_indptr, w_neighbors,
                    w_mid_near, gradients, num_threads)

            # Update the embedding based on the gradients, via the Adam
            # optimizer
            update_embedding_adam_parallel(embedding, gradients, momentum,
                                           velocity, num_cells, beta1, beta2,
                                           learning_rate, iteration,
                                           num_threads)

            # Re-sample further pairs every 10 iterations in phase 3
            if iteration >= num_phase_1_iterations + num_phase_2_iterations \
                    and iteration % 10 == 0:
                sample_further_pairs_nearby(embedding, neighbor_pairs,
                                            further_pairs, seed + iteration,
                                            max_distance_squared, num_threads)
                reformat_for_parallel(further_pairs, further_pair_indices,
                                      further_pair_indptr, num_threads)

            # Check for KeyboardInterrupts
            elif iteration % 8 == 7:
                with gil:
                    PyErr_CheckSignals()


def localmap(const float[:, ::1] PCs,
             float[:, ::1] embedding,
             float[:, ::1] momentum,
             float[:, ::1] velocity,
             float[:, ::1] gradients,
             float[::1] total_distances,
             unsigned[:, ::1] neighbor_pairs,
             unsigned[:, ::1] mid_near_pairs,
             unsigned[:, ::1] further_pairs,
             unsigned[::1] neighbor_pair_indices,
             unsigned[::1] neighbor_pair_indptr,
             unsigned[::1] mid_near_pair_indices,
             unsigned[::1] mid_near_pair_indptr,
             unsigned[::1] further_pair_indices,
             unsigned[::1] further_pair_indptr,
             const unsigned[:, ::1] neighbors,
             const float[:, ::1] distances,
             const unsigned num_neighbors,
             const unsigned num_extra_neighbors,
             const unsigned num_mid_near_pairs,
             const unsigned num_further_pairs,
             const unsigned num_phase_1_iterations,
             const unsigned num_phase_2_iterations,
             const unsigned num_phase_3_iterations,
             const float learning_rate,
             const float max_distance,
             const unsigned long long seed,
             const bint match_parallel,
             str neighbors_key,
             unsigned num_threads):
    cdef unsigned i, num_cells = PCs.shape[0]
    cdef bint too_large = False, self_neighbors = False
    cdef str error_message

    # Select the `num_neighbors` of the `num_total_neighbors`
    # nearest-neighbor pairs with the lowest scaled distances
    get_neighbor_pairs(PCs, neighbors, distances, neighbor_pairs,
                       total_distances, num_threads, too_large,
                       self_neighbors, match_parallel)

    # If any nearest-neighbor indices were out of bounds or equal to the
    # cell's own index, raise an error
    if too_large:
        error_message = (
            f'some nearest-neighbor indices in obsm[{neighbors_key!r}] '
            f'are >= the total number of cells, '
            f'{neighbors.shape[0]:,}. This may happen if '
            f'you subset this SingleCell dataset between neighbors() and '
            f'localmap(); if so, make sure to run neighbors() after, not '
            f'before, subsetting.')
        raise ValueError(error_message)
    elif self_neighbors:
        error_message = (
            f'some nearest-neighbor indices in obsm[{neighbors_key!r}] '
            f'indicate that a cell is its own neighbor, i.e. '
            f'obsm[{neighbors_key!r}][i, j] == i for some i and j. This '
            f'may happen if you created obsm[{neighbors_key!r}] manually '
            f'rather than following the recommended approach of running '
            f'neighbors().')
        raise ValueError(error_message)

    # Sample mid-near pairs
    sample_mid_near_pairs(PCs, mid_near_pairs, seed, num_threads,
                          match_parallel)

    # Sample further pairs
    sample_further_pairs(PCs, neighbor_pairs, further_pairs,
                         seed + num_cells * num_mid_near_pairs, num_threads,
                         match_parallel)

    # If multithreaded, or single-threaded with `match_parallel=True`,
    # reformat the three lists of pairs to ensure deterministic
    # parallelism. Specifically, transform pairs of cell indices from the
    # original format of a 2D array `pairs` where `pairs[i]` contains all
    # js for which (i, j) is a pair, to a pair of 1D arrays `pair_indices`
    # and `pair_indptr` forming a sparse array, where
    # `pair_indices[pair_indptr[i]:pair_indptr[i + 1]]` contains all js for
    # which (i, j) is a pair or (j, i) is a pair. `pair_indices` must have
    # length `2 * pairs.size`, since each pair will appear twice, once for
    # (i, j) and once for (j, i). `pair_indptr` must have length equal to
    # the number of cells plus one, just like for scipy sparse matrices.
    if num_threads == 1 and not match_parallel:
        localmap_serial_fast(PCs, embedding, momentum, velocity, gradients,
                             neighbor_pairs, mid_near_pairs, further_pairs,
                             num_phase_1_iterations, num_phase_2_iterations,
                             num_phase_3_iterations, learning_rate,
                             max_distance, seed)
    else:
        num_threads = min(num_threads, num_cells)
        if num_threads < 3:
            reformat_for_parallel(neighbor_pairs, neighbor_pair_indices,
                                  neighbor_pair_indptr, num_threads)
            reformat_for_parallel(mid_near_pairs, mid_near_pair_indices,
                                  mid_near_pair_indptr, num_threads)
            reformat_for_parallel(further_pairs, further_pair_indices,
                                  further_pair_indptr, num_threads)
        else:
            for i in prange(3, nogil=True, num_threads=3):
                if i == 0:
                    reformat_for_parallel(neighbor_pairs,
                                          neighbor_pair_indices,
                                          neighbor_pair_indptr, num_threads)
                elif i == 1:
                    reformat_for_parallel(mid_near_pairs,
                                          mid_near_pair_indices,
                                          mid_near_pair_indptr, num_threads)
                else:
                    reformat_for_parallel(further_pairs, further_pair_indices,
                                          further_pair_indptr, num_threads)
        PyErr_CheckSignals()

        # Use the parallel implementation even when single-threaded, to
        # ensure identical results regardless of the number of threads
        localmap_parallel(PCs, embedding, momentum, velocity, gradients,
                          neighbor_pairs, neighbor_pair_indices,
                          neighbor_pair_indptr, mid_near_pair_indices,
                          mid_near_pair_indptr, further_pairs,
                          further_pair_indices, further_pair_indptr,
                          num_phase_1_iterations, num_phase_2_iterations,
                          num_phase_3_iterations, learning_rate,
                          max_distance, seed, num_threads)


def umap_fuzzy_weights(const float[:, ::1] distances,
                       float[::1] data,
                       const unsigned num_threads):

    cdef unsigned i, j, n, N = distances.shape[0], K = distances.shape[1]
    cdef float lo, mid, hi, psum, d, rho, val, mean_ith, ith_sum, \
        target = log2(<float> K)

    if num_threads == 1:
        for i in range(N):
            ith_sum = 0
            rho = distances[i, 0]
            for j in range(K):
                ith_sum += distances[i, j]
            mean_ith = ith_sum / K
            lo = 0
            mid = 1
            hi = FLT_MAX
            for n in range(64):
                psum = 0
                for j in range(K):
                    d = distances[i, j] - rho
                    if d > <float> 0:
                        psum += exp(-(d / mid))
                    else:
                        psum += <float> 1
                if abs(psum - target) < <float> 1e-5:
                    break
                if psum > target:
                    hi = mid
                    mid = <float> 0.5 * (lo + hi)
                else:
                    lo = mid
                    if hi == FLT_MAX:
                        mid *= <float> 2
                    else:
                        mid = <float> 0.5 * (lo + hi)
            if mid < <float> 0.001 * mean_ith:
                mid = <float> 0.001 * mean_ith
            for j in range(K):
                d = distances[i, j] - rho
                if d <= <float> 0 or not mid:
                    val = 1
                else:
                    val = exp(-(d / mid))
                data[i * K + j] = val
    else:
        for i in prange(N, nogil=True, num_threads=num_threads):
            ith_sum = 0
            rho = distances[i, 0]
            for j in range(K):
                ith_sum = ith_sum + distances[i, j]
            mean_ith = ith_sum / K
            lo = 0
            mid = 1
            hi = FLT_MAX
            for n in range(64):
                psum = 0
                for j in range(K):
                    d = distances[i, j] - rho
                    if d > <float> 0:
                        psum = psum + exp(-(d / mid))
                    else:
                        psum = psum + <float> 1
                if abs(psum - target) < <float> 1e-5:
                    break
                if psum > target:
                    hi = mid
                    mid = <float> 0.5 * (lo + hi)
                else:
                    lo = mid
                    if hi == FLT_MAX:
                        mid = mid * <float> 2
                    else:
                        mid = <float> 0.5 * (lo + hi)
            if mid < <float> 0.001 * mean_ith:
                mid = <float> 0.001 * mean_ith
            for j in range(K):
                d = distances[i, j] - rho
                if d <= <float> 0 or not mid:
                    val = 1
                else:
                    val = exp(-(d / mid))
                data[i * K + j] = val


cdef inline void umap_noisy_scale(float[:, ::1] embedding,
                                  unsigned long long[::1] rng_state) \
        noexcept nogil:
    cdef unsigned i, N = embedding.shape[0]
    cdef float val, scale, min0, max0, min1, max1, scale0, scale1, max_val = 0

    # Find global absolute maximum
    for i in range(N):
        val = abs(embedding[i, 0])
        if val > max_val:
            max_val = val
        val = abs(embedding[i, 1])
        if val > max_val:
            max_val = val
    scale = <float> 10 / max_val

    # Scale and add noise
    for i in range(N):
        embedding[i, 0] = (embedding[i, 0] * scale) + \
            <float> 0.0001 * random_normal(&rng_state[i])
        embedding[i, 1] = (embedding[i, 1] * scale) + \
            <float> 0.0001 * random_normal(&rng_state[i])

    # Compute min/max for each dimension
    min0 = max0 = embedding[0, 0]
    min1 = max1 = embedding[0, 1]
    for i in range(1, N):
        val = embedding[i, 0]
        if val < min0: min0 = val
        elif val > max0: max0 = val
        val = embedding[i, 1]
        if val < min1: min1 = val
        elif val > max1: max1 = val

    # Min-max scale to [0, 10]
    scale0 = <float> 10 / (max0 - min0) if max0 > min0 else 0
    scale1 = <float> 10 / (max1 - min1) if max1 > min1 else 0
    for i in range(N):
        embedding[i, 0] = (embedding[i, 0] - min0) * scale0
        embedding[i, 1] = (embedding[i, 1] - min1) * scale1


def umap_optimize(float[:, ::1] embedding,
                  const signed_integer[::1] head,
                  const signed_integer[::1] tail,
                  const float[::1] weights,
                  const unsigned num_iterations,
                  const float a,
                  const float b,
                  const float gamma,
                  const float initial_alpha,
                  const unsigned negative_sample_rate,
                  const unsigned seed,
                  unsigned num_threads):
    # The main SGD optimizer for UMAP

    cdef unsigned epoch, i, u, v, start, end, current_negative_sample_rate, \
        num_vertices = embedding.shape[0], num_edges = head.shape[0]
    cdef float epoch_modifier, alpha, weight, movement
    cdef uninitialized_vector[unsigned long long] rng_state_buffer
    cdef unsigned long long[::1] rng_state

    # Initialize per-vertex random number generators
    rng_state_buffer.resize(num_vertices)
    rng_state = <unsigned long long[:num_vertices]> rng_state_buffer.data()
    for i in range(num_vertices):
        rng_state[i] = srand(seed + i)

    # Add noise and scale the initial embedding
    umap_noisy_scale(embedding, rng_state)

    num_threads = min(num_threads, num_vertices)
    if num_threads == 1:
        # Perform SGD
        for epoch in range(num_iterations):
            epoch_modifier = <float> 1 - epoch / <float> num_iterations
            alpha = initial_alpha * epoch_modifier
            for i in range(num_edges):
                u = head[i]
                v = tail[i]
                weight = weights[i]
                umap_single_edge_gradient_update(
                    embedding, u, v, weight, a, b, gamma, alpha,
                    negative_sample_rate, &rng_state[u], num_vertices)
            PyErr_CheckSignals()
    else:
        # Hogwild! Re-initialize `rng_state` to be per-thread RNGs, not
        # per-vertex.
        for i in range(num_threads):
            rng_state[i] = srand(seed + i)

        # Perform SGD in parallel, without locks
        for epoch in range(num_iterations):
            epoch_modifier = <float> 1 - epoch / <float> num_iterations
            alpha = initial_alpha * epoch_modifier
            for i in prange(num_edges, num_threads=num_threads, nogil=True):
                u = head[i]
                v = tail[i]
                weight = weights[i]
                umap_single_edge_gradient_update(
                    embedding, u, v, weight, a, b, gamma, alpha,
                    negative_sample_rate, &rng_state[threadid()], num_vertices)
            PyErr_CheckSignals()


cdef inline void umap_single_edge_gradient_update(
        float[:, ::1] embedding,
        const unsigned u,
        const unsigned v,
        const float weight,
        const float a,
        const float b,
        const float gamma,
        const float alpha,
        const unsigned negative_sample_rate,
        unsigned long long *rng_state,
        const unsigned num_vertices) noexcept nogil:
    # Update the gradients for the two cells linked by a single edge.
    # Attractive forces update both endpoints (`u` and `v`) symmetrically so
    # the edge pulls the two cells together in the embedding. Repulsive forces
    # are implemented using negative sampling: we draw random vertices and push
    # them away from `u` only. This asymmetric update is intentional and
    # matches the stochastic approximation used in the original UMAP optimizer.
    # In expectation the repulsion is symmetric because every vertex appears as
    # the source vertex of some edge update, but updating only `u` keeps the
    # computation cheaper.

    cdef unsigned k, other_v
    cdef float ux, uy, vx, vy, dx, dy, dist_squared, dist_b_minus_1, dist_b, \
        denom, scale, gradx, grady, \
        attraction_constant = <float> -2 * a * b * alpha, \
        repulsion_constant = alpha * (<float> 2 * gamma * b)

    # Randomly drop the edge update with probability `1 - weight`.
    # This is mathematically equivalent to UMAP's fractional epoch updates!
    if weight < random_uniform(rng_state): return

    # Attractive gradients
    ux = embedding[u, 0]; uy = embedding[u, 1]
    vx = embedding[v, 0]; vy = embedding[v, 1]
    dx = ux - vx; dy = uy - vy
    dist_squared = dx * dx + dy * dy
    if dist_squared:
        dist_b_minus_1 = pow(dist_squared, b - <float> 1)
        dist_b = dist_b_minus_1 * dist_squared
        denom = <float> 1 + a * dist_b
        scale = attraction_constant * dist_b_minus_1 / denom
        gradx = min(<float> 4, max(<float> -4, scale * dx))
        grady = min(<float> 4, max(<float> -4, scale * dy))
        ux += gradx; uy += grady
        vx -= gradx; vy -= grady
    embedding[v, 0] = vx; embedding[v, 1] = vy

    # Repulsive gradients, using negative sampling
    for k in range(negative_sample_rate):
        other_v = randint(num_vertices, rng_state)
        if other_v == u: continue
        vx = embedding[other_v, 0]; vy = embedding[other_v, 1]
        dx = ux - vx; dy = uy - vy
        dist_squared = dx * dx + dy * dy
        if dist_squared:
            denom = (<float> 0.001 + dist_squared) * \
                (<float> 1 + a * pow(dist_squared, b))
            scale = repulsion_constant / denom
            gradx = min(<float> 4, max(<float> -4, scale * dx))
            grady = min(<float> 4, max(<float> -4, scale * dy))
            ux += gradx; uy += grady
    embedding[u, 0] = ux; embedding[u, 1] = uy