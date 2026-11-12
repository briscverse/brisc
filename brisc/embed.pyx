# Functions for embedding with PaCMAP, LocalMAP and UMAP

from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libc.float cimport FLT_MAX
from libc.limits cimport UINT_MAX
from libc.math cimport M_PI
from libc.string cimport memcpy
from libcpp.algorithm cimport min_element, nth_element, sort
from libcpp.cmath cimport atan2, cos, exp, log, log2, log10, pow, sin, sqrt, \
    tanh
from libcpp.vector cimport vector
from .cyutils cimport PREFETCH, advise_huge_pages, atomic_or, \
    get_thread_offsets, max_heap_replace_top, max_heap_sort, randint, \
    random_normal, random_uniform, srand, uninitialized_vector


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
    cdef unsigned long long state, lookahead_state
    cdef float dx, dy

    # Use a parallel loop even when single-threaded, to ensure identical
    # results regardless of the number of threads
    for i in prange(n, nogil=True, num_threads=num_threads):
        state = srand(seed + i)

        # Draw candidates 16 ahead of `state` to prefetch their positions
        lookahead_state = state
        for k in range(16):
            PREFETCH(&embedding[randint(n, &lookahead_state), 0])

        for j in range(num_further_pairs):
            # Give up after 100 trials, and keep the further pair as-is;
            # this corrects the original implementation's logic by counting
            # all types of failures towards the 100 trials, including ones
            # where the candidate further pair is too far away
            for count in range(100):
                # Sample a random cell...
                further_pair_index = randint(n, &state)
                PREFETCH(&embedding[randint(n, &lookahead_state), 0])

                # ...that is not this cell...
                if further_pair_index == i:
                    continue

                # ...nor farther than `max_distance` away (in embedding
                # space), the most common reason for rejection...
                dx = embedding[i, 0] - embedding[further_pair_index, 0]
                dy = embedding[i, 1] - embedding[further_pair_index, 1]
                if dx * dx + dy * dy > max_distance_squared:
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


cdef inline void reformat_for_parallel(
        const unsigned[:, ::1] pairs,
        unsigned[::1] pair_indices,
        unsigned[::1] pair_indptr,
        const unsigned num_threads) noexcept nogil:
    cdef unsigned i, j, k, t, r, own, num_pairs_per_cell = pairs.shape[1]
    cdef unsigned long long index, p, start, end, chunk_size, total, \
        num_cells = pairs.shape[0], stride = num_threads + 16
    cdef uninitialized_vector[unsigned] dest_indices, bucket_sources, \
        bucket_targets
    cdef uninitialized_vector[unsigned long long] cursors, range_offsets
    dest_indices.resize(num_cells)

    # Each pair (i, j) appears in two cells' lists: i's, as part of
    # `pairs[i]`, and j's, as a "reverse" pair. Each cell's list contains
    # its pairs in the order they would be appended by going through `pairs`
    # in order and appending j to i's list and i to j's list. This assumes no
    # self-neighbors (checked earlier in `get_neighbor_pairs()`).

    if num_threads == 1:
        # Count how many pairs each cell is in, cumsum the counts in
        # `pair_indptr`, then go through `pairs` in order
        pair_indptr[0] = 0
        pair_indptr[1:] = num_pairs_per_cell
        for i in range(num_cells):
            for k in range(num_pairs_per_cell):
                pair_indptr[pairs[i, k] + 1] += 1
        for i in range(2, pair_indptr.shape[0]):
            pair_indptr[i] += pair_indptr[i - 1]
        memcpy(dest_indices.data(), &pair_indptr[0],
               num_cells * sizeof(unsigned))
        for i in range(num_cells):
            for k in range(num_pairs_per_cell):
                j = pairs[i, k]
                pair_indices[dest_indices[i]] = j
                pair_indices[dest_indices[j]] = i
                dest_indices[i] += 1
                dest_indices[j] += 1
        return

    # Perform first-touch NUMA page allocation for the outputs
    for index in prange(pair_indices.shape[0], num_threads=num_threads):
        pair_indices[index] = 0
    for index in prange(pair_indptr.shape[0], num_threads=num_threads):
        pair_indptr[index] = 0

    # Split the cells into `num_threads` contiguous ranges of `chunk_size`
    # cells. Bucket the reverse pairs by the range containing j, keeping them
    # in order of i: count the pairs going from each range of i's to each
    # range of j's, then lay out the buckets for each range of j's
    # consecutively, ordered by range of i.
    bucket_sources.resize(num_cells * num_pairs_per_cell)
    bucket_targets.resize(num_cells * num_pairs_per_cell)
    cursors.resize(num_threads * stride)
    range_offsets.resize(num_threads + 1)
    chunk_size = (num_cells + num_threads - 1) / num_threads
    for t in prange(num_threads, num_threads=num_threads):
        for r in range(num_threads):
            cursors[t * stride + r] = 0
        start = min(t * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            for k in range(num_pairs_per_cell):
                r = pairs[i, k] / chunk_size
                cursors[t * stride + r] += 1
    total = 0
    for r in range(num_threads):
        range_offsets[r] = total
        for t in range(num_threads):
            p = cursors[t * stride + r]
            cursors[t * stride + r] = total
            total += p
    range_offsets[num_threads] = total
    for t in prange(num_threads, num_threads=num_threads):
        start = min(t * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            for k in range(num_pairs_per_cell):
                j = pairs[i, k]
                r = j / chunk_size
                p = cursors[t * stride + r]
                cursors[t * stride + r] = p + 1
                bucket_sources[p] = i
                bucket_targets[p] = j

    # Count how many pairs each cell is in (`pairs.shape[1]` as the i, plus
    # its reverse pairs), then cumsum the counts in `pair_indptr`
    pair_indptr[0] = 0
    for r in prange(num_threads, num_threads=num_threads):
        start = min(r * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            pair_indptr[i + 1] = num_pairs_per_cell
        for p in range(range_offsets[r], range_offsets[r + 1]):
            pair_indptr[bucket_targets[p] + 1] += 1
    for i in range(2, pair_indptr.shape[0]):
        pair_indptr[i] += pair_indptr[i - 1]

    # Fill in each cell's list: each range goes through its reverse pairs in
    # order of i, and appends each of its cells' own pairs, `pairs[own]`,
    # just before the first reverse pair with i >= own
    for r in prange(num_threads, num_threads=num_threads):
        start = min(r * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            dest_indices[i] = pair_indptr[i]
        own = start
        for p in range(range_offsets[r], range_offsets[r + 1]):
            i = bucket_sources[p]
            while own < end and own <= i:
                for k in range(num_pairs_per_cell):
                    pair_indices[dest_indices[own]] = pairs[own, k]
                    dest_indices[own] += 1
                own = own + 1
            j = bucket_targets[p]
            pair_indices[dest_indices[j]] = i
            dest_indices[j] += 1
        while own < end:
            for k in range(num_pairs_per_cell):
                pair_indices[dest_indices[own]] = pairs[own, k]
                dest_indices[own] += 1
            own = own + 1


cdef inline void get_gradients_fast(const float[:, ::1] embedding,
                                    const unsigned[:, ::1] neighbor_pairs,
                                    const unsigned[:, ::1] mid_near_pairs,
                                    const unsigned[:, ::1] further_pairs,
                                    const float w_neighbors,
                                    const float w_mid_near,
                                    float[:, ::1] gradients):
    cdef unsigned i, j, num_neighbors = neighbor_pairs.shape[1], \
        num_mid_near_pairs = mid_near_pairs.shape[1], \
        num_further_pairs = further_pairs.shape[1]
    cdef unsigned long long num_cells = neighbor_pairs.shape[0]
    cdef float embedding_i0, embedding_i1, gradients_i0, gradients_i1, \
        embedding_ij_0, embedding_ij_1, distance_ij, w
    cdef const unsigned[::1] neighbor_pairs_flat = \
        <unsigned[:num_cells * num_neighbors]> &neighbor_pairs[0, 0]
    cdef const unsigned[::1] mid_near_pairs_flat = \
        <unsigned[:num_cells * num_mid_near_pairs]> &mid_near_pairs[0, 0]
    cdef const unsigned[::1] further_pairs_flat = \
        <unsigned[:num_cells * num_further_pairs]> &further_pairs[0, 0]
    cdef unsigned long long index

    gradients[:] = 0
    for i in range(num_cells):
        embedding_i0 = embedding[i, 0]
        embedding_i1 = embedding[i, 1]
        gradients_i0 = 0
        gradients_i1 = 0

        # Nearest-neighbor pairs
        for index in range(<unsigned long long> i * num_neighbors,
                           <unsigned long long> (i + 1) * num_neighbors):
            if index + 8 < num_cells * num_neighbors:
                PREFETCH(&embedding[neighbor_pairs_flat[index + 8], 0])
                PREFETCH(&gradients[neighbor_pairs_flat[index + 8], 0])
            j = neighbor_pairs_flat[index]
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

        # Mid-near pairs (skipped in phase 3, where their weight is 0)
        if w_mid_near:
            for index in range(
                    <unsigned long long> i * num_mid_near_pairs,
                    <unsigned long long> (i + 1) * num_mid_near_pairs):
                if index + 8 < num_cells * num_mid_near_pairs:
                    PREFETCH(&embedding[mid_near_pairs_flat[index + 8], 0])
                    PREFETCH(&gradients[mid_near_pairs_flat[index + 8], 0])
                j = mid_near_pairs_flat[index]
                embedding_ij_0 = embedding_i0 - embedding[j, 0]
                embedding_ij_1 = embedding_i1 - embedding[j, 1]
                distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
                    embedding_ij_1 * embedding_ij_1
                w = w_mid_near * (<float> 20000 / (
                    (<float> 10000 + distance_ij) *
                    (<float> 10000 + distance_ij)))
                gradients_i0 += w * embedding_ij_0
                gradients[j, 0] -= w * embedding_ij_0
                gradients_i1 += w * embedding_ij_1
                gradients[j, 1] -= w * embedding_ij_1

        # Further pairs
        for index in range(<unsigned long long> i * num_further_pairs,
                           <unsigned long long> (i + 1) * num_further_pairs):
            if index + 8 < num_cells * num_further_pairs:
                PREFETCH(&embedding[further_pairs_flat[index + 8], 0])
                PREFETCH(&gradients[further_pairs_flat[index + 8], 0])
            j = further_pairs_flat[index]
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
        const unsigned[:, ::1] further_pairs,
        const float half_max_distance,
        float[:, ::1] gradients):
    # LocalMAP's phase-3 gradients, where `w_neighbors` is 1 and `w_mid_near`
    # is 0 (so mid-near pairs are skipped), and nearest-neighbor forces are
    # scaled by `half_max_distance / sqrt(distance_ij)`
    cdef unsigned i, j, num_neighbors = neighbor_pairs.shape[1], \
        num_further_pairs = further_pairs.shape[1]
    cdef unsigned long long num_cells = neighbor_pairs.shape[0]
    cdef float embedding_i0, embedding_i1, gradients_i0, gradients_i1, \
        embedding_ij_0, embedding_ij_1, distance_ij, w
    cdef const unsigned[::1] neighbor_pairs_flat = \
        <unsigned[:num_cells * num_neighbors]> &neighbor_pairs[0, 0]
    cdef const unsigned[::1] further_pairs_flat = \
        <unsigned[:num_cells * num_further_pairs]> &further_pairs[0, 0]
    cdef unsigned long long index

    gradients[:] = 0
    for i in range(num_cells):
        embedding_i0 = embedding[i, 0]
        embedding_i1 = embedding[i, 1]
        gradients_i0 = 0
        gradients_i1 = 0

        # Nearest-neighbor pairs
        for index in range(<unsigned long long> i * num_neighbors,
                           <unsigned long long> (i + 1) * num_neighbors):
            if index + 8 < num_cells * num_neighbors:
                PREFETCH(&embedding[neighbor_pairs_flat[index + 8], 0])
                PREFETCH(&gradients[neighbor_pairs_flat[index + 8], 0])
            j = neighbor_pairs_flat[index]
            embedding_ij_0 = embedding_i0 - embedding[j, 0]
            embedding_ij_1 = embedding_i1 - embedding[j, 1]
            distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
                embedding_ij_1 * embedding_ij_1
            w = <float> 20 / ((<float> 10 + distance_ij) *
                              (<float> 10 + distance_ij))
            w *= half_max_distance / sqrt(distance_ij)
            gradients_i0 += w * embedding_ij_0
            gradients[j, 0] -= w * embedding_ij_0
            gradients_i1 += w * embedding_ij_1
            gradients[j, 1] -= w * embedding_ij_1

        # Further pairs
        for index in range(<unsigned long long> i * num_further_pairs,
                           <unsigned long long> (i + 1) * num_further_pairs):
            if index + 8 < num_cells * num_further_pairs:
                PREFETCH(&embedding[further_pairs_flat[index + 8], 0])
                PREFETCH(&gradients[further_pairs_flat[index + 8], 0])
            j = further_pairs_flat[index]
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
        if k + 8 < neighbor_pair_indices.shape[0]:
            PREFETCH(&embedding[neighbor_pair_indices[k + 8], 0])
        j = neighbor_pair_indices[k]
        embedding_ij_0 = embedding_i0 - embedding[j, 0]
        embedding_ij_1 = embedding_i1 - embedding[j, 1]
        distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
            embedding_ij_1 * embedding_ij_1
        w = w_neighbors * (<float> 20 /
            ((<float> 10 + distance_ij) * (<float> 10 + distance_ij)))
        gradient_i0 = gradient_i0 + w * embedding_ij_0
        gradient_i1 = gradient_i1 + w * embedding_ij_1

    # Mid-near pairs (skipped in phase 3, where their weight is 0)
    if w_mid_near:
        for k in range(mid_near_pair_indptr[i],
                       mid_near_pair_indptr[i + 1]):
            if k + 8 < mid_near_pair_indices.shape[0]:
                PREFETCH(&embedding[mid_near_pair_indices[k + 8], 0])
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
        if k + 8 < further_pair_indices.shape[0]:
            PREFETCH(&embedding[further_pair_indices[k + 8], 0])
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
                                     const unsigned[::1] further_pair_indices,
                                     const unsigned[::1] further_pair_indptr,
                                     const float half_max_distance,
                                     float[:, ::1] gradients,
                                     const unsigned i) noexcept nogil:
    # LocalMAP's phase-3 gradients, where `w_neighbors` is 1 and `w_mid_near`
    # is 0 (so mid-near pairs are skipped), and nearest-neighbor forces are
    # scaled by `half_max_distance / sqrt(distance_ij)`
    cdef unsigned j, k
    cdef unsigned long long num_cells = embedding.shape[0]
    cdef float embedding_ij_0, embedding_ij_1, distance_ij, w, \
        embedding_i0 = embedding[i, 0], embedding_i1 = embedding[i, 1], \
        gradient_i0 = 0, gradient_i1 = 0

    # Nearest-neighbor pairs
    for k in range(neighbor_pair_indptr[i],
                   neighbor_pair_indptr[i + 1]):
        if k + 8 < neighbor_pair_indices.shape[0]:
            PREFETCH(&embedding[neighbor_pair_indices[k + 8], 0])
        j = neighbor_pair_indices[k]
        embedding_ij_0 = embedding_i0 - embedding[j, 0]
        embedding_ij_1 = embedding_i1 - embedding[j, 1]
        distance_ij = <float> 1 + embedding_ij_0 * embedding_ij_0 + \
            embedding_ij_1 * embedding_ij_1
        w = <float> 20 / ((<float> 10 + distance_ij) *
                          (<float> 10 + distance_ij))
        w *= half_max_distance / sqrt(distance_ij)
        gradient_i0 = gradient_i0 + w * embedding_ij_0
        gradient_i1 = gradient_i1 + w * embedding_ij_1

    # Further pairs
    for k in range(further_pair_indptr[i],
                   further_pair_indptr[i + 1]):
        if k + 8 < further_pair_indices.shape[0]:
            PREFETCH(&embedding[further_pair_indices[k + 8], 0])
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
        const unsigned[::1] further_pair_indices,
        const unsigned[::1] further_pair_indptr,
        const float half_max_distance,
        float[:, ::1] gradients,
        const unsigned num_threads) noexcept nogil:
    cdef unsigned i
    cdef unsigned long long num_cells = embedding.shape[0]

    for i in prange(num_cells, num_threads=num_threads):
        get_scaled_gradient(embedding, neighbor_pair_indices,
                            neighbor_pair_indptr, further_pair_indices,
                            further_pair_indptr, half_max_distance,
                            gradients, i)


cdef inline void update_cell_embedding_adam(
        float& embedding,
        const float gradient,
        float& momentum,
        float& velocity,
        const float beta1,
        const float beta2,
        const float learning_rate,
        const float epsilon) noexcept nogil:
    momentum += (<float> 1 - beta1) * (gradient - momentum)
    velocity += (<float> 1 - beta2) * (gradient * gradient - velocity)
    embedding -= learning_rate * momentum / (sqrt(velocity) + epsilon)


cdef inline void update_embedding_adam(
        float[:, ::1] embedding,
        const float[:, ::1] gradients,
        float[:, ::1] momentum,
        float[:, ::1] velocity,
        const unsigned long long num_cells,
        const float beta1,
        const float beta2,
        float learning_rate,
        const float epsilon,
        const unsigned iteration):
    cdef unsigned i
    learning_rate *= sqrt(<float> 1 - pow(beta2, <float> (iteration + 1))) / \
        (<float> 1 - pow(beta1, <float> (iteration + 1)))
    for i in range(num_cells):
        update_cell_embedding_adam(embedding[i, 0], gradients[i, 0],
                                   momentum[i, 0], velocity[i, 0], beta1,
                                   beta2, learning_rate, epsilon)
        update_cell_embedding_adam(embedding[i, 1], gradients[i, 1],
                                   momentum[i, 1], velocity[i, 1], beta1,
                                   beta2, learning_rate, epsilon)


cdef inline void update_embedding_adam_parallel(
        float[:, ::1] embedding,
        const float[:, ::1] gradients,
        float[:, ::1] momentum,
        float[:, ::1] velocity,
        const unsigned long long num_cells,
        const float beta1,
        const float beta2,
        float learning_rate,
        const float epsilon,
        const unsigned iteration,
        const unsigned num_threads) noexcept nogil:
    cdef unsigned i
    learning_rate *= sqrt(<float> 1 - pow(beta2, <float> (iteration + 1))) / \
        (<float> 1 - pow(beta1, <float> (iteration + 1)))
    for i in prange(num_cells, num_threads=num_threads):
        update_cell_embedding_adam(embedding[i, 0], gradients[i, 0],
                                   momentum[i, 0], velocity[i, 0], beta1,
                                   beta2, learning_rate, epsilon)
        update_cell_embedding_adam(embedding[i, 1], gradients[i, 1],
                                   momentum[i, 1], velocity[i, 1], beta1,
                                   beta2, learning_rate, epsilon)


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
                              <float> 1e-7, iteration)

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
                                           learning_rate, <float> 1e-7,
                                           iteration, num_threads)

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
        reformat_for_parallel(neighbor_pairs, neighbor_pair_indices,
                              neighbor_pair_indptr, num_threads)
        reformat_for_parallel(mid_near_pairs, mid_near_pair_indices,
                              mid_near_pair_indptr, num_threads)
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

        # Calculate gradients
        if iteration >= num_phase_1_iterations + num_phase_2_iterations:
            get_scaled_gradients_fast(embedding, neighbor_pairs,
                                      further_pairs, half_max_distance,
                                      gradients)
        else:
            get_gradients_fast(embedding, neighbor_pairs, mid_near_pairs,
                               further_pairs, w_neighbors, w_mid_near,
                               gradients)

        # Update the embedding based on the gradients, via the Adam
        # optimizer
        update_embedding_adam(embedding, gradients, momentum, velocity,
                              num_cells, beta1, beta2, learning_rate,
                              <float> 1e-7, iteration)

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

            # Calculate gradients
            if iteration >= num_phase_1_iterations + num_phase_2_iterations:
                get_scaled_gradients_parallel(
                    embedding, neighbor_pair_indices, neighbor_pair_indptr,
                    further_pair_indices, further_pair_indptr,
                    half_max_distance, gradients, num_threads)
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
                                           learning_rate, <float> 1e-7,
                                           iteration, num_threads)

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
        reformat_for_parallel(neighbor_pairs, neighbor_pair_indices,
                              neighbor_pair_indptr, num_threads)
        reformat_for_parallel(mid_near_pairs, mid_near_pair_indices,
                              mid_near_pair_indptr, num_threads)
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


# UMAP. brisc's implementation follows umap-learn 0.6's new pipeline
# (`compatibility_layout=False`): a recursive initialization that lays out
# successively coarser versions of the nearest-neighbor graph, then an Adam
# optimizer that updates every cell's position synchronously each epoch. Since
# each cell's update only reads the previous epoch's positions and only writes
# its own position, and all random numbers are drawn from per-cell streams, the
# results do not depend on the number of threads.

cdef inline void umap_fuzzy_weights(const float* distances,
                                    float* weights,
                                    const unsigned num_neighbors,
                                    const float target) noexcept nogil:
    # Convert one cell's squared nearest-neighbor distances to UMAP's directed
    # fuzzy membership weights, as in umap-learn's `smooth_knn_dist()`:
    # subtract the distance to the nearest neighbor (`rho`), then binary-search
    # for the bandwidth (`mid`) at which the weights sum to `target`

    cdef unsigned j, n
    cdef float d, rho, mean_distance, weight_sum, distance_sum = 0, lo = 0, \
        mid = 1, hi = FLT_MAX

    for j in range(num_neighbors):
        d = sqrt(max(distances[j], <float> 0))
        weights[j] = d
        distance_sum += d
    rho = 0
    for j in range(num_neighbors):
        if weights[j] > <float> 0:
            rho = weights[j]
            break
    mean_distance = distance_sum / (num_neighbors + 1)
    for n in range(64):
        weight_sum = 0
        for j in range(num_neighbors):
            d = max(weights[j] - rho, <float> 0)
            weight_sum += exp(-(d / mid))
        if abs(weight_sum - target) < <float> 1e-5:
            break
        if weight_sum > target:
            hi = mid
            mid = <float> 0.5 * (lo + hi)
        else:
            lo = mid
            if hi == FLT_MAX:
                mid *= <float> 2
            else:
                mid = <float> 0.5 * (lo + hi)
    if mid < <float> 0.001 * mean_distance:
        mid = <float> 0.001 * mean_distance
    for j in range(num_neighbors):
        d = max(weights[j] - rho, <float> 0)
        weights[j] = exp(-(d / mid))


cdef inline void umap_union_weights(const unsigned* neighbors_i,
                                    const float* directed,
                                    const unsigned long long* reverse,
                                    const unsigned long long reverse_start,
                                    const unsigned long long reverse_end,
                                    const unsigned i,
                                    const unsigned num_neighbors,
                                    float* forward_weights) noexcept nogil:
    # Calculate cell `i`'s fuzzy-union weights to each of its own nearest
    # neighbors `j`: `w_ij + w_ji - w_ij * w_ji` if `i` is also one of `j`'s
    # nearest neighbors, and just `w_ij` otherwise. `reverse[reverse_start:
    # reverse_end]` holds the flat indices `j * num_neighbors + k` of every
    # directed edge pointing to `i`.

    cdef unsigned k, j
    cdef unsigned long long p, e
    cdef float w

    for k in range(num_neighbors):
        forward_weights[k] = \
            directed[<unsigned long long> i * num_neighbors + k]
    for p in range(reverse_start, reverse_end):
        e = reverse[p]
        j = <unsigned> (e / num_neighbors)
        k = 0
        while k < num_neighbors and neighbors_i[k] != j:
            k += 1
        if k < num_neighbors:
            w = directed[e]
            forward_weights[k] = \
                forward_weights[k] + w - forward_weights[k] * w


cdef void umap_fuzzy_graph(const unsigned[:, ::1] neighbors,
                           const float[:, ::1] distances,
                           const unsigned num_pruning_iterations,
                           uninitialized_vector[long long]& indptr,
                           uninitialized_vector[unsigned]& indices,
                           uninitialized_vector[float]& weights,
                           bint& too_large,
                           bint& self_neighbors,
                           unsigned num_threads):
    # Build UMAP's fuzzy simplicial set: convert each cell's nearest-neighbor
    # distances to directed membership weights `w_ij`, symmetrize them with the
    # fuzzy union `w_ij + w_ji - w_ij * w_ji`, and prune weights below
    # `max_weight / num_pruning_iterations`, which would be sampled less than
    # once during optimization. Each row of the resulting CSR graph lists the
    # cell's own nearest neighbors (in order), followed by the cells that have
    # it as a nearest neighbor but are not among its own nearest neighbors (in
    # ascending order).

    cdef unsigned i, j, k, t, r, thread_index, start, end, chunk_size, \
        stride, num_cells = neighbors.shape[0], \
        num_neighbors = neighbors.shape[1]
    cdef unsigned long long e, p, q, total, count, num_edges = \
        <unsigned long long> num_cells * num_neighbors
    cdef float w, max_weight, threshold, \
        target = log2(<float> (num_neighbors + 1))
    cdef const unsigned* neighbors_flat = &neighbors[0, 0]
    cdef uninitialized_vector[float] directed_buffer, forward_weights_buffer, \
        thread_max_weights
    cdef uninitialized_vector[unsigned long long] counts, range_offsets, \
        buckets_buffer, reverse_buffer, reverse_indptr_buffer, cursors, \
        thread_totals
    cdef float* directed
    cdef float* forward_weights
    cdef unsigned long long* buckets
    cdef unsigned long long* reverse
    cdef unsigned long long* reverse_indptr

    # Split cells into `num_threads` contiguous chunks of `chunk_size` cells.
    # Pad per-thread arrays to avoid false sharing.
    num_threads = min(num_threads, num_cells)
    chunk_size = (num_cells + num_threads - 1) / num_threads
    stride = num_threads + 16

    # Calculate directed weights and check that nearest-neighbor indices are
    # valid. Also count the edges going from each chunk of source cells to
    # each chunk of target cells.
    directed_buffer.resize(num_edges)
    directed = directed_buffer.data()
    advise_huge_pages(directed, num_edges * sizeof(float))
    counts.resize(<unsigned long long> num_threads * stride)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        for r in range(num_threads):
            counts[thread_index * stride + r] = 0
        start = min(thread_index * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            umap_fuzzy_weights(&distances[i, 0],
                               &directed[<unsigned long long> i *
                                         num_neighbors],
                               num_neighbors, target)
            for k in range(num_neighbors):
                j = neighbors[i, k]
                if j >= num_cells:
                    atomic_or(too_large, True)
                elif j == i:
                    atomic_or(self_neighbors, True)
                else:
                    counts[thread_index * stride + j / chunk_size] += 1
    if too_large or self_neighbors:
        return

    # Transpose the directed edges to get, for each cell, the edges pointing
    # to it, in ascending order of source cell. Do this deterministically and
    # in parallel by first bucketing edges by the chunk containing their
    # target, with the buckets for each chunk of targets laid out
    # consecutively and ordered by chunk of source.
    range_offsets.resize(num_threads + 1)
    total = 0
    for r in range(num_threads):
        range_offsets[r] = total
        for t in range(num_threads):
            count = counts[t * stride + r]
            counts[t * stride + r] = total
            total += count
    range_offsets[num_threads] = total
    buckets_buffer.resize(num_edges)
    buckets = buckets_buffer.data()
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        start = min(thread_index * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            for k in range(num_neighbors):
                e = <unsigned long long> i * num_neighbors + k
                r = neighbors_flat[e] / chunk_size
                p = counts[thread_index * stride + r]
                counts[thread_index * stride + r] = p + 1
                buckets[p] = e

    # Count the edges pointing to each cell and cumsum the counts in
    # `reverse_indptr`; each thread handles one chunk of targets, whose cumsum
    # starts at `range_offsets[thread_index]`
    reverse_indptr_buffer.resize(num_cells + 1)
    reverse_indptr = reverse_indptr_buffer.data()
    reverse_indptr[0] = 0
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        start = min(thread_index * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for j in range(start, end):
            reverse_indptr[j + 1] = 0
        for p in range(range_offsets[thread_index],
                       range_offsets[thread_index + 1]):
            reverse_indptr[neighbors_flat[buckets[p]] + 1] += 1
        q = range_offsets[thread_index]
        for j in range(start, end):
            q = q + reverse_indptr[j + 1]
            reverse_indptr[j + 1] = q

    # Fill in the edges pointing to each cell
    reverse_buffer.resize(num_edges)
    reverse = reverse_buffer.data()
    cursors.resize(num_cells)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        start = min(thread_index * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for j in range(start, end):
            cursors[j] = reverse_indptr[j]
        for p in range(range_offsets[thread_index],
                       range_offsets[thread_index + 1]):
            e = buckets[p]
            j = neighbors_flat[e]
            reverse[cursors[j]] = e
            cursors[j] += 1
    buckets_buffer.clear()
    buckets_buffer.shrink_to_fit()
    cursors.clear()
    cursors.shrink_to_fit()

    # Find the maximum fuzzy-union weight
    forward_weights_buffer.resize(num_threads * (num_neighbors + 16))
    thread_max_weights.resize(num_threads * 16)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        forward_weights = \
            forward_weights_buffer.data() + thread_index * (num_neighbors + 16)
        w = 0
        start = min(thread_index * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            umap_union_weights(&neighbors[i, 0], directed, reverse,
                               reverse_indptr[i], reverse_indptr[i + 1], i,
                               num_neighbors, forward_weights)
            for k in range(num_neighbors):
                if forward_weights[k] > w:
                    w = forward_weights[k]
            for p in range(reverse_indptr[i], reverse_indptr[i + 1]):
                e = reverse[p]
                j = <unsigned> (e / num_neighbors)
                k = 0
                while k < num_neighbors and neighbors[i, k] != j:
                    k = k + 1
                if k == num_neighbors and directed[e] > w:
                    w = directed[e]
        thread_max_weights[thread_index * 16] = w
    max_weight = 0
    for t in range(num_threads):
        max_weight = max(max_weight, thread_max_weights[t * 16])
    threshold = max_weight / num_pruning_iterations

    # Count the edges each cell keeps after pruning, then cumsum the counts in
    # `indptr`
    indptr.resize(num_cells + 1)
    thread_totals.resize(num_threads * 16)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        forward_weights = \
            forward_weights_buffer.data() + thread_index * (num_neighbors + 16)
        q = 0
        start = min(thread_index * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            umap_union_weights(&neighbors[i, 0], directed, reverse,
                               reverse_indptr[i], reverse_indptr[i + 1], i,
                               num_neighbors, forward_weights)
            count = 0
            for k in range(num_neighbors):
                if forward_weights[k] >= threshold:
                    count = count + 1
            for p in range(reverse_indptr[i], reverse_indptr[i + 1]):
                e = reverse[p]
                j = <unsigned> (e / num_neighbors)
                k = 0
                while k < num_neighbors and neighbors[i, k] != j:
                    k = k + 1
                if k == num_neighbors and directed[e] >= threshold:
                    count = count + 1
            q = q + count
            indptr[i + 1] = q
        thread_totals[thread_index * 16] = q
    indptr[0] = 0
    total = 0
    for t in range(num_threads):
        count = thread_totals[t * 16]
        thread_totals[t * 16] = total
        total += count
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        start = min(thread_index * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            indptr[i + 1] += thread_totals[thread_index * 16]

    # Fill in the pruned graph
    indices.resize(total)
    weights.resize(total)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        forward_weights = \
            forward_weights_buffer.data() + thread_index * (num_neighbors + 16)
        start = min(thread_index * chunk_size, num_cells)
        end = min(start + chunk_size, num_cells)
        for i in range(start, end):
            umap_union_weights(&neighbors[i, 0], directed, reverse,
                               reverse_indptr[i], reverse_indptr[i + 1], i,
                               num_neighbors, forward_weights)
            q = indptr[i]
            for k in range(num_neighbors):
                if forward_weights[k] >= threshold:
                    indices[q] = neighbors[i, k]
                    weights[q] = forward_weights[k]
                    q = q + 1
            for p in range(reverse_indptr[i], reverse_indptr[i + 1]):
                e = reverse[p]
                j = <unsigned> (e / num_neighbors)
                k = 0
                while k < num_neighbors and neighbors[i, k] != j:
                    k = k + 1
                if k == num_neighbors and directed[e] >= threshold:
                    indices[q] = j
                    weights[q] = directed[e]
                    q = q + 1


cdef void umap_projection_order(const float* embedding,
                                const float cos_angle,
                                const float sin_angle,
                                unsigned* bins,
                                unsigned* order,
                                unsigned* rank,
                                unsigned long long* histograms,
                                float* thread_extremes,
                                const unsigned num_cells,
                                const unsigned num_threads) noexcept nogil:
    # Sort cells by their projection onto the direction (`cos_angle`,
    # `sin_angle`), storing the sorted order in `order` and each cell's
    # position in the sorted order in `rank`. Unlike umap-learn, which sorts
    # the projections exactly, quantize them into 2048 equal-width bins and
    # counting-sort the bins, which takes a single pass. Ties are broken by
    # index; since bins are much narrower than the windows negative samples
    # are drawn from, this barely changes the windows. Each thread counts and
    # then scatters one contiguous chunk of cells, so the result does not
    # depend on the number of threads.

    cdef unsigned i, bin, thread_index, start, end
    cdef unsigned long long count, total, position
    cdef unsigned long long* histogram
    cdef float minimum, maximum, projection, scale

    # Find the range of the projections
    with parallel(num_threads=num_threads):
        thread_index = threadid()
        start = <unsigned> (<unsigned long long> thread_index * num_cells /
                            num_threads)
        end = <unsigned> (<unsigned long long> (thread_index + 1) *
                          num_cells / num_threads)
        minimum = FLT_MAX
        maximum = -FLT_MAX
        for i in range(start, end):
            projection = embedding[2 * i] * cos_angle + \
                embedding[2 * i + 1] * sin_angle
            minimum = min(minimum, projection)
            maximum = max(maximum, projection)
        thread_extremes[thread_index * 16] = minimum
        thread_extremes[thread_index * 16 + 1] = maximum
    minimum = FLT_MAX
    maximum = -FLT_MAX
    for thread_index in range(num_threads):
        minimum = min(minimum, thread_extremes[thread_index * 16])
        maximum = max(maximum, thread_extremes[thread_index * 16 + 1])
    scale = <float> 2048 / (maximum - minimum) if maximum > minimum else \
        <float> 0

    # Bin each cell's projection, and count each thread's cells in each bin
    with parallel(num_threads=num_threads):
        thread_index = threadid()
        histogram = histograms + <unsigned long long> thread_index * 2048
        for bin in range(2048):
            histogram[bin] = 0
        start = <unsigned> (<unsigned long long> thread_index * num_cells /
                            num_threads)
        end = <unsigned> (<unsigned long long> (thread_index + 1) *
                          num_cells / num_threads)
        for i in range(start, end):
            projection = embedding[2 * i] * cos_angle + \
                embedding[2 * i + 1] * sin_angle
            bin = min(<unsigned> ((projection - minimum) * scale),
                      <unsigned> 2047)
            bins[i] = bin
            histogram[bin] += 1

    # Convert the counts to each thread's first position in each bin
    total = 0
    for bin in range(2048):
        for thread_index in range(num_threads):
            position = <unsigned long long> thread_index * 2048 + bin
            count = histograms[position]
            histograms[position] = total
            total += count

    # Scatter cells to their sorted positions
    with parallel(num_threads=num_threads):
        thread_index = threadid()
        histogram = histograms + <unsigned long long> thread_index * 2048
        start = <unsigned> (<unsigned long long> thread_index * num_cells /
                            num_threads)
        end = <unsigned> (<unsigned long long> (thread_index + 1) *
                          num_cells / num_threads)
        for i in range(start, end):
            position = histogram[bins[i]]
            histogram[bins[i]] = position + 1
            order[position] = i
            rank[i] = <unsigned> position


cdef inline void umap_adaptation_repulsion(const float* embedding,
                                           const unsigned source,
                                           const unsigned negative,
                                           const float a,
                                           const float b,
                                           const float gamma,
                                           float& resultant_x,
                                           float& resultant_y) noexcept nogil:
    # Add the clipped repulsive force on `source` from `negative` to the
    # resultant, as in umap-learn's `_sample_negative_force_ratio()`

    cdef float dx = embedding[2 * source] - embedding[2 * negative], \
        dy = embedding[2 * source + 1] - embedding[2 * negative + 1], \
        dist_squared = dx * dx + dy * dy, distance, coefficient, scale
    if dist_squared > <float> 0:
        distance = sqrt(dist_squared)
        coefficient = <float> 2 * gamma * b / (
            (<float> 0.001 + dist_squared) *
            (a * pow(dist_squared, b) + <float> 1))
        scale = gamma * tanh(coefficient * distance / gamma) / distance
        resultant_x += scale * dx
        resultant_y += scale * dy


cdef inline float sorted_quantile(const float* values,
                                  const unsigned num_values,
                                  const float quantile) noexcept nogil:
    # Linearly interpolated quantile of an already-sorted array, as in
    # `np.quantile()`

    cdef float position = (num_values - 1) * quantile
    cdef unsigned lo = <unsigned> position, \
        hi = min(lo + 1, num_values - 1)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


cdef float umap_negative_force_ratio(const float* embedding,
                                     const unsigned* order,
                                     const unsigned num_cells,
                                     const unsigned window,
                                     const float a,
                                     const float b,
                                     const float gamma,
                                     unsigned long long* state,
                                     float* log_ratios) noexcept nogil:
    # Estimate how much larger the resultant repulsive force on a cell is
    # from uniformly sampled negatives than from negatives sampled from its
    # window, as in umap-learn's `_sample_negative_force_ratio()`: for 128
    # random cells, compute the resultants of the forces from 5 negatives
    # sampled each way, and return the exponentiated mean log ratio of the
    # two, after clipping the log ratios to their 10th and 90th percentiles

    cdef unsigned source_index, k, position, window_start, source, \
        num_valid = 0
    cdef float local_x, local_y, global_x, global_y, local_resultant, \
        global_resultant, lower, upper, total = 0

    for source_index in range(128):
        position = randint(num_cells, state)
        source = order[position]
        window_start = min(position - min(position, window / 2),
                           num_cells - window)
        local_x = local_y = global_x = global_y = 0
        for k in range(5):
            umap_adaptation_repulsion(
                embedding, source,
                order[window_start + randint(window, state)], a, b, gamma,
                local_x, local_y)
        for k in range(5):
            umap_adaptation_repulsion(
                embedding, source, randint(num_cells, state), a, b, gamma,
                global_x, global_y)
        local_resultant = sqrt(local_x * local_x + local_y * local_y)
        global_resultant = sqrt(global_x * global_x + global_y * global_y)
        if local_resultant > <float> 0 and \
                global_resultant > <float> 0:
            log_ratios[num_valid] = \
                log(global_resultant) - log(local_resultant)
            num_valid += 1
    if num_valid == 0:
        return <float> 1
    sort(log_ratios, log_ratios + num_valid)
    lower = sorted_quantile(log_ratios, num_valid, <float> 0.1)
    upper = sorted_quantile(log_ratios, num_valid, <float> 0.9)
    for k in range(num_valid):
        total += min(max(log_ratios[k], lower), upper)
    return exp(total / num_valid)


cdef inline float umap_learning_rate(const unsigned epoch,
                                     const unsigned num_epochs,
                                     const float initial_alpha,
                                     const bint good_initialization) \
        noexcept nogil:
    # umap-learn 0.6's Adam learning-rate schedule: a quadratic decay to 0.1
    # over a warm-up period of up to 100 epochs, starting from
    # `initial_alpha / 2` (after a good initialization) or `2 * initial_alpha`,
    # then a linear decay from 0.2 to 0.05

    cdef unsigned num_warmup_epochs
    cdef float fraction
    if good_initialization:
        num_warmup_epochs = \
            <unsigned> min(<float> 100, num_epochs / <float> 8)
    else:
        num_warmup_epochs = \
            <unsigned> min(num_epochs / <float> 2, <float> 100)
    if epoch < num_warmup_epochs:
        fraction = <float> 1 - epoch / <float> num_warmup_epochs
        return ((<float> 0.5 if good_initialization else <float> 2) *
                initial_alpha - <float> 0.1) * fraction * fraction + \
            <float> 0.1
    return <float> 0.15 * (<float> 1 - (epoch - num_warmup_epochs) /
                           <float> (num_epochs - num_warmup_epochs)) + \
        <float> 0.05


cdef inline void umap_repulsion(const float ux,
                                const float uy,
                                const float vx,
                                const float vy,
                                const float a,
                                const float b,
                                const float repulsion_constant,
                                const float clip_constant,
                                const float inverse_gamma,
                                float& gradient_x,
                                float& gradient_y) noexcept nogil:
    # Add the repulsive gradient on (`ux`, `uy`) from a negative sample at
    # (`vx`, `vy`). Like umap-learn 0.6, softly clip the gradient's norm to at
    # most `clip_constant` via `tanh()`.

    cdef float dx = ux - vx, dy = uy - vy, dist_squared = dx * dx + dy * dy, \
        inverse_distance, scale
    if dist_squared > <float> 0:
        inverse_distance = <float> 1 / sqrt(dist_squared)
        scale = repulsion_constant / ((<float> 0.001 + dist_squared) *
                                      (<float> 1 + a * pow(dist_squared, b)))
        scale = clip_constant * inverse_distance * tanh(
            scale * dist_squared * inverse_distance * inverse_gamma)
        gradient_x -= scale * dx
        gradient_y -= scale * dy


cdef inline void umap_cell_prepare(
        const float* embedding,
        const float* negative_embedding,
        const unsigned u,
        const long long* indptr,
        const unsigned* indices,
        const float* weights,
        const float max_weight,
        const float negatives_per_edge,
        const unsigned window,
        const unsigned window_start,
        unsigned long long state,
        unsigned* neighbor_buffer,
        unsigned* negative_buffer,
        unsigned& num_accepted,
        unsigned& num_negatives) noexcept nogil:
    # Draw cell `u`'s random numbers and prefetch the positions it will need
    # for `umap_cell_compute()`, so their cache misses overlap with the
    # previous cell's computation. Sample each of `u`'s edges with
    # probability proportional to its weight, storing the neighbors at the
    # other ends of sampled edges in `neighbor_buffer`. Then draw
    # `negatives_per_edge` negative samples per sampled edge (rounded
    # randomly) uniformly from positions `window_start` to
    # `window_start + window` of `negative_embedding`, storing their
    # positions in `negative_buffer`.

    cdef long long i
    cdef unsigned k, v, accepted = 0, negatives

    PREFETCH(&embedding[2 * u])
    for i in range(indptr[u], indptr[u + 1]):
        # Drop the edge with probability `1 - weight / max_weight`
        if weights[i] < max_weight * random_uniform(&state):
            continue
        v = indices[i]
        PREFETCH(&embedding[2 * v])
        neighbor_buffer[accepted] = v
        accepted += 1
    negatives = <unsigned> (accepted * negatives_per_edge +
                            random_uniform(&state))
    for k in range(negatives):
        v = window_start + randint(window, &state)
        PREFETCH(&negative_embedding[2 * v])
        negative_buffer[k] = v
    num_accepted = accepted
    num_negatives = negatives


cdef inline void umap_cell_compute(const float* embedding,
                                   const float* negative_embedding,
                                   const unsigned u,
                                   const unsigned* neighbor_buffer,
                                   const unsigned num_accepted,
                                   const unsigned* negative_buffer,
                                   const unsigned num_negatives,
                                   const float a,
                                   const float b,
                                   const float attraction_constant,
                                   const float repulsion_constant,
                                   const float clip_constant,
                                   const float inverse_gamma,
                                   float* gradient) noexcept nogil:
    # Calculate the gradient on cell `u`'s position from the neighbors and
    # negative samples drawn by `umap_cell_prepare()`, storing it in
    # `gradient`: attract `u` towards each neighbor and repel it from each
    # negative sample. Like umap-learn, only `u`'s position is updated, not
    # its neighbors' or negative samples'.

    cdef unsigned k, v
    cdef float ux = embedding[2 * u], uy = embedding[2 * u + 1], dx, dy, \
        dist_squared, dist_b_minus_1, scale, gx = 0, gy = 0, \
        b_minus_1 = b - <float> 1

    # Attractive gradients
    for k in range(num_accepted):
        v = neighbor_buffer[k]
        dx = ux - embedding[2 * v]
        dy = uy - embedding[2 * v + 1]
        dist_squared = dx * dx + dy * dy
        if dist_squared > <float> 0:
            dist_b_minus_1 = pow(dist_squared, b_minus_1)
            scale = attraction_constant * dist_b_minus_1 / \
                (<float> 1 + a * dist_b_minus_1 * dist_squared)
            gx += scale * dx
            gy += scale * dy

    # Repulsive gradients
    for k in range(num_negatives):
        v = negative_buffer[k]
        umap_repulsion(ux, uy, negative_embedding[2 * v],
                       negative_embedding[2 * v + 1], a, b,
                       repulsion_constant, clip_constant, inverse_gamma, gx,
                       gy)
    gradient[0] = gx
    gradient[1] = gy


cdef void umap_layout(float[:, ::1] embedding,
                      const long long[::1] indptr,
                      const unsigned[::1] indices,
                      const float[::1] weights,
                      const unsigned num_epochs,
                      const float a,
                      const float b,
                      const float gamma,
                      const float initial_alpha,
                      const float negative_sample_rate,
                      unsigned window,
                      const bint good_initialization,
                      const unsigned long long seed,
                      unsigned num_threads):
    # Optimize `embedding` with umap-learn 0.6's Adam optimizer. Each epoch,
    # every cell calculates the gradient on its position from the current
    # positions, then every cell takes an Adam step. Cell `u`'s random
    # numbers in epoch `epoch` come from the stream
    # `srand(seed + epoch * num_cells + u)`, so results do not depend on the
    # number of threads.
    #
    # When `window` is less than the number of cells (and there are more than
    # 1024 cells), negative samples are "hard negatives" drawn from each
    # cell's window of `window` nearby cells in a random 1D projection of the
    # embedding, recomputed each epoch. Their repulsion is scaled down by a
    # factor that starts at `(window / num_cells) ** 0.75` and adapts every 10
    # epochs during the first half of optimization, based on an estimate of
    # how much weaker the repulsion from window negatives is than from
    # uniformly sampled negatives.

    cdef unsigned epoch, u, thread_index, window_start, buffer_size, \
        num_accepted, num_negatives, next_num_accepted, next_num_negatives, \
        num_cells = embedding.shape[0], max_degree = 0
    cdef long long i
    cdef unsigned long long state
    cdef bint use_window
    cdef float alpha, epsilon, max_weight, angle, attraction_constant, \
        repulsion_constant, clip_constant, inverse_gamma, \
        negatives_per_edge, negative_scale, scale_prior, scale_anchor = -1, \
        force_ratio, raw_scale, log_change, beta1 = <float> 0.9, \
        beta2 = <float> 0.99
    cdef float* positions = &embedding[0, 0]
    cdef float* gradients
    cdef float* window_embedding = NULL
    cdef float* negative_embedding = &embedding[0, 0]
    cdef unsigned* order = NULL
    cdef unsigned* rank = NULL
    cdef unsigned* neighbor_buffer
    cdef unsigned* negative_buffer
    cdef unsigned* next_neighbor_buffer
    cdef unsigned* next_negative_buffer
    cdef unsigned* swap
    cdef float[:, ::1] gradients_view, momentum, velocity
    cdef uninitialized_vector[float] gradients_buffer, momentum_buffer, \
        velocity_buffer, window_embedding_buffer, thread_floats, log_ratios
    cdef uninitialized_vector[unsigned] thread_offsets, thread_unsigneds, \
        bins_buffer, order_buffer, rank_buffer, cell_buffers
    cdef uninitialized_vector[unsigned long long] histograms

    # Use at most one thread per 1024 cells, since each thread's work per
    # epoch is otherwise too small to be worth parallelizing. Split cells
    # into contiguous ranges with about the same number of edges per thread.
    num_threads = max(1, min(num_threads, num_cells / 1024))
    thread_offsets.resize(num_threads + 1)
    get_thread_offsets(indptr, thread_offsets.data(), num_threads)

    # Get the maximum edge weight and degree
    thread_floats.resize(num_threads * 16)
    thread_unsigneds.resize(num_threads * 16)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        max_weight = 0
        max_degree = 0
        for u in range(thread_offsets[thread_index],
                       thread_offsets[thread_index + 1]):
            max_degree = max(max_degree,
                             <unsigned> (indptr[u + 1] - indptr[u]))
            for i in range(indptr[u], indptr[u + 1]):
                max_weight = max(max_weight, weights[i])
        thread_floats[thread_index * 16] = max_weight
        thread_unsigneds[thread_index * 16] = max_degree
    max_weight = 0
    max_degree = 0
    for thread_index in range(num_threads):
        max_weight = max(max_weight, thread_floats[thread_index * 16])
        max_degree = max(max_degree, thread_unsigneds[thread_index * 16])

    # Allocate the gradients and Adam's momentum and velocity, initializing
    # the latter two in parallel (with the same schedule as
    # `update_embedding_adam_parallel()`) for NUMA first-touch reasons
    gradients_buffer.resize(2 * <unsigned long long> num_cells)
    momentum_buffer.resize(2 * <unsigned long long> num_cells)
    velocity_buffer.resize(2 * <unsigned long long> num_cells)
    advise_huge_pages(gradients_buffer.data(),
                      2 * <unsigned long long> num_cells * sizeof(float))
    advise_huge_pages(momentum_buffer.data(),
                      2 * <unsigned long long> num_cells * sizeof(float))
    advise_huge_pages(velocity_buffer.data(),
                      2 * <unsigned long long> num_cells * sizeof(float))
    gradients = gradients_buffer.data()
    gradients_view = <float[:num_cells, :2]> gradients
    momentum = <float[:num_cells, :2]> momentum_buffer.data()
    velocity = <float[:num_cells, :2]> velocity_buffer.data()
    with nogil:
        for u in prange(num_cells, num_threads=num_threads):
            momentum[u, 0] = 0
            momentum[u, 1] = 0
            velocity[u, 0] = 0
            velocity[u, 1] = 0

    # Set up hard-negative windows, if used
    window = min(num_cells, max(200, window))
    use_window = num_cells > 1024 and window < num_cells
    if not use_window:
        window = num_cells
    negative_scale = pow(window / <float> num_cells, <float> 0.75) \
        if use_window else <float> 1
    scale_prior = negative_scale
    if use_window:
        bins_buffer.resize(num_cells)
        order_buffer.resize(num_cells)
        rank_buffer.resize(num_cells)
        window_embedding_buffer.resize(2 * <unsigned long long> num_cells)
        advise_huge_pages(bins_buffer.data(), num_cells * sizeof(unsigned))
        advise_huge_pages(order_buffer.data(), num_cells * sizeof(unsigned))
        advise_huge_pages(rank_buffer.data(), num_cells * sizeof(unsigned))
        advise_huge_pages(window_embedding_buffer.data(),
                          2 * <unsigned long long> num_cells * sizeof(float))
        order = order_buffer.data()
        rank = rank_buffer.data()
        window_embedding = window_embedding_buffer.data()
        negative_embedding = window_embedding
        histograms.resize(<unsigned long long> num_threads * 2048)
        log_ratios.resize(128)

    # Like umap-learn, use half-strength attraction for graphs with at most
    # 1024 cells, and draw `negative_sample_rate / 1.5` negative samples per
    # sampled edge
    attraction_constant = <float> 2 * a * b * \
        (<float> 0.5 if num_cells <= 1024 else <float> 1)
    clip_constant = min(gamma, <float> 16)
    inverse_gamma = <float> 1 / gamma
    negatives_per_edge = negative_sample_rate / <float> 1.5

    # Each thread needs two sets of buffers for the neighbors and negative
    # samples of the current and next cells
    buffer_size = max_degree + 16 + \
        <unsigned> (max_degree * negatives_per_edge) + 16
    cell_buffers.resize(num_threads * 2 * buffer_size)
    state = srand(seed - 1)

    for epoch in range(num_epochs):
        # Get the learning rate, and scale Adam's epsilon to match
        # umap-learn's placement of it after bias correction
        alpha = umap_learning_rate(epoch, num_epochs, initial_alpha,
                                   good_initialization)
        epsilon = <float> 1e-4 * \
            sqrt(<float> 1 - pow(beta2, <float> (epoch + 1)))

        # Sort cells by their projection onto a random direction, gather the
        # embedding into this order so each cell's window is contiguous, and
        # adapt the scale of the repulsion from window negatives
        if use_window:
            angle = <float> 2 * <float> M_PI * random_uniform(&state)
            with nogil:
                umap_projection_order(
                    positions, cos(angle), sin(angle), bins_buffer.data(),
                    order, rank, histograms.data(), thread_floats.data(),
                    num_cells, num_threads)
                for u in prange(num_cells, num_threads=num_threads):
                    window_embedding[2 * u] = positions[2 * order[u]]
                    window_embedding[2 * u + 1] = positions[2 * order[u] + 1]
            if 10 <= epoch <= num_epochs / 2 and epoch % 10 == 0:
                force_ratio = umap_negative_force_ratio(
                    positions, order, num_cells, window, a, b, gamma, &state,
                    log_ratios.data())
                if scale_anchor < <float> 0:
                    scale_anchor = negative_scale / force_ratio
                else:
                    raw_scale = min(max(scale_anchor * force_ratio,
                                        <float> 0.5 * scale_prior),
                                    <float> 2 * scale_prior)
                    log_change = min(max(log(raw_scale / negative_scale),
                                         log(<float> 0.9)), log(<float> 1.1))
                    negative_scale *= exp(<float> 0.1 * log_change)
        repulsion_constant = <float> 2 * gamma * b * negative_scale

        # Calculate the gradient on each cell's position. Draw each cell's
        # random numbers and prefetch its neighbors' and negative samples'
        # positions one cell ahead, so their cache misses overlap with the
        # previous cell's computation.
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            neighbor_buffer = \
                cell_buffers.data() + thread_index * 2 * buffer_size
            negative_buffer = neighbor_buffer + max_degree + 16
            next_neighbor_buffer = neighbor_buffer + buffer_size
            next_negative_buffer = negative_buffer + buffer_size
            num_accepted = 0
            num_negatives = 0
            next_num_accepted = 0
            next_num_negatives = 0
            for u in range(thread_offsets[thread_index],
                           thread_offsets[thread_index + 1]):
                if u == thread_offsets[thread_index]:
                    window_start = 0 if not use_window else min(
                        rank[u] - min(rank[u], window / 2),
                        num_cells - window)
                    umap_cell_prepare(
                        positions, negative_embedding, u, &indptr[0],
                        &indices[0], &weights[0], max_weight,
                        negatives_per_edge, window, window_start,
                        srand(seed + <unsigned long long> epoch * num_cells +
                              u), neighbor_buffer, negative_buffer,
                        num_accepted, num_negatives)
                if u + 1 < thread_offsets[thread_index + 1]:
                    window_start = 0 if not use_window else min(
                        rank[u + 1] - min(rank[u + 1], window / 2),
                        num_cells - window)
                    umap_cell_prepare(
                        positions, negative_embedding, u + 1, &indptr[0],
                        &indices[0], &weights[0], max_weight,
                        negatives_per_edge, window, window_start,
                        srand(seed + <unsigned long long> epoch * num_cells +
                              u + 1), next_neighbor_buffer,
                        next_negative_buffer, next_num_accepted,
                        next_num_negatives)
                umap_cell_compute(
                    positions, negative_embedding, u, neighbor_buffer,
                    num_accepted, negative_buffer, num_negatives, a, b,
                    attraction_constant, repulsion_constant, clip_constant,
                    inverse_gamma, &gradients[2 * u])
                swap = neighbor_buffer
                neighbor_buffer = next_neighbor_buffer
                next_neighbor_buffer = swap
                swap = negative_buffer
                negative_buffer = next_negative_buffer
                next_negative_buffer = swap
                num_accepted = next_num_accepted
                num_negatives = next_num_negatives

        # Take an Adam step
        with nogil:
            update_embedding_adam_parallel(
                embedding, gradients_view, momentum, velocity, num_cells,
                beta1, beta2, alpha, epsilon, epoch, num_threads)

        # Check for KeyboardInterrupts
        PyErr_CheckSignals()


cdef void umap_center(float* positions,
                      const unsigned num_rows,
                      const unsigned num_threads) noexcept nogil:
    # Subtract the mean of each column of the `num_rows` × 2 array
    # `positions`. Sum in fixed-size chunks so the result does not depend on
    # the number of threads.

    cdef unsigned i, chunk, num_chunks = (num_rows + 65535) / 65536
    cdef float mean_x = 0, mean_y = 0, sum_x, sum_y
    cdef vector[float] chunk_sums
    chunk_sums.resize(2 * num_chunks)
    for chunk in prange(num_chunks, num_threads=num_threads):
        sum_x = 0
        sum_y = 0
        for i in range(chunk * 65536, min(chunk * 65536 + 65536, num_rows)):
            sum_x = sum_x + positions[2 * i]
            sum_y = sum_y + positions[2 * i + 1]
        chunk_sums[2 * chunk] = sum_x
        chunk_sums[2 * chunk + 1] = sum_y
    for chunk in range(num_chunks):
        mean_x += chunk_sums[2 * chunk]
        mean_y += chunk_sums[2 * chunk + 1]
    mean_x /= num_rows
    mean_y /= num_rows
    for i in prange(num_rows, num_threads=num_threads):
        positions[2 * i] -= mean_x
        positions[2 * i + 1] -= mean_y


cdef float umap_quantile(float* values,
                         const unsigned num_values,
                         const unsigned percentile) noexcept nogil:
    # The linearly interpolated `percentile`th percentile of `values`, as in
    # `np.quantile()`. Partially sorts `values` in place.

    cdef unsigned long long numerator = \
        <unsigned long long> (num_values - 1) * percentile
    cdef unsigned lo = <unsigned> (numerator / 100)
    cdef float result, fraction = (numerator % 100) / <float> 100
    nth_element(values, values + lo, values + num_values)
    result = values[lo]
    if fraction > <float> 0:
        result += fraction * (min_element(values + lo + 1,
                                          values + num_values)[0] - result)
    return result


cdef void umap_robust_spreads(const float* positions,
                              const unsigned num_rows,
                              float& spread_x,
                              float& spread_y):
    # The difference between the 95th and 5th percentiles of each column of
    # the `num_rows` × 2 array `positions`, or 1 if the difference is 0

    cdef unsigned i, dimension
    cdef float spread
    cdef uninitialized_vector[float] values
    values.resize(num_rows)
    for dimension in range(2):
        for i in range(num_rows):
            values[i] = positions[2 * i + dimension]
        spread = umap_quantile(values.data(), num_rows, 95) - \
            umap_quantile(values.data(), num_rows, 5)
        if spread == <float> 0:
            spread = 1
        if dimension == 0:
            spread_x = spread
        else:
            spread_y = spread


cdef void umap_align(float* positions,
                     const float* anchor,
                     const unsigned num_rows,
                     const unsigned num_threads) noexcept nogil:
    # Rotate the centered `num_rows` × 2 array `positions` to best match
    # `anchor`, via the orthogonal Procrustes rotation, which in 2D has the
    # closed form `atan2(c10 - c01, c00 + c11)` in terms of the cross-
    # covariance matrix `c = positions.T @ (anchor - anchor.mean(axis=0))`.
    # Sum in fixed-size chunks so the result does not depend on the number of
    # threads.

    cdef unsigned i, chunk, num_chunks = (num_rows + 65535) / 65536
    cdef float mean_x = 0, mean_y = 0, c00 = 0, c01 = 0, c10 = 0, c11 = 0, \
        sum_x, sum_y, sum_00, sum_01, sum_10, sum_11, angle, cos_angle, \
        sin_angle, x, y
    cdef vector[float] chunk_sums
    chunk_sums.resize(4 * num_chunks)
    for chunk in prange(num_chunks, num_threads=num_threads):
        sum_x = 0
        sum_y = 0
        for i in range(chunk * 65536, min(chunk * 65536 + 65536, num_rows)):
            sum_x = sum_x + anchor[2 * i]
            sum_y = sum_y + anchor[2 * i + 1]
        chunk_sums[4 * chunk] = sum_x
        chunk_sums[4 * chunk + 1] = sum_y
    for chunk in range(num_chunks):
        mean_x += chunk_sums[4 * chunk]
        mean_y += chunk_sums[4 * chunk + 1]
    mean_x /= num_rows
    mean_y /= num_rows
    for chunk in prange(num_chunks, num_threads=num_threads):
        sum_00 = 0
        sum_01 = 0
        sum_10 = 0
        sum_11 = 0
        for i in range(chunk * 65536, min(chunk * 65536 + 65536, num_rows)):
            sum_00 = sum_00 + positions[2 * i] * (anchor[2 * i] - mean_x)
            sum_01 = sum_01 + positions[2 * i] * (anchor[2 * i + 1] - mean_y)
            sum_10 = sum_10 + positions[2 * i + 1] * (anchor[2 * i] - mean_x)
            sum_11 = sum_11 + \
                positions[2 * i + 1] * (anchor[2 * i + 1] - mean_y)
        chunk_sums[4 * chunk] = sum_00
        chunk_sums[4 * chunk + 1] = sum_01
        chunk_sums[4 * chunk + 2] = sum_10
        chunk_sums[4 * chunk + 3] = sum_11
    for chunk in range(num_chunks):
        c00 += chunk_sums[4 * chunk]
        c01 += chunk_sums[4 * chunk + 1]
        c10 += chunk_sums[4 * chunk + 2]
        c11 += chunk_sums[4 * chunk + 3]
    angle = atan2(c10 - c01, c00 + c11)
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    for i in prange(num_rows, num_threads=num_threads):
        x = positions[2 * i]
        y = positions[2 * i + 1]
        positions[2 * i] = x * cos_angle + y * sin_angle
        positions[2 * i + 1] = y * cos_angle - x * sin_angle


cdef void umap_init_level(const long long[::1] indptr,
                          const unsigned[::1] indices,
                          const float[::1] label_weights,
                          const float[::1] summed_weights,
                          const float[:, ::1] anchor,
                          float[:, ::1] result,
                          const float a,
                          const float b,
                          const unsigned depth,
                          unsigned num_parts,
                          const unsigned num_epochs,
                          const unsigned long long seed,
                          const unsigned num_threads):
    # One level of umap-learn 0.6's recursive initialization
    # (`label_propagation_init()`), writing the initial positions of this
    # level's graph's nodes to `result`. Graphs with at most 1024 nodes are
    # placed directly from `anchor`. Otherwise:
    # 1. Partition the graph into `num_parts` parts by label propagation from
    #    its `num_parts` highest-degree nodes, using `label_weights` as edge
    #    weights.
    # 2. Build the coarse graph between parts. Its `summed_weights` are the
    #    total `summed_weights` between parts' members (which, below the top
    #    level, are themselves summed from the level above, so they always
    #    total the original graph's weights). Its `label_weights` are the
    #    fuzzy union of the two directed "transition" weights between each
    #    pair of parts, i.e. the fraction of each part's summed weight that
    #    goes to the other part.
    # 3. Recursively initialize the coarse graph with about a quarter as many
    #    parts, then optimize it with a short UMAP run.
    # 4. Place each node halfway between its part's position and the
    #    weighted average of its neighbors' parts' positions, then center and
    #    rotate to best match `anchor`.

    cdef unsigned i, j, k, p, q, label, best_label, num_votes, \
        num_unlabeled, previous_unlabeled, iteration, num_coarse_edges, \
        thread_index, num_iterations, window, level_threads, \
        num_nodes = anchor.shape[0], max_degree = 0
    cdef long long e, total
    cdef unsigned long long m, position, count, \
        level_seed = seed + (<unsigned long long> depth << 40)
    cdef unsigned long long state
    cdef bint found, use_share = depth > 1
    cdef float w, total_weight, max_vote, sum_x, sum_y, spread_x = 1, \
        spread_y = 1, scale, d_pq, d_qp
    cdef unsigned* labels
    cdef unsigned* new_labels
    cdef unsigned* vote_labels
    cdef float* vote_weights
    cdef unsigned long long* keys
    cdef vector[unsigned] stack
    cdef uninitialized_vector[unsigned long long] keys_buffer, part_indptr, \
        cursors
    cdef uninitialized_vector[long long] part_work, coarse_indptr, \
        thread_totals
    cdef long long[::1] part_work_view
    cdef uninitialized_vector[unsigned] labels_buffer, new_labels_buffer, \
        vote_labels_buffer, thread_offsets, thread_unsigneds, members, \
        coarse_indices
    cdef uninitialized_vector[float] vote_weights_buffer, coarse_summed, \
        coarse_label_weights, out_weights, coarse_anchor, coarse_layout
    cdef vector[vector[unsigned long long]] thread_keys
    cdef vector[vector[float]] thread_weights, thread_out_weights
    cdef vector[vector[unsigned]] thread_out_indices

    # Base case: place nodes at their (centered) anchor positions, scaled to
    # a robust spread that grows with the number of nodes and the depth
    if num_nodes <= 1024:
        for i in range(num_nodes):
            result[i, 0] = anchor[i, 0]
            result[i, 1] = anchor[i, 1]
        umap_center(&result[0, 0], num_nodes, 1)
        umap_robust_spreads(&result[0, 0], num_nodes, spread_x, spread_y)
        scale = log10(<float> num_nodes) * <float> 3 * \
            log2(<float> (depth + 1))
        for i in range(num_nodes):
            result[i, 0] *= scale / spread_x
            result[i, 1] *= scale / spread_y
        return

    num_parts = max(2, min(num_parts, num_nodes / 2))
    level_threads = max(1, min(num_threads, num_nodes / 1024))
    thread_offsets.resize(level_threads + 1)
    get_thread_offsets(indptr, thread_offsets.data(), level_threads)

    # Select the `num_parts` highest-degree nodes as hubs, breaking ties by
    # index. Encode each node's degree (whose bits sort like unsigned
    # integers, since it is non-negative) in the upper 32 bits of a key and
    # its index in the lower 32, and partially sort the keys.
    keys_buffer.resize(num_nodes)
    keys = keys_buffer.data()
    thread_unsigneds.resize(level_threads * 16)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        max_degree = 0
        for i in range(thread_offsets[thread_index],
                       thread_offsets[thread_index + 1]):
            w = 0
            for e in range(indptr[i], indptr[i + 1]):
                w = w + label_weights[e]
            # Assign `label` before `memcpy()` so Cython makes it
            # thread-private
            label = 0
            memcpy(&label, &w, sizeof(unsigned))
            keys[i] = (<unsigned long long> label << 32) | i
            max_degree = \
                max(max_degree, <unsigned> (indptr[i + 1] - indptr[i]))
        thread_unsigneds[thread_index * 16] = max_degree
    max_degree = 0
    for thread_index in range(level_threads):
        max_degree = max(max_degree, thread_unsigneds[thread_index * 16])
    nth_element(keys, keys + (num_nodes - num_parts), keys + num_nodes)

    # Give each hub its own label (in ascending order of index), and leave
    # other nodes unlabeled (`UINT_MAX`)
    labels_buffer.resize(num_nodes)
    labels = labels_buffer.data()
    advise_huge_pages(labels, num_nodes * sizeof(unsigned))
    for i in range(num_nodes):
        labels[i] = UINT_MAX
    for e in range(num_nodes - num_parts, num_nodes):
        labels[<unsigned> keys[e]] = 0
    label = 0
    for i in range(num_nodes):
        if labels[i] == 0:
            labels[i] = label
            label += 1
    keys_buffer.clear()
    keys_buffer.shrink_to_fit()

    # Propagate labels for up to 100 iterations. Each unlabeled node adopts
    # the label with the largest total edge weight among its neighbors (the
    # last one among ties, in order of first appearance), if that total is at
    # least 1 (at the top level) or 10% of the node's total edge weight
    # (below the top level, where weights are normalized). Stop early if no
    # nodes became labeled over the last 5 iterations.
    new_labels_buffer.resize(num_nodes)
    new_labels = new_labels_buffer.data()
    advise_huge_pages(new_labels, num_nodes * sizeof(unsigned))
    vote_labels_buffer.resize(level_threads * (max_degree + 16))
    vote_weights_buffer.resize(level_threads * (max_degree + 16))
    previous_unlabeled = num_nodes - num_parts
    for iteration in range(100):
        with nogil, parallel(num_threads=level_threads):
            thread_index = threadid()
            vote_labels = \
                vote_labels_buffer.data() + thread_index * (max_degree + 16)
            vote_weights = \
                vote_weights_buffer.data() + thread_index * (max_degree + 16)
            num_unlabeled = 0
            for i in range(thread_offsets[thread_index],
                           thread_offsets[thread_index + 1]):
                if labels[i] != UINT_MAX:
                    new_labels[i] = labels[i]
                    continue
                total_weight = 0
                num_votes = 0
                for e in range(indptr[i], indptr[i + 1]):
                    w = label_weights[e]
                    total_weight = total_weight + w
                    label = labels[indices[e]]
                    k = 0
                    while k < num_votes and vote_labels[k] != label:
                        k = k + 1
                    if k == num_votes:
                        vote_labels[k] = label
                        vote_weights[k] = w
                        num_votes = num_votes + 1
                    else:
                        vote_weights[k] += w
                max_vote = \
                    <float> 0.1 * total_weight if use_share else <float> 1
                best_label = UINT_MAX
                for k in range(num_votes):
                    if vote_labels[k] != UINT_MAX and \
                            vote_weights[k] >= max_vote:
                        max_vote = vote_weights[k]
                        best_label = vote_labels[k]
                new_labels[i] = best_label
                if best_label == UINT_MAX:
                    num_unlabeled = num_unlabeled + 1
            thread_unsigneds[thread_index * 16] = num_unlabeled
        labels_buffer.swap(new_labels_buffer)
        labels = labels_buffer.data()
        new_labels = new_labels_buffer.data()
        if iteration % 5 == 0:
            num_unlabeled = 0
            for thread_index in range(level_threads):
                num_unlabeled += thread_unsigneds[thread_index * 16]
            if num_unlabeled == 0 or num_unlabeled == previous_unlabeled:
                break
            previous_unlabeled = num_unlabeled
    new_labels_buffer.clear()
    new_labels_buffer.shrink_to_fit()

    # Give each remaining unlabeled node the label of the first labeled node
    # found by a depth-first search of up to 100 steps, or a random label if
    # none is found
    state = srand(level_seed - 2)
    for i in range(num_nodes):
        if labels[i] != UINT_MAX:
            continue
        stack.clear()
        stack.push_back(i)
        found = False
        num_iterations = 0
        while not found and num_iterations < 100 and stack.size() > 0:
            num_iterations += 1
            j = stack.back()
            stack.pop_back()
            for e in range(indptr[j], indptr[j + 1]):
                if labels[indices[e]] != UINT_MAX:
                    labels[i] = labels[indices[e]]
                    found = True
                    break
                stack.push_back(indices[e])
        if not found:
            labels[i] = randint(num_parts, &state)

    # Group nodes by part, in ascending order of index within each part, and
    # count each part's total degree to balance work across threads
    part_indptr.resize(num_parts + 1)
    members.resize(num_nodes)
    part_work.resize(num_parts + 1)
    for p in range(num_parts + 1):
        part_indptr[p] = 0
        part_work[p] = 0
    for i in range(num_nodes):
        part_indptr[labels[i] + 1] += 1
        part_work[labels[i] + 1] += indptr[i + 1] - indptr[i]
    for p in range(num_parts):
        part_indptr[p + 1] += part_indptr[p]
        part_work[p + 1] += part_work[p]
    cursors.resize(num_parts)
    for p in range(num_parts):
        cursors[p] = part_indptr[p]
    for i in range(num_nodes):
        p = labels[i]
        members[cursors[p]] = i
        cursors[p] += 1
    cursors.clear()
    cursors.shrink_to_fit()

    # Build the coarse graph, omitting edges within parts: for each part,
    # sort the edges from its members by the part they point to (and then by
    # order of appearance), and sum the weights of edges pointing to the
    # same part. Each thread appends its parts' edges to its own buffers,
    # which are concatenated at the end.
    thread_offsets.resize(level_threads + 1)
    part_work_view = <long long[:num_parts + 1]> part_work.data()
    get_thread_offsets(part_work_view, thread_offsets.data(), level_threads)
    coarse_indptr.resize(num_parts + 1)
    out_weights.resize(num_parts)
    thread_keys.resize(level_threads)
    thread_weights.resize(level_threads)
    thread_out_indices.resize(level_threads)
    thread_out_weights.resize(level_threads)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        for p in range(thread_offsets[thread_index],
                       thread_offsets[thread_index + 1]):
            thread_keys[thread_index].clear()
            thread_weights[thread_index].clear()
            for m in range(part_indptr[p], part_indptr[p + 1]):
                i = members[m]
                for e in range(indptr[i], indptr[i + 1]):
                    q = labels[indices[e]]
                    if q != p:
                        thread_keys[thread_index].push_back(
                            (<unsigned long long> q << 32) |
                            thread_weights[thread_index].size())
                        thread_weights[thread_index].push_back(
                            summed_weights[e])
            sort(thread_keys[thread_index].begin(),
                 thread_keys[thread_index].end())
            count = 0
            total_weight = 0
            label = UINT_MAX
            for position in range(thread_keys[thread_index].size()):
                q = <unsigned> (thread_keys[thread_index][position] >> 32)
                w = thread_weights[thread_index][
                    <unsigned> thread_keys[thread_index][position]]
                total_weight = total_weight + w
                if q == label:
                    thread_out_weights[thread_index][
                        thread_out_weights[thread_index].size() - 1] += w
                else:
                    thread_out_indices[thread_index].push_back(q)
                    thread_out_weights[thread_index].push_back(w)
                    count = count + 1
                    label = q
            coarse_indptr[p + 1] = count
            out_weights[p] = total_weight

    # Cumsum the coarse graph's row lengths and concatenate the threads'
    # buffers into the coarse graph
    thread_totals.resize(level_threads + 1)
    total = 0
    for thread_index in range(level_threads):
        thread_totals[thread_index] = total
        total += thread_out_indices[thread_index].size()
    num_coarse_edges = max(1, <unsigned> total)
    coarse_indices.resize(num_coarse_edges)
    coarse_summed.resize(num_coarse_edges)
    coarse_label_weights.resize(num_coarse_edges)
    coarse_indptr[0] = 0
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        total = thread_totals[thread_index]
        for p in range(thread_offsets[thread_index],
                       thread_offsets[thread_index + 1]):
            total = total + coarse_indptr[p + 1]
            coarse_indptr[p + 1] = total
        for position in range(thread_out_indices[thread_index].size()):
            e = thread_totals[thread_index] + <long long> position
            coarse_indices[e] = thread_out_indices[thread_index][position]
            coarse_summed[e] = thread_out_weights[thread_index][position]
    thread_keys.clear()
    thread_weights.clear()
    thread_out_indices.clear()
    thread_out_weights.clear()

    # Calculate the coarse graph's label weights: the fuzzy union of the
    # transition weights `d_pq = W_pq / out_p` and `d_qp = W_pq / out_q`. Also
    # calculate the coarse anchor: the mean of each part's members' anchors.
    coarse_anchor.resize(2 * num_parts)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        for p in range(thread_offsets[thread_index],
                       thread_offsets[thread_index + 1]):
            for e in range(coarse_indptr[p], coarse_indptr[p + 1]):
                d_pq = coarse_summed[e] / out_weights[p]
                d_qp = coarse_summed[e] / out_weights[coarse_indices[e]]
                coarse_label_weights[e] = \
                    min(max(d_pq + d_qp - d_pq * d_qp, <float> 0), <float> 1)
            sum_x = 0
            sum_y = 0
            for m in range(part_indptr[p], part_indptr[p + 1]):
                sum_x = sum_x + anchor[members[m], 0]
                sum_y = sum_y + anchor[members[m], 1]
            coarse_anchor[2 * p] = \
                sum_x / <float> (part_indptr[p + 1] - part_indptr[p])
            coarse_anchor[2 * p + 1] = \
                sum_y / <float> (part_indptr[p + 1] - part_indptr[p])

    # Recursively initialize the coarse graph, with exponents of the curve
    # parameters closer to 1, a quarter as many parts, and √2 times as many
    # epochs
    coarse_layout.resize(2 * num_parts)
    umap_init_level(
        <long long[:num_parts + 1]> coarse_indptr.data(),
        <unsigned[:num_coarse_edges]> coarse_indices.data(),
        <float[:num_coarse_edges]> coarse_label_weights.data(),
        <float[:num_coarse_edges]> coarse_summed.data(),
        <float[:num_parts, :2]> coarse_anchor.data(),
        <float[:num_parts, :2]> coarse_layout.data(),
        pow(a, <float> 0.25), pow(b, <float> 0.25), depth + 1, num_parts / 4,
        <unsigned> (num_epochs * sqrt(<float> 2)), seed, num_threads)

    # Optimize the coarse graph's layout with a repulsion strength of 4, one
    # negative sample per sampled edge, and windows of half the coarse graph
    # (rounded like `np.round()`)
    window = num_parts / 2
    if num_parts % 2 == 1 and window % 2 == 1:
        window += 1
    umap_layout(<float[:num_parts, :2]> coarse_layout.data(),
                <long long[:num_parts + 1]> coarse_indptr.data(),
                <unsigned[:num_coarse_edges]> coarse_indices.data(),
                <float[:num_coarse_edges]> coarse_label_weights.data(),
                num_epochs, a, b, 4, <float> 0.5, 1, max(1, window),
                num_parts / 4 > 1024, level_seed, num_threads)

    # Place each node halfway between its part's position and the weighted
    # average of its neighbors' parts' positions, then center and align to
    # the anchor
    thread_offsets.resize(level_threads + 1)
    get_thread_offsets(indptr, thread_offsets.data(), level_threads)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        for i in range(thread_offsets[thread_index],
                       thread_offsets[thread_index + 1]):
            sum_x = 0
            sum_y = 0
            total_weight = 0
            for e in range(indptr[i], indptr[i + 1]):
                w = label_weights[e]
                total_weight = total_weight + w
                p = labels[indices[e]]
                sum_x = sum_x + w * coarse_layout[2 * p]
                sum_y = sum_y + w * coarse_layout[2 * p + 1]
            if total_weight > <float> 0:
                sum_x = sum_x / total_weight
                sum_y = sum_y / total_weight
            p = labels[i]
            result[i, 0] = <float> 0.5 * (sum_x + coarse_layout[2 * p])
            result[i, 1] = <float> 0.5 * (sum_y + coarse_layout[2 * p + 1])
    with nogil:
        umap_center(&result[0, 0], num_nodes, level_threads)
        umap_align(&result[0, 0], &anchor[0, 0], num_nodes, level_threads)


cdef void umap_embed(float[:, ::1] embedding,
                     const float[:, ::1] PCs,
                     const long long[::1] indptr,
                     const unsigned[::1] indices,
                     const float[::1] weights,
                     const unsigned num_iterations,
                     const float a,
                     const float b,
                     const float alpha,
                     const float gamma,
                     const unsigned negative_sample_rate,
                     const unsigned negative_window,
                     const unsigned long long seed,
                     const unsigned num_threads):
    # Initialize the embedding recursively, rescale it, then optimize it

    cdef unsigned i, num_cells = embedding.shape[0]
    cdef float min_x = FLT_MAX, max_x = -FLT_MAX, min_y = FLT_MAX, \
        max_y = -FLT_MAX, scale_x, scale_y, spread_x = 1, spread_y = 1, scale
    cdef uninitialized_vector[float] anchor

    # Use the first two PCs, min-max scaled to [0, 10], as the anchor that
    # orients each level of the recursive initialization
    anchor.resize(2 * <unsigned long long> num_cells)
    for i in range(num_cells):
        min_x = min(min_x, PCs[i, 0])
        max_x = max(max_x, PCs[i, 0])
        min_y = min(min_y, PCs[i, 1])
        max_y = max(max_y, PCs[i, 1])
    scale_x = <float> 10 / (max_x - min_x) if max_x > min_x else <float> 10
    scale_y = <float> 10 / (max_y - min_y) if max_y > min_y else <float> 10
    for i in range(num_cells):
        anchor[2 * i] = (PCs[i, 0] - min_x) * scale_x
        anchor[2 * i + 1] = (PCs[i, 1] - min_y) * scale_y

    # Initialize recursively, starting with a quarter as many parts as cells
    # (but at least 1024), 50 epochs, and the curve parameters' fourth roots
    umap_init_level(indptr, indices, weights, weights,
                    <float[:num_cells, :2]> anchor.data(), embedding,
                    pow(a, <float> 0.25), pow(b, <float> 0.25), 1,
                    max(1024, num_cells / 4), 50, seed, num_threads)

    # Center the initialization and scale it so the geometric mean of its two
    # robust spreads (the differences between their 95th and 5th
    # percentiles) is `3 * log10(num_cells) * log2(gamma + 1)`
    with nogil:
        umap_center(&embedding[0, 0], num_cells, num_threads)
    umap_robust_spreads(&embedding[0, 0], num_cells, spread_x, spread_y)
    scale = <float> 3 * log10(<float> num_cells) * \
        log2(gamma + <float> 1) / \
        sqrt(spread_x * spread_y)
    with nogil:
        for i in prange(num_cells, num_threads=num_threads):
            embedding[i, 0] *= scale
            embedding[i, 1] *= scale

    # Optimize
    umap_layout(embedding, indptr, indices, weights, num_iterations, a, b,
                gamma, alpha, negative_sample_rate, negative_window, True,
                seed, num_threads)


def umap(float[:, ::1] embedding,
         const float[:, ::1] PCs,
         const unsigned[:, ::1] neighbors,
         const float[:, ::1] distances,
         const unsigned num_iterations,
         const unsigned num_pruning_iterations,
         const float a,
         const float b,
         const float alpha,
         const float gamma,
         const unsigned negative_sample_rate,
         const unsigned negative_window,
         const unsigned long long seed,
         str neighbors_key,
         const unsigned num_threads):
    cdef unsigned long long num_edges
    cdef bint too_large = False, self_neighbors = False
    cdef str error_message
    cdef uninitialized_vector[long long] indptr
    cdef uninitialized_vector[unsigned] indices
    cdef uninitialized_vector[float] weights

    # Build the fuzzy simplicial set
    umap_fuzzy_graph(neighbors, distances, num_pruning_iterations, indptr,
                     indices, weights, too_large, self_neighbors, num_threads)

    # If any nearest-neighbor indices were out of bounds or equal to the
    # cell's own index, raise an error
    if too_large:
        error_message = (
            f'some nearest-neighbor indices in obsm[{neighbors_key!r}] are '
            f'>= the total number of cells, {neighbors.shape[0]:,}. This may '
            f'happen if you subset this SingleCell dataset between '
            f'neighbors() and umap(); if so, make sure to run neighbors() '
            f'after, not before, subsetting.')
        raise ValueError(error_message)
    elif self_neighbors:
        error_message = (
            f'some nearest-neighbor indices in obsm[{neighbors_key!r}] '
            f'indicate that a cell is its own neighbor, i.e. '
            f'obsm[{neighbors_key!r}][i, j] == i for some i and j. This may '
            f'happen if you created obsm[{neighbors_key!r}] manually rather '
            f'than following the recommended approach of running '
            f'neighbors().')
        raise ValueError(error_message)

    # Embed
    num_edges = indices.size()
    umap_embed(embedding, PCs,
               <long long[:embedding.shape[0] + 1]> indptr.data(),
               <unsigned[:num_edges]> indices.data(),
               <float[:num_edges]> weights.data(), num_iterations, a, b,
               alpha, gamma, negative_sample_rate, negative_window, seed,
               num_threads)
