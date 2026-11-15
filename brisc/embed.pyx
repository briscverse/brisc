# Functions for embedding with PaCMAP, LocalMAP and UMAP

from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libc.float cimport FLT_MAX
from libc.limits cimport UINT_MAX
from libc.math cimport M_PI
from libc.string cimport memcpy
from libcpp.algorithm cimport nth_element, sort
from libcpp.cmath cimport atan2, cos, exp, log, log2, log10, pow, sin, sqrt, \
    tanh
from libcpp.vector cimport vector
from .cyutils cimport PREFETCH, atomic_or, get_thread_offsets, long_long_, \
    max_heap_replace_top, max_heap_sort, numa_vector, randint, \
    random_uniform, srand, uninitialized_vector, unsigned_, \
    unsigned_long_long_


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
    cdef numa_vector[unsigned] dest_indices, bucket_sources, bucket_targets
    cdef uninitialized_vector[unsigned long long] cursors, range_offsets
    dest_indices = numa_vector[unsigned_](num_threads)
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
    bucket_sources = numa_vector[unsigned_](num_threads)
    bucket_sources.resize(num_cells * num_pairs_per_cell)
    bucket_targets = numa_vector[unsigned_](num_threads)
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


cdef inline (float, float) get_gradient(
        const float[:, ::1] embedding,
        const unsigned[::1] neighbor_pair_indices,
        const unsigned[::1] neighbor_pair_indptr,
        const unsigned[::1] mid_near_pair_indices,
        const unsigned[::1] mid_near_pair_indptr,
        const unsigned[::1] further_pair_indices,
        const unsigned[::1] further_pair_indptr,
        const float w_neighbors,
        const float w_mid_near,
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

    return gradient_i0, gradient_i1


cdef inline void update_embedding_parallel(
        const float[:, ::1] embedding,
        const unsigned[::1] neighbor_pair_indices,
        const unsigned[::1] neighbor_pair_indptr,
        const unsigned[::1] mid_near_pair_indices,
        const unsigned[::1] mid_near_pair_indptr,
        const unsigned[::1] further_pair_indices,
        const unsigned[::1] further_pair_indptr,
        const float w_neighbors,
        const float w_mid_near,
        float[:, ::1] new_embedding,
        float[:, ::1] momentum,
        float[:, ::1] velocity,
        const float beta1,
        const float beta2,
        float learning_rate,
        const unsigned iteration,
        const unsigned num_threads) noexcept nogil:
    # Calculate each cell's gradient from `embedding` and immediately apply
    # it via the Adam optimizer, writing the cell's new position to
    # `new_embedding`, since other cells' gradients need its old position
    cdef unsigned i
    cdef unsigned long long num_cells = embedding.shape[0]
    cdef float gradient_0, gradient_1
    learning_rate *= sqrt(<float> 1 - pow(beta2, <float> (iteration + 1))) / \
        (<float> 1 - pow(beta1, <float> (iteration + 1)))
    for i in prange(num_cells, num_threads=num_threads):
        gradient_0, gradient_1 = get_gradient(
            embedding, neighbor_pair_indices, neighbor_pair_indptr,
            mid_near_pair_indices, mid_near_pair_indptr, further_pair_indices,
            further_pair_indptr, w_neighbors, w_mid_near, i)
        new_embedding[i, 0] = embedding[i, 0]
        new_embedding[i, 1] = embedding[i, 1]
        update_cell_embedding_adam(new_embedding[i, 0], gradient_0,
                                   momentum[i, 0], velocity[i, 0], beta1,
                                   beta2, learning_rate, <float> 1e-7)
        update_cell_embedding_adam(new_embedding[i, 1], gradient_1,
                                   momentum[i, 1], velocity[i, 1], beta1,
                                   beta2, learning_rate, <float> 1e-7)


cdef inline (float, float) get_scaled_gradient(
        const float[:, ::1] embedding,
        const unsigned[::1] neighbor_pair_indices,
        const unsigned[::1] neighbor_pair_indptr,
        const unsigned[::1] further_pair_indices,
        const unsigned[::1] further_pair_indptr,
        const float half_max_distance,
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

    return gradient_i0, gradient_i1


cdef inline void update_embedding_scaled_parallel(
        const float[:, ::1] embedding,
        const unsigned[::1] neighbor_pair_indices,
        const unsigned[::1] neighbor_pair_indptr,
        const unsigned[::1] further_pair_indices,
        const unsigned[::1] further_pair_indptr,
        const float half_max_distance,
        float[:, ::1] new_embedding,
        float[:, ::1] momentum,
        float[:, ::1] velocity,
        const float beta1,
        const float beta2,
        float learning_rate,
        const unsigned iteration,
        const unsigned num_threads) noexcept nogil:
    # Like `update_embedding_parallel()`, but with LocalMAP's phase-3
    # gradients
    cdef unsigned i
    cdef unsigned long long num_cells = embedding.shape[0]
    cdef float gradient_0, gradient_1
    learning_rate *= sqrt(<float> 1 - pow(beta2, <float> (iteration + 1))) / \
        (<float> 1 - pow(beta1, <float> (iteration + 1)))
    for i in prange(num_cells, num_threads=num_threads):
        gradient_0, gradient_1 = get_scaled_gradient(
            embedding, neighbor_pair_indices, neighbor_pair_indptr,
            further_pair_indices, further_pair_indptr, half_max_distance, i)
        new_embedding[i, 0] = embedding[i, 0]
        new_embedding[i, 1] = embedding[i, 1]
        update_cell_embedding_adam(new_embedding[i, 0], gradient_0,
                                   momentum[i, 0], velocity[i, 0], beta1,
                                   beta2, learning_rate, <float> 1e-7)
        update_cell_embedding_adam(new_embedding[i, 1], gradient_1,
                                   momentum[i, 1], velocity[i, 1], beta1,
                                   beta2, learning_rate, <float> 1e-7)


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
        const unsigned iteration):
    cdef unsigned i
    learning_rate *= sqrt(<float> 1 - pow(beta2, <float> (iteration + 1))) / \
        (<float> 1 - pow(beta1, <float> (iteration + 1)))
    for i in range(num_cells):
        update_cell_embedding_adam(embedding[i, 0], gradients[i, 0],
                                   momentum[i, 0], velocity[i, 0], beta1,
                                   beta2, learning_rate, <float> 1e-7)
        update_cell_embedding_adam(embedding[i, 1], gradients[i, 1],
                                   momentum[i, 1], velocity[i, 1], beta1,
                                   beta2, learning_rate, <float> 1e-7)


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
                                 float[:, ::1] embedding_buffer,
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
    cdef float[:, ::1] old_embedding = embedding, \
        new_embedding = embedding_buffer

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

            # Calculate gradients and update the embedding via the Adam
            # optimizer, then swap the old and new embeddings
            update_embedding_parallel(
                old_embedding, neighbor_pair_indices, neighbor_pair_indptr,
                mid_near_pair_indices, mid_near_pair_indptr,
                further_pair_indices, further_pair_indptr, w_neighbors,
                w_mid_near, new_embedding, momentum, velocity, beta1, beta2,
                learning_rate, iteration, num_threads)
            old_embedding, new_embedding = new_embedding, old_embedding

            # Check for KeyboardInterrupts
            if iteration % 8 == 7:
                with gil:
                    PyErr_CheckSignals()


def pacmap(const float[:, ::1] PCs,
           float[:, ::1] embedding,
           float[:, ::1] gradients,
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
    cdef numa_vector[float] momentum_buffer, velocity_buffer, \
        total_distances_buffer
    cdef numa_vector[unsigned] neighbor_pairs_buffer, mid_near_pairs_buffer, \
        further_pairs_buffer, neighbor_pair_indices_buffer, \
        neighbor_pair_indptr_buffer, mid_near_pair_indices_buffer, \
        mid_near_pair_indptr_buffer, further_pair_indices_buffer, \
        further_pair_indptr_buffer
    cdef float[:, ::1] momentum, velocity
    cdef float[::1] total_distances
    cdef unsigned[:, ::1] neighbor_pairs, mid_near_pairs, further_pairs
    cdef unsigned[::1] neighbor_pair_indices, neighbor_pair_indptr, \
        mid_near_pair_indices, mid_near_pair_indptr, further_pair_indices, \
        further_pair_indptr

    momentum_buffer = numa_vector[float](num_threads)
    momentum_buffer.resize(2 * num_cells)
    momentum = <float[:num_cells, :2]> momentum_buffer.data()
    velocity_buffer = numa_vector[float](num_threads)
    velocity_buffer.resize(2 * num_cells)
    velocity = <float[:num_cells, :2]> velocity_buffer.data()
    total_distances_buffer = numa_vector[float](num_threads)
    total_distances_buffer.resize(num_cells)
    total_distances = <float[:num_cells]> total_distances_buffer.data()
    neighbor_pairs_buffer = numa_vector[unsigned_](num_threads)
    neighbor_pairs_buffer.resize(
        <unsigned long long> num_cells * num_neighbors)
    neighbor_pairs = <unsigned[:num_cells, :num_neighbors]> \
        neighbor_pairs_buffer.data()
    mid_near_pairs_buffer = numa_vector[unsigned_](num_threads)
    mid_near_pairs_buffer.resize(
        <unsigned long long> num_cells * num_mid_near_pairs)
    mid_near_pairs = <unsigned[:num_cells, :num_mid_near_pairs]> \
        mid_near_pairs_buffer.data()
    further_pairs_buffer = numa_vector[unsigned_](num_threads)
    further_pairs_buffer.resize(
        <unsigned long long> num_cells * num_further_pairs)
    further_pairs = <unsigned[:num_cells, :num_further_pairs]> \
        further_pairs_buffer.data()

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

        neighbor_pair_indices_buffer = numa_vector[unsigned_](num_threads)
        neighbor_pair_indices_buffer.resize(2ULL * num_cells * num_neighbors)
        neighbor_pair_indices = <unsigned[:2ULL * num_cells * num_neighbors]> \
            neighbor_pair_indices_buffer.data()
        neighbor_pair_indptr_buffer = numa_vector[unsigned_](num_threads)
        neighbor_pair_indptr_buffer.resize(num_cells + 1)
        neighbor_pair_indptr = <unsigned[:num_cells + 1]> \
            neighbor_pair_indptr_buffer.data()
        mid_near_pair_indices_buffer = numa_vector[unsigned_](num_threads)
        mid_near_pair_indices_buffer.resize(
            2ULL * num_cells * num_mid_near_pairs)
        mid_near_pair_indices = \
            <unsigned[:2ULL * num_cells * num_mid_near_pairs]> \
            mid_near_pair_indices_buffer.data()
        mid_near_pair_indptr_buffer = numa_vector[unsigned_](num_threads)
        mid_near_pair_indptr_buffer.resize(num_cells + 1)
        mid_near_pair_indptr = <unsigned[:num_cells + 1]> \
            mid_near_pair_indptr_buffer.data()
        further_pair_indices_buffer = numa_vector[unsigned_](num_threads)
        further_pair_indices_buffer.resize(
            2ULL * num_cells * num_further_pairs)
        further_pair_indices = \
            <unsigned[:2ULL * num_cells * num_further_pairs]> \
            further_pair_indices_buffer.data()
        further_pair_indptr_buffer = numa_vector[unsigned_](num_threads)
        further_pair_indptr_buffer.resize(num_cells + 1)
        further_pair_indptr = <unsigned[:num_cells + 1]> \
            further_pair_indptr_buffer.data()

        reformat_for_parallel(neighbor_pairs, neighbor_pair_indices,
                              neighbor_pair_indptr, num_threads)
        reformat_for_parallel(mid_near_pairs, mid_near_pair_indices,
                              mid_near_pair_indptr, num_threads)
        reformat_for_parallel(further_pairs, further_pair_indices,
                              further_pair_indptr, num_threads)
        PyErr_CheckSignals()

        # Use the parallel implementation even when single-threaded, to
        # ensure identical results regardless of the number of threads. It
        # uses `gradients` as a second embedding buffer and swaps the two
        # every iteration, so after an odd number of iterations, the final
        # embedding is in `gradients`.
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
                              iteration)

        # Re-sample further pairs every 10 iterations in phase 3
        if iteration >= num_phase_1_iterations + num_phase_2_iterations \
                and iteration % 10 == 0:
            sample_further_pairs_nearby(embedding, neighbor_pairs,
                                        further_pairs,
                                        seed - (iteration + 1) * num_cells,
                                        max_distance_squared, num_threads=1)

        # Check for KeyboardInterrupts
        elif iteration % 8 == 7:
            PyErr_CheckSignals()


cdef inline void localmap_parallel(const float[:, ::1] PCs,
                                   float[:, ::1] embedding,
                                   float[:, ::1] momentum,
                                   float[:, ::1] velocity,
                                   float[:, ::1] embedding_buffer,
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
    cdef float[:, ::1] old_embedding = embedding, \
        new_embedding = embedding_buffer

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

            # Calculate gradients and update the embedding via the Adam
            # optimizer, then swap the old and new embeddings
            if iteration >= num_phase_1_iterations + num_phase_2_iterations:
                update_embedding_scaled_parallel(
                    old_embedding, neighbor_pair_indices, neighbor_pair_indptr,
                    further_pair_indices, further_pair_indptr,
                    half_max_distance, new_embedding, momentum, velocity,
                    beta1, beta2, learning_rate, iteration, num_threads)
            else:
                update_embedding_parallel(
                    old_embedding, neighbor_pair_indices, neighbor_pair_indptr,
                    mid_near_pair_indices, mid_near_pair_indptr,
                    further_pair_indices, further_pair_indptr, w_neighbors,
                    w_mid_near, new_embedding, momentum, velocity, beta1,
                    beta2, learning_rate, iteration, num_threads)
            old_embedding, new_embedding = new_embedding, old_embedding

            # Re-sample further pairs every 10 iterations in phase 3
            if iteration >= num_phase_1_iterations + num_phase_2_iterations \
                    and iteration % 10 == 0:
                sample_further_pairs_nearby(old_embedding, neighbor_pairs,
                                            further_pairs,
                                            seed - (iteration + 1) * num_cells,
                                            max_distance_squared, num_threads)
                reformat_for_parallel(further_pairs, further_pair_indices,
                                      further_pair_indptr, num_threads)

            # Check for KeyboardInterrupts
            elif iteration % 8 == 7:
                with gil:
                    PyErr_CheckSignals()


def localmap(const float[:, ::1] PCs,
             float[:, ::1] embedding,
             float[:, ::1] gradients,
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
    cdef numa_vector[float] momentum_buffer, velocity_buffer, \
        total_distances_buffer
    cdef numa_vector[unsigned] neighbor_pairs_buffer, mid_near_pairs_buffer, \
        further_pairs_buffer, neighbor_pair_indices_buffer, \
        neighbor_pair_indptr_buffer, mid_near_pair_indices_buffer, \
        mid_near_pair_indptr_buffer, further_pair_indices_buffer, \
        further_pair_indptr_buffer
    cdef float[:, ::1] momentum, velocity
    cdef float[::1] total_distances
    cdef unsigned[:, ::1] neighbor_pairs, mid_near_pairs, further_pairs
    cdef unsigned[::1] neighbor_pair_indices, neighbor_pair_indptr, \
        mid_near_pair_indices, mid_near_pair_indptr, further_pair_indices, \
        further_pair_indptr

    momentum_buffer = numa_vector[float](num_threads)
    momentum_buffer.resize(2 * num_cells)
    momentum = <float[:num_cells, :2]> momentum_buffer.data()
    velocity_buffer = numa_vector[float](num_threads)
    velocity_buffer.resize(2 * num_cells)
    velocity = <float[:num_cells, :2]> velocity_buffer.data()
    total_distances_buffer = numa_vector[float](num_threads)
    total_distances_buffer.resize(num_cells)
    total_distances = <float[:num_cells]> total_distances_buffer.data()
    neighbor_pairs_buffer = numa_vector[unsigned_](num_threads)
    neighbor_pairs_buffer.resize(
        <unsigned long long> num_cells * num_neighbors)
    neighbor_pairs = <unsigned[:num_cells, :num_neighbors]> \
        neighbor_pairs_buffer.data()
    mid_near_pairs_buffer = numa_vector[unsigned_](num_threads)
    mid_near_pairs_buffer.resize(
        <unsigned long long> num_cells * num_mid_near_pairs)
    mid_near_pairs = <unsigned[:num_cells, :num_mid_near_pairs]> \
        mid_near_pairs_buffer.data()
    further_pairs_buffer = numa_vector[unsigned_](num_threads)
    further_pairs_buffer.resize(
        <unsigned long long> num_cells * num_further_pairs)
    further_pairs = <unsigned[:num_cells, :num_further_pairs]> \
        further_pairs_buffer.data()

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

        neighbor_pair_indices_buffer = numa_vector[unsigned_](num_threads)
        neighbor_pair_indices_buffer.resize(2ULL * num_cells * num_neighbors)
        neighbor_pair_indices = <unsigned[:2ULL * num_cells * num_neighbors]> \
            neighbor_pair_indices_buffer.data()
        neighbor_pair_indptr_buffer = numa_vector[unsigned_](num_threads)
        neighbor_pair_indptr_buffer.resize(num_cells + 1)
        neighbor_pair_indptr = <unsigned[:num_cells + 1]> \
            neighbor_pair_indptr_buffer.data()
        mid_near_pair_indices_buffer = numa_vector[unsigned_](num_threads)
        mid_near_pair_indices_buffer.resize(
            2ULL * num_cells * num_mid_near_pairs)
        mid_near_pair_indices = \
            <unsigned[:2ULL * num_cells * num_mid_near_pairs]> \
            mid_near_pair_indices_buffer.data()
        mid_near_pair_indptr_buffer = numa_vector[unsigned_](num_threads)
        mid_near_pair_indptr_buffer.resize(num_cells + 1)
        mid_near_pair_indptr = <unsigned[:num_cells + 1]> \
            mid_near_pair_indptr_buffer.data()
        further_pair_indices_buffer = numa_vector[unsigned_](num_threads)
        further_pair_indices_buffer.resize(
            2ULL * num_cells * num_further_pairs)
        further_pair_indices = \
            <unsigned[:2ULL * num_cells * num_further_pairs]> \
            further_pair_indices_buffer.data()
        further_pair_indptr_buffer = numa_vector[unsigned_](num_threads)
        further_pair_indptr_buffer.resize(num_cells + 1)
        further_pair_indptr = <unsigned[:num_cells + 1]> \
            further_pair_indptr_buffer.data()

        reformat_for_parallel(neighbor_pairs, neighbor_pair_indices,
                              neighbor_pair_indptr, num_threads)
        reformat_for_parallel(mid_near_pairs, mid_near_pair_indices,
                              mid_near_pair_indptr, num_threads)
        reformat_for_parallel(further_pairs, further_pair_indices,
                              further_pair_indptr, num_threads)
        PyErr_CheckSignals()

        # Use the parallel implementation even when single-threaded, to
        # ensure identical results regardless of the number of threads. It
        # uses `gradients` as a second embedding buffer and swaps the two
        # every iteration, so after an odd number of iterations, the final
        # embedding is in `gradients`.
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
# results do not depend on the number of threads. Sums over cells are
# calculated over fixed-size chunks of `chunk_size` cells, which are then
# combined in order, for the same reason.

cdef inline void umap_group_by_key(const unsigned[::1] keys,
                                   const unsigned num_keys,
                                   numa_vector[long long]& group_indptr,
                                   numa_vector[unsigned long long]& groups,
                                   unsigned num_threads):
    # Group the items `0, 1, ..., len(keys) - 1` by their keys, which are all
    # less than `num_keys`: `groups[group_indptr[k]:group_indptr[k + 1]]` will
    # contain the items whose key is `k`, in ascending order. Do this
    # deterministically and in parallel by first bucketing items by which of
    # `num_threads` contiguous ranges of keys their key falls in, with the
    # buckets for each range of keys laid out consecutively and ordered by
    # which of `num_threads` contiguous chunks of items they come from.

    cdef unsigned key, item_chunk, key_range, thread_index, \
        stride = num_threads + 16
    cdef unsigned long long item, start, end, position, count, total, \
        num_items = keys.shape[0], \
        item_chunk_size = (num_items + num_threads - 1) / num_threads, \
        key_range_size = (num_keys + num_threads - 1) / num_threads
    cdef uninitialized_vector[unsigned long long] cursors, range_offsets
    cdef numa_vector[unsigned long long] buckets_buffer, key_cursors_buffer
    cdef unsigned long long[::1] buckets, key_cursors

    # Count the items going from each chunk of items to each range of keys
    cursors.resize(<unsigned long long> num_threads * stride)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        for key_range in range(num_threads):
            cursors[thread_index * stride + key_range] = 0
        start = min(thread_index * item_chunk_size, num_items)
        end = min(start + item_chunk_size, num_items)
        for item in range(start, end):
            cursors[thread_index * stride + keys[item] / key_range_size] += 1

    # Lay out the buckets, and convert the counts to each bucket's start
    range_offsets.resize(num_threads + 1)
    total = 0
    for key_range in range(num_threads):
        range_offsets[key_range] = total
        for item_chunk in range(num_threads):
            count = cursors[item_chunk * stride + key_range]
            cursors[item_chunk * stride + key_range] = total
            total += count
    range_offsets[num_threads] = total

    # Fill in the buckets
    buckets_buffer = numa_vector[unsigned_long_long_](num_threads)
    buckets_buffer.resize(num_items)
    buckets = <unsigned long long[:num_items]> buckets_buffer.data()
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        start = min(thread_index * item_chunk_size, num_items)
        end = min(start + item_chunk_size, num_items)
        for item in range(start, end):
            key_range = keys[item] / key_range_size
            position = cursors[thread_index * stride + key_range]
            cursors[thread_index * stride + key_range] = position + 1
            buckets[position] = item

    # Count the items with each key and cumsum the counts in `group_indptr`;
    # each thread handles one range of keys, whose cumsum starts at
    # `range_offsets[thread_index]`
    group_indptr = numa_vector[long_long_](num_threads)
    group_indptr.resize(num_keys + 1)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        start = min(thread_index * key_range_size, num_keys)
        end = min(start + key_range_size, num_keys)
        for position in range(range_offsets[thread_index],
                              range_offsets[thread_index + 1]):
            group_indptr[keys[buckets[position]] + 1] += 1
        total = range_offsets[thread_index]
        for key in range(start, end):
            total = total + group_indptr[key + 1]
            group_indptr[key + 1] = total

    # Fill in each key's group, in ascending order of item
    groups = numa_vector[unsigned_long_long_](num_threads)
    groups.resize(num_items)
    key_cursors_buffer = numa_vector[unsigned_long_long_](num_threads)
    key_cursors_buffer.resize(num_keys)
    key_cursors = <unsigned long long[:num_keys]> key_cursors_buffer.data()
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        start = min(thread_index * key_range_size, num_keys)
        end = min(start + key_range_size, num_keys)
        for key in range(start, end):
            key_cursors[key] = group_indptr[key]
        for position in range(range_offsets[thread_index],
                              range_offsets[thread_index + 1]):
            item = buckets[position]
            key = keys[item]
            groups[key_cursors[key]] = item
            key_cursors[key] += 1


cdef inline void umap_union_weights(const unsigned[:, ::1] neighbors,
                                    const float[::1] directed_weights,
                                    const unsigned long long[::1] reverse,
                                    const long long[::1] reverse_indptr,
                                    const unsigned cell,
                                    float* forward_weights) noexcept nogil:
    # Calculate `cell`'s fuzzy-union weights to each of its own nearest
    # neighbors `j`: `w_ij + w_ji - w_ij * w_ji` if `cell` is also one of
    # `j`'s nearest neighbors, and just `w_ij` otherwise.
    # `reverse[reverse_indptr[cell]:reverse_indptr[cell + 1]]` holds the flat
    # indices `j * num_neighbors + k` of every directed edge pointing to
    # `cell`. (`forward_weights` is a pointer into a per-thread buffer, since
    # Cython does not allow slicing a memoryview per thread in a parallel
    # block.)

    cdef unsigned neighbor_index, source, num_neighbors = neighbors.shape[1]
    cdef long long position
    cdef unsigned long long edge
    cdef float weight

    for neighbor_index in range(num_neighbors):
        forward_weights[neighbor_index] = directed_weights[
            <unsigned long long> cell * num_neighbors + neighbor_index]
    for position in range(reverse_indptr[cell], reverse_indptr[cell + 1]):
        edge = reverse[position]
        source = <unsigned> (edge / num_neighbors)
        neighbor_index = 0
        while neighbor_index < num_neighbors and \
                neighbors[cell, neighbor_index] != source:
            neighbor_index += 1
        if neighbor_index < num_neighbors:
            weight = directed_weights[edge]
            forward_weights[neighbor_index] = \
                forward_weights[neighbor_index] + weight - \
                forward_weights[neighbor_index] * weight


cdef inline void umap_fuzzy_graph(const unsigned[:, ::1] neighbors,
                                  const float[:, ::1] distances,
                                  const unsigned num_pruning_iterations,
                                  numa_vector[long long]& indptr,
                                  numa_vector[unsigned]& indices,
                                  numa_vector[float]& weights,
                                  bint& too_large,
                                  bint& self_neighbors,
                                  unsigned num_threads):
    # Build UMAP's fuzzy simplicial set: convert each cell's nearest-neighbor
    # distances to directed membership weights `w_ij`, symmetrize them with the
    # fuzzy union `w_ij + w_ji - w_ij * w_ji`, and prune small weights. Each
    # row of the resulting CSR graph lists the cell's own nearest neighbors
    # (in order), followed by the cells that have it as a nearest neighbor but
    # are not among its own nearest neighbors (in ascending order).

    cdef unsigned cell, neighbor, neighbor_index, source, thread_index, \
        start, end, iteration, num_cells = neighbors.shape[0], \
        num_neighbors = neighbors.shape[1]
    cdef long long position
    cdef unsigned long long edge, output_position, num_kept, total, \
        num_edges = <unsigned long long> num_cells * num_neighbors
    cdef float distance, rho, mean_distance, distance_sum, weight_sum, \
        low, middle, high, threshold, \
        target = log2(<float> (num_neighbors + 1))
    cdef float* forward_weights
    cdef uninitialized_vector[float] forward_weights_buffer
    cdef uninitialized_vector[unsigned] thread_offsets
    cdef uninitialized_vector[unsigned long long] thread_num_edges
    cdef numa_vector[float] directed_weights_buffer
    cdef numa_vector[long long] reverse_indptr_buffer
    cdef numa_vector[unsigned long long] reverse_buffer
    cdef float[::1] directed_weights
    cdef long long[::1] reverse_indptr
    cdef unsigned long long[::1] reverse

    # Convert each cell's nearest-neighbor distances to directed weights, as
    # in umap-learn's `smooth_knn_dist()`: subtract the distance to the
    # nearest neighbor (`rho`), then binary-search for the bandwidth
    # (`middle`) at which the weights sum to `target`. Also check that
    # nearest-neighbor indices are valid.
    directed_weights_buffer = numa_vector[float](num_threads)
    directed_weights_buffer.resize(num_edges)
    directed_weights = <float[:num_edges]> directed_weights_buffer.data()
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        start = <unsigned> (<unsigned long long> thread_index * num_cells /
                            num_threads)
        end = <unsigned> (<unsigned long long> (thread_index + 1) *
                          num_cells / num_threads)
        for cell in range(start, end):
            edge = <unsigned long long> cell * num_neighbors
            distance_sum = 0
            for neighbor_index in range(num_neighbors):
                distance = sqrt(max(distances[cell, neighbor_index],
                                    <float> 0))
                directed_weights[edge + neighbor_index] = distance
                distance_sum = distance_sum + distance
                neighbor = neighbors[cell, neighbor_index]
                if neighbor >= num_cells:
                    atomic_or(too_large, True)
                elif neighbor == cell:
                    atomic_or(self_neighbors, True)
            rho = 0
            for neighbor_index in range(num_neighbors):
                if directed_weights[edge + neighbor_index] > <float> 0:
                    rho = directed_weights[edge + neighbor_index]
                    break
            mean_distance = distance_sum / (num_neighbors + 1)
            low = 0
            middle = 1
            high = FLT_MAX
            for iteration in range(64):
                weight_sum = 0
                for neighbor_index in range(num_neighbors):
                    distance = max(
                        directed_weights[edge + neighbor_index] - rho,
                        <float> 0)
                    weight_sum = weight_sum + exp(-(distance / middle))
                if abs(weight_sum - target) < <float> 1e-5:
                    break
                if weight_sum > target:
                    high = middle
                    middle = <float> 0.5 * (low + high)
                else:
                    low = middle
                    if high == FLT_MAX:
                        middle = middle * <float> 2
                    else:
                        middle = <float> 0.5 * (low + high)
            if middle < <float> 0.001 * mean_distance:
                middle = <float> 0.001 * mean_distance
            for neighbor_index in range(num_neighbors):
                distance = max(directed_weights[edge + neighbor_index] - rho,
                               <float> 0)
                directed_weights[edge + neighbor_index] = \
                    exp(-(distance / middle))
    if too_large or self_neighbors:
        return
    PyErr_CheckSignals()

    # Get the directed edges pointing to each cell, in ascending order of
    # source cell
    umap_group_by_key(
        <unsigned[:num_edges]> <unsigned*> &neighbors[0, 0], num_cells,
        reverse_indptr_buffer, reverse_buffer, num_threads)
    reverse_indptr = <long long[:num_cells + 1]> reverse_indptr_buffer.data()
    reverse = <unsigned long long[:num_edges]> reverse_buffer.data()
    PyErr_CheckSignals()

    # Symmetrizing a cell's edges takes time proportional to the number of
    # edges pointing to it, which varies widely (e.g. for hubs), so balance
    # work across threads by that number
    thread_offsets.resize(num_threads + 1)
    get_thread_offsets(reverse_indptr, thread_offsets.data(), num_threads)

    # Prune weights below `1 / num_pruning_iterations`. Like umap-learn,
    # this is the maximum weight divided by the number of iterations, but the
    # maximum weight is always 1 (up to rounding): each cell's directed weight
    # to its nearest neighbor is `exp(0) = 1`, and the fuzzy union of weights
    # between 0 and 1 is also between 0 and 1. So pruning removes the edges
    # that would be sampled less than once in expectation during
    # optimization, and each remaining edge is sampled as many times in
    # expectation as in umap-learn.
    threshold = <float> 1 / num_pruning_iterations
    forward_weights_buffer.resize(num_threads * (num_neighbors + 16))

    # Count the edges each cell keeps after pruning, then cumsum the counts in
    # `indptr`
    indptr = numa_vector[long_long_](num_threads)
    indptr.resize(num_cells + 1)
    thread_num_edges.resize(num_threads * 16)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        forward_weights = \
            forward_weights_buffer.data() + thread_index * (num_neighbors + 16)
        total = 0
        for cell in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            umap_union_weights(neighbors, directed_weights, reverse,
                               reverse_indptr, cell, forward_weights)
            num_kept = 0
            for neighbor_index in range(num_neighbors):
                if forward_weights[neighbor_index] >= threshold:
                    num_kept = num_kept + 1
            for position in range(reverse_indptr[cell],
                                  reverse_indptr[cell + 1]):
                edge = reverse[position]
                source = <unsigned> (edge / num_neighbors)
                neighbor_index = 0
                while neighbor_index < num_neighbors and \
                        neighbors[cell, neighbor_index] != source:
                    neighbor_index = neighbor_index + 1
                if neighbor_index == num_neighbors and \
                        directed_weights[edge] >= threshold:
                    num_kept = num_kept + 1
            total = total + num_kept
            indptr[cell + 1] = total
        thread_num_edges[thread_index * 16] = total
    total = 0
    for thread_index in range(num_threads):
        num_kept = thread_num_edges[thread_index * 16]
        thread_num_edges[thread_index * 16] = total
        total += num_kept
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        for cell in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            indptr[cell + 1] += thread_num_edges[thread_index * 16]
    PyErr_CheckSignals()

    # Fill in the pruned graph
    indices = numa_vector[unsigned_](num_threads)
    indices.resize(total)
    weights = numa_vector[float](num_threads)
    weights.resize(total)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        forward_weights = \
            forward_weights_buffer.data() + thread_index * (num_neighbors + 16)
        for cell in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            umap_union_weights(neighbors, directed_weights, reverse,
                               reverse_indptr, cell, forward_weights)
            output_position = indptr[cell]
            for neighbor_index in range(num_neighbors):
                if forward_weights[neighbor_index] >= threshold:
                    indices[output_position] = neighbors[cell, neighbor_index]
                    weights[output_position] = forward_weights[neighbor_index]
                    output_position = output_position + 1
            for position in range(reverse_indptr[cell],
                                  reverse_indptr[cell + 1]):
                edge = reverse[position]
                source = <unsigned> (edge / num_neighbors)
                neighbor_index = 0
                while neighbor_index < num_neighbors and \
                        neighbors[cell, neighbor_index] != source:
                    neighbor_index = neighbor_index + 1
                if neighbor_index == num_neighbors and \
                        directed_weights[edge] >= threshold:
                    indices[output_position] = source
                    weights[output_position] = directed_weights[edge]
                    output_position = output_position + 1


cdef inline (float, float) umap_projection_range(
        const float[::1] positions,
        const float cos_angle,
        const float sin_angle,
        const unsigned num_threads) noexcept nogil:
    # Return the minimum and maximum over cells of the projection of their
    # positions onto the direction (`cos_angle`, `sin_angle`). `positions`
    # is flattened: cell `i`'s position is `positions[2 * i:2 * i + 2]`.
    # Minima and maxima are exact, so unlike sums, they do not depend on how
    # cells are split across threads.

    cdef unsigned cell, thread_index, start, end, \
        num_cells = positions.shape[0] / 2
    cdef float projection, minimum, maximum
    cdef uninitialized_vector[float] thread_minima, thread_maxima
    thread_minima.resize(num_threads * 16)
    thread_maxima.resize(num_threads * 16)
    with parallel(num_threads=num_threads):
        thread_index = threadid()
        start = <unsigned> (<unsigned long long> thread_index * num_cells /
                            num_threads)
        end = <unsigned> (<unsigned long long> (thread_index + 1) *
                          num_cells / num_threads)
        minimum = FLT_MAX
        maximum = -FLT_MAX
        for cell in range(start, end):
            projection = positions[2 * cell] * cos_angle + \
                positions[2 * cell + 1] * sin_angle
            minimum = min(minimum, projection)
            maximum = max(maximum, projection)
        thread_minima[thread_index * 16] = minimum
        thread_maxima[thread_index * 16] = maximum
    minimum = FLT_MAX
    maximum = -FLT_MAX
    for thread_index in range(num_threads):
        minimum = min(minimum, thread_minima[thread_index * 16])
        maximum = max(maximum, thread_maxima[thread_index * 16])
    return minimum, maximum


cdef inline (float, float) umap_mean(const float[:, ::1] positions,
                                     const unsigned chunk_size,
                                     const unsigned num_threads) \
        noexcept nogil:
    # Return the mean of each column of the `num_rows` × 2 array `positions`

    cdef unsigned row, chunk, num_rows = positions.shape[0], \
        num_chunks = (num_rows + chunk_size - 1) / chunk_size
    cdef float sum_x, sum_y, total_x = 0, total_y = 0
    cdef uninitialized_vector[float] chunk_sums
    chunk_sums.resize(2 * num_chunks)
    for chunk in prange(num_chunks, num_threads=num_threads):
        sum_x = 0
        sum_y = 0
        for row in range(chunk * chunk_size,
                         min((chunk + 1) * chunk_size, num_rows)):
            sum_x = sum_x + positions[row, 0]
            sum_y = sum_y + positions[row, 1]
        chunk_sums[2 * chunk] = sum_x
        chunk_sums[2 * chunk + 1] = sum_y
    for chunk in range(num_chunks):
        total_x += chunk_sums[2 * chunk]
        total_y += chunk_sums[2 * chunk + 1]
    return total_x / num_rows, total_y / num_rows


cdef inline float sorted_quantile(const float* values,
                                  const unsigned num_values,
                                  const float quantile) noexcept nogil:
    # Linearly interpolated quantile of an already-sorted array, as in
    # `np.quantile()`

    cdef float position = (num_values - 1) * quantile
    cdef unsigned low = <unsigned> position, \
        high = min(low + 1, num_values - 1)
    return values[low] + (position - low) * (values[high] - values[low])


cdef inline float umap_negative_force_ratio(
        const float[::1] sorted_positions,
        const unsigned window,
        const float a,
        const float b,
        const float gamma,
        unsigned long long* state) noexcept nogil:
    # Estimate how much larger the resultant repulsive force on a cell is
    # from uniformly sampled negatives than from negatives sampled from its
    # window, as in umap-learn's `_sample_negative_force_ratio()`: for 128
    # random cells, compute the resultants of the forces from 5 negatives
    # sampled from the cell's window and 5 sampled uniformly, and return the
    # exponentiated mean log ratio of the two, after clipping the log ratios
    # to their 10th and 90th percentiles. Forces are softly clipped to a norm
    # of at most `gamma`. `sorted_positions` contains cells' (flattened)
    # positions in the current projection order.

    cdef unsigned source_index, sample, position, negative, window_start, \
        num_cells = sorted_positions.shape[0] / 2, num_valid = 0
    cdef float dx, dy, dist_squared, distance, coefficient, scale, \
        local_resultant, global_resultant, lower, upper, total = 0
    cdef float resultant_x[2]
    cdef float resultant_y[2]
    cdef float log_ratios[128]

    for source_index in range(128):
        position = randint(num_cells, state)
        window_start = min(position - min(position, window / 2),
                           num_cells - window)
        resultant_x[0] = resultant_x[1] = resultant_y[0] = resultant_y[1] = 0

        # Samples 0-4 are from the window (and contribute to resultant 0) and
        # samples 5-9 are uniform (and contribute to resultant 1)
        for sample in range(10):
            negative = window_start + randint(window, state) \
                if sample < 5 else randint(num_cells, state)
            dx = sorted_positions[2 * position] - \
                sorted_positions[2 * negative]
            dy = sorted_positions[2 * position + 1] - \
                sorted_positions[2 * negative + 1]
            dist_squared = dx * dx + dy * dy
            if dist_squared > <float> 0:
                distance = sqrt(dist_squared)
                coefficient = <float> 2 * gamma * b / (
                    (<float> 0.001 + dist_squared) *
                    (a * pow(dist_squared, b) + <float> 1))
                scale = gamma * tanh(coefficient * distance / gamma) / distance
                resultant_x[sample / 5] += scale * dx
                resultant_y[sample / 5] += scale * dy
        local_resultant = sqrt(resultant_x[0] * resultant_x[0] +
                               resultant_y[0] * resultant_y[0])
        global_resultant = sqrt(resultant_x[1] * resultant_x[1] +
                                resultant_y[1] * resultant_y[1])
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
    for sample in range(num_valid):
        total += min(max(log_ratios[sample], lower), upper)
    return exp(total / num_valid)


cdef inline void umap_layout(float[::1] embedding,
                             float[::1] embedding_buffer,
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
    # positions and immediately takes an Adam step, writing its new position
    # to a second buffer, since other cells' gradients need its old position.
    # `embedding` and `embedding_buffer` swap roles every epoch, so after an
    # odd number of epochs, the final positions are in `embedding_buffer`.
    #
    # Positions (and Adam's momentum and velocity) are flattened: cell `i`'s
    # position is `embedding[2 * i:2 * i + 2]`. With 2D memoryviews, each
    # access to another cell's position would multiply by the row stride in
    # bytes, which prevents the compiler from vectorizing the gradient loops
    # with gather instructions.
    #
    # When `window` is less than the number of cells (and there are more than
    # 1024 cells), negative samples are "hard negatives" drawn from each
    # cell's window of `window` nearby cells in a random 1D projection of the
    # embedding, recomputed each epoch. Their repulsion is scaled down by a
    # factor that starts at `(window / num_cells) ** 0.75` and adapts every 10
    # epochs during the first half of optimization, based on an estimate of
    # how much weaker the repulsion from window negatives is than from
    # uniformly sampled negatives.

    cdef unsigned epoch, cell, previous_cell, neighbor, negative, sample, \
        bin, other_thread, thread_index, start, end, window_start, slot, \
        buffer_size, num_bins, num_warmup_epochs, num_accepted, \
        num_negatives, num_cells = embedding.shape[0] / 2, max_degree = 0
    cdef long long edge
    cdef unsigned long long state, position, count, total
    cdef bint use_window
    cdef float alpha, learning_rate, epsilon, max_weight, fraction, angle, \
        cos_angle, sin_angle, minimum, maximum, projection, bin_scale, \
        attraction_constant, repulsion_constant, clip_constant, \
        inverse_gamma, negatives_per_edge, negative_scale, scale_prior, \
        scale_anchor = -1, force_ratio, raw_scale, log_change, position_x, \
        position_y, dx, dy, dist_squared, dist_b_minus_1, inverse_distance, \
        scale, gradient_x, gradient_y, b_minus_1 = b - <float> 1, \
        beta1 = <float> 0.9, beta2 = <float> 0.99
    cdef unsigned long long* histogram
    cdef unsigned* neighbor_buffer
    cdef unsigned* negative_buffer
    cdef float[::1] old_positions = embedding, \
        new_positions = embedding_buffer, momentum, velocity, \
        window_positions, negative_positions = embedding
    cdef unsigned[::1] bins, rank
    cdef unsigned long long[::1] histograms, bin_starts
    cdef uninitialized_vector[float] thread_max_weights
    cdef uninitialized_vector[unsigned] thread_offsets, thread_max_degrees, \
        sample_buffers, sample_counts
    cdef uninitialized_vector[unsigned long long] histograms_buffer, \
        bin_starts_buffer
    cdef numa_vector[float] momentum_buffer, velocity_buffer, \
        window_positions_buffer
    cdef numa_vector[unsigned] bins_buffer, rank_buffer

    # Split cells into contiguous ranges with about the same number of edges
    # per thread
    num_threads = min(num_threads, num_cells)
    thread_offsets.resize(num_threads + 1)
    get_thread_offsets(indptr, thread_offsets.data(), num_threads)

    # Get the maximum edge weight and degree
    thread_max_weights.resize(num_threads * 16)
    thread_max_degrees.resize(num_threads * 16)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        max_weight = 0
        max_degree = 0
        for cell in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            max_degree = max(max_degree,
                             <unsigned> (indptr[cell + 1] - indptr[cell]))
            for edge in range(indptr[cell], indptr[cell + 1]):
                max_weight = max(max_weight, weights[edge])
        thread_max_weights[thread_index * 16] = max_weight
        thread_max_degrees[thread_index * 16] = max_degree
    max_weight = 0
    max_degree = 0
    for thread_index in range(num_threads):
        max_weight = max(max_weight, thread_max_weights[thread_index * 16])
        max_degree = max(max_degree, thread_max_degrees[thread_index * 16])

    # Allocate Adam's momentum and velocity
    momentum_buffer = numa_vector[float](num_threads)
    momentum_buffer.resize(2 * <unsigned long long> num_cells)
    momentum = <float[:2 * num_cells]> momentum_buffer.data()
    velocity_buffer = numa_vector[float](num_threads)
    velocity_buffer.resize(2 * <unsigned long long> num_cells)
    velocity = <float[:2 * num_cells]> velocity_buffer.data()

    # Set up hard-negative windows, if used. Quantize projections into enough
    # bins that each window spans at least 32 of them: a power of 2 between
    # 16 and 2048.
    window = min(num_cells, max(200, window))
    use_window = num_cells > 1024 and window < num_cells
    if not use_window:
        window = num_cells
    negative_scale = pow(window / <float> num_cells, <float> 0.75) \
        if use_window else <float> 1
    scale_prior = negative_scale
    if use_window:
        num_bins = 16
        while num_bins < 2048 and \
                num_bins < 32 * <unsigned long long> num_cells / window:
            num_bins *= 2
        bins_buffer = numa_vector[unsigned_](num_threads)
        bins_buffer.resize(num_cells)
        bins = <unsigned[:num_cells]> bins_buffer.data()
        rank_buffer = numa_vector[unsigned_](num_threads)
        rank_buffer.resize(num_cells)
        rank = <unsigned[:num_cells]> rank_buffer.data()
        window_positions_buffer = numa_vector[float](num_threads)
        window_positions_buffer.resize(2 * <unsigned long long> num_cells)
        window_positions = \
            <float[:2 * num_cells]> window_positions_buffer.data()
        negative_positions = window_positions
        histograms_buffer.resize(<unsigned long long> num_threads * num_bins)
        histograms = <unsigned long long[:num_threads * num_bins]> \
            histograms_buffer.data()
        bin_starts_buffer.resize(num_bins)
        bin_starts = <unsigned long long[:num_bins]> bin_starts_buffer.data()

    # Like umap-learn, use half-strength attraction for graphs with at most
    # 1024 cells, and draw `negative_sample_rate / 1.5` negative samples per
    # sampled edge
    attraction_constant = <float> 2 * a * b * \
        (<float> 0.5 if num_cells <= 1024 else <float> 1)
    clip_constant = min(gamma, <float> 16)
    inverse_gamma = <float> 1 / gamma
    negatives_per_edge = negative_sample_rate / <float> 1.5

    # Each thread alternates between two buffers for the sampled neighbors
    # and negative samples of consecutive cells, and stores the number of
    # each in `sample_counts`
    buffer_size = max_degree + 16 + \
        <unsigned> (max_degree * negatives_per_edge) + 16
    sample_buffers.resize(<unsigned long long> num_threads * 2 * buffer_size)
    sample_counts.resize(num_threads * 16)

    # umap-learn's Adam learning-rate schedule: a quadratic decay to 0.1 over
    # a warm-up period of up to 100 epochs, starting from `initial_alpha / 2`
    # (after a good initialization) or `2 * initial_alpha`, then a linear
    # decay from 0.2 to 0.05
    num_warmup_epochs = <unsigned> min(<float> 100, num_epochs / <float> 8) \
        if good_initialization else \
        <unsigned> min(num_epochs / <float> 2, <float> 100)
    state = srand(seed - 1)
    with nogil:
        for epoch in range(num_epochs):
            # Get the learning rate, folding in Adam's bias correction, and
            # scale Adam's epsilon to match umap-learn's placement of it after
            # bias correction
            if epoch < num_warmup_epochs:
                fraction = <float> 1 - epoch / <float> num_warmup_epochs
                alpha = ((<float> 0.5 if good_initialization else <float> 2) *
                         initial_alpha - <float> 0.1) * fraction * fraction + \
                    <float> 0.1
            else:
                alpha = <float> 0.15 * (
                    <float> 1 - (epoch - num_warmup_epochs) /
                    <float> (num_epochs - num_warmup_epochs)) + <float> 0.05
            learning_rate = alpha * \
                sqrt(<float> 1 - pow(beta2, <float> (epoch + 1))) / \
                (<float> 1 - pow(beta1, <float> (epoch + 1)))
            epsilon = <float> 1e-4 * \
                sqrt(<float> 1 - pow(beta2, <float> (epoch + 1)))

            if use_window:
                # Sort cells by their projection onto a random direction,
                # storing each cell's position in the sorted order in `rank`
                # and gathering cells' positions into `window_positions` in
                # this order, so each cell's window is contiguous. Unlike
                # umap-learn, which sorts the projections exactly, quantize
                # them into `num_bins` equal-width bins and counting-sort the
                # bins. Ties are broken by index; since bins are much
                # narrower than windows, this barely changes the windows.
                # First, bin each cell's projection, and count each thread's
                # cells in each bin.
                angle = <float> 2 * <float> M_PI * random_uniform(&state)
                cos_angle = cos(angle)
                sin_angle = sin(angle)
                minimum, maximum = umap_projection_range(
                    old_positions, cos_angle, sin_angle, num_threads)
                bin_scale = num_bins / (maximum - minimum) \
                    if maximum > minimum else <float> 0
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    histogram = &histograms[
                        <unsigned long long> thread_index * num_bins]
                    for bin in range(num_bins):
                        histogram[bin] = 0
                    start = <unsigned> (<unsigned long long> thread_index *
                                        num_cells / num_threads)
                    end = <unsigned> (<unsigned long long> (thread_index + 1) *
                                      num_cells / num_threads)
                    for cell in range(start, end):
                        projection = old_positions[2 * cell] * cos_angle + \
                            old_positions[2 * cell + 1] * sin_angle
                        bin = min(<unsigned> ((projection - minimum) *
                                              bin_scale), num_bins - 1)
                        bins[cell] = bin
                        histogram[bin] += 1

                # Convert the counts to each thread's first position in each
                # bin: cumsum each bin's counts over threads, then add the
                # bin's start
                for bin in prange(num_bins, num_threads=num_threads):
                    total = 0
                    for other_thread in range(num_threads):
                        count = histograms[
                            <unsigned long long> other_thread * num_bins + bin]
                        histograms[<unsigned long long> other_thread *
                                   num_bins + bin] = total
                        total = total + count
                    bin_starts[bin] = total
                total = 0
                for bin in range(num_bins):
                    count = bin_starts[bin]
                    bin_starts[bin] = total
                    total += count
                for bin in prange(num_bins, num_threads=num_threads):
                    for other_thread in range(num_threads):
                        histograms[<unsigned long long> other_thread *
                                   num_bins + bin] += bin_starts[bin]

                # Scatter each cell's position to its sorted position. Each
                # thread handles the same contiguous range of cells it
                # counted, so the result does not depend on the number of
                # threads.
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    histogram = &histograms[
                        <unsigned long long> thread_index * num_bins]
                    start = <unsigned> (<unsigned long long> thread_index *
                                        num_cells / num_threads)
                    end = <unsigned> (<unsigned long long> (thread_index + 1) *
                                      num_cells / num_threads)
                    for cell in range(start, end):
                        position = histogram[bins[cell]]
                        histogram[bins[cell]] = position + 1
                        rank[cell] = <unsigned> position
                        window_positions[2 * position] = \
                            old_positions[2 * cell]
                        window_positions[2 * position + 1] = \
                            old_positions[2 * cell + 1]

                # Adapt the scale of the repulsion from window negatives
                if 10 <= epoch <= num_epochs / 2 and epoch % 10 == 0:
                    force_ratio = umap_negative_force_ratio(
                        window_positions, window, a, b, gamma, &state)
                    if scale_anchor < <float> 0:
                        scale_anchor = negative_scale / force_ratio
                    else:
                        raw_scale = min(max(scale_anchor * force_ratio,
                                            <float> 0.5 * scale_prior),
                                        <float> 2 * scale_prior)
                        log_change = min(max(log(raw_scale / negative_scale),
                                             log(<float> 0.9)),
                                         log(<float> 1.1))
                        negative_scale *= exp(<float> 0.1 * log_change)
            else:
                negative_positions = old_positions
            repulsion_constant = <float> 2 * gamma * b * negative_scale

            # Update each cell's position. Each iteration draws one cell's
            # samples, then calculates the previous cell's gradient from the
            # samples drawn in the previous iteration and takes its Adam step.
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                start = thread_offsets[thread_index]
                end = thread_offsets[thread_index + 1]
                for cell in range(start, end + 1):
                    if cell < end:
                        # Sample each of the cell's edges with probability
                        # proportional to its weight, then draw
                        # `negatives_per_edge` negative samples per sampled
                        # edge (rounded randomly) uniformly from the cell's
                        # window. The cell's random numbers come from the
                        # stream `srand(seed + epoch * num_cells + cell)`.
                        slot = (cell - start) & 1
                        neighbor_buffer = &sample_buffers[
                            (2 * <unsigned long long> thread_index + slot) *
                            buffer_size]
                        negative_buffer = neighbor_buffer + max_degree + 16
                        state = srand(seed + <unsigned long long> epoch *
                                      num_cells + cell)
                        num_accepted = 0
                        for edge in range(indptr[cell], indptr[cell + 1]):
                            # Drop the edge with probability
                            # `1 - weight / max_weight`
                            if weights[edge] < \
                                    max_weight * random_uniform(&state):
                                continue
                            neighbor = indices[edge]
                            # Prefetch the neighbor's position, so the cache
                            # miss overlaps with the previous cell's gradient
                            # calculation
                            PREFETCH(&old_positions[2 * neighbor])
                            neighbor_buffer[num_accepted] = neighbor
                            num_accepted = num_accepted + 1
                        num_negatives = <unsigned> (
                            num_accepted * negatives_per_edge +
                            random_uniform(&state))
                        window_start = 0 if not use_window else min(
                            rank[cell] - min(rank[cell], window / 2),
                            num_cells - window)
                        for sample in range(num_negatives):
                            negative = window_start + randint(window, &state)
                            # Same for the negative sample's position
                            PREFETCH(&negative_positions[2 * negative])
                            negative_buffer[sample] = negative
                        sample_counts[thread_index * 16 + 2 * slot] = \
                            num_accepted
                        sample_counts[thread_index * 16 + 2 * slot + 1] = \
                            num_negatives
                    if cell == start:
                        continue

                    # Calculate the previous cell's gradient: attract it
                    # towards each sampled neighbor and repel it from each
                    # negative sample. Like umap-learn, softly clip each
                    # repulsive gradient's norm to at most `clip_constant` via
                    # `tanh()`, and only update the cell's own position, not
                    # its neighbors' or negative samples'.
                    previous_cell = cell - 1
                    slot = (previous_cell - start) & 1
                    neighbor_buffer = &sample_buffers[
                        (2 * <unsigned long long> thread_index + slot) *
                        buffer_size]
                    negative_buffer = neighbor_buffer + max_degree + 16
                    num_accepted = sample_counts[thread_index * 16 + 2 * slot]
                    num_negatives = \
                        sample_counts[thread_index * 16 + 2 * slot + 1]
                    position_x = old_positions[2 * previous_cell]
                    position_y = old_positions[2 * previous_cell + 1]
                    gradient_x = 0
                    gradient_y = 0
                    for sample in range(num_accepted):
                        neighbor = neighbor_buffer[sample]
                        dx = position_x - old_positions[2 * neighbor]
                        dy = position_y - old_positions[2 * neighbor + 1]
                        dist_squared = dx * dx + dy * dy
                        if dist_squared > <float> 0:
                            dist_b_minus_1 = pow(dist_squared, b_minus_1)
                            scale = attraction_constant * dist_b_minus_1 / (
                                <float> 1 +
                                a * dist_b_minus_1 * dist_squared)
                            gradient_x = gradient_x + scale * dx
                            gradient_y = gradient_y + scale * dy
                    for sample in range(num_negatives):
                        negative = negative_buffer[sample]
                        dx = position_x - negative_positions[2 * negative]
                        dy = position_y - \
                            negative_positions[2 * negative + 1]
                        dist_squared = dx * dx + dy * dy
                        if dist_squared > <float> 0:
                            inverse_distance = <float> 1 / sqrt(dist_squared)
                            scale = repulsion_constant / (
                                (<float> 0.001 + dist_squared) *
                                (<float> 1 + a * pow(dist_squared, b)))
                            scale = clip_constant * inverse_distance * tanh(
                                scale * dist_squared * inverse_distance *
                                inverse_gamma)
                            gradient_x = gradient_x - scale * dx
                            gradient_y = gradient_y - scale * dy

                    # Take the Adam step
                    new_positions[2 * previous_cell] = position_x
                    new_positions[2 * previous_cell + 1] = position_y
                    update_cell_embedding_adam(
                        new_positions[2 * previous_cell], gradient_x,
                        momentum[2 * previous_cell],
                        velocity[2 * previous_cell], beta1, beta2,
                        learning_rate, epsilon)
                    update_cell_embedding_adam(
                        new_positions[2 * previous_cell + 1], gradient_y,
                        momentum[2 * previous_cell + 1],
                        velocity[2 * previous_cell + 1], beta1, beta2,
                        learning_rate, epsilon)
            old_positions, new_positions = new_positions, old_positions

            # Check for KeyboardInterrupts
            if epoch % 8 == 7:
                with gil:
                    PyErr_CheckSignals()


cdef inline float umap_select(const float[:, ::1] positions,
                              const unsigned dimension,
                              unsigned long long k,
                              unsigned long long[::1] histograms,
                              const unsigned num_threads) noexcept nogil:
    # Return the `k`th smallest value in column `dimension` of `positions`,
    # via an exact parallel radix select on the values' bits, mapped to
    # unsigned integers with the same order (by flipping the sign bit of
    # non-negative values and all bits of negative values). Each pass
    # histograms the next (at most) 11 bits of the values whose earlier bits
    # match those of the `k`th smallest value so far: bits 22-31, then 11-21,
    # then 0-10.

    cdef unsigned row, key, digit, other_thread, thread_index, start, end, \
        radix_pass, shift, prefix = 0, num_rows = positions.shape[0]
    cdef unsigned long long count
    cdef unsigned long long* histogram
    cdef float value

    for radix_pass in range(3):
        shift = 22 - 11 * radix_pass
        with parallel(num_threads=num_threads):
            thread_index = threadid()
            histogram = &histograms[<unsigned long long> thread_index * 2048]
            for digit in range(2048):
                histogram[digit] = 0
            start = <unsigned> (<unsigned long long> thread_index * num_rows /
                                num_threads)
            end = <unsigned> (<unsigned long long> (thread_index + 1) *
                              num_rows / num_threads)
            for row in range(start, end):
                # (Assigned first so Cython makes it thread-private)
                key = 0
                memcpy(&key, &positions[row, dimension], sizeof(unsigned))
                key = key ^ ((<unsigned> 0 - (key >> 31)) | 0x80000000u)
                if radix_pass == 0 or \
                        key >> (shift + 11) == prefix >> (shift + 11):
                    histogram[(key >> shift) & 2047] += 1
        for digit in range(2048):
            count = 0
            for other_thread in range(num_threads):
                count += histograms[
                    <unsigned long long> other_thread * 2048 + digit]
            if k < count:
                break
            k -= count
        prefix |= digit << shift
    prefix ^= ((prefix >> 31) - 1) | 0x80000000u
    memcpy(&value, &prefix, sizeof(float))
    return value


cdef inline (float, float) umap_robust_spreads(
        const float[:, ::1] positions,
        unsigned long long[::1] histograms,
        const unsigned num_threads) noexcept nogil:
    # Return the difference between the 95th and 5th percentiles of each
    # column of `positions`, or 1 if the difference is 0. Percentiles are
    # linearly interpolated, as in `np.quantile()`.

    cdef unsigned dimension, percentile
    cdef unsigned long long numerator, low, num_rows = positions.shape[0]
    cdef float fraction, quantile
    cdef float spreads[2]
    cdef float quantiles[2]
    for dimension in range(2):
        for percentile in range(2):
            numerator = (num_rows - 1) * <unsigned long long> (
                5 if percentile == 0 else 95)
            low = numerator / 100
            fraction = (numerator % 100) / <float> 100
            quantile = umap_select(positions, dimension, low, histograms,
                                   num_threads)
            if fraction > <float> 0:
                quantile += fraction * (umap_select(
                    positions, dimension, low + 1, histograms, num_threads) -
                    quantile)
            quantiles[percentile] = quantile
        spreads[dimension] = quantiles[1] - quantiles[0]
        if spreads[dimension] == <float> 0:
            spreads[dimension] = 1
    return spreads[0], spreads[1]


cdef inline void umap_tally(unsigned* lookup_table,
                            const unsigned lookup_table_mask,
                            unsigned* tally_keys,
                            float* tally_weights,
                            unsigned* tally_slots,
                            unsigned& num_tallied,
                            const unsigned key,
                            const float weight) noexcept nogil:
    # Add `weight` to the tally for `key`. Tallies are kept in order of first
    # appearance in `tally_keys` and `tally_weights`, and `lookup_table` maps
    # keys to their positions in `tally_keys`. It is an open-addressing table
    # with linear probing whose size is a power of 2 (`lookup_table_mask` + 1)
    # at least twice the number of distinct keys tallied at once; empty slots
    # hold `UINT_MAX`, and `tally_slots` records which slots are used so they
    # can be reset afterwards. (A dense array indexed by key, as in Leiden,
    # would need one array per thread with an entry for every possible key,
    # which is too much memory with many threads and many keys.)

    cdef unsigned slot = (key * 2654435761u) & lookup_table_mask
    while lookup_table[slot] != UINT_MAX and \
            tally_keys[lookup_table[slot]] != key:
        slot = (slot + 1) & lookup_table_mask
    if lookup_table[slot] == UINT_MAX:
        lookup_table[slot] = num_tallied
        tally_slots[num_tallied] = slot
        tally_keys[num_tallied] = key
        tally_weights[num_tallied] = weight
        num_tallied += 1
    else:
        tally_weights[lookup_table[slot]] += weight


# A node's weighted degree and index, for selecting hubs (as a `ctypedef`,
# since Cython does not allow ctuples as template arguments)
ctypedef (float, unsigned) umap_weighted_node


cdef inline bint umap_lighter_node(const umap_weighted_node first,
                                   const umap_weighted_node second) \
        noexcept nogil:
    # Whether `first` comes before `second` when sorting nodes by weighted
    # degree, breaking ties by index

    return first[0] < second[0] or \
        (first[0] == second[0] and first[1] < second[1])


cdef inline void umap_init_level(const long long[::1] indptr,
                                 const unsigned[::1] indices,
                                 const float[::1] label_weights,
                                 numa_vector[float]& summed_weights_buffer,
                                 const bint free_summed_weights,
                                 const float[:, ::1] anchor,
                                 float[:, ::1] result,
                                 const float a,
                                 const float b,
                                 const unsigned depth,
                                 unsigned num_parts,
                                 const unsigned num_epochs,
                                 const unsigned long long seed,
                                 const unsigned chunk_size,
                                 const unsigned num_threads):
    # One level of umap-learn 0.6's recursive initialization
    # (`label_propagation_init()`), writing the initial positions of this
    # level's graph's nodes to `result`. Graphs with at most 1024 nodes are
    # placed directly from `anchor`. Otherwise:
    # 1. Partition the graph into `num_parts` parts by label propagation from
    #    its `num_parts` highest-degree nodes, using `label_weights` as edge
    #    weights.
    # 2. Build the coarse graph between parts. Its summed weights are the
    #    total summed weights (`summed_weights_buffer`) between parts'
    #    members (which, below the top level, are themselves summed from the
    #    level above, so they always total the original graph's weights). Its
    #    label weights are the fuzzy union of the two directed "transition"
    #    weights between each pair of parts, i.e. the fraction of each part's
    #    summed weight that goes to the other part.
    # 3. Recursively initialize the coarse graph with about a quarter as many
    #    parts, then optimize it with a short UMAP run.
    # 4. Place each node halfway between its part's position and the
    #    weighted average of its neighbors' parts' positions, then center and
    #    rotate to best match `anchor`.
    # If `free_summed_weights` is true, `summed_weights_buffer` is freed as
    # soon as the coarse graph is built, instead of being kept while the
    # levels below run.

    cdef unsigned node, other_node, part, other_part, tally_index, label, \
        best_label, num_tallied, chunk, num_unlabeled, previous_unlabeled, \
        iteration, num_coarse_edges, thread_index, start, end, \
        num_hubs, num_steps, window, level_threads, lookup_table_size, \
        num_chunks, num_nodes = anchor.shape[0], max_degree = 0
    cdef long long edge, member, total, max_part_degree
    cdef unsigned long long position, state, \
        level_seed = seed + (<unsigned long long> depth << 40)
    cdef bint found, use_share = depth > 1
    cdef float weight, total_weight, max_vote, sum_x, sum_y, mean_x, \
        mean_y, anchor_sum_x, anchor_sum_y, anchor_mean_x, anchor_mean_y, \
        spread_x, spread_y, scale, transition_forward, transition_backward, \
        c00, c01, c10, c11, angle, cos_angle, sin_angle, x, y
    cdef umap_weighted_node threshold
    cdef unsigned* lookup_table
    cdef unsigned* tally_keys
    cdef unsigned* tally_slots
    cdef float* tally_weights
    cdef vector[vector[unsigned]] thread_stacks
    cdef uninitialized_vector[umap_weighted_node] keys
    cdef uninitialized_vector[float] chunk_sums
    cdef uninitialized_vector[long long] thread_num_coarse_edges
    cdef uninitialized_vector[unsigned long long] histograms_buffer
    cdef uninitialized_vector[unsigned] thread_offsets, thread_max_degrees, \
        thread_num_unlabeled, thread_num_hubs
    cdef numa_vector[long long] part_indptr_buffer, part_degrees_buffer, \
        coarse_indptr
    cdef numa_vector[unsigned long long] members_buffer
    cdef numa_vector[unsigned] labels_buffer, new_labels_buffer, \
        coarse_indices
    cdef numa_vector[float] weighted_degrees_buffer, coarse_summed, \
        coarse_label_weights, out_weights, coarse_anchor, coarse_layout
    cdef vector[uninitialized_vector[unsigned]] thread_lookup_tables, \
        thread_tally_keys, thread_tally_slots, thread_coarse_indices
    cdef vector[uninitialized_vector[float]] thread_tally_weights, \
        thread_coarse_weights
    cdef unsigned[::1] labels, new_labels
    cdef unsigned long long[::1] members
    cdef long long[::1] part_indptr, part_degrees
    cdef unsigned long long[::1] histograms
    cdef float[::1] summed_weights, weighted_degrees
    cdef float[:, ::1] coarse_positions

    # Base case: place nodes at their centered anchor positions, scaled to a
    # robust spread that grows with the number of nodes and the depth
    level_threads = min(num_threads, num_nodes)
    if num_nodes <= 1024:
        histograms_buffer.resize(<unsigned long long> level_threads * 2048)
        histograms = <unsigned long long[:level_threads * 2048]> \
            histograms_buffer.data()
        with nogil:
            mean_x, mean_y = umap_mean(anchor, chunk_size, level_threads)
            spread_x, spread_y = umap_robust_spreads(anchor, histograms,
                                                     level_threads)
        scale = log10(<float> num_nodes) * <float> 3 * \
            log2(<float> (depth + 1))
        for node in range(num_nodes):
            result[node, 0] = (anchor[node, 0] - mean_x) * (scale / spread_x)
            result[node, 1] = (anchor[node, 1] - mean_y) * (scale / spread_y)
        return

    num_parts = max(2, min(num_parts, num_nodes / 2))
    thread_offsets.resize(level_threads + 1)
    get_thread_offsets(indptr, thread_offsets.data(), level_threads)

    # Select the `num_parts` nodes with the highest weighted degree as hubs,
    # breaking ties by index: partially sort (weighted degree, index) pairs
    # to find the `num_parts`th largest, `threshold`
    weighted_degrees_buffer = numa_vector[float](level_threads)
    weighted_degrees_buffer.resize(num_nodes)
    weighted_degrees = <float[:num_nodes]> weighted_degrees_buffer.data()
    keys.resize(num_nodes)
    thread_max_degrees.resize(level_threads * 16)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        max_degree = 0
        for node in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            total_weight = 0
            for edge in range(indptr[node], indptr[node + 1]):
                total_weight = total_weight + label_weights[edge]
            weighted_degrees[node] = total_weight
            keys[node] = total_weight, node
            max_degree = \
                max(max_degree, <unsigned> (indptr[node + 1] - indptr[node]))
        thread_max_degrees[thread_index * 16] = max_degree
    max_degree = 0
    for thread_index in range(level_threads):
        max_degree = max(max_degree, thread_max_degrees[thread_index * 16])
    nth_element(keys.data(), keys.data() + (num_nodes - num_parts),
                keys.data() + num_nodes, umap_lighter_node)
    threshold = keys[num_nodes - num_parts]
    keys.clear()
    keys.shrink_to_fit()

    # Give each hub its own label, in ascending order of index, and leave
    # other nodes unlabeled (`UINT_MAX`): mark hubs with 0 and count each
    # thread's hubs, then number each thread's hubs starting after the
    # previous threads' hubs
    labels_buffer = numa_vector[unsigned_](level_threads)
    labels_buffer.resize(num_nodes)
    labels = <unsigned[:num_nodes]> labels_buffer.data()
    thread_num_hubs.resize(level_threads * 16)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        start = <unsigned> (<unsigned long long> thread_index * num_nodes /
                            level_threads)
        end = <unsigned> (<unsigned long long> (thread_index + 1) *
                          num_nodes / level_threads)
        num_hubs = 0
        for node in range(start, end):
            if not umap_lighter_node((weighted_degrees[node], node),
                                     threshold):
                labels[node] = 0
                num_hubs = num_hubs + 1
            else:
                labels[node] = UINT_MAX
        thread_num_hubs[thread_index * 16] = num_hubs
    num_hubs = 0
    for thread_index in range(level_threads):
        label = thread_num_hubs[thread_index * 16]
        thread_num_hubs[thread_index * 16] = num_hubs
        num_hubs += label
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        start = <unsigned> (<unsigned long long> thread_index * num_nodes /
                            level_threads)
        end = <unsigned> (<unsigned long long> (thread_index + 1) *
                          num_nodes / level_threads)
        label = thread_num_hubs[thread_index * 16]
        for node in range(start, end):
            if labels[node] == 0:
                labels[node] = label
                label = label + 1
    weighted_degrees_buffer.clear()
    weighted_degrees_buffer.shrink_to_fit()

    # Propagate labels for up to 100 iterations. Each unlabeled node adopts
    # the label with the largest total edge weight among its neighbors (the
    # last one among ties, in order of first appearance), if that total is at
    # least 1 (at the top level) or 10% of the node's total edge weight
    # (below the top level, where weights are normalized). Stop early if no
    # nodes became labeled over the last 5 iterations.
    new_labels_buffer = numa_vector[unsigned_](level_threads)
    new_labels_buffer.resize(num_nodes)
    new_labels = <unsigned[:num_nodes]> new_labels_buffer.data()
    lookup_table_size = 16
    while lookup_table_size < 2 * max_degree:
        lookup_table_size *= 2
    thread_lookup_tables.resize(level_threads)
    thread_tally_keys.resize(level_threads)
    thread_tally_weights.resize(level_threads)
    thread_tally_slots.resize(level_threads)
    thread_num_unlabeled.resize(level_threads * 16)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        thread_lookup_tables[thread_index].assign(lookup_table_size, UINT_MAX)
        thread_tally_keys[thread_index].resize(max_degree)
        thread_tally_weights[thread_index].resize(max_degree)
        thread_tally_slots[thread_index].resize(max_degree)
    previous_unlabeled = num_nodes - num_parts
    with nogil:
        for iteration in range(100):
            with parallel(num_threads=level_threads):
                thread_index = threadid()
                lookup_table = thread_lookup_tables[thread_index].data()
                tally_keys = thread_tally_keys[thread_index].data()
                tally_weights = thread_tally_weights[thread_index].data()
                tally_slots = thread_tally_slots[thread_index].data()
                num_unlabeled = 0
                for node in range(thread_offsets[thread_index],
                                  thread_offsets[thread_index + 1]):
                    if labels[node] != UINT_MAX:
                        new_labels[node] = labels[node]
                        continue
                    total_weight = 0
                    num_tallied = 0
                    for edge in range(indptr[node], indptr[node + 1]):
                        # Prefetch the label of the neighbor 8 edges ahead
                        if edge + 8 < indptr[num_nodes]:
                            PREFETCH(&labels[indices[edge + 8]])
                        weight = label_weights[edge]
                        total_weight = total_weight + weight
                        label = labels[indices[edge]]
                        if label != UINT_MAX:
                            umap_tally(lookup_table, lookup_table_size - 1,
                                       tally_keys, tally_weights, tally_slots,
                                       num_tallied, label, weight)
                    max_vote = \
                        <float> 0.1 * total_weight if use_share else <float> 1
                    best_label = UINT_MAX
                    for tally_index in range(num_tallied):
                        lookup_table[tally_slots[tally_index]] = UINT_MAX
                        if tally_weights[tally_index] >= max_vote:
                            max_vote = tally_weights[tally_index]
                            best_label = tally_keys[tally_index]
                    new_labels[node] = best_label
                    if best_label == UINT_MAX:
                        num_unlabeled = num_unlabeled + 1
                thread_num_unlabeled[thread_index * 16] = num_unlabeled
            labels, new_labels = new_labels, labels

            # Check for KeyboardInterrupts
            if iteration % 8 == 7:
                with gil:
                    PyErr_CheckSignals()

            if iteration % 5 == 0:
                num_unlabeled = 0
                for thread_index in range(level_threads):
                    num_unlabeled += thread_num_unlabeled[thread_index * 16]
                if num_unlabeled == 0 or num_unlabeled == previous_unlabeled:
                    break
                previous_unlabeled = num_unlabeled

    # Give each remaining unlabeled node the label of the first labeled node
    # found by a depth-first search of up to 100 steps, or a random label if
    # none is found. Unlike umap-learn, which searches nodes in order and lets
    # later searches find labels assigned by earlier ones, every search only
    # finds labels assigned by label propagation, so nodes can be searched in
    # parallel. Node `node`'s random label comes from the stream
    # `srand(level_seed - 2 - node)`.
    thread_stacks.resize(level_threads)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        for node in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            new_labels[node] = labels[node]
            if labels[node] != UINT_MAX:
                continue
            thread_stacks[thread_index].clear()
            thread_stacks[thread_index].push_back(node)
            found = False
            num_steps = 0
            while not found and num_steps < 100 and \
                    thread_stacks[thread_index].size() > 0:
                num_steps = num_steps + 1
                other_node = thread_stacks[thread_index].back()
                thread_stacks[thread_index].pop_back()
                for edge in range(indptr[other_node], indptr[other_node + 1]):
                    if labels[indices[edge]] != UINT_MAX:
                        new_labels[node] = labels[indices[edge]]
                        found = True
                        break
                    thread_stacks[thread_index].push_back(indices[edge])
            if not found:
                state = srand(level_seed - 2 - node)
                new_labels[node] = randint(num_parts, &state)
    labels, new_labels = new_labels, labels
    thread_stacks.clear()

    # Group nodes by part (in ascending order of index within each part), and
    # cumsum each part's total degree, to balance work across threads
    umap_group_by_key(labels, num_parts, part_indptr_buffer, members_buffer,
                      level_threads)
    part_indptr = <long long[:num_parts + 1]> part_indptr_buffer.data()
    members = <unsigned long long[:num_nodes]> members_buffer.data()
    part_degrees_buffer = numa_vector[long_long_](level_threads)
    part_degrees_buffer.resize(num_parts + 1)
    part_degrees = <long long[:num_parts + 1]> part_degrees_buffer.data()
    with nogil:
        for part in prange(num_parts, num_threads=level_threads):
            total = 0
            for member in range(part_indptr[part], part_indptr[part + 1]):
                total = total + indptr[members[member] + 1] - \
                    indptr[members[member]]
            part_degrees[part + 1] = total
    for part in range(num_parts):
        part_degrees[part + 1] += part_degrees[part]
    PyErr_CheckSignals()

    # Build the coarse graph, omitting edges within parts: for each part,
    # tally the summed weights of the edges from its members to each other
    # part, in order of first appearance, and the part's total summed weight
    # to other parts (`out_weights`). Each thread sizes its tallies for the
    # largest part in its range (some parts are much larger than the rest)
    # and appends its parts' rows to its own buffers, which are concatenated
    # below.
    summed_weights = <float[:indices.shape[0]]> summed_weights_buffer.data()
    thread_offsets.resize(level_threads + 1)
    get_thread_offsets(part_degrees, thread_offsets.data(), level_threads)
    coarse_indptr = numa_vector[long_long_](level_threads)
    coarse_indptr.resize(num_parts + 1)
    out_weights = numa_vector[float](level_threads)
    out_weights.resize(num_parts)
    thread_coarse_indices.resize(level_threads)
    thread_coarse_weights.resize(level_threads)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        max_part_degree = 0
        for part in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            max_part_degree = max(max_part_degree,
                                  part_degrees[part + 1] - part_degrees[part])
        max_part_degree = min(max_part_degree, <long long> num_parts)
        lookup_table_size = 16
        while lookup_table_size < 2 * max_part_degree:
            lookup_table_size = lookup_table_size * 2
        thread_lookup_tables[thread_index].assign(lookup_table_size, UINT_MAX)
        thread_tally_keys[thread_index].resize(max_part_degree)
        thread_tally_weights[thread_index].resize(max_part_degree)
        thread_tally_slots[thread_index].resize(max_part_degree)
        lookup_table = thread_lookup_tables[thread_index].data()
        tally_keys = thread_tally_keys[thread_index].data()
        tally_weights = thread_tally_weights[thread_index].data()
        tally_slots = thread_tally_slots[thread_index].data()
        for part in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            num_tallied = 0
            total_weight = 0
            for member in range(part_indptr[part], part_indptr[part + 1]):
                node = members[member]
                for edge in range(indptr[node], indptr[node + 1]):
                    other_part = labels[indices[edge]]
                    if other_part == part:
                        continue
                    weight = summed_weights[edge]
                    total_weight = total_weight + weight
                    umap_tally(lookup_table, lookup_table_size - 1,
                               tally_keys, tally_weights, tally_slots,
                               num_tallied, other_part, weight)
            for tally_index in range(num_tallied):
                lookup_table[tally_slots[tally_index]] = UINT_MAX
                thread_coarse_indices[thread_index].push_back(
                    tally_keys[tally_index])
                thread_coarse_weights[thread_index].push_back(
                    tally_weights[tally_index])
            coarse_indptr[part + 1] = num_tallied
            out_weights[part] = total_weight
    thread_lookup_tables.clear()
    thread_tally_keys.clear()
    thread_tally_weights.clear()
    thread_tally_slots.clear()
    if free_summed_weights:
        summed_weights_buffer.clear()
        summed_weights_buffer.shrink_to_fit()

    # Concatenate the threads' buffers into the coarse graph, cumsumming its
    # row lengths
    thread_num_coarse_edges.resize(level_threads + 1)
    total = 0
    for thread_index in range(level_threads):
        thread_num_coarse_edges[thread_index] = total
        total += thread_coarse_indices[thread_index].size()
    num_coarse_edges = max(1, <unsigned> total)
    coarse_indices = numa_vector[unsigned_](level_threads)
    coarse_indices.resize(num_coarse_edges)
    coarse_summed = numa_vector[float](level_threads)
    coarse_summed.resize(num_coarse_edges)
    coarse_indptr[0] = 0
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        total = thread_num_coarse_edges[thread_index]
        for part in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            total = total + coarse_indptr[part + 1]
            coarse_indptr[part + 1] = total
        for position in range(thread_coarse_indices[thread_index].size()):
            edge = thread_num_coarse_edges[thread_index] + <long long> position
            coarse_indices[edge] = \
                thread_coarse_indices[thread_index][position]
            coarse_summed[edge] = thread_coarse_weights[thread_index][position]
    thread_coarse_indices.clear()
    thread_coarse_weights.clear()

    # Calculate the coarse graph's label weights: the fuzzy union of the
    # transition weights `W_pq / out_p` and `W_pq / out_q` for each edge
    # between parts `p` and `q` with summed weight `W_pq`. Also calculate the
    # coarse anchor: the mean of each part's members' anchors. (This is a
    # separate pass from the one above so that the threads' buffers are freed
    # before the label weights are allocated, lowering peak memory.)
    coarse_label_weights = numa_vector[float](level_threads)
    coarse_label_weights.resize(num_coarse_edges)
    coarse_anchor = numa_vector[float](level_threads)
    coarse_anchor.resize(2 * num_parts)
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        for part in range(thread_offsets[thread_index],
                          thread_offsets[thread_index + 1]):
            for edge in range(coarse_indptr[part], coarse_indptr[part + 1]):
                transition_forward = coarse_summed[edge] / out_weights[part]
                transition_backward = \
                    coarse_summed[edge] / out_weights[coarse_indices[edge]]
                coarse_label_weights[edge] = min(max(
                    transition_forward + transition_backward -
                    transition_forward * transition_backward, <float> 0),
                    <float> 1)
            sum_x = 0
            sum_y = 0
            for member in range(part_indptr[part], part_indptr[part + 1]):
                sum_x = sum_x + anchor[members[member], 0]
                sum_y = sum_y + anchor[members[member], 1]
            coarse_anchor[2 * part] = sum_x / <float> (
                part_indptr[part + 1] - part_indptr[part])
            coarse_anchor[2 * part + 1] = sum_y / <float> (
                part_indptr[part + 1] - part_indptr[part])
    PyErr_CheckSignals()

    # Recursively initialize the coarse graph, with exponents of the curve
    # parameters closer to 1, a quarter as many parts, and √2 times as many
    # epochs
    coarse_layout = numa_vector[float](level_threads)
    coarse_layout.resize(2 * num_parts)
    coarse_positions = <float[:num_parts, :2]> coarse_layout.data()
    umap_init_level(
        <long long[:num_parts + 1]> coarse_indptr.data(),
        <unsigned[:num_coarse_edges]> coarse_indices.data(),
        <float[:num_coarse_edges]> coarse_label_weights.data(),
        coarse_summed, True, <float[:num_parts, :2]> coarse_anchor.data(),
        coarse_positions, pow(a, <float> 0.25), pow(b, <float> 0.25),
        depth + 1, num_parts / 4, <unsigned> (num_epochs * sqrt(<float> 2)),
        seed, chunk_size, num_threads)

    # Optimize the coarse graph's layout with a repulsion strength of 4, one
    # negative sample per sampled edge, and windows of half the coarse graph
    # (rounded like `np.round()`). The coarse anchor is no longer needed, so
    # use it as `umap_layout()`'s second buffer.
    window = num_parts / 2
    if num_parts % 2 == 1 and window % 2 == 1:
        window += 1
    umap_layout(<float[:2 * num_parts]> coarse_layout.data(),
                <float[:2 * num_parts]> coarse_anchor.data(),
                <long long[:num_parts + 1]> coarse_indptr.data(),
                <unsigned[:num_coarse_edges]> coarse_indices.data(),
                <float[:num_coarse_edges]> coarse_label_weights.data(),
                num_epochs, a, b, 4, <float> 0.5, 1, max(1, window),
                num_parts / 4 > 1024, level_seed, num_threads)
    if num_epochs & 1:
        coarse_positions = <float[:num_parts, :2]> coarse_anchor.data()

    # Place each node halfway between its part's position and the weighted
    # average of its neighbors' parts' positions, and sum each chunk of
    # `chunk_size` nodes' positions and anchors. Give each thread a
    # contiguous range of whole chunks with about the same number of edges,
    # by rounding the boundaries of edge-balanced ranges of nodes to the
    # nearest chunk boundary.
    num_chunks = (num_nodes + chunk_size - 1) / chunk_size
    chunk_sums.resize(4 * num_chunks)
    get_thread_offsets(indptr, thread_offsets.data(), level_threads)
    for thread_index in range(1, level_threads):
        thread_offsets[thread_index] = <unsigned> min(
            (thread_offsets[thread_index] +
             <unsigned long long> chunk_size / 2) / chunk_size,
            <unsigned long long> num_chunks)
    thread_offsets[level_threads] = num_chunks
    with nogil, parallel(num_threads=level_threads):
        thread_index = threadid()
        for chunk in range(thread_offsets[thread_index],
                           thread_offsets[thread_index + 1]):
            sum_x = 0
            sum_y = 0
            anchor_sum_x = 0
            anchor_sum_y = 0
            for node in range(chunk * chunk_size,
                              min((chunk + 1) * chunk_size, num_nodes)):
                x = 0
                y = 0
                total_weight = 0
                for edge in range(indptr[node], indptr[node + 1]):
                    # Prefetch the label of the neighbor 8 edges ahead
                    if edge + 8 < indptr[num_nodes]:
                        PREFETCH(&labels[indices[edge + 8]])
                    weight = label_weights[edge]
                    total_weight = total_weight + weight
                    part = labels[indices[edge]]
                    x = x + weight * coarse_positions[part, 0]
                    y = y + weight * coarse_positions[part, 1]
                if total_weight > <float> 0:
                    x = x / total_weight
                    y = y / total_weight
                part = labels[node]
                x = <float> 0.5 * (x + coarse_positions[part, 0])
                y = <float> 0.5 * (y + coarse_positions[part, 1])
                result[node, 0] = x
                result[node, 1] = y
                sum_x = sum_x + x
                sum_y = sum_y + y
                anchor_sum_x = anchor_sum_x + anchor[node, 0]
                anchor_sum_y = anchor_sum_y + anchor[node, 1]
            chunk_sums[4 * chunk] = sum_x
            chunk_sums[4 * chunk + 1] = sum_y
            chunk_sums[4 * chunk + 2] = anchor_sum_x
            chunk_sums[4 * chunk + 3] = anchor_sum_y
    with nogil:
        mean_x = 0
        mean_y = 0
        anchor_mean_x = 0
        anchor_mean_y = 0
        for chunk in range(num_chunks):
            mean_x += chunk_sums[4 * chunk]
            mean_y += chunk_sums[4 * chunk + 1]
            anchor_mean_x += chunk_sums[4 * chunk + 2]
            anchor_mean_y += chunk_sums[4 * chunk + 3]
        mean_x /= num_nodes
        mean_y /= num_nodes
        anchor_mean_x /= num_nodes
        anchor_mean_y /= num_nodes

        # Center the positions and rotate them to best match `anchor`, via
        # the orthogonal Procrustes rotation, which in 2D has the closed form
        # `atan2(c10 - c01, c00 + c11)` in terms of the cross-covariance
        # matrix `c` between the centered positions and the centered anchor.
        # (Centering and rotating takes a separate pass from calculating `c`,
        # since the rotation depends on `c` summed over all nodes.)
        for chunk in prange(num_chunks, num_threads=level_threads):
            c00 = 0
            c01 = 0
            c10 = 0
            c11 = 0
            for node in range(chunk * chunk_size,
                              min((chunk + 1) * chunk_size, num_nodes)):
                x = result[node, 0] - mean_x
                y = result[node, 1] - mean_y
                c00 = c00 + x * (anchor[node, 0] - anchor_mean_x)
                c01 = c01 + x * (anchor[node, 1] - anchor_mean_y)
                c10 = c10 + y * (anchor[node, 0] - anchor_mean_x)
                c11 = c11 + y * (anchor[node, 1] - anchor_mean_y)
            chunk_sums[4 * chunk] = c00
            chunk_sums[4 * chunk + 1] = c01
            chunk_sums[4 * chunk + 2] = c10
            chunk_sums[4 * chunk + 3] = c11
        c00 = 0
        c01 = 0
        c10 = 0
        c11 = 0
        for chunk in range(num_chunks):
            c00 += chunk_sums[4 * chunk]
            c01 += chunk_sums[4 * chunk + 1]
            c10 += chunk_sums[4 * chunk + 2]
            c11 += chunk_sums[4 * chunk + 3]
        angle = atan2(c10 - c01, c00 + c11)
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        for node in prange(num_nodes, num_threads=level_threads):
            x = result[node, 0] - mean_x
            y = result[node, 1] - mean_y
            result[node, 0] = x * cos_angle + y * sin_angle
            result[node, 1] = y * cos_angle - x * sin_angle


def umap(float[:, ::1] embedding,
         float[:, ::1] embedding_buffer,
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
         const unsigned chunk_size,
         str neighbors_key,
         unsigned num_threads):
    # `embedding` and `embedding_buffer` swap roles every iteration, so after
    # an odd number of iterations, the final embedding is in
    # `embedding_buffer`
    cdef unsigned cell, num_cells = embedding.shape[0]
    cdef unsigned long long num_edges
    cdef bint too_large = False, self_neighbors = False
    cdef float min_x, max_x, min_y, max_y, scale_x, scale_y, mean_x, \
        mean_y, spread_x, spread_y, scale
    cdef str error_message
    cdef numa_vector[long long] indptr_buffer
    cdef numa_vector[unsigned] indices_buffer
    cdef numa_vector[float] weights_buffer
    cdef uninitialized_vector[unsigned long long] histograms_buffer
    cdef long long[::1] indptr
    cdef unsigned[::1] indices
    cdef float[::1] weights, anchor_flat
    cdef float[:, ::1] anchor
    cdef unsigned long long[::1] histograms

    num_threads = min(num_threads, num_cells)

    # Build the fuzzy simplicial set
    umap_fuzzy_graph(neighbors, distances, num_pruning_iterations,
                     indptr_buffer, indices_buffer, weights_buffer, too_large,
                     self_neighbors, num_threads)

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
    num_edges = indices_buffer.size()
    indptr = <long long[:num_cells + 1]> indptr_buffer.data()
    indices = <unsigned[:num_edges]> indices_buffer.data()
    weights = <float[:num_edges]> weights_buffer.data()

    # Use the first two PCs, min-max scaled to [0, 10], as the anchor that
    # orients each level of the recursive initialization. `embedding_buffer`
    # is not needed until the final layout, so store the anchor in it.
    anchor = embedding_buffer
    anchor_flat = <float[:2 * num_cells]> &embedding_buffer[0, 0]
    with nogil:
        for cell in prange(num_cells, num_threads=num_threads):
            anchor[cell, 0] = PCs[cell, 0]
            anchor[cell, 1] = PCs[cell, 1]
        min_x, max_x = umap_projection_range(anchor_flat, 1, 0, num_threads)
        min_y, max_y = umap_projection_range(anchor_flat, 0, 1, num_threads)
        scale_x = <float> 10 / (max_x - min_x) if max_x > min_x else \
            <float> 10
        scale_y = <float> 10 / (max_y - min_y) if max_y > min_y else \
            <float> 10
        for cell in prange(num_cells, num_threads=num_threads):
            anchor[cell, 0] = (anchor[cell, 0] - min_x) * scale_x
            anchor[cell, 1] = (anchor[cell, 1] - min_y) * scale_y

    # Initialize recursively, starting with a quarter as many parts as cells
    # (but at least 1024), 50 epochs, and the curve parameters' fourth roots.
    # The graph's weights are still needed for the final layout, so do not
    # free them.
    umap_init_level(indptr, indices, weights, weights_buffer, False, anchor,
                    embedding, pow(a, <float> 0.25), pow(b, <float> 0.25), 1,
                    max(1024, num_cells / 4), 50, seed, chunk_size,
                    num_threads)

    # Center the initialization and scale it so the geometric mean of its two
    # robust spreads (the differences between their 95th and 5th
    # percentiles, which do not depend on centering) is
    # `3 * log10(num_cells) * log2(gamma + 1)`
    histograms_buffer.resize(<unsigned long long> num_threads * 2048)
    histograms = <unsigned long long[:num_threads * 2048]> \
        histograms_buffer.data()
    with nogil:
        mean_x, mean_y = umap_mean(embedding, chunk_size, num_threads)
        spread_x, spread_y = umap_robust_spreads(embedding, histograms,
                                                 num_threads)
        scale = <float> 3 * log10(<float> num_cells) * \
            log2(gamma + <float> 1) / sqrt(spread_x * spread_y)
        for cell in prange(num_cells, num_threads=num_threads):
            embedding[cell, 0] = (embedding[cell, 0] - mean_x) * scale
            embedding[cell, 1] = (embedding[cell, 1] - mean_y) * scale

    # Optimize
    umap_layout(<float[:2 * num_cells]> &embedding[0, 0],
                <float[:2 * num_cells]> &embedding_buffer[0, 0], indptr,
                indices, weights, num_iterations, a, b, gamma, alpha,
                negative_sample_rate, negative_window, True, seed, num_threads)
