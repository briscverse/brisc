# Functionality for k-means clustering

from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libc.float cimport FLT_MAX
from libc.string cimport memcpy
from libcpp.algorithm cimport fill
from libcpp.vector cimport vector
from .cyutils cimport min_heap_replace_top, min_heap_sort, partial_distances, \
    randint, random_uniform, sgemm, srand, uninitialized_vector


cdef inline void kmeans_random_init(const float[:, ::1] X,
                                    float[:, ::1] centroids,
                                    const unsigned long long num_cells,
                                    const unsigned num_clusters,
                                    const unsigned long long seed):
    # Initialize centroids with random cells. Use a bitset to efficiently keep
    # track of which cells have been sampled already, to avoid sampling the
    # same cell twice.

    cdef unsigned i, j
    cdef unsigned long long word_index, bit_index, state = srand(seed)
    cdef vector[unsigned long long] bitset_buffer
    bitset_buffer.resize((num_cells + 63) / 64)

    cdef unsigned long long[::1] bitset = \
        <unsigned long long[:(num_cells + 63) / 64]> bitset_buffer.data()

    for i in range(num_clusters):
        while True:
            j = randint(num_cells, &state)
            word_index = j >> 6
            bit_index = j & 63
            if not bitset[word_index] & (1LL << bit_index):
                bitset[word_index] |= 1LL << bit_index
                centroids[i] = X[j]
                break


cdef inline void kmeans_barbar_init(
        const float[:, ::1] X,
        float[:, ::1] centroids,
        float[::1] min_distances,
        const unsigned num_init_iterations,
        const float oversampling_factor,
        const unsigned long long num_cells,
        const unsigned num_clusters,
        const unsigned num_dimensions,
        const unsigned chunk_size,
        const unsigned long long seed):
    cdef unsigned i, j, k, random_cell, c0, c1, iteration, \
        num_newly_selected_cells, num_previously_selected_cells, \
        selected_cell, chunk_index, start, chunk_num_cells, \
        num_selected_cells, end, cluster_index, selected_centroid, \
        num_chunks = (num_cells + chunk_size - 1) / chunk_size
    cdef int best_selected_cell
    cdef unsigned long long state
    cdef float cost, difference, distance, norm, l_over_cost, chunk_cost, \
        min_distance, inverse_cost, probability, alpha = -2, beta = 1, \
        l = oversampling_factor * num_clusters
    cdef char transA = b'T', transB = b'N'
    cdef uninitialized_vector[float] X_norms_buffer, newly_selected_X_buffer, \
        distances_buffer, chunk_costs_buffer
    cdef vector[unsigned] best_selected_cells_buffer, selected_cells, \
        selected_cell_weights_buffer
    cdef uninitialized_vector[unsigned] centroid_indices_buffer
    cdef float[::1] X_norms, chunk_costs
    cdef float[:, ::1] newly_selected_X, distances
    cdef unsigned[::1] best_selected_cells, selected_cell_weights, \
        centroid_indices

    # Reserve 25% more than the expected number to be safe
    selected_cells.reserve(<unsigned>(<float> 1.25 * num_init_iterations * l))

    # Sample a random cell from `X`, and add it to our list of selected cells.
    # This will constitute a shortlist from which we will select the final
    # centroids to initialize k-means with.
    state = srand(seed - 1)
    random_cell = randint(num_cells, &state)
    selected_cells.push_back(random_cell)

    # Calculate the (squared Euclidean) distance from each cell to the random
    # cell, storing it in `min_distances`. In the same loop, also calculate the
    # squared L2 norm of each cell, ||X||².
    X_norms_buffer.resize(num_cells)
    X_norms = <float[:num_cells]> X_norms_buffer.data()
    for i in range(num_cells):
        difference = X[i, 0] - X[random_cell, 0]
        distance = difference * difference
        norm = X[i, 0] * X[i, 0]
        for j in range(1, num_dimensions):
            difference = X[i, j] - X[random_cell, j]
            distance += difference * difference
            norm += X[i, j] * X[i, j]
        min_distances[i] = distance
        X_norms[i] = norm

    # Sum the `min_distances` separately at the end, to ensure deterministic
    # parallelism
    cost = 0
    for i in range(num_cells):
        cost += min_distances[i]
    if not cost:
        error_message = \
            f'all cells have the same principal component loadings'
        raise ValueError(error_message)

    # Sample each cell with probability `l * min_distances[i] / cost`.
    # Set `min_distances` to zero for sampled cells to reflect that
    # each sampled cell's nearest centroid candidate is now itself.
    l_over_cost = l / cost
    for i in range(num_cells):
        state = srand(seed + i)
        if l_over_cost * min_distances[i] >= random_uniform(&state):
            selected_cells.push_back(i)
            min_distances[i] = 0

    # Keep track of how many cells were selected, newly selected on this past
    # iteration (for this first iteration, all cells except the first random
    # cell), and selected on a previous iteration (just the first random cell).
    num_selected_cells = selected_cells.size()
    num_newly_selected_cells = num_selected_cells - 1
    num_previously_selected_cells = 1

    # `best_selected_cells` maps each cell to the index in `selected_cells`
    # with the cell's nearest candidate centroid selected so far. For each
    # newly selected cell, set `best_selected_cells[selected_cell]` to the
    # cell's index, to reflect that the cell's nearest centroid candidate is
    # now itself. `best_selected_cells` starts off at 0 for non-newly selected
    # cells, i.e. they still have the initial random cell (`selected_cells[0]`)
    # listed as their closest centroid candidate.
    best_selected_cells_buffer.resize(num_cells)
    best_selected_cells = <unsigned[:num_cells]> \
        best_selected_cells_buffer.data()
    for i in range(num_previously_selected_cells, num_selected_cells):
        selected_cell = selected_cells[i]
        best_selected_cells[selected_cell] = i

    # For each remaining iteration...
    chunk_costs_buffer.resize(num_chunks)
    chunk_costs = <float[:num_chunks]> chunk_costs_buffer.data()
    iteration = 1
    while True:
        # For very small datasets, it's possible no cells are selected on a
        # given iteration. In that case, skip the distance and cost
        # calculations and go straight to sampling more cells.
        if num_newly_selected_cells > 0:
            # Copy newly selected cells into a temporary buffer
            newly_selected_X_buffer.resize(
                num_newly_selected_cells * num_dimensions)
            newly_selected_X = \
                <float[:num_newly_selected_cells, :num_dimensions]> \
                newly_selected_X_buffer.data()
            for i in range(num_newly_selected_cells):
                selected_cell = \
                    selected_cells[num_previously_selected_cells + i]
                newly_selected_X[i, :] = X[selected_cell, :]

            # Update each cell's nearest candidate centroid selected so far
            # (`best_selected_cells`) and distance to this candidate centroid
            # (`min_distances`) to account for the newly selected cells. We
            # only have to compute distances from each cell to the newly
            # selected cells, rather than to all selected cells. Use the
            # identity:
            # ||X - C||² = ||X||² - 2 * X.dot(C.T) + ||C||²
            # but add the ||X||² at the end since the minimum distance for a
            # given cell doesn't depend on ||X||².
            distances_buffer.resize(chunk_size * num_newly_selected_cells)
            distances = <float[:chunk_size, :num_newly_selected_cells]> \
                distances_buffer.data()
            for chunk_index in range(num_chunks):
                chunk_cost = 0
                start = chunk_index * chunk_size
                chunk_num_cells = num_cells - start \
                    if chunk_index == num_chunks - 1 else chunk_size
                for i in range(chunk_num_cells):
                    for j in range(num_newly_selected_cells):
                        # distances = ||C||²
                        selected_cell = \
                            selected_cells[num_previously_selected_cells + j]
                        distances[i, j] = X_norms[selected_cell]

                # distances -= 2 * X.dot(C.T)
                sgemm(transA, transB, num_newly_selected_cells,
                      chunk_num_cells, num_dimensions, alpha,
                      &newly_selected_X[0, 0], num_dimensions, &X[start, 0],
                      num_dimensions, beta, &distances[0, 0],
                      num_newly_selected_cells)

                # distances += ||X||², and find the best distance. As an
                # optimization, to avoid having to add `X_norms[start + i]`
                # (i.e. ||X||²) to `min_distance` and
                # `num_previously_selected_cells` to `best_selected_cell` to
                # every candidate inside the inner loop (which would be
                # necessary to make them comparable to the previous minimum
                # distance and best selected cell), subtract these terms at the
                # start and add them back at the end. (`best_selected_cell` is
                # declared as a signed integer to allow for underflow.)
                for i in range(chunk_num_cells):
                    min_distance = \
                        min_distances[start + i] - X_norms[start + i]
                    best_selected_cell = best_selected_cells[start + i] - \
                        num_previously_selected_cells
                    for j in range(num_newly_selected_cells):
                        distance = distances[i, j]
                        if distance < min_distance:
                            min_distance = distance
                            best_selected_cell = j
                    min_distance += X_norms[start + i]
                    min_distances[start + i] = min_distance
                    best_selected_cells[start + i] = best_selected_cell + \
                        num_previously_selected_cells
                    chunk_cost += min_distance
                chunk_costs[chunk_index] = chunk_cost

            # Sum costs for each chunk separately at the end, to ensure
            # deterministic parallelism
            cost = 0
            for chunk_index in range(num_chunks):
                cost += chunk_costs[chunk_index]

        # Sample each cell `i` with probability `l * min_distances[i] / cost`.
        # Note that since we set `min_distances` to 0 for cells that were
        # sampled, we will avoid sampling them twice. As before, set
        # `min_distances` to zero for newly sampled cells, to reflect that each
        # sampled cell's nearest centroid candidate is now itself.
        l_over_cost = l / cost
        for i in range(num_cells):
            state = srand(seed + i * iteration)
            probability = random_uniform(&state)
            if l_over_cost * min_distances[i] >= probability:
                selected_cells.push_back(i)
                min_distances[i] = 0
                best_selected_cells[i] = i

        # Update the number of selected, previously selected and newly selected
        # cells
        num_previously_selected_cells = num_selected_cells
        num_selected_cells = selected_cells.size()
        num_newly_selected_cells = \
            num_selected_cells - num_previously_selected_cells

        # As before, set `best_selected_cells` for each newly selected cell to
        # the cell's index, to reflect that the cell's nearest centroid
        # candidate is now itself
        for i in range(num_previously_selected_cells, num_selected_cells):
            selected_cell = selected_cells[i]
            best_selected_cells[selected_cell] = i

        # Check for KeyboardInterrupts after each iteration
        PyErr_CheckSignals()

        # Check if progress has frozen (i.e. `num_newly_selected_cells == 0`
        # when `cost == 0`), indicating there are fewer distinct cells than
        # clusters requested
        if num_newly_selected_cells == 0 and not cost:
            error_message = (
                f'num_clusters ({num_clusters:,}) is greater than the number '
                f'of cells with distinct principal component loadings; '
                f'decrease num_clusters')
            raise ValueError(error_message)

        # Stop after `num_init_iterations` iterations, unless we have found
        # fewer than `num_clusters` centroid candidates so far. Also stop if we
        # have already selected every cell as a centroid candidate, which can
        # happen if the dataset is very small.
        iteration += 1
        if iteration == num_init_iterations and \
                num_selected_cells >= num_clusters or \
                num_selected_cells == num_cells:
            break

    # Now we are done selecting cells as candidate centroids and need to
    # whittle down to the final centroids. Get the weight for each selected
    # cell: the number of cells that are closer to the selected cell than to
    # any other selected cell.
    selected_cell_weights_buffer.resize(num_selected_cells)
    selected_cell_weights = <unsigned[:num_selected_cells]> \
        selected_cell_weights_buffer.data()
    for i in range(num_cells):
        best_selected_cell = best_selected_cells[i]
        selected_cell_weights[best_selected_cell] += 1

    # Run k-means++ to select `num_clusters` of the selected cells as the
    # centroids, using `selected_cell_weights` as weights. Start by selecting a
    # random cell from our selected cells as the first centroid.
    state = srand(seed - 2)
    random_cell = selected_cells[randint(num_selected_cells, &state)]
    centroid_indices_buffer.resize(num_clusters)
    centroid_indices = <unsigned[:num_clusters]> centroid_indices_buffer.data()
    centroid_indices[0] = random_cell
    centroids[0] = X[random_cell]

    # Return if only one cluster was requested
    if num_clusters == 1:
        return

    # Find each selected cell's distance to this centroid, weighted by the
    # cell's weight in `selected_cell_weights`. Store these distances in the
    # first `num_selected_cells` entries of `min_distances`.
    for i in range(num_selected_cells):
        selected_cell = selected_cells[i]
        difference = X[selected_cell, 0] - X[random_cell, 0]
        distance = difference * difference
        for j in range(1, num_dimensions):
            difference = X[selected_cell, j] - X[random_cell, j]
            distance += difference * difference
        distance *= selected_cell_weights[i]
        min_distances[i] = distance

    # Sum the weighted distances separately at the end, to ensure deterministic
    # parallelism
    cost = 0
    for i in range(num_selected_cells):
        cost += min_distances[i]

    # Iteratively select the remaining centroids
    cluster_index = 1
    while True:
        # Sample a single cell `i` with probability `min_distances[i] / cost`.
        # Set `min_distances` to 0 for the sampled cell, to avoid sampling it
        # twice.
        inverse_cost = <float> 1 / cost
        probability = random_uniform(&state)
        for i in range(num_selected_cells):
            probability -= min_distances[i] * inverse_cost
            if probability < <float> 0:
                break
        min_distances[i] = 0
        centroid_indices[cluster_index] = selected_cells[i]
        centroids[cluster_index] = X[selected_cells[i]]

        # Stop once all centroids have been selected
        if cluster_index == num_clusters - 1:
            break

        # Update each selected cell's weighted distance to its nearest
        # centroid, if it is closer to this new centroid than to any we have
        # selected so far.
        for i in range(num_selected_cells):
            selected_cell = selected_cells[i]
            difference = X[selected_cell, 0] - centroids[cluster_index, 0]
            distance = difference * difference
            for j in range(1, num_dimensions):
                difference = X[selected_cell, j] - centroids[cluster_index, j]
                distance += difference * difference
            distance *= selected_cell_weights[i]
            if distance < min_distances[i]:
                min_distances[i] = distance

        # Sum the weighted distances separately at the end, to ensure
        # deterministic parallelism
        cost = 0
        for i in range(num_selected_cells):
            cost += min_distances[i]

        # Increment the centroid counter
        cluster_index += 1


cdef inline void kmeans_barbar_init_parallel(
        const float[:, ::1] X,
        float[:, ::1] centroids,
        float[::1] min_distances,
        const unsigned num_init_iterations,
        const float oversampling_factor,
        const unsigned long long num_cells,
        const unsigned num_clusters,
        const unsigned num_dimensions,
        const unsigned long long seed,
        const unsigned chunk_size,
        const unsigned num_threads):
    cdef unsigned i, j, k, random_cell, thread_index, c0, c1, iteration, \
        num_newly_selected_cells, num_previously_selected_cells, \
        selected_cell, chunk_index, start, chunk_num_cells, chunk_size_2, \
        num_selected_cells, end, cluster_index, selected_centroid, \
        num_chunks = (num_cells + chunk_size - 1) / chunk_size
    cdef int best_selected_cell
    cdef unsigned long long state
    cdef float cost, difference, distance, norm, l_over_cost, chunk_cost, \
        min_distance, inverse_cost, probability, alpha = -2, beta = 1, \
        l = oversampling_factor * num_clusters
    cdef char transA = b'T', transB = b'N'
    cdef uninitialized_vector[float] X_norms_buffer, newly_selected_X, \
        chunk_costs_buffer
    cdef vector[unsigned] best_selected_cells_buffer, selected_cells, \
        selected_cell_weights
    cdef uninitialized_vector[unsigned] centroid_indices_buffer
    cdef vector[uninitialized_vector[float]] thread_distances
    cdef vector[vector[unsigned]] thread_selected_cells, \
        thread_selected_cell_weights
    cdef float[::1] X_norms, chunk_costs
    cdef unsigned[::1] best_selected_cells, centroid_indices
    cdef unsigned* cell_weights
    cdef float* distance_pointer

    # Reserve 25% more than the expected number to be safe
    selected_cells.reserve(<unsigned>(<float> 1.25 * num_init_iterations * l))

    # Sample a random cell from `X`, and add it to our list of selected cells.
    # This will constitute a shortlist from which we will select the final
    # centroids to initialize k-means with.
    state = srand(seed - 1)
    random_cell = randint(num_cells, &state)
    selected_cells.push_back(random_cell)

    # Calculate the (squared Euclidean) distance from each cell to the random
    # cell, storing it in `min_distances`. In the same loop, also calculate the
    # squared L2 norm of each cell, ||X||².
    X_norms_buffer.resize(num_cells)
    X_norms = <float[:num_cells]> X_norms_buffer.data()
    with nogil:
        for i in prange(num_cells, num_threads=num_threads):
            difference = X[i, 0] - X[random_cell, 0]
            distance = difference * difference
            norm = X[i, 0] * X[i, 0]
            for j in range(1, num_dimensions):
                difference = X[i, j] - X[random_cell, j]
                distance = distance + difference * difference
                norm = norm + X[i, j] * X[i, j]
            min_distances[i] = distance
            X_norms[i] = norm

    # Sum the `min_distances` single-threaded at the end, to ensure
    # deterministic parallelism
    cost = 0
    for i in range(num_cells):
        cost += min_distances[i]
    if not cost:
        error_message = \
            f'all cells have the same principal component loadings'
        raise ValueError(error_message)

    # Sample each cell with probability `l * min_distances[i] / cost`.
    # Set `min_distances` to zero for sampled cells to reflect that each
    # sampled cell's nearest centroid candidate is now itself.
    l_over_cost = l / cost
    thread_selected_cells.resize(num_threads)
    with nogil, parallel(num_threads=num_threads):
        thread_index = threadid()
        thread_selected_cells[thread_index].reserve(
            <unsigned>(<float> 1.25 * l / num_threads))
        c0 = num_cells * thread_index / num_threads
        c1 = num_cells * (thread_index + 1) / num_threads
        for i in range(c0, c1):
            state = srand(seed + i)
            if l_over_cost * min_distances[i] >= random_uniform(&state):
                thread_selected_cells[thread_index].push_back(i)
                min_distances[i] = 0

    # Aggregate each thread's selected cells into a single vector
    for thread_index in range(num_threads):
        selected_cells.insert(
            selected_cells.end(),
            thread_selected_cells[thread_index].begin(),
            thread_selected_cells[thread_index].end())

    # Keep track of how many cells were selected, newly selected on this past
    # iteration (for this first iteration, all cells except the first random
    # cell), and selected on a previous iteration (just the first random cell).
    num_selected_cells = selected_cells.size()
    num_newly_selected_cells = num_selected_cells - 1
    num_previously_selected_cells = 1

    # `best_selected_cells` maps each cell to the index in `selected_cells`
    # with the cell's nearest candidate centroid selected so far. For each
    # newly selected cell, set `best_selected_cells[selected_cell]` to the
    # cell's index, to reflect that the cell's nearest centroid candidate is
    # now itself. `best_selected_cells` starts off at 0 for non-newly selected
    # cells, i.e. they still have the initial random cell (`selected_cells[0]`)
    # listed as their closest centroid candidate.
    best_selected_cells_buffer.resize(num_cells)
    best_selected_cells = <unsigned[:num_cells]> \
        best_selected_cells_buffer.data()
    for i in range(num_previously_selected_cells, num_selected_cells):
        selected_cell = selected_cells[i]
        best_selected_cells[selected_cell] = i

    # For each remaining iteration...
    chunk_costs_buffer.resize(num_chunks)
    chunk_costs = <float[:num_chunks]> chunk_costs_buffer.data()
    thread_distances.resize(num_threads)
    iteration = 1
    while True:
        with nogil:
            # For very small datasets, it's possible no cells are selected on a
            # given iteration. In that case, skip the distance and cost
            # calculations and go straight to sampling more cells.
            if num_newly_selected_cells > 0:
                # Copy newly selected cells into a temporary buffer
                newly_selected_X.resize(
                    num_newly_selected_cells * num_dimensions)
                for i in prange(num_newly_selected_cells,
                                num_threads=num_threads):
                    selected_cell = selected_cells[
                        num_previously_selected_cells + i]
                    memcpy(newly_selected_X.data() + i * num_dimensions,
                           &X[selected_cell, 0],
                           num_dimensions * sizeof(float))

                # Update each cell's nearest candidate centroid selected so far
                # (`best_selected_cells`) and distance to this candidate
                # centroid (`min_distances`) to account for the newly selected
                # cells. We only have to compute distances from each cell to
                # the newly selected cells, rather than to all selected cells.
                # Use the identity:
                # ||X - C||² = ||X||² - 2 * X.dot(C.T) + ||C||²
                # but add the ||X||² at the end since the minimum distance for
                # a given cell doesn't depend on ||X||².
                with parallel(num_threads=num_threads):
                    thread_index = threadid()
                    thread_distances[thread_index].resize(
                        chunk_size * num_newly_selected_cells)
                    distance_pointer = thread_distances[thread_index].data()
                    for chunk_index in prange(num_chunks):
                        chunk_cost = 0
                        start = chunk_index * chunk_size
                        chunk_num_cells = num_cells - start \
                            if chunk_index == num_chunks - 1 else chunk_size
                        for i in range(chunk_num_cells):
                            for j in range(num_newly_selected_cells):
                                # distances = ||C||²
                                selected_cell = selected_cells[
                                    num_previously_selected_cells + j]
                                distance_pointer[
                                    i * num_newly_selected_cells + j] = \
                                    X_norms[selected_cell]

                        # distances -= 2 * X.dot(C.T)
                        sgemm(transA, transB, num_newly_selected_cells,
                              chunk_num_cells, num_dimensions, alpha,
                              newly_selected_X.data(), num_dimensions,
                              &X[start, 0], num_dimensions, beta,
                              distance_pointer, num_newly_selected_cells)

                        # distances += ||X||², and find the best distance. As
                        # an optimization, to avoid having to add
                        # `X_norms[start + i]` (i.e. ||X||²) to `min_distance`
                        # and `num_previously_selected_cells` to
                        # `best_selected_cell` to every candidate inside the
                        # inner loop (which would be necessary to make them
                        # comparable to the previous minimum distance and best
                        # selected cell), subtract these terms at the start and
                        # add them back at the end. (`best_selected_cell` is
                        # declared as a signed integer to allow for underflow.)
                        for i in range(chunk_num_cells):
                            min_distance = \
                                min_distances[start + i] - X_norms[start + i]
                            best_selected_cell = \
                                best_selected_cells[start + i] - \
                                num_previously_selected_cells
                            for j in range(num_newly_selected_cells):
                                distance = distance_pointer[
                                    i * num_newly_selected_cells + j]
                                if distance < min_distance:
                                    min_distance = distance
                                    best_selected_cell = j
                            min_distance = min_distance + X_norms[start + i]
                            min_distances[start + i] = min_distance
                            best_selected_cells[start + i] = \
                                best_selected_cell + \
                                num_previously_selected_cells
                            chunk_cost = chunk_cost + min_distance
                        chunk_costs[chunk_index] = chunk_cost

                # Sum costs for each chunk single-threaded at the end, to
                # ensure deterministic parallelism
                cost = 0
                for chunk_index in range(num_chunks):
                    cost += chunk_costs[chunk_index]

            # Sample each cell `i` with probability
            # `l * min_distances[i] / cost`. Note that since we set
            # `min_distances` to 0 for cells that were sampled, we will avoid
            # sampling them twice. As before, set `min_distances` to zero for
            # newly sampled cells, to reflect that each sampled cell's nearest
            # centroid candidate is now itself.
            l_over_cost = l / cost
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_selected_cells[thread_index].clear()  # reset
                c0 = num_cells * thread_index / num_threads
                c1 = num_cells * (thread_index + 1) / num_threads
                for i in range(c0, c1):
                    state = srand(seed + i * iteration)
                    probability = random_uniform(&state)
                    if l_over_cost * min_distances[i] >= probability:
                        thread_selected_cells[thread_index].push_back(i)
                        min_distances[i] = 0
                        best_selected_cells[i] = i

        # Aggregate each thread's selected cells into a single vector. Update
        # the number of selected, previously selected and newly selected cells.
        num_previously_selected_cells = num_selected_cells
        for thread_index in range(num_threads):
            selected_cells.insert(
                selected_cells.end(),
                thread_selected_cells[thread_index].begin(),
                thread_selected_cells[thread_index].end())
        num_selected_cells = selected_cells.size()
        num_newly_selected_cells = \
            num_selected_cells - num_previously_selected_cells

        # As before, set `best_selected_cells` for each newly selected cell to
        # the cell's index, to reflect that the cell's nearest centroid
        # candidate is now itself
        for i in range(num_previously_selected_cells, num_selected_cells):
            selected_cell = selected_cells[i]
            best_selected_cells[selected_cell] = i

        # Check for KeyboardInterrupts after each iteration
        PyErr_CheckSignals()

        # Check if progress has frozen (i.e. `num_newly_selected_cells == 0`
        # when `cost == 0`), indicating there are fewer distinct cells than
        # clusters requested
        if num_newly_selected_cells == 0 and not cost:
            error_message = (
                f'num_clusters ({num_clusters:,}) is greater than the number '
                f'of cells with distinct principal component loadings; '
                f'decrease num_clusters')
            raise ValueError(error_message)

        # Stop after `num_init_iterations` iterations, unless we have found
        # fewer than `num_clusters` centroid candidates so far. Also stop if we
        # have already selected every cell as a centroid candidate, which can
        # happen if the dataset is very small.
        iteration += 1
        if iteration == num_init_iterations and \
                num_selected_cells >= num_clusters or \
                num_selected_cells == num_cells:
            break

    centroid_indices_buffer.resize(num_clusters)
    centroid_indices = <unsigned[:num_clusters]> centroid_indices_buffer.data()
    with nogil:
        # Now we are done selecting cells as candidate centroids and need to
        # whittle down to the final centroids. Get the weight for each selected
        # cell: the number of cells that are closer to the selected cell than
        # to any other selected cell. Store weights for each thread in a
        # temporary buffer, then aggregate at the end. As an optimization, put
        # the row sums for the last thread (`thread_index == num_threads - 1`)
        # directly into the final `selected_cell_weights` vector.
        thread_selected_cell_weights.resize(num_threads - 1)
        chunk_size_2 = (num_cells + num_threads - 1) / num_threads
        with parallel(num_threads=num_threads):
            thread_index = threadid()
            start = thread_index * chunk_size_2
            if thread_index == num_threads - 1:
                end = num_cells
                selected_cell_weights.resize(num_selected_cells)
                for i in range(start, end):
                    best_selected_cell = best_selected_cells[i]
                    selected_cell_weights[best_selected_cell] += 1
            else:
                thread_selected_cell_weights[thread_index].resize(
                    num_selected_cells)
                cell_weights = \
                    thread_selected_cell_weights[thread_index].data()
                end = min(start + chunk_size_2, num_cells)
                for i in range(start, end):
                    best_selected_cell = best_selected_cells[i]
                    cell_weights[best_selected_cell] += 1
        for thread_index in range(num_threads - 1):
            cell_weights = thread_selected_cell_weights[thread_index].data()
            for i in range(num_selected_cells):
                selected_cell_weights[i] += cell_weights[i]

        # Run k-means++ to select `num_clusters` of the selected cells as the
        # centroids, using `selected_cell_weights` as weights. Start by
        # selecting a random cell from our selected cells as the first
        # centroid.
        state = srand(seed - 2)
        random_cell = selected_cells[randint(num_selected_cells, &state)]
        centroid_indices[0] = random_cell
        centroids[0] = X[random_cell]

        # Return if only one cluster was requested
        if num_clusters == 1:
            return

        # Find each selected cell's distance to this centroid, weighted by the
        # cell's weight in `selected_cell_weights`. Store these distances in
        # the first `num_selected_cells` entries of `min_distances`.
        for i in prange(num_selected_cells, num_threads=num_threads):
            selected_cell = selected_cells[i]
            difference = X[selected_cell, 0] - X[random_cell, 0]
            distance = difference * difference
            for j in range(1, num_dimensions):
                difference = X[selected_cell, j] - X[random_cell, j]
                distance = distance + difference * difference
            min_distances[i] = distance * selected_cell_weights[i]

        # Sum the weighted distances single-threaded at the end, to ensure
        # deterministic parallelism
        cost = 0
        for i in range(num_selected_cells):
            cost += min_distances[i]

        # Iteratively select the remaining centroids
        cluster_index = 1
        while True:
            # Sample a single cell `i` with probability
            # `min_distances[i] / cost`. Set `min_distances` to 0 for the
            # sampled cell, to avoid sampling it twice.
            inverse_cost = <float> 1 / cost
            probability = random_uniform(&state)
            for i in range(num_selected_cells):
                probability -= min_distances[i] * inverse_cost
                if probability < <float> 0:
                    break
            min_distances[i] = 0
            centroid_indices[cluster_index] = selected_cells[i]
            centroids[cluster_index] = X[selected_cells[i]]

            # Stop once all centroids have been selected
            if cluster_index == num_clusters - 1:
                break

            # Update each selected cell's weighted distance to its nearest
            # centroid, if it is closer to this new centroid than to any we
            # have selected so far.
            cost = 0
            for i in prange(num_selected_cells, num_threads=num_threads):
                selected_cell = selected_cells[i]
                difference = X[selected_cell, 0] - centroids[cluster_index, 0]
                distance = difference * difference
                for j in range(1, num_dimensions):
                    difference = \
                        X[selected_cell, j] - centroids[cluster_index, j]
                    distance = distance + difference * difference
                distance = distance * selected_cell_weights[i]
                if distance < min_distances[i]:
                    min_distances[i] = distance

            # Sum the weighted distances single-threaded at the end, to ensure
            # deterministic parallelism
            cost = 0
            for i in range(num_selected_cells):
                cost += min_distances[i]

            # Increment the centroid counter
            cluster_index += 1


cdef inline void relocate_empty_clusters(
        const float[:, ::1] X,
        const unsigned[::1] cluster_labels,
        const float[:, ::1] centroids,
        float[:, ::1] centroids_new,
        unsigned[::1] num_cells_per_cluster):
    # Relocate centroids with no cells assigned to them

    cdef unsigned i, j, k, num_empty, new_cluster_label, old_cluster_label, \
        num_dimensions = X.shape[1], num_clusters = centroids.shape[0]
    cdef unsigned long long num_cells = X.shape[0]
    cdef float distance, difference
    cdef vector[unsigned] empty_cluster_indices
    cdef uninitialized_vector[unsigned] farthest_cells_buffer
    cdef uninitialized_vector[float] farthest_distances_buffer
    cdef unsigned[::1] farthest_cells
    cdef float[::1] farthest_distances
    cdef str error_message

    # Collect indices of empty clusters
    for i in range(num_clusters):
        if num_cells_per_cluster[i] == 0:
            empty_cluster_indices.push_back(i)

    # Return if no clusters are empty
    num_empty = empty_cluster_indices.size()
    if num_empty == 0:
        return

    # Find the `num_empty` farthest points from their assigned centroids,
    # using a min-heap to keep track of the `num_empty` largest distances
    farthest_cells_buffer.resize(num_empty)
    farthest_cells = <unsigned[:num_empty]> farthest_cells_buffer.data()
    farthest_distances_buffer.resize(num_empty)
    farthest_distances = <float[:num_empty]> farthest_distances_buffer.data()
    for i in range(num_empty):
        farthest_distances[i] = -FLT_MAX
    for i in range(num_cells):
        j = cluster_labels[i]
        difference = X[i, 0] - centroids[j, 0]
        distance = difference * difference
        for k in range(1, num_dimensions):
            difference = X[i, k] - centroids[j, k]
            distance = distance + difference * difference
        if distance > farthest_distances[0]:
            min_heap_replace_top(&farthest_cells[0], &farthest_distances[0], i,
                                 distance, num_empty)

    # Sort the heap to get distances in descending order
    min_heap_sort(&farthest_cells[0], &farthest_distances[0], num_empty)

    # Check if any of the farthest distances are 0
    if not farthest_distances[0]:
        error_message = (
            f'num_clusters ({num_clusters:,}) is greater than the number '
            f'of cells with distinct principal component loadings '
            f'({num_cells - num_empty:,}); decrease num_clusters')
        raise ValueError(error_message)

    # Relocate empty clusters to points
    for i in range(num_empty):
        new_cluster_label = empty_cluster_indices[i]
        j = farthest_cells[i]
        old_cluster_label = cluster_labels[j]

        # Move the cell from the old cluster to the new cluster
        for k in range(num_dimensions):
            centroids_new[old_cluster_label, k] -= X[j, k]
            centroids_new[new_cluster_label, k] = X[j, k]

        # Keep counts accurate for the normalization step
        num_cells_per_cluster[old_cluster_label] -= 1
        num_cells_per_cluster[new_cluster_label] += 1


cdef inline unsigned relocate_empty_clusters_parallel(
        const float[:, ::1] X,
        const unsigned[::1] cluster_labels,
        const float[:, ::1] centroids,
        float[:, ::1] centroids_new,
        unsigned[::1] num_cells_per_cluster,
        const unsigned num_threads) noexcept nogil:
    # Relocate centroids with no cells assigned to them

    cdef unsigned i, j, k, num_empty, new_cluster_label, old_cluster_label, \
        num_dimensions = X.shape[1], num_clusters = centroids.shape[0]
    cdef unsigned long long num_cells = X.shape[0]
    cdef float distance, difference
    cdef vector[unsigned] empty_cluster_indices
    cdef uninitialized_vector[unsigned] farthest_cells
    cdef uninitialized_vector[float] farthest_distances

    # Collect indices of empty clusters
    for i in range(num_clusters):
        if num_cells_per_cluster[i] == 0:
            empty_cluster_indices.push_back(i)

    # Return if no clusters are empty
    num_empty = empty_cluster_indices.size()
    if num_empty == 0:
        return 0

    # Find the `num_empty` farthest points from their assigned centroids,
    # using a min-heap to keep track of the `num_empty` largest distances
    farthest_cells.resize(num_empty)
    farthest_distances.resize(num_empty)
    for i in range(num_empty):
        farthest_distances[i] = -FLT_MAX
    for i in range(num_cells):
        j = cluster_labels[i]
        difference = X[i, 0] - centroids[j, 0]
        distance = difference * difference
        for k in range(1, num_dimensions):
            difference = X[i, k] - centroids[j, k]
            distance = distance + difference * difference
        if distance > farthest_distances[0]:
            min_heap_replace_top(&farthest_cells[0], &farthest_distances[0], i,
                                 distance, num_empty)

    # Sort the heap to get distances in descending order
    min_heap_sort(&farthest_cells[0], &farthest_distances[0], num_empty)

    # Check if any of the farthest distances are 0
    if not farthest_distances[0]:
        return num_empty  # error code

    # Relocate empty clusters to points
    for i in range(num_empty):
        new_cluster_label = empty_cluster_indices[i]
        j = farthest_cells[i]
        old_cluster_label = cluster_labels[j]

        # Move the cell from the old cluster to the new cluster
        for k in range(num_dimensions):
            centroids_new[old_cluster_label, k] -= X[j, k]
            centroids_new[new_cluster_label, k] = X[j, k]

        # Keep counts accurate for the normalization step
        num_cells_per_cluster[old_cluster_label] -= 1
        num_cells_per_cluster[new_cluster_label] += 1


def kmeans(const float[:, ::1] X,
           unsigned[::1] cluster_labels,
           float[:, ::1] centroids,
           float[:, ::1] centroids_new,
           unsigned[::1] num_cells_per_cluster,
           float[::1] min_distances,
           float[::1] cell_norms,
           const bint kmeans_barbar,
           const unsigned num_init_iterations,
           const unsigned num_kmeans_iterations,
           const float tolerance,
           const float oversampling_factor,
           const unsigned chunk_size,
           const unsigned long long seed,
           const bint is_mac,
           unsigned num_threads):
    cdef unsigned long long num_cells = X.shape[0], state = srand(seed)
    cdef unsigned i, j, k, l, cluster_tile_size, iteration, best_cluster, \
        thread_index, chunk_index, start, chunk_num_cells, num_empty, \
        tile_start, current_tile_size, num_clusters = centroids.shape[0], \
        num_dimensions = centroids.shape[1], \
        num_chunks = (num_cells + chunk_size - 1) / chunk_size
    cdef float squared_norm, sum_of_squared_norms, difference, distance, \
        min_distance, norm, total_min_distance, old_total_min_distance
    cdef uninitialized_vector[float] centroid_norms_buffer, \
        chunk_centroids_new_buffer, distances_buffer
    cdef vector[vector[unsigned]] thread_num_cells_per_cluster
    cdef vector[uninitialized_vector[float]] thread_distances
    cdef uninitialized_vector[unsigned] remap_buffer, compact_buffer
    cdef float[::1] centroid_norms
    cdef float[:, ::1] distances, temp
    cdef float[:, :, ::1] chunk_centroids_new
    cdef unsigned[::1] remap, compact
    cdef float* distance_pointer
    cdef unsigned* num_cells_per_cluster_pointer
    cdef str error_message

    # Pick a size to tile over clusters that keeps the `partial_distances()`
    # output in cache for the subsequent scan, rather than evicting to main
    # memory
    cluster_tile_size = 1024 * 1024 / (chunk_size * sizeof(float))
    if cluster_tile_size == 0:
        cluster_tile_size = 1
    if cluster_tile_size > num_clusters:
        cluster_tile_size = num_clusters

    # Force the parallel path for Mac for floating-point consistency between 1
    # and multiple threads.

    num_threads = min(num_threads, num_chunks)
    if num_threads <= 1 and not is_mac:
        if kmeans_barbar:
            # Initialize centroids with k-means||
            kmeans_barbar_init(X, centroids, min_distances,
                               num_init_iterations, oversampling_factor,
                               num_cells, num_clusters, num_dimensions,
                               chunk_size, seed)
        else:
            # Initialize centroids with random points
            kmeans_random_init(X, centroids, num_cells, num_clusters, seed)

        # Calculate ||X||², the squared L2 norm of each cell, and store in
        # `cell_norms` for reuse by `knn_self()`/`knn_cross()`. Also accumulate
        # their sum, used in the convergence check. Or, if `cell_norms` is
        # empty, indicating k-nearest neighbors does not need to be run
        # afterwards, just accumulate the sum.
        sum_of_squared_norms = 0
        if cell_norms.shape[0] != 0:
            for i in range(num_cells):
                squared_norm = X[i, 0] * X[i, 0]
                for j in range(1, num_dimensions):
                    squared_norm += X[i, j] * X[i, j]
                cell_norms[i] = squared_norm
                sum_of_squared_norms += squared_norm
        else:
            for i in range(num_cells):
                squared_norm = X[i, 0] * X[i, 0]
                for j in range(1, num_dimensions):
                    squared_norm += X[i, j] * X[i, j]
                sum_of_squared_norms += squared_norm

        # 1. Run the E and M steps of k-means for `num_kmeans_iterations`
        # iterations

        centroid_norms_buffer.resize(num_clusters)
        centroid_norms = <float[:num_clusters]> centroid_norms_buffer.data()
        chunk_centroids_new_buffer.resize(
            <unsigned long long> num_chunks * num_clusters * num_dimensions)
        chunk_centroids_new = \
            <float[:num_chunks, :num_clusters, :num_dimensions]> \
            chunk_centroids_new_buffer.data()
        distances_buffer.resize(chunk_size * cluster_tile_size)
        distances = \
            <float[:chunk_size, :cluster_tile_size]> distances_buffer.data()

        iteration = 0
        while True:
            centroids_new[:] = 0
            num_cells_per_cluster[:] = 0
            chunk_centroids_new[:] = 0

            # Calculate the squared L2 norm of each centroid, ||C||²
            for i in range(num_clusters):
                norm = centroids[i, 0] * centroids[i, 0]
                for j in range(1, num_dimensions):
                    norm += centroids[i, j] * centroids[i, j]
                centroid_norms[i] = norm

            # Run the E and M steps of Lloyd's algorithm in chunks
            for chunk_index in range(num_chunks):
                start = chunk_index * chunk_size
                chunk_num_cells = num_cells - start \
                    if chunk_index == num_chunks - 1 else chunk_size

                # Find the closest centroid to each cell in the chunk, i.e. the
                # cell's cluster assignment, and the cell's distance to the
                # closest centroid, used to assess convergence.
                for i in range(chunk_num_cells):
                    min_distances[start + i] = FLT_MAX
                tile_start = 0
                while tile_start < num_clusters:
                    current_tile_size = cluster_tile_size \
                        if tile_start + cluster_tile_size <= num_clusters \
                        else num_clusters - tile_start

                    # Calculate the distance from each cell in the chunk to
                    # each centroid in the tile. Use the identity:
                    # ||X - C||² = ||X||² - 2 * X.dot(C.T) + ||C||²
                    # but skip calculating the ||X||² term since the best
                    # cluster for a given cell does not depend on ||X||².
                    partial_distances(&X[start, 0], &centroids[tile_start, 0],
                                      &centroid_norms[tile_start],
                                      &distances[0, 0],
                                      chunk_num_cells, current_tile_size,
                                      cluster_tile_size, num_dimensions)

                    for i in range(chunk_num_cells):
                        min_distance = min_distances[start + i]
                        best_cluster = cluster_labels[start + i]
                        for j in range(current_tile_size):
                            distance = distances[i, j]
                            if distance < min_distance:
                                min_distance = distance
                                best_cluster = tile_start + j
                        min_distances[start + i] = min_distance
                        cluster_labels[start + i] = best_cluster
                    tile_start = tile_start + cluster_tile_size

                # Keep track of how many cells were assigned to each cluster,
                # and calculate the total contribution of the cells in the
                # chunk to the new centroids (i.e. the sum of the cells that
                # were assigned to a centroid's cluster; we will normalize to
                # get the mean later). Aggregate centroids by chunk, to allow
                # deterministic parallelism.
                for i in range(chunk_num_cells):
                    best_cluster = cluster_labels[start + i]
                    num_cells_per_cluster[best_cluster] += 1
                    for j in range(num_dimensions):
                        chunk_centroids_new[chunk_index, best_cluster, j] += \
                            X[start + i, j]

                # Check for KeyboardInterrupts.
                # (This check is not present in the parallel version.)
                if chunk_index % 128 == 127:
                    PyErr_CheckSignals()

            # Aggregate the contributions of each chunk to the new centroids
            for i in range(num_clusters):
                for chunk_index in range(num_chunks):
                    for j in range(num_dimensions):
                        centroids_new[i, j] += \
                            chunk_centroids_new[chunk_index, i, j]

            # Handle empty clusters
            relocate_empty_clusters(X, cluster_labels, centroids,
                                    centroids_new, num_cells_per_cluster)

            # Normalize the new centroids by the number of cells in the cluster
            # to get the mean instead of the sum
            for i in range(num_clusters):
                if num_cells_per_cluster[i] > 0:
                    norm = <float> 1 / num_cells_per_cluster[i]
                    for j in range(num_dimensions):
                        centroids_new[i, j] *= norm

            # Swap `centroids` and `centroids_new` after each k-means iteration
            temp = centroids
            centroids = centroids_new
            centroids_new = temp

            # Check for KeyboardInterrupts
            PyErr_CheckSignals()

            # Stop if we have reached `num_kmeans_iterations` iterations,
            # without bothering to do the convergence check
            iteration += 1
            if iteration == num_kmeans_iterations:
                break

            # Stop early if we have converged to the desired tolerance.
            # Convergence is based on the relative change in the sum of the min
            # distances; this sum needs to use an intermediate array to be
            # consistent with the parallel version.
            total_min_distance = sum_of_squared_norms
            for i in range(num_cells):
                total_min_distance += min_distances[i]
            if iteration > 1 and \
                    old_total_min_distance - total_min_distance < \
                    tolerance * old_total_min_distance:
                break
            old_total_min_distance = total_min_distance

        # 2. Run one last iteration of the E step, to get the right cluster
        # assignments

        num_cells_per_cluster[:] = 0

        # Calculate the squared L2 norm of each centroid, ||C||²
        for i in range(num_clusters):
            norm = centroids[i, 0] * centroids[i, 0]
            for j in range(1, num_dimensions):
                norm += centroids[i, j] * centroids[i, j]
            centroid_norms[i] = norm

        # Run the E step of Lloyd's algorithm in chunks
        for chunk_index in range(num_chunks):
            start = chunk_index * chunk_size
            chunk_num_cells = num_cells - start \
                if chunk_index == num_chunks - 1 else chunk_size

            # Find the closest centroid to each cell in the chunk, i.e. the
            # cell's cluster assignment.
            for i in range(chunk_num_cells):
                min_distances[start + i] = FLT_MAX
            tile_start = 0
            while tile_start < num_clusters:
                current_tile_size = cluster_tile_size \
                    if tile_start + cluster_tile_size <= num_clusters \
                    else num_clusters - tile_start

                # Calculate the distance from each cell in the chunk to each
                # centroid in the tile. Use the identity:
                # ||X - C||² = ||X||² - 2 * X.dot(C.T) + ||C||²
                # but skip calculating the ||X||² term since the best
                # cluster for a given cell does not depend on ||X||².
                partial_distances(&X[start, 0], &centroids[tile_start, 0],
                                  &centroid_norms[tile_start],
                                  &distances[0, 0],
                                  chunk_num_cells, current_tile_size,
                                  cluster_tile_size, num_dimensions)

                for i in range(chunk_num_cells):
                    min_distance = min_distances[start + i]
                    best_cluster = cluster_labels[start + i]
                    for j in range(current_tile_size):
                        distance = distances[i, j]
                        if distance < min_distance:
                            min_distance = distance
                            best_cluster = tile_start + j
                    min_distances[start + i] = min_distance
                    cluster_labels[start + i] = best_cluster
                tile_start = tile_start + cluster_tile_size

            # Keep track of how many cells were assigned to each cluster
            for i in range(chunk_num_cells):
                num_cells_per_cluster[cluster_labels[start + i]] += 1
    else:
        # Same as the single-threaded version, but the centroid-finding step
        # needs each thread to scan through every cell and only process the
        # cells that match a particular cluster, to avoid expensive
        # synchronization. The k-means|| has similar additional complexity.
        # Also, skip the KeyboardInterrupt check after each chunk.

        if kmeans_barbar:
            # Initialize centroids with k-means||
            kmeans_barbar_init_parallel(X, centroids, min_distances,
                                        num_init_iterations,
                                        oversampling_factor, num_cells,
                                        num_clusters, num_dimensions, seed,
                                        chunk_size, num_threads)
        else:
            # Initialize centroids with random points
            kmeans_random_init(X, centroids, num_cells, num_clusters, seed)

        # Calculate ||X||², the squared L2 norm of each cell, and store in
        # `cell_norms` for reuse by `knn_self()`/`knn_cross()`. Also accumulate
        # their sum, used in the convergence check. Or, if `cell_norms` is
        # empty, indicating k-nearest neighbors does not need to be run
        # afterwards, just accumulate the sum.
        sum_of_squared_norms = 0
        if cell_norms.shape[0] != 0:
            for i in prange(num_cells, num_threads=num_threads, nogil=True):
                squared_norm = X[i, 0] * X[i, 0]
                for j in range(1, num_dimensions):
                    squared_norm = squared_norm + X[i, j] * X[i, j]
                cell_norms[i] = squared_norm
            for i in range(num_cells):
                sum_of_squared_norms += cell_norms[i]
        else:
            for i in range(num_cells):
                squared_norm = X[i, 0] * X[i, 0]
                for j in range(1, num_dimensions):
                    squared_norm += X[i, j] * X[i, j]
                sum_of_squared_norms += squared_norm

        # 1. Run the E and M steps of k-means for `num_kmeans_iterations`
        # iterations

        centroid_norms_buffer.resize(num_clusters)
        centroid_norms = <float[:num_clusters]> centroid_norms_buffer.data()
        chunk_centroids_new_buffer.resize(
            <unsigned long long> num_chunks * num_clusters * num_dimensions)
        chunk_centroids_new = \
            <float[:num_chunks, :num_clusters, :num_dimensions]> \
            chunk_centroids_new_buffer.data()
        thread_num_cells_per_cluster.resize(num_threads)
        thread_distances.resize(num_threads)

        iteration = 0
        while True:
            with nogil:
                # Calculate the squared L2 norm of each centroid, ||C||²
                for i in prange(num_clusters, num_threads=num_threads):
                    norm = centroids[i, 0] * centroids[i, 0]
                    for j in range(1, num_dimensions):
                        norm = norm + centroids[i, j] * centroids[i, j]
                    centroid_norms[i] = norm

                # Run the E and M steps of Lloyd's algorithm in chunks
                with parallel(num_threads=num_threads):
                    thread_index = threadid()

                    # Allocate thread-local buffers to store temporary data for
                    # each chunk
                    if iteration == 0:
                        thread_num_cells_per_cluster[thread_index].resize(
                            num_clusters)
                        thread_distances[thread_index].resize(
                            chunk_size * cluster_tile_size)
                    else:
                        fill(
                            thread_num_cells_per_cluster[thread_index].begin(),
                            thread_num_cells_per_cluster[thread_index].end(),
                            0)
                    distance_pointer = thread_distances[thread_index].data()
                    num_cells_per_cluster_pointer = \
                        thread_num_cells_per_cluster[thread_index].data()

                    for chunk_index in prange(num_chunks):
                        start = chunk_index * chunk_size
                        chunk_num_cells = num_cells - start \
                            if chunk_index == num_chunks - 1 else chunk_size

                        # Find the closest centroid to each cell in the chunk,
                        # i.e. the cell's cluster assignment, and the cell's
                        # distance to the closest centroid, used to assess
                        # convergence.
                        for i in range(chunk_num_cells):
                            min_distances[start + i] = FLT_MAX
                        tile_start = 0
                        while tile_start < num_clusters:
                            current_tile_size = cluster_tile_size \
                                if tile_start + cluster_tile_size <= \
                                num_clusters else num_clusters - tile_start

                            # Calculate the distance from each cell in the
                            # chunk to each centroid in the tile. Use the
                            # identity:
                            # ||X - C||² = ||X||² - 2 * X.dot(C.T) + ||C||²
                            # but skip calculating the ||X||² term since the
                            # best cluster for a given cell does not depend
                            # on ||X||².
                            partial_distances(
                                &X[start, 0], &centroids[tile_start, 0],
                                &centroid_norms[tile_start], distance_pointer,
                                chunk_num_cells, current_tile_size, 
                                cluster_tile_size, num_dimensions)

                            for i in range(chunk_num_cells):
                                min_distance = min_distances[start + i]
                                best_cluster = cluster_labels[start + i]
                                for j in range(current_tile_size):
                                    distance = distance_pointer[
                                        i * cluster_tile_size + j]
                                    if distance < min_distance:
                                        min_distance = distance
                                        best_cluster = tile_start + j
                                min_distances[start + i] = min_distance
                                cluster_labels[start + i] = best_cluster
                            tile_start = tile_start + cluster_tile_size

                        # Keep track of how many cells were assigned to each
                        # cluster, and calculate the total contribution of the
                        # cells in the chunk to the new centroids (i.e. the sum
                        # of the cells that were assigned to a centroid's
                        # cluster; we will normalize to get the mean later).
                        # Aggregate centroids by chunk, to allow deterministic
                        # parallelism, but just aggregate the number of cells
                        # in each cluster by thread since they are not
                        # floating-point.
                        chunk_centroids_new[chunk_index, :] = 0
                        for i in range(chunk_num_cells):
                            best_cluster = cluster_labels[start + i]
                            num_cells_per_cluster_pointer[best_cluster] += 1
                            for j in range(num_dimensions):
                                chunk_centroids_new[
                                    chunk_index, best_cluster, j] += \
                                    X[start + i, j]

                # Aggregate the contributions of each thread/chunk to
                # a) the number of cells assigned to each cluster, and
                # b) the new centroids
                for i in prange(num_clusters, num_threads=num_threads):
                    num_cells_per_cluster[i] = 0
                    for thread_index in range(num_threads):
                        num_cells_per_cluster[i] += \
                            thread_num_cells_per_cluster[thread_index][i]
                    centroids_new[i, :] = 0
                    for chunk_index in range(num_chunks):
                        for j in range(num_dimensions):
                            centroids_new[i, j] += \
                                chunk_centroids_new[chunk_index, i, j]

                # Handle empty clusters
                num_empty = relocate_empty_clusters_parallel(
                    X, cluster_labels, centroids, centroids_new,
                    num_cells_per_cluster, num_threads)

                # Normalize the new centroids by the number of cells in the
                # cluster to get the mean instead of the sum
                for i in prange(num_clusters, num_threads=num_threads):
                    if num_cells_per_cluster[i] > 0:
                        norm = <float> 1 / num_cells_per_cluster[i]
                        for j in range(num_dimensions):
                            centroids_new[i, j] *= norm

            if num_empty:
                error_message = (
                    f'num_clusters ({num_clusters:,}) is greater than the '
                    f'number of cells with distinct principal component '
                    f'loadings ({num_cells - num_empty:,}); decrease '
                    f'num_clusters')
                raise ValueError(error_message)

            # Swap `centroids` and `centroids_new` after each k-means iteration
            temp = centroids
            centroids = centroids_new
            centroids_new = temp

            # Check for KeyboardInterrupts
            PyErr_CheckSignals()

            # Stop if we have reached `num_kmeans_iterations` iterations,
            # without bothering to do the convergence check
            iteration += 1
            if iteration == num_kmeans_iterations:
                break

            # Stop early if we have converged to the desired tolerance.
            # Convergence is based on the relative change in the sum of the min
            # distances.
            total_min_distance = sum_of_squared_norms
            for i in range(num_cells):
                total_min_distance += min_distances[i]
            if iteration > 1 and \
                    old_total_min_distance - total_min_distance < \
                    tolerance * old_total_min_distance:
                break
            old_total_min_distance = total_min_distance

        # 2. Run one last iteration of the E step, to get the right cluster
        # assignments

        with nogil:
            # Calculate the squared L2 norm of each centroid, ||C||²
            for i in prange(num_clusters, num_threads=num_threads):
                norm = centroids[i, 0] * centroids[i, 0]
                for j in range(1, num_dimensions):
                    norm = norm + centroids[i, j] * centroids[i, j]
                centroid_norms[i] = norm

            # Run the E step of Lloyd's algorithm in chunks
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                fill(thread_num_cells_per_cluster[thread_index].begin(),
                     thread_num_cells_per_cluster[thread_index].end(), 0)
                distance_pointer = thread_distances[thread_index].data()
                num_cells_per_cluster_pointer = \
                    thread_num_cells_per_cluster[thread_index].data()

                for chunk_index in prange(num_chunks):
                    start = chunk_index * chunk_size
                    chunk_num_cells = num_cells - start \
                        if chunk_index == num_chunks - 1 else chunk_size

                    # Find the closest centroid to each cell in the chunk,
                    # i.e. the cell's cluster assignment.
                    for i in range(chunk_num_cells):
                        min_distances[start + i] = FLT_MAX
                    tile_start = 0
                    while tile_start < num_clusters:
                        current_tile_size = cluster_tile_size \
                            if tile_start + cluster_tile_size <= num_clusters \
                            else num_clusters - tile_start

                        # Calculate the distance from each cell in the chunk
                        # to each centroid in the tile. Use the identity:
                        # ||X - C||² = ||X||² - 2 * X.dot(C.T) + ||C||²
                        # but skip calculating the ||X||² term since the best
                        # cluster for a given cell does not depend on ||X||².
                        partial_distances(&X[start, 0],
                                          &centroids[tile_start, 0],
                                          &centroid_norms[tile_start],
                                          distance_pointer,
                                          chunk_num_cells, current_tile_size,
                                          cluster_tile_size, num_dimensions)

                        for i in range(chunk_num_cells):
                            min_distance = min_distances[start + i]
                            best_cluster = cluster_labels[start + i]
                            for j in range(current_tile_size):
                                distance = distance_pointer[
                                    i * cluster_tile_size + j]
                                if distance < min_distance:
                                    min_distance = distance
                                    best_cluster = tile_start + j
                            min_distances[start + i] = min_distance
                            cluster_labels[start + i] = best_cluster
                        tile_start = tile_start + cluster_tile_size

                    # Keep track of how many cells were assigned to each
                    # cluster
                    for i in range(chunk_num_cells):
                        num_cells_per_cluster_pointer[
                            cluster_labels[start + i]] += 1

            # Aggregate the contributions of each thread/chunk to
            # the number of cells assigned to each cluster.
            for i in prange(num_clusters, num_threads=num_threads):
                num_cells_per_cluster[i] = 0
                for thread_index in range(num_threads):
                    num_cells_per_cluster[i] += \
                        thread_num_cells_per_cluster[thread_index][i]

    return iteration
