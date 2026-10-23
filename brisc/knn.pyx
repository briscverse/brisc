# Functionality for k-nearest neighbor search

from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libc.float cimport FLT_MAX
from libc.string cimport memcpy
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from .cyutils cimport max_heap_replace_top, max_heap_sort, partial_distances, \
    uninitialized_vector


def knn_self(const float[:, ::1] X,
             const unsigned[::1] sorted_order,
             const float[:, ::1] centroids,
             const unsigned[::1] num_cells_per_cluster,
             unsigned[:, ::1] neighbors,
             float[:, ::1] distances,
             const float[::1] cell_norms,
             float[:, ::1] centroid_distances,
             unsigned[:, ::1] nearest_clusters,
             const unsigned num_neighbors,
             const unsigned num_clusters_searched,
             const unsigned chunk_size_kmeans,
             const unsigned chunk_size_search,
             const bint is_mac,
             unsigned num_threads):
    # Find the `num_neighbors` nearest neighbors of each cell in `X` among the
    # cells in `X`. Store the nearest-neighbor indices in `neighbors` and
    # distances in `distances`. The caller must have grouped `X` by cluster;
    # this avoids the need for an explicit inverted-file index.

    cdef unsigned long long num_X = X.shape[0]
    cdef unsigned i, j, k, cluster_tile_size, thread_index, chunk_index, \
        start, end, chunk_num_X, cluster_index, cluster_label, neighbor, \
        tile_start, current_tile_size, block_index, block_start, \
        num_cells_in_block, query_index_in_block, query_sorted, num_pairs, \
        pair_index, next_pair_index, query_in_cluster_index, \
        num_queries_in_cluster, max_cluster_size, cluster_start, \
        cluster_size, neighbor_offset, block_offset, block_end, \
        num_dimensions = X.shape[1], \
        num_clusters = centroids.shape[0], \
        num_chunks = (num_X + chunk_size_kmeans - 1) / chunk_size_kmeans, \
        num_blocks = (num_X + chunk_size_search - 1) / chunk_size_search
    cdef float norm, worst_distance, distance, query_norm, true_distance, \
        block_min_distance
    cdef const float* row
    cdef uninitialized_vector[unsigned] cluster_offsets_buffer
    cdef uninitialized_vector[float] centroid_norms_buffer, \
        temp_distances_buffer, query_X_buffer, block_distances_buffer
    cdef uninitialized_vector[unsigned long long] pairs_buffer
    cdef vector[uninitialized_vector[float]] thread_distances, \
        thread_query_X, thread_block_distances
    cdef vector[uninitialized_vector[unsigned long long]] thread_pairs
    centroid_norms_buffer.resize(num_clusters)
    cluster_offsets_buffer.resize(num_clusters + 1)
    cdef float[::1] block_distances, centroid_norms = \
        <float[:num_clusters]> centroid_norms_buffer.data()
    cdef float[:, ::1] temp_distances, query_X
    cdef unsigned[::1] cluster_offsets = \
        <unsigned[:num_clusters + 1]> cluster_offsets_buffer.data()
    cdef unsigned long long[::1] pairs

    # Build a prefix-sum array of cluster offsets, and find the largest cluster
    # (used to size the per-block sgemm output matrix)
    cluster_offsets[0] = 0
    max_cluster_size = 0
    for i in range(num_clusters):
        cluster_offsets[i + 1] = cluster_offsets[i] + num_cells_per_cluster[i]
        if num_cells_per_cluster[i] > max_cluster_size:
            max_cluster_size = num_cells_per_cluster[i]

    # Pick a size to tile over clusters that keeps the partial_distances output
    # in cache for the subsequent scan, rather than evicting to main memory.
    # Used only in the centroid-ranking phase.
    cluster_tile_size = 1024 * 1024 / (chunk_size_kmeans * sizeof(float))
    if cluster_tile_size == 0:
        cluster_tile_size = 1
    if cluster_tile_size > num_clusters:
        cluster_tile_size = num_clusters

    # The neighbor-search phase processes queries in blocks of
    # `chunk_size_search`. Because `X` is sorted by k-means cluster, a block of
    # consecutive queries is typically drawn from one or two clusters and
    # therefore shares many of its `num_clusters_searched` nearest clusters.
    # For each block we pack each `(cluster, query_in_block)` pair into an
    # unsigned long long with the cluster label in the upper 32 bits, sort to
    # obtain cluster-major order, then go through the sorted list grouping
    # consecutive pairs with the same cluster. Each unique cluster in the block
    # is visited exactly once; we gather the queries that want it into a
    # contiguous buffer and invoke sgemm via `partial_distances()` to compute
    # all `(query, cell)` partial distances ||C||² - 2 X·C at once. We then add
    # each query's squared norm ||X||² (precomputed in `kmeans()` and passed in
    # via `cell_norms`) to recover the true squared Euclidean distance for the
    # heap comparison.

    # Force the parallel path for Mac for floating-point consistency between 1
    # and multiple threads.

    num_threads = min(num_threads, num_chunks)
    if num_threads <= 1 and not is_mac:
        PyErr_CheckSignals()

        # Calculate the squared L2 norm of each centroid, ||C||²
        for i in range(num_clusters):
            norm = centroids[i, 0] * centroids[i, 0]
            for j in range(1, num_dimensions):
                norm += centroids[i, j] * centroids[i, j]
            centroid_norms[i] = norm

        # Find the `num_clusters_searched` nearest centroids of each cell in
        # `X`, storing their indices in `nearest_clusters`. Use a max-heap to
        # keep track of the `num_clusters_searched` smallest distances.
        temp_distances_buffer.resize(chunk_size_kmeans * cluster_tile_size)
        temp_distances = <float[:chunk_size_kmeans, :cluster_tile_size]> \
            temp_distances_buffer.data()
        for chunk_index in range(num_chunks):
            start = chunk_index * chunk_size_kmeans
            chunk_num_X = num_X - start if chunk_index == num_chunks - 1 else \
                chunk_size_kmeans

            for i in range(chunk_num_X):
                for j in range(num_clusters_searched):
                    centroid_distances[start + i, j] = FLT_MAX

            tile_start = 0
            while tile_start < num_clusters:
                current_tile_size = cluster_tile_size \
                    if tile_start + cluster_tile_size <= num_clusters \
                    else num_clusters - tile_start

                # Calculate the distance from each cell in the chunk to each
                # centroid in the tile. Use the identity:
                # ||X - C||² = ||X||² - 2 * X.dot(C.T) + ||C||²
                # but skip calculating ||X||² since the ranking of centroid
                # distances for a given cell does not depend on ||X||².
                partial_distances(&X[start, 0], &centroids[tile_start, 0],
                                  &centroid_norms[tile_start],
                                  &temp_distances[0, 0],
                                  chunk_num_X, current_tile_size,
                                  cluster_tile_size, num_dimensions)
                for i in range(chunk_num_X):
                    worst_distance = centroid_distances[start + i, 0]
                    for cluster_index in range(current_tile_size):
                        distance = temp_distances[i, cluster_index]

                        # If this centroid is one of the
                        # `num_clusters_searched` nearest centroids found so
                        # far, add it to the heap, and remove the formerly
                        # `num_clusters_searched`th-nearest centroid (which
                        # is now no longer in the top
                        # `num_clusters_searched` centroids)
                        if distance < worst_distance:
                            max_heap_replace_top(
                                &nearest_clusters[start + i, 0],
                                &centroid_distances[start + i, 0],
                                tile_start + cluster_index, distance,
                                num_clusters_searched)
                            worst_distance = centroid_distances[start + i, 0]
                tile_start = tile_start + cluster_tile_size

            # Sort the heap to get nearest clusters in ascending order of
            # distance
            for i in range(chunk_num_X):
                max_heap_sort(&nearest_clusters[start + i, 0],
                              &centroid_distances[start + i, 0],
                              num_clusters_searched)

        PyErr_CheckSignals()

        # Process queries block by block, sharing cluster loads across the
        # block via a cluster-sorted pair list
        pairs_buffer.resize(chunk_size_search * num_clusters_searched)
        pairs = \
            <unsigned long long[:chunk_size_search * num_clusters_searched]> \
            pairs_buffer.data()
        query_X_buffer.resize(chunk_size_search * num_dimensions)
        query_X = <float[:chunk_size_search, :num_dimensions]> \
            query_X_buffer.data()
        block_distances_buffer.resize(chunk_size_search * max_cluster_size)
        block_distances = <float[:chunk_size_search * max_cluster_size]> \
            block_distances_buffer.data()
        for block_index in range(num_blocks):
            block_start = block_index * chunk_size_search
            num_cells_in_block = num_X - block_start \
                if block_start + chunk_size_search > num_X \
                else chunk_size_search

            # Initialize each query's heap to a sentinel distance
            for query_index_in_block in range(num_cells_in_block):
                query_sorted = block_start + query_index_in_block
                for j in range(num_neighbors):
                    distances[query_sorted, j] = FLT_MAX

            # Pack each `(cluster, query_in_block)` pair into the upper and
            # lower 32 bits of an unsigned long long so that sorting yields
            # cluster-major order
            num_pairs = 0
            for query_index_in_block in range(num_cells_in_block):
                query_sorted = block_start + query_index_in_block
                for cluster_index in range(num_clusters_searched):
                    cluster_label = \
                        nearest_clusters[query_sorted, cluster_index]
                    pairs[num_pairs] = \
                        (<unsigned long long> cluster_label << 32) | \
                        query_index_in_block
                    num_pairs = num_pairs + 1

            sort(&pairs[0], &pairs[num_pairs])

            # Go through the sorted pair list, processing each unique cluster
            # exactly once per block. Gather queries into a contiguous buffer
            # and compute ||C||² - 2 X·C for every (query, cell) pair via
            # sgemm; then add each query's ||X||² to recover the true squared
            # distance.
            pair_index = 0
            while pair_index < num_pairs:
                cluster_label = <unsigned>(pairs[pair_index] >> 32)
                cluster_start = cluster_offsets[cluster_label]
                cluster_size = \
                    cluster_offsets[cluster_label + 1] - cluster_start
                next_pair_index = pair_index + 1
                while next_pair_index < num_pairs and \
                        <unsigned>(pairs[next_pair_index] >> 32) == \
                        cluster_label:
                    next_pair_index = next_pair_index + 1
                num_queries_in_cluster = next_pair_index - pair_index

                for query_in_cluster_index in range(num_queries_in_cluster):
                    query_index_in_block = <unsigned> pairs[
                        pair_index + query_in_cluster_index]
                    query_sorted = block_start + query_index_in_block
                    memcpy(&query_X[query_in_cluster_index, 0],
                           &X[query_sorted, 0], num_dimensions * sizeof(float))

                partial_distances(&query_X[0, 0], &X[cluster_start, 0],
                                  &cell_norms[cluster_start],
                                  &block_distances[0], num_queries_in_cluster,
                                  cluster_size, cluster_size, num_dimensions)

                for query_in_cluster_index in range(num_queries_in_cluster):
                    query_index_in_block = <unsigned> pairs[
                        pair_index + query_in_cluster_index]
                    query_sorted = block_start + query_index_in_block
                    query_norm = cell_norms[query_sorted]
                    worst_distance = distances[query_sorted, 0]
                    row = &block_distances[
                        query_in_cluster_index * cluster_size]
                    # Skip blocks of 64 cells where even the nearest one
                    # does not beat the current worst neighbor
                    for block_offset in range(0, cluster_size, 64):
                        block_end = min(block_offset + 64, cluster_size)
                        block_min_distance = row[block_offset]
                        for neighbor_offset in range(block_offset + 1,
                                                     block_end):
                            block_min_distance = min(block_min_distance,
                                                     row[neighbor_offset])
                        if query_norm + block_min_distance >= worst_distance:
                            continue
                        for neighbor_offset in range(block_offset,
                                                     block_end):
                            neighbor = cluster_start + neighbor_offset
                            if query_sorted == neighbor:
                                continue  # skip self-neighbors
                            true_distance = query_norm + row[neighbor_offset]
                            if true_distance < worst_distance:
                                max_heap_replace_top(
                                    &neighbors[query_sorted, 0],
                                    &distances[query_sorted, 0],
                                    sorted_order[neighbor], true_distance,
                                    num_neighbors)
                                worst_distance = distances[query_sorted, 0]

                pair_index = next_pair_index

            # Sort each query's heap to get neighbors in ascending order of
            # distance
            for query_index_in_block in range(num_cells_in_block):
                query_sorted = block_start + query_index_in_block
                max_heap_sort(&neighbors[query_sorted, 0],
                              &distances[query_sorted, 0], num_neighbors)

            # Check for KeyboardInterrupts
            if block_index % 128 == 127:
                PyErr_CheckSignals()
    else:
        with nogil:
            # Calculate the squared L2 norm of each centroid, ||C||²
            for i in prange(num_clusters, num_threads=num_threads):
                norm = centroids[i, 0] * centroids[i, 0]
                for j in range(1, num_dimensions):
                    norm = norm + centroids[i, j] * centroids[i, j]
                centroid_norms[i] = norm

            # Find the `num_clusters_searched` nearest centroids of each cell
            # in `X`, storing their indices in `nearest_clusters`. Use a
            # max-heap to keep track of the `num_clusters_searched` smallest
            # distances.
            thread_distances.resize(num_threads)
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_distances[thread_index].resize(
                    chunk_size_kmeans * cluster_tile_size)
                for chunk_index in prange(num_chunks):
                    start = chunk_index * chunk_size_kmeans
                    chunk_num_X = num_X - start \
                        if chunk_index == num_chunks - 1 \
                        else chunk_size_kmeans

                    for i in range(chunk_num_X):
                        for j in range(num_clusters_searched):
                            centroid_distances[start + i, j] = FLT_MAX

                    tile_start = 0
                    while tile_start < num_clusters:
                        current_tile_size = cluster_tile_size \
                            if tile_start + cluster_tile_size <= num_clusters \
                            else num_clusters - tile_start

                        # Calculate the distance from each cell in the chunk
                        # to each centroid in the tile. Use the identity:
                        # ||X - C||² = ||X||² - 2 * X.dot(C.T) + ||C||²
                        # but skip calculating ||X||² since the ranking of
                        # centroid distances for a given cell does not depend
                        # on ||X||².
                        partial_distances(
                            &X[start, 0], &centroids[tile_start, 0],
                            &centroid_norms[tile_start],
                            thread_distances[thread_index].data(), chunk_num_X,
                            current_tile_size, cluster_tile_size,
                            num_dimensions)
                        for i in range(chunk_num_X):
                            worst_distance = centroid_distances[start + i, 0]
                            for cluster_index in range(current_tile_size):
                                distance = thread_distances[thread_index][
                                    i * cluster_tile_size + cluster_index]

                                # If this centroid is one of the
                                # `num_clusters_searched` nearest centroids
                                # found so far, add it to the heap, and remove
                                # the formerly `num_clusters_searched`th-
                                # nearest centroid (which is now no longer in
                                # the top `num_clusters_searched` centroids)
                                if distance < worst_distance:
                                    max_heap_replace_top(
                                        &nearest_clusters[start + i, 0],
                                        &centroid_distances[start + i, 0],
                                        tile_start + cluster_index, distance,
                                        num_clusters_searched)
                                    worst_distance = \
                                        centroid_distances[start + i, 0]
                        tile_start = tile_start + cluster_tile_size

                    # Sort the heap to get nearest clusters in ascending
                    # order of distance
                    for i in range(chunk_num_X):
                        max_heap_sort(&nearest_clusters[start + i, 0],
                                      &centroid_distances[start + i, 0],
                                      num_clusters_searched)

                    # Check for KeyboardInterrupts
                    if chunk_index % 8 == 7:
                        with gil:
                            PyErr_CheckSignals()

            # Process queries block by block, sharing cluster loads across the
            # block via a cluster-sorted pair list
            thread_pairs.resize(num_threads)
            thread_query_X.resize(num_threads)
            thread_block_distances.resize(num_threads)
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_pairs[thread_index].resize(
                    chunk_size_search * num_clusters_searched)
                thread_query_X[thread_index].resize(
                    chunk_size_search * num_dimensions)
                thread_block_distances[thread_index].resize(
                    chunk_size_search * max_cluster_size)
                for block_index in prange(num_blocks):
                    block_start = block_index * chunk_size_search
                    num_cells_in_block = num_X - block_start \
                        if block_start + chunk_size_search > num_X \
                        else chunk_size_search

                    # Initialize each query's heap to a sentinel distance
                    for query_index_in_block in range(num_cells_in_block):
                        query_sorted = block_start + query_index_in_block
                        for j in range(num_neighbors):
                            distances[query_sorted, j] = FLT_MAX

                    # Pack each `(cluster, query_in_block)` pair into the upper
                    # and lower 32 bits of an unsigned long long so that
                    # sorting yields cluster-major order
                    num_pairs = 0
                    for query_index_in_block in range(num_cells_in_block):
                        query_sorted = block_start + query_index_in_block
                        for cluster_index in range(num_clusters_searched):
                            cluster_label = \
                                nearest_clusters[query_sorted, cluster_index]
                            thread_pairs[thread_index][num_pairs] = \
                                (<unsigned long long> cluster_label << 32) | \
                                query_index_in_block
                            num_pairs = num_pairs + 1

                    sort(thread_pairs[thread_index].data(),
                         thread_pairs[thread_index].data() + num_pairs)

                    # Go through the sorted pair list, processing each unique
                    # cluster exactly once per block. Gather queries into a
                    # contiguous buffer and compute ||C||² - 2 X·C for every
                    # (query, cell) pair via sgemm; then add each query's
                    # ||X||² to recover the true squared distance.
                    pair_index = 0
                    while pair_index < num_pairs:
                        cluster_label = <unsigned>(
                            thread_pairs[thread_index][pair_index] >> 32)
                        cluster_start = cluster_offsets[cluster_label]
                        cluster_size = \
                            cluster_offsets[cluster_label + 1] - cluster_start
                        next_pair_index = pair_index + 1
                        while next_pair_index < num_pairs and <unsigned>(
                                thread_pairs[
                                    thread_index][next_pair_index] >> 32) == \
                                cluster_label:
                            next_pair_index = next_pair_index + 1
                        num_queries_in_cluster = next_pair_index - pair_index

                        for query_in_cluster_index in \
                                range(num_queries_in_cluster):
                            query_index_in_block = \
                                <unsigned> thread_pairs[thread_index][
                                pair_index + query_in_cluster_index]
                            query_sorted = block_start + query_index_in_block
                            memcpy(thread_query_X[thread_index].data() +
                                   query_in_cluster_index * num_dimensions,
                                   &X[query_sorted, 0],
                                   num_dimensions * sizeof(float))

                        partial_distances(
                            thread_query_X[thread_index].data(),
                            &X[cluster_start, 0], &cell_norms[cluster_start],
                            thread_block_distances[thread_index].data(),
                            num_queries_in_cluster, cluster_size, cluster_size,
                            num_dimensions)

                        for query_in_cluster_index in \
                                range(num_queries_in_cluster):
                            query_index_in_block = \
                                <unsigned> thread_pairs[thread_index][
                                pair_index + query_in_cluster_index]
                            query_sorted = block_start + query_index_in_block
                            query_norm = cell_norms[query_sorted]
                            worst_distance = distances[query_sorted, 0]
                            row = \
                                thread_block_distances[thread_index].data() + \
                                query_in_cluster_index * cluster_size
                            for block_offset in range(0, cluster_size, 64):
                                block_end = \
                                    min(block_offset + 64, cluster_size)
                                block_min_distance = row[block_offset]
                                for neighbor_offset in range(
                                        block_offset + 1, block_end):
                                    block_min_distance = min(
                                        block_min_distance,
                                        row[neighbor_offset])
                                if query_norm + block_min_distance >= \
                                        worst_distance:
                                    continue
                                for neighbor_offset in range(block_offset,
                                                             block_end):
                                    neighbor = cluster_start + neighbor_offset
                                    if query_sorted == neighbor:
                                        continue  # skip self-neighbors
                                    true_distance = \
                                        query_norm + row[neighbor_offset]
                                    if true_distance < worst_distance:
                                        max_heap_replace_top(
                                            &neighbors[query_sorted, 0],
                                            &distances[query_sorted, 0],
                                            sorted_order[neighbor],
                                            true_distance, num_neighbors)
                                        worst_distance = \
                                            distances[query_sorted, 0]

                        pair_index = next_pair_index

                    # Sort each query's heap to get neighbors in ascending
                    # order of distance
                    for query_index_in_block in range(num_cells_in_block):
                        query_sorted = block_start + query_index_in_block
                        max_heap_sort(&neighbors[query_sorted, 0],
                                      &distances[query_sorted, 0],
                                      num_neighbors)

                    # Check for KeyboardInterrupts
                    if block_index % 128 == 127:
                        with gil:
                            PyErr_CheckSignals()


def knn_cross(const float[:, ::1] Y,
              const float[:, ::1] X,
              const float[:, ::1] centroids,
              const unsigned[::1] num_cells_per_cluster,
              const float[::1] cell_norms,
              unsigned[:, ::1] neighbors,
              float[:, ::1] distances,
              float[:, ::1] centroid_distances,
              unsigned[:, ::1] nearest_clusters,
              float[::1] query_norms,
              const unsigned num_neighbors,
              const unsigned num_clusters_searched,
              const unsigned chunk_size_kmeans,
              const unsigned chunk_size_search,
              const bint is_mac,
              unsigned num_threads):
    # Find the `num_neighbors` nearest neighbors of each cell in `Y` among the
    # cells in `X`. Store the nearest-neighbor indices in `neighbors` and
    # distances in `distances`. The caller must have grouped `X` by cluster;
    # this avoids the need for an explicit inverted-file index. `cell_norms`
    # holds ||X||² (precomputed in `kmeans()`); `query_norms` is a scratch
    # buffer that this function fills with ||Y||².

    cdef unsigned long long num_Y = Y.shape[0], num_X = X.shape[0]
    cdef unsigned i, j, k, cluster_tile_size, thread_index, chunk_index, \
        start, end, chunk_num_Y, cluster_index, cluster_label, neighbor, \
        tile_start, current_tile_size, block_index, block_start, \
        num_cells_in_block, query_index_in_block, query_sorted, num_pairs, \
        pair_index, next_pair_index, query_in_cluster_index, \
        num_queries_in_cluster, max_cluster_size, cluster_start, \
        cluster_size, neighbor_offset, block_offset, block_end, \
        num_dimensions = X.shape[1], \
        num_clusters = centroids.shape[0], \
        num_chunks = (num_Y + chunk_size_kmeans - 1) / \
            chunk_size_kmeans, \
        num_blocks = (num_Y + chunk_size_search - 1) / \
            chunk_size_search
    cdef float norm, worst_distance, distance, query_norm, true_distance, \
        block_min_distance
    cdef const float* row
    cdef uninitialized_vector[unsigned] cluster_offsets_buffer
    cdef uninitialized_vector[float] centroid_norms_buffer, \
        temp_distances_buffer, query_Y_buffer, block_distances_buffer
    cdef uninitialized_vector[unsigned long long] pairs_buffer
    cdef vector[uninitialized_vector[float]] thread_distances, \
        thread_query_Y, thread_block_distances
    cdef vector[uninitialized_vector[unsigned long long]] thread_pairs
    centroid_norms_buffer.resize(num_clusters)
    cluster_offsets_buffer.resize(num_clusters + 1)
    cdef float[::1] block_distances, centroid_norms = \
        <float[:num_clusters]> centroid_norms_buffer.data()
    cdef float[:, ::1] temp_distances, query_Y
    cdef unsigned[::1] cluster_offsets = \
        <unsigned[:num_clusters + 1]> cluster_offsets_buffer.data()
    cdef unsigned long long[::1] pairs

    # Build a prefix-sum array of cluster offsets, and find the largest cluster
    # (used to size the per-block sgemm output matrix)
    cluster_offsets[0] = 0
    max_cluster_size = 0
    for i in range(num_clusters):
        cluster_offsets[i + 1] = cluster_offsets[i] + num_cells_per_cluster[i]
        if num_cells_per_cluster[i] > max_cluster_size:
            max_cluster_size = num_cells_per_cluster[i]

    # Pick a size to tile over clusters that keeps the partial_distances output
    # in cache for the subsequent scan, rather than evicting to main memory.
    # Used only in the centroid-ranking phase.
    cluster_tile_size = 1024 * 1024 / (chunk_size_kmeans * sizeof(float))
    if cluster_tile_size == 0:
        cluster_tile_size = 1
    if cluster_tile_size > num_clusters:
        cluster_tile_size = num_clusters

    # Force the parallel path for Mac for floating-point consistency between 1
    # and multiple threads

    num_threads = min(num_threads, num_chunks)
    if num_threads <= 1 and not is_mac:
        PyErr_CheckSignals()

        # Calculate the squared L2 norm of each centroid, ||C||²
        for i in range(num_clusters):
            norm = centroids[i, 0] * centroids[i, 0]
            for j in range(1, num_dimensions):
                norm += centroids[i, j] * centroids[i, j]
            centroid_norms[i] = norm

        # Calculate the squared L2 norm of each query in `Y`, ||Y||²
        for i in range(num_Y):
            norm = Y[i, 0] * Y[i, 0]
            for j in range(1, num_dimensions):
                norm += Y[i, j] * Y[i, j]
            query_norms[i] = norm

        # Find the `num_clusters_searched` nearest centroids of each cell in
        # `Y`, storing their indices in `nearest_clusters`. Use a max-heap to
        # keep track of the `num_clusters_searched` smallest distances.
        temp_distances_buffer.resize(chunk_size_kmeans * cluster_tile_size)
        temp_distances = <float[:chunk_size_kmeans, :cluster_tile_size]> \
            temp_distances_buffer.data()
        for chunk_index in range(num_chunks):
            start = chunk_index * chunk_size_kmeans
            chunk_num_Y = num_Y - start if chunk_index == num_chunks - 1 else \
                chunk_size_kmeans

            for i in range(chunk_num_Y):
                for j in range(num_clusters_searched):
                    centroid_distances[start + i, j] = FLT_MAX

            tile_start = 0
            while tile_start < num_clusters:
                current_tile_size = cluster_tile_size \
                    if tile_start + cluster_tile_size <= num_clusters \
                    else num_clusters - tile_start

                # Calculate the distance from each cell in the chunk to each
                # centroid in the tile. Use the identity:
                # ||Y - C||² = ||Y||² - 2 * Y.dot(C.T) + ||C||²
                # but skip calculating ||Y||² since the ranking of centroid
                # distances for a given cell does not depend on ||Y||².
                partial_distances(&Y[start, 0], &centroids[tile_start, 0],
                                  &centroid_norms[tile_start],
                                  &temp_distances[0, 0],
                                  chunk_num_Y, current_tile_size,
                                  cluster_tile_size, num_dimensions)
                for i in range(chunk_num_Y):
                    worst_distance = centroid_distances[start + i, 0]
                    for cluster_index in range(current_tile_size):
                        distance = temp_distances[i, cluster_index]

                        # If this centroid is one of the
                        # `num_clusters_searched` nearest centroids found so
                        # far, add it to the heap, and remove the formerly
                        # `num_clusters_searched`th-nearest centroid (which is
                        # now no longer in the top `num_clusters_searched`
                        # centroids)
                        if distance < worst_distance:
                            max_heap_replace_top(
                                &nearest_clusters[start + i, 0],
                                &centroid_distances[start + i, 0],
                                tile_start + cluster_index, distance,
                                num_clusters_searched)
                            worst_distance = centroid_distances[start + i, 0]
                tile_start = tile_start + cluster_tile_size

            # Sort the heap to get nearest clusters in ascending order of
            # distance
            for i in range(chunk_num_Y):
                max_heap_sort(&nearest_clusters[start + i, 0],
                              &centroid_distances[start + i, 0],
                              num_clusters_searched)

        PyErr_CheckSignals()

        # Process queries block by block, sharing cluster loads across the
        # block via a cluster-sorted pair list
        pairs_buffer.resize(chunk_size_search * num_clusters_searched)
        pairs = \
            <unsigned long long[:chunk_size_search * num_clusters_searched]> \
            pairs_buffer.data()
        query_Y_buffer.resize(chunk_size_search * num_dimensions)
        query_Y = <float[:chunk_size_search, :num_dimensions]> \
            query_Y_buffer.data()
        block_distances_buffer.resize(chunk_size_search * max_cluster_size)
        block_distances = <float[:chunk_size_search * max_cluster_size]> \
            block_distances_buffer.data()
        for block_index in range(num_blocks):
            block_start = block_index * chunk_size_search
            num_cells_in_block = num_Y - block_start \
                if block_start + chunk_size_search > num_Y \
                else chunk_size_search

            # Initialize each query's heap to a sentinel distance
            for query_index_in_block in range(num_cells_in_block):
                query_sorted = block_start + query_index_in_block
                for j in range(num_neighbors):
                    distances[query_sorted, j] = FLT_MAX

            # Pack each `(cluster, query_in_block)` pair into the upper and
            # lower 32 bits of an unsigned long long so that sorting yields
            # cluster-major order
            num_pairs = 0
            for query_index_in_block in range(num_cells_in_block):
                query_sorted = block_start + query_index_in_block
                for cluster_index in range(num_clusters_searched):
                    cluster_label = \
                        nearest_clusters[query_sorted, cluster_index]
                    pairs[num_pairs] = \
                        (<unsigned long long> cluster_label << 32) | \
                        query_index_in_block
                    num_pairs = num_pairs + 1

            sort(&pairs[0], &pairs[num_pairs])

            # Go through the sorted pair list, processing each unique cluster
            # exactly once per block. Gather queries into a contiguous buffer
            # and compute ||C||² - 2 Y·C for every (query, cell) pair via
            # sgemm; then add each query's ||Y||² to recover the true squared
            # distance. Unlike `knn_self()`, do not check for self-neighbors.
            pair_index = 0
            while pair_index < num_pairs:
                cluster_label = <unsigned>(pairs[pair_index] >> 32)
                cluster_start = cluster_offsets[cluster_label]
                cluster_size = \
                    cluster_offsets[cluster_label + 1] - cluster_start
                next_pair_index = pair_index + 1
                while next_pair_index < num_pairs and \
                        <unsigned>(pairs[next_pair_index] >> 32) == \
                        cluster_label:
                    next_pair_index = next_pair_index + 1
                num_queries_in_cluster = next_pair_index - pair_index

                for query_in_cluster_index in range(num_queries_in_cluster):
                    query_index_in_block = <unsigned> pairs[
                        pair_index + query_in_cluster_index]
                    query_sorted = block_start + query_index_in_block
                    memcpy(&query_Y[query_in_cluster_index, 0],
                           &Y[query_sorted, 0], num_dimensions * sizeof(float))

                partial_distances(&query_Y[0, 0], &X[cluster_start, 0],
                                  &cell_norms[cluster_start],
                                  &block_distances[0], num_queries_in_cluster,
                                  cluster_size, cluster_size, num_dimensions)

                for query_in_cluster_index in range(num_queries_in_cluster):
                    query_index_in_block = <unsigned> pairs[
                        pair_index + query_in_cluster_index]
                    query_sorted = block_start + query_index_in_block
                    query_norm = query_norms[query_sorted]
                    worst_distance = distances[query_sorted, 0]
                    row = &block_distances[
                        query_in_cluster_index * cluster_size]
                    # Skip blocks of 64 cells where even the nearest one
                    # does not beat the current worst neighbor
                    for block_offset in range(0, cluster_size, 64):
                        block_end = min(block_offset + 64, cluster_size)
                        block_min_distance = row[block_offset]
                        for neighbor_offset in range(block_offset + 1,
                                                     block_end):
                            block_min_distance = min(block_min_distance,
                                                     row[neighbor_offset])
                        if query_norm + block_min_distance >= worst_distance:
                            continue
                        for neighbor_offset in range(block_offset,
                                                     block_end):
                            neighbor = cluster_start + neighbor_offset
                            true_distance = query_norm + row[neighbor_offset]
                            if true_distance < worst_distance:
                                max_heap_replace_top(
                                    &neighbors[query_sorted, 0],
                                    &distances[query_sorted, 0], neighbor,
                                    true_distance, num_neighbors)
                                worst_distance = distances[query_sorted, 0]

                pair_index = next_pair_index

            # Sort each query's heap to get neighbors in ascending order of
            # distance
            for query_index_in_block in range(num_cells_in_block):
                query_sorted = block_start + query_index_in_block
                max_heap_sort(&neighbors[query_sorted, 0],
                              &distances[query_sorted, 0], num_neighbors)

            # Check for KeyboardInterrupts
            if block_index % 128 == 127:
                PyErr_CheckSignals()
    else:
        with nogil:
            # Calculate the squared L2 norm of each centroid, ||C||²
            for i in prange(num_clusters, num_threads=num_threads):
                norm = centroids[i, 0] * centroids[i, 0]
                for j in range(1, num_dimensions):
                    norm = norm + centroids[i, j] * centroids[i, j]
                centroid_norms[i] = norm

            # Calculate the squared L2 norm of each query in `Y`, ||Y||²
            for i in prange(num_Y, num_threads=num_threads):
                norm = Y[i, 0] * Y[i, 0]
                for j in range(1, num_dimensions):
                    norm = norm + Y[i, j] * Y[i, j]
                query_norms[i] = norm

            # Find the `num_clusters_searched` nearest centroids of each cell
            # in `Y`, storing their indices in `nearest_clusters`. Use a
            # max-heap to keep track of the `num_clusters_searched` smallest
            # distances.
            thread_distances.resize(num_threads)
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_distances[thread_index].resize(
                    chunk_size_kmeans * cluster_tile_size)
                for chunk_index in prange(num_chunks):
                    start = chunk_index * chunk_size_kmeans
                    chunk_num_Y = num_Y - start \
                        if chunk_index == num_chunks - 1 \
                        else chunk_size_kmeans

                    for i in range(chunk_num_Y):
                        for j in range(num_clusters_searched):
                            centroid_distances[start + i, j] = FLT_MAX

                    tile_start = 0
                    while tile_start < num_clusters:
                        current_tile_size = cluster_tile_size \
                            if tile_start + cluster_tile_size <= num_clusters \
                            else num_clusters - tile_start

                        # Calculate the distance from each cell in the chunk
                        # to each centroid in the tile. Use the identity:
                        # ||Y - C||² = ||Y||² - 2 * Y.dot(C.T) + ||C||²
                        # but skip calculating ||Y||² since the ranking of
                        # centroid distances for a given cell does not depend
                        # on ||Y||².
                        partial_distances(
                            &Y[start, 0], &centroids[tile_start, 0],
                            &centroid_norms[tile_start],
                            thread_distances[thread_index].data(), chunk_num_Y,
                            current_tile_size, cluster_tile_size,
                            num_dimensions)
                        for i in range(chunk_num_Y):
                            worst_distance = centroid_distances[start + i, 0]
                            for cluster_index in range(current_tile_size):
                                distance = thread_distances[thread_index][
                                    i * cluster_tile_size + cluster_index]

                                # If this centroid is one of the
                                # `num_clusters_searched` nearest centroids
                                # found so far, add it to the heap, and remove
                                # the formerly `num_clusters_searched`th-
                                # nearest centroid (which is now no longer in
                                # the top `num_clusters_searched` centroids)
                                if distance < worst_distance:
                                    max_heap_replace_top(
                                        &nearest_clusters[start + i, 0],
                                        &centroid_distances[start + i, 0],
                                        tile_start + cluster_index, distance,
                                        num_clusters_searched)
                                    worst_distance = \
                                        centroid_distances[start + i, 0]
                        tile_start = tile_start + cluster_tile_size

                    # Sort the heap to get nearest clusters in ascending order
                    # of distance
                    for i in range(chunk_num_Y):
                        max_heap_sort(&nearest_clusters[start + i, 0],
                                      &centroid_distances[start + i, 0],
                                      num_clusters_searched)

                    # Check for KeyboardInterrupts
                    if chunk_index % 8 == 7:
                        with gil:
                            PyErr_CheckSignals()

            # Process queries block by block, sharing cluster loads across the
            # block via a cluster-sorted pair list
            thread_pairs.resize(num_threads)
            thread_query_Y.resize(num_threads)
            thread_block_distances.resize(num_threads)
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_pairs[thread_index].resize(
                    chunk_size_search * num_clusters_searched)
                thread_query_Y[thread_index].resize(
                    chunk_size_search * num_dimensions)
                thread_block_distances[thread_index].resize(
                    chunk_size_search * max_cluster_size)
                for block_index in prange(num_blocks):
                    block_start = block_index * chunk_size_search
                    num_cells_in_block = num_Y - block_start \
                        if block_start + chunk_size_search > num_Y \
                        else chunk_size_search

                    # Initialize each query's heap to a sentinel distance
                    for query_index_in_block in range(num_cells_in_block):
                        query_sorted = block_start + query_index_in_block
                        for j in range(num_neighbors):
                            distances[query_sorted, j] = FLT_MAX

                    # Pack each `(cluster, query_in_block)` pair into the upper
                    # and lower 32 bits of an unsigned long long so that
                    # sorting yields cluster-major order
                    num_pairs = 0
                    for query_index_in_block in range(num_cells_in_block):
                        query_sorted = block_start + query_index_in_block
                        for cluster_index in range(num_clusters_searched):
                            cluster_label = \
                                nearest_clusters[query_sorted, cluster_index]
                            thread_pairs[thread_index][num_pairs] = \
                                (<unsigned long long> cluster_label << 32) | \
                                query_index_in_block
                            num_pairs = num_pairs + 1

                    sort(thread_pairs[thread_index].data(),
                         thread_pairs[thread_index].data() + num_pairs)

                    # Go through the sorted pair list, processing each unique
                    # cluster exactly once per block. Gather queries into a
                    # contiguous buffer and compute ||C||² - 2 Y·C for every
                    # (query, cell) pair via sgemm; then add each query's
                    # ||Y||² to recover the true squared distance. Unlike
                    # `knn_self()`, do not check for self-neighbors.
                    pair_index = 0
                    while pair_index < num_pairs:
                        cluster_label = <unsigned>(
                            thread_pairs[thread_index][pair_index] >> 32)
                        cluster_start = cluster_offsets[cluster_label]
                        cluster_size = \
                            cluster_offsets[cluster_label + 1] - cluster_start
                        next_pair_index = pair_index + 1
                        while next_pair_index < num_pairs and <unsigned>(
                                thread_pairs[
                                    thread_index][next_pair_index] >> 32) == \
                                cluster_label:
                            next_pair_index = next_pair_index + 1
                        num_queries_in_cluster = next_pair_index - pair_index

                        for query_in_cluster_index in \
                                range(num_queries_in_cluster):
                            query_index_in_block = \
                                <unsigned> thread_pairs[thread_index][
                                pair_index + query_in_cluster_index]
                            query_sorted = block_start + query_index_in_block
                            memcpy(thread_query_Y[thread_index].data() +
                                   query_in_cluster_index * num_dimensions,
                                   &Y[query_sorted, 0],
                                   num_dimensions * sizeof(float))

                        partial_distances(
                            thread_query_Y[thread_index].data(),
                            &X[cluster_start, 0], &cell_norms[cluster_start],
                            thread_block_distances[thread_index].data(),
                            num_queries_in_cluster, cluster_size, cluster_size,
                            num_dimensions)

                        for query_in_cluster_index in \
                                range(num_queries_in_cluster):
                            query_index_in_block = \
                                <unsigned> thread_pairs[thread_index][
                                pair_index + query_in_cluster_index]
                            query_sorted = block_start + query_index_in_block
                            query_norm = query_norms[query_sorted]
                            worst_distance = distances[query_sorted, 0]
                            row = \
                                thread_block_distances[thread_index].data() + \
                                query_in_cluster_index * cluster_size
                            for block_offset in range(0, cluster_size, 64):
                                block_end = \
                                    min(block_offset + 64, cluster_size)
                                block_min_distance = row[block_offset]
                                for neighbor_offset in range(
                                        block_offset + 1, block_end):
                                    block_min_distance = min(
                                        block_min_distance,
                                        row[neighbor_offset])
                                if query_norm + block_min_distance >= \
                                        worst_distance:
                                    continue
                                for neighbor_offset in range(block_offset,
                                                             block_end):
                                    neighbor = cluster_start + neighbor_offset
                                    true_distance = \
                                        query_norm + row[neighbor_offset]
                                    if true_distance < worst_distance:
                                        max_heap_replace_top(
                                            &neighbors[query_sorted, 0],
                                            &distances[query_sorted, 0],
                                            neighbor, true_distance,
                                            num_neighbors)
                                        worst_distance = \
                                            distances[query_sorted, 0]

                        pair_index = next_pair_index

                    # Sort each query's heap to get neighbors in ascending
                    # order of distance
                    for query_index_in_block in range(num_cells_in_block):
                        query_sorted = block_start + query_index_in_block
                        max_heap_sort(&neighbors[query_sorted, 0],
                                      &distances[query_sorted, 0],
                                      num_neighbors)

                    # Check for KeyboardInterrupts
                    if block_index % 128 == 127:
                        with gil:
                            PyErr_CheckSignals()
