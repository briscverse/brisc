# Functionality for shared-nearest neighbor graph construction

cimport numpy as np
np.import_array()
from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libc.string cimport memcpy
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from .cyutils cimport get_thread_offset, uninitialized_vector


def snn(const unsigned[:, ::1] neighbors,
        const long long[::1] QCed_to_full_map,
        long long[::1] indptr,
        const unsigned min_shared_neighbors,
        str neighbors_key,
        unsigned num_threads):
    # Loosely based on the build_snn_graph function from
    # github.com/libscran/scran_graph_cluster

    cdef bint has_QC_column = QCed_to_full_map.shape[0] != 0
    cdef unsigned i, j, k, thread_index, num_unique_pruned_ks, \
        num_shared, size, twice_num_neighbors_plus_one = \
        2 * (neighbors.shape[1] + 1)  # +1 for self-neighbors
    cdef unsigned long long num_reverse_neighbors, cumsum, count, start, end, \
        l, num_cells = QCed_to_full_map.shape[0] if has_QC_column else \
            neighbors.shape[0], \
        reserve_size = (<unsigned long long> neighbors.shape[1] + 1) * \
            (neighbors.shape[1] + 1)
    cdef np.npy_intp nnz
    cdef uninitialized_vector[unsigned] reverse_indices_buffer
    cdef vector[unsigned long long] reverse_indptr_buffer
    cdef vector[unsigned] shared_neighbors
    cdef vector[vector[unsigned]] thread_shared_neighbors
    cdef np.ndarray[np.uint32_t, ndim=1] indices
    cdef np.ndarray[np.float32_t, ndim=1] data
    cdef unsigned[::1] reverse_indices, indices_view
    cdef unsigned long long[::1] reverse_indptr
    cdef float[::1] data_view
    cdef str error_message

    # Build the "reverse" mapping: which cells have each cell as a neighbor.
    # Also check that all neighbor indices are in-range.

    reverse_indptr_buffer.resize(num_cells + 1)
    reverse_indptr = <unsigned long long[:num_cells + 1]> \
        reverse_indptr_buffer.data()
    # First pass: count reverse neighbors and validate indices.
    # `reverse_indptr` is used as a temporary workspace for counts.
    num_reverse_neighbors = 0
    for i in range(num_cells):
        for j in neighbors[i]:
            if j >= num_cells:
                error_message = (
                    f'some nearest-neighbor indices in '
                    f'obsm[{neighbors_key!r}] are >= the total number of '
                    f'cells, {num_cells:,}. This may happen if you subset '
                    f'this SingleCell dataset between neighbors() and '
                    f'shared_neighbors(); if so, make sure to run neighbors() '
                    f'after, not before, subsetting.')
                raise ValueError(error_message)
            reverse_indptr[j] += 1
            num_reverse_neighbors += 1
    # Cumsum `reverse_indptr`, shifting right by 1, to convert it from an array
    # of counts into a true indptr
    cumsum = 0
    for i in range(num_cells):
        count = reverse_indptr[i]
        reverse_indptr[i] = cumsum
        cumsum += count
    reverse_indptr[num_cells] = num_reverse_neighbors
    # Second pass: populate `reverse_indices`. For each edge (i, j), place `i`
    # at the current position pointed to by `reverse_indptr[j]`, then increment
    # that pointer.
    reverse_indices_buffer.resize(num_reverse_neighbors)
    reverse_indices = \
        <unsigned[:num_reverse_neighbors]> reverse_indices_buffer.data()
    for i in range(num_cells):
        for j in neighbors[i]:
            # Get the next available write position for cell j
            count = reverse_indptr[j]
            # Write cell i into that position
            reverse_indices[count] = i
            # Increment the write position for cell j
            reverse_indptr[j] += 1
    # Reset `reverse_indptr`.
    # The pointers in `reverse_indptr` now point to the end of each block.
    # Shift all the values (except the last) one to the right, so that they
    # point to the start of each block instead.
    i = num_cells
    while i > 0:
        reverse_indptr[i] = reverse_indptr[i - 1]
        i -= 1
    reverse_indptr[0] = 0

    PyErr_CheckSignals()

    # The main SNN graph calculation also involves two passes: first to
    # calculate the number of shared nearest neighbors each cell has so we can
    # allocate `data` and `indices`, second to fill in the graph. This requires
    # repeating the calculation of the shared-nearest neighbor graph weights,
    # but ultimately saves time by avoiding temporary memory allocations for
    # each cell.
    #
    # Rather than maintaining a num_cells-sized counter array for each thread,
    # we collect all shared-neighbor candidate indices (with duplicates) into
    # a small vector, sort it, and count consecutive runs. The vector size per
    # cell is bounded by ~(num_neighbors + 1)^2.

    # First pass: calculate each cell's number of shared nearest neighbors

    num_threads = min(num_threads, num_cells)
    if not has_QC_column:
        if num_threads <= 1:
            shared_neighbors.reserve(reserve_size)
            for i in range(num_cells):
                # For this cell `i`, find all pairs of cells `j` and `k` where
                # `j` is a shared nearest neighbor of `i` and `k`. Do this by
                # looking up all neighbors (`j`) of this cell (`i`), then
                # looking up each of these neighbors' reverse neighbors (`k`).
                #
                # A twist: we want to include self-neighbors, but `neighbors`
                # does not include them. Handle self-neighbors in two ways:

                # 1) Perform an extra iteration of the outer loop (below) with
                #    `j = i`.
                start = reverse_indptr[i]
                end = reverse_indptr[i + 1]
                for l in range(start, end):
                    k = reverse_indices[l]
                    shared_neighbors.push_back(k)

                for j in neighbors[i]:  # outer loop
                    # 2) Perform an extra iteration of the inner loop (below)
                    #    with `k = j`
                    shared_neighbors.push_back(j)

                    start = reverse_indptr[j]
                    end = reverse_indptr[j + 1]
                    for l in range(start, end):  # inner loop
                        k = reverse_indices[l]
                        # Skip the case where `i == k`, so that the output SNN
                        # graph does not include self-edges.
                        if i == k:
                            continue

                        shared_neighbors.push_back(k)

                # For each other cell `k`, if the SNN weight between `i` and `k`
                # exceeds `min_shared_neighbors`, the SNN graph will contain an
                # edge between `i` and `k`; it will not be pruned. Tally up how
                # many `k`s this is true for (`num_unique_pruned_ks`).
                sort(shared_neighbors.begin(), shared_neighbors.end())
                num_unique_pruned_ks = 0
                l = 0
                size = shared_neighbors.size()
                while l < size:
                    k = shared_neighbors[l]
                    num_shared = 1
                    l += 1
                    while l < size and shared_neighbors[l] == k:
                        num_shared += 1
                        l += 1
                    if num_shared >= min_shared_neighbors:
                        num_unique_pruned_ks += 1

                # Store the number of unique `k`s after pruning in
                # `indptr` (we will take the cumsum later)
                indptr[i + 1] = num_unique_pruned_ks

                # Clear the list of the unique `k`s, so it can be reused for
                # the next cell. Crucially, do not deallocate its memory when
                # clearing, to avoid allocating new memory for each cell.
                shared_neighbors.clear()
        else:
            # Same as the single-threaded version except that
            # `shared_neighbors` is now `thread_shared_neighbors`

            thread_shared_neighbors.resize(num_threads)
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_shared_neighbors[thread_index].reserve(
                    reserve_size)
                for i in prange(num_cells):
                    start = reverse_indptr[i]
                    end = reverse_indptr[i + 1]
                    for l in range(start, end):
                        k = reverse_indices[l]
                        thread_shared_neighbors[thread_index].push_back(k)
                    for j in neighbors[i]:
                        thread_shared_neighbors[thread_index].push_back(j)
                        start = reverse_indptr[j]
                        end = reverse_indptr[j + 1]
                        for l in range(start, end):
                            k = reverse_indices[l]
                            if i == k:
                                continue
                            thread_shared_neighbors[
                                thread_index].push_back(k)
                    sort(thread_shared_neighbors[thread_index].begin(),
                         thread_shared_neighbors[thread_index].end())
                    num_unique_pruned_ks = 0
                    l = 0
                    size = thread_shared_neighbors[thread_index].size()
                    while l < size:
                        k = thread_shared_neighbors[thread_index][l]
                        num_shared = 1
                        l = l + 1
                        while l < size and \
                                thread_shared_neighbors[
                                    thread_index][l] == k:
                            num_shared = num_shared + 1
                            l = l + 1
                        if num_shared >= min_shared_neighbors:
                            num_unique_pruned_ks = \
                                num_unique_pruned_ks + 1
                    indptr[i + 1] = num_unique_pruned_ks
                    thread_shared_neighbors[thread_index].clear()
    else:
        # Same as the version without a QC column, but store in
        # `indptr[QCed_to_full_map[i] + 1]` instead of `indptr[i + 1]`.
        # Also, initialize `indptr` to 0 so that cells failing QC are not
        # left uninitialized.

        indptr[:] = 0
        if num_threads <= 1:
            shared_neighbors.reserve(reserve_size)
            for i in range(num_cells):
                start = reverse_indptr[i]
                end = reverse_indptr[i + 1]
                for l in range(start, end):
                    k = reverse_indices[l]
                    shared_neighbors.push_back(k)
                for j in neighbors[i]:
                    shared_neighbors.push_back(j)
                    start = reverse_indptr[j]
                    end = reverse_indptr[j + 1]
                    for l in range(start, end):
                        k = reverse_indices[l]
                        if i == k:
                            continue
                        shared_neighbors.push_back(k)
                sort(shared_neighbors.begin(), shared_neighbors.end())
                num_unique_pruned_ks = 0
                l = 0
                size = shared_neighbors.size()
                while l < size:
                    k = shared_neighbors[l]
                    num_shared = 1
                    l += 1
                    while l < size and shared_neighbors[l] == k:
                        num_shared += 1
                        l += 1
                    if num_shared >= min_shared_neighbors:
                        num_unique_pruned_ks += 1
                indptr[QCed_to_full_map[i] + 1] = num_unique_pruned_ks
                shared_neighbors.clear()
        else:
            thread_shared_neighbors.resize(num_threads)
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_shared_neighbors[thread_index].reserve(
                    reserve_size)
                for i in prange(num_cells):
                    start = reverse_indptr[i]
                    end = reverse_indptr[i + 1]
                    for l in range(start, end):
                        k = reverse_indices[l]
                        thread_shared_neighbors[thread_index].push_back(k)
                    for j in neighbors[i]:
                        thread_shared_neighbors[thread_index].push_back(j)
                        start = reverse_indptr[j]
                        end = reverse_indptr[j + 1]
                        for l in range(start, end):
                            k = reverse_indices[l]
                            if i == k:
                                continue
                            thread_shared_neighbors[
                                thread_index].push_back(k)
                    sort(thread_shared_neighbors[thread_index].begin(),
                         thread_shared_neighbors[thread_index].end())
                    num_unique_pruned_ks = 0
                    l = 0
                    size = thread_shared_neighbors[thread_index].size()
                    while l < size:
                        k = thread_shared_neighbors[thread_index][l]
                        num_shared = 1
                        l = l + 1
                        while l < size and \
                                thread_shared_neighbors[
                                    thread_index][l] == k:
                            num_shared = num_shared + 1
                            l = l + 1
                        if num_shared >= min_shared_neighbors:
                            num_unique_pruned_ks = \
                                num_unique_pruned_ks + 1
                    indptr[QCed_to_full_map[i] + 1] = \
                        num_unique_pruned_ks
                    thread_shared_neighbors[thread_index].clear()

    PyErr_CheckSignals()

    # Cumsum the values in `indptr`; initialize the first element to 0
    indptr[0] = 0
    for i in range(2, indptr.shape[0]):
        indptr[i] += indptr[i - 1]

    # Allocate `indices` and `data`: their length is the sum of the numbers of
    # unique `k`s across all cells. Access them via memoryviews with
    # C-contiguity specified to force Cython to avoid generating slower code
    # that accounts for stride (not sure if this is necessary).
    nnz = indptr[indptr.shape[0] - 1]
    # indices = np.empty(nnz, dtype=np.uint32)
    indices = np.PyArray_EMPTY(1, &nnz, np.NPY_UINT32, 0)
    # data = np.empty(nnz, dtype=np.float32)
    data = np.PyArray_EMPTY(1, &nnz, np.NPY_FLOAT32, 0)
    indices_view = indices
    data_view = data

    # Second pass: populate the SNN graph. If `QC_column` was specified, map
    # each element of `indices` through `QCed_to_full_map` so the indices are
    # with respect to all cells, not just QCed cells. Similarly, use
    # `indptr[QCed_to_full_map[i]]` instead of `indptr[i]` when indexing
    # `indptr`.

    if not has_QC_column:
        if num_threads <= 1:
            for i in range(num_cells):
                # Gather unpruned shared nearest neighbors and their counts,
                # using identical logic to the first pass.
                start = reverse_indptr[i]
                end = reverse_indptr[i + 1]
                for l in range(start, end):
                    k = reverse_indices[l]
                    shared_neighbors.push_back(k)
                for j in neighbors[i]:
                    shared_neighbors.push_back(j)
                    start = reverse_indptr[j]
                    end = reverse_indptr[j + 1]
                    for l in range(start, end):
                        k = reverse_indices[l]
                        if i == k:
                            continue
                        shared_neighbors.push_back(k)
                # Sort the unpruned shared nearest neighbors
                sort(shared_neighbors.begin(), shared_neighbors.end())
                # Iterate through the sorted shared nearest neighbors, prune,
                # calculate SNN weights, and write the SNN graph to the final
                # `indices` and `data` arrays. The SNN weight between `i` and
                # each `k` is the number of shared neighbors, divided by the
                # total number of unique cells in `i` and `k`'s neighbor lists
                # (which is twice the number of total neighbors, minus the
                # number of shared neighbors).
                count = indptr[i]
                l = 0
                size = shared_neighbors.size()
                while l < size:
                    k = shared_neighbors[l]
                    num_shared = 1
                    l += 1
                    while l < size and shared_neighbors[l] == k:
                        num_shared += 1
                        l += 1
                    if num_shared >= min_shared_neighbors:
                        indices_view[count] = k
                        data_view[count] = <float> num_shared / (
                            twice_num_neighbors_plus_one - num_shared)
                        count += 1

                # Clear `shared_neighbors` so it can be reused for the next
                # cell
                shared_neighbors.clear()
        else:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                for i in prange(num_cells):
                    start = reverse_indptr[i]
                    end = reverse_indptr[i + 1]
                    for l in range(start, end):
                        k = reverse_indices[l]
                        thread_shared_neighbors[thread_index].push_back(k)
                    for j in neighbors[i]:
                        thread_shared_neighbors[thread_index].push_back(j)
                        start = reverse_indptr[j]
                        end = reverse_indptr[j + 1]
                        for l in range(start, end):
                            k = reverse_indices[l]
                            if i == k:
                                continue
                            thread_shared_neighbors[
                                thread_index].push_back(k)
                    sort(thread_shared_neighbors[thread_index].begin(),
                         thread_shared_neighbors[thread_index].end())
                    count = indptr[i]
                    l = 0
                    size = thread_shared_neighbors[thread_index].size()
                    while l < size:
                        k = thread_shared_neighbors[thread_index][l]
                        num_shared = 1
                        l = l + 1
                        while l < size and \
                                thread_shared_neighbors[
                                    thread_index][l] == k:
                            num_shared = num_shared + 1
                            l = l + 1
                        if num_shared >= min_shared_neighbors:
                            indices_view[count] = k
                            data_view[count] = <float> num_shared / (
                                twice_num_neighbors_plus_one - num_shared)
                            count = count + 1
                    thread_shared_neighbors[thread_index].clear()
    else:
        if num_threads <= 1:
            for i in range(num_cells):
                start = reverse_indptr[i]
                end = reverse_indptr[i + 1]
                for l in range(start, end):
                    k = reverse_indices[l]
                    shared_neighbors.push_back(k)
                for j in neighbors[i]:
                    shared_neighbors.push_back(j)
                    start = reverse_indptr[j]
                    end = reverse_indptr[j + 1]
                    for l in range(start, end):
                        k = reverse_indices[l]
                        if i == k:
                            continue
                        shared_neighbors.push_back(k)
                sort(shared_neighbors.begin(), shared_neighbors.end())
                count = indptr[QCed_to_full_map[i]]
                l = 0
                size = shared_neighbors.size()
                while l < size:
                    k = shared_neighbors[l]
                    num_shared = 1
                    l += 1
                    while l < size and shared_neighbors[l] == k:
                        num_shared += 1
                        l += 1
                    if num_shared >= min_shared_neighbors:
                        indices_view[count] = QCed_to_full_map[k]
                        data_view[count] = <float> num_shared / (
                            twice_num_neighbors_plus_one - num_shared)
                        count += 1
                shared_neighbors.clear()
        else:
            with nogil, parallel(num_threads=num_threads):
                thread_index = threadid()
                for i in prange(num_cells):
                    start = reverse_indptr[i]
                    end = reverse_indptr[i + 1]
                    for l in range(start, end):
                        k = reverse_indices[l]
                        thread_shared_neighbors[thread_index].push_back(k)
                    for j in neighbors[i]:
                        thread_shared_neighbors[thread_index].push_back(j)
                        start = reverse_indptr[j]
                        end = reverse_indptr[j + 1]
                        for l in range(start, end):
                            k = reverse_indices[l]
                            if i == k:
                                continue
                            thread_shared_neighbors[
                                thread_index].push_back(k)
                    sort(thread_shared_neighbors[thread_index].begin(),
                         thread_shared_neighbors[thread_index].end())
                    count = indptr[QCed_to_full_map[i]]
                    l = 0
                    size = thread_shared_neighbors[thread_index].size()
                    while l < size:
                        k = thread_shared_neighbors[thread_index][l]
                        num_shared = 1
                        l = l + 1
                        while l < size and \
                                thread_shared_neighbors[
                                    thread_index][l] == k:
                            num_shared = num_shared + 1
                            l = l + 1
                        if num_shared >= min_shared_neighbors:
                            indices_view[count] = QCed_to_full_map[k]
                            data_view[count] = <float> num_shared / (
                                twice_num_neighbors_plus_one - num_shared)
                            count = count + 1
                    thread_shared_neighbors[thread_index].clear()

    return indices, data