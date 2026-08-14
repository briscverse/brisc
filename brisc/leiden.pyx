# Functionality for Leiden clustering

import threading
from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport prange
from libc.limits cimport UINT_MAX
from libcpp.algorithm cimport fill
from libcpp.vector cimport vector
from signal import set_wakeup_fd
from socket import socketpair
from .cyutils cimport atomic_or, randint, recv, signed_integer, srand, \
    uninitialized_vector, PREFETCH


cdef inline unsigned next_pow2_minus_1(unsigned x) noexcept nogil:
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x


def leiden(float[::1] data,
           signed_integer[::1] indices,
           signed_integer[::1] indptr,
           const unsigned[:, ::1] neighbors,
           unsigned[::1] final_communities,
           const float resolution,
           const unsigned min_cluster_size,
           const unsigned long long seed,
           const bint verbose):

    # Note: this code extensively exploits the fact that the number of nodes,
    # edges, communities etc. gets smaller on each iteration. This lets us
    # preallocate everything once and only use a fraction of what we've
    # allocated on subsequent iterations.

    # Note: Leiden expects the input graph to not have self-edges, and our SNN
    # graph created by `shared_neighbors()` excludes them. However, in the
    # aggregated graph, self-edges are important for Leiden.

    cdef unsigned i, k, leiden_iteration, queue_head, queue_tail, community, \
        other, other_community, new_community, num_refined_communities, \
        num_final_communities, community_size, neighbor, future_i, \
        unrefined_community, contiguous_index, original_index, current_sum, \
        count, num_touched, edges_written
    cdef unsigned long long num_nodes, word_index, bit_index, num_edges, \
        num_cells = indptr.shape[0] - 1, \
        queue_size_minus_1 = next_pow2_minus_1(num_cells), \
        queue_size = queue_size_minus_1 + 1, state = srand(seed)
    cdef signed_integer start_index, end_index, j
    cdef double weighted_degree, total_weighted_degree, scaled_resolution, \
        community_weight, node_to_community_weight, best_delta_objective, \
        scaled_resolution_times_weighted_degree, base_delta, \
        other_node_to_community_weight, other_community_weight, \
        delta_objective, edge_weight
    cdef bint any_moved
    cdef str error_message
    cdef uninitialized_vector[unsigned] communities_buffer, \
        node_order_buffer, queue_buffer, refined_communities_buffer, \
        community_node_indptr_buffer
    cdef uninitialized_vector[double] weighted_degrees_buffer, \
        community_weights_buffer, refined_community_weights_buffer
    cdef uninitialized_vector[float] data_buffer, data_buffer_alt
    cdef uninitialized_vector[signed_integer] indices_buffer, \
        indices_buffer_alt, indptr_buffer, indptr_buffer_alt
    cdef vector[double] node_to_community_weights_buffer
    cdef vector[unsigned long long] bitset_buffer

    communities_buffer.resize(num_cells)
    node_order_buffer.resize(num_cells)
    queue_buffer.resize(queue_size)
    weighted_degrees_buffer.resize(num_cells)
    community_weights_buffer.resize(num_cells)
    refined_community_weights_buffer.resize(num_cells)
    node_to_community_weights_buffer.resize(num_cells)
    bitset_buffer.resize((num_cells + 63) / 64)

    # For the first Leiden iteration, `refined_communities` points to
    # `final_communities`, but on subsequent iterations it will point to a
    # separate buffer, `refined_communities_buffer`. This separation allows
    # `final_communities` to track each cell's refined community, while
    # `refined_communities` tracks each node's refined community. (On the
    # first iteration, nodes are cells, but on subsequent iterations they are
    # groups of cells that formed a refined community on the previous
    # iteration.) Also, share memory between arrays used in mutually exclusive
    # phases: `node_order` and `queue` are only used in the move phase,
    # `original_to_contiguous` only in the refinement phase and at the
    # very end, `contiguous_to_original` only in the refinement phase,
    # `community_nodes` and `touched_communities` only in the aggregation
    # phase, and `community_sizes` only at the very end. Another sharing
    # happens below between `refined_communities_buffer` and
    # `contiguous_community_sizes`.
    cdef unsigned[::1] contiguous_community_sizes, community_node_indptr, \
        refined_communities = final_communities, \
        communities = <unsigned[:num_cells]> communities_buffer.data(), \
        node_order = <unsigned[:num_cells]> node_order_buffer.data(), \
        queue = <unsigned[:queue_size]> queue_buffer.data(), \
        original_to_contiguous = queue, contiguous_to_original = node_order, \
        community_nodes = node_order, touched_communities = queue, \
        community_sizes = node_order
    cdef double[::1] \
        weighted_degrees = \
            <double[:num_cells]> weighted_degrees_buffer.data(), \
        community_weights = \
            <double[:num_cells]> community_weights_buffer.data(), \
        refined_community_weights = \
            <double[:num_cells]> refined_community_weights_buffer.data(), \
        node_to_community_weights = \
            <double[:num_cells]> node_to_community_weights_buffer.data()
    cdef unsigned long long[::1] bitset = \
        <unsigned long long[:(num_cells + 63) / 64]> bitset_buffer.data()

    cdef float* next_data
    cdef signed_integer* next_indices
    cdef signed_integer* next_indptr

    # Loop over cells and:
    total_weighted_degree = 0
    for i in range(num_cells):
        # 1) Get the weighted degree (total edge weight) of each cell in the
        #    SNN graph.
        weighted_degree = 0
        start_index = indptr[i]
        end_index = indptr[i + 1]
        for j in range(start_index, end_index):
            edge_weight = data[j]
            weighted_degree += edge_weight
        weighted_degrees[i] = weighted_degree
        # 2) Get `2m`, twice the total edge weight of the SNN graph. Since the
        #    graph is symmetric, `2m` is simply the sum of the weighted
        #    degrees: we deliberately double-count i -> j and j -> i even
        #    though they are the same edge. Note that the total edge weight
        #    remains constant across Leiden iterations, since aggregation is
        #    just a matter of summing the edges.
        total_weighted_degree += weighted_degree
        # 3) Initialize `communities` (the mapping from nodes to communities)
        #    and `community_weights` (the total weighted degree of all nodes in
        #    the community) so that each cell is its own community.
        communities[i] = i
        community_weights[i] = weighted_degree

    # Get the 'scaled' resolution (resolution divided by `2m`), used in the
    # move and refinement phases
    scaled_resolution = resolution / total_weighted_degree

    # Start the Leiden iterations
    leiden_iteration = 0
    num_nodes = num_cells
    while True:
        # 1) Move phase:
        #    Starting from each nodes as its own community (on the first
        #    iteration), or the previous iteration's clustering (on subsequent
        #    iterations), greedily move nodes from one community to another to
        #    maximize the objective, until no more nodes can be moved.

        if verbose:
            print(f'Leiden iteration {leiden_iteration:,}, move phase: '
                  f'{num_nodes:,} nodes')

        # Define the random order to iterate over nodes in, via the
        # "inside-out" variant of the Fisher-Yates shuffle. The uninitialized
        # variable is intentional!
        for k in range(num_nodes):
            i = randint(k + 1, &state)
            node_order[k] = node_order[i]
            node_order[i] = k

        # For the first move iteration, visit all nodes in the order specified
        # by `node_order`. Add nodes to the queue if they need to be revisited.
        # Use a bitset to track which nodes are already in the queue, to avoid
        # adding them twice.

        queue_head = queue_tail = 0
        any_moved = False
        for k in range(num_nodes):
            # Nodes are visited in random order (`node_order`), so hardware
            # prefetchers may not predict it. Instead, manually request
            # `indptr[node_order[k]]` 24 nodes beforehand; by 16 nodes
            # beforehand it should be available, so request
            # `indices[indptr[node_order[k]]]`; by 8 nodes beforehand it
            # should be available, so request
            # `communities[indices[indptr[node_order[k]]]]`, as well as the
            # node's own community, weighted degree, and community weight.
            if k + 24 < num_nodes:
                PREFETCH(&indptr[node_order[k + 24]])
            if k + 16 < num_nodes:
                PREFETCH(&indices[indptr[node_order[k + 16]]])
            if k + 8 < num_nodes:
                PREFETCH(&communities[indices[indptr[node_order[k + 8]]]])
                PREFETCH(&communities[node_order[k + 8]])
                PREFETCH(&weighted_degrees[node_order[k + 8]])
                PREFETCH(&community_weights[node_order[k + 8]])

            # Get the next node
            i = node_order[k]

            # Get the node's community, weighted degree, and community weight
            community = communities[i]
            weighted_degree = weighted_degrees[i]
            community_weight = community_weights[community]

            # Tabulate `node_to_community_weights`, the total edge weight
            # connecting this node to each community it's connected to. Store
            # the value for the node's own community separately, in
            # `node_to_community_weight`.
            start_index = indptr[i]
            end_index = indptr[i + 1]
            j = start_index
            while j < end_index - 16:
                # `indices` is sequential and will be hardware-prefetched, but
                # `communities[indices[j]]` is a random access that hardware
                # prefetching may not predict
                PREFETCH(&communities[indices[j + 16]])
                other = indices[j]
                node_to_community_weights[communities[other]] += data[j]
                j += 1
            while j < end_index:
                other = indices[j]
                node_to_community_weights[communities[other]] += data[j]
                j += 1
            node_to_community_weight = node_to_community_weights[community]
            node_to_community_weights[community] = 0

            # Find `new_community`, the community that would lead to the
            # largest positive change in the objective (`delta_objective`) if
            # we moved this node to it. Reset `node_to_community_weights` so it
            # can be used for the next node. Loop-hoisting optimization:
            # initialize `best_delta_objective` to `base_delta` instead of 0,
            # to avoid subtracting `base_delta` from `delta_objective` within
            # the loop.
            scaled_resolution_times_weighted_degree = \
                scaled_resolution * weighted_degree
            base_delta = node_to_community_weight + \
                scaled_resolution_times_weighted_degree * \
                (weighted_degree - community_weight)
            best_delta_objective = base_delta
            start_index = indptr[i]
            end_index = indptr[i + 1]
            for j in range(start_index, end_index):
                other = indices[j]
                other_community = communities[other]
                other_node_to_community_weight = \
                    node_to_community_weights[other_community]
                if other_node_to_community_weight > 0:
                    # First time seeing this community in this pass; reset
                    # `node_to_community_weights` (so it's fresh for the next
                    # node and to avoid evaluating the same community twice)
                    # and evaluate whether to move the node to this community
                    node_to_community_weights[other_community] = 0
                    other_community_weight = \
                        community_weights[other_community]
                    delta_objective = other_node_to_community_weight - \
                        scaled_resolution_times_weighted_degree * \
                        other_community_weight
                    if delta_objective > best_delta_objective:
                        best_delta_objective = delta_objective
                        new_community = other_community

            # If this node's community assignment is suboptimal (i.e. at least
            # one community had `delta_objective` larger than `base_delta`),
            # move it to the community with the largest `delta_objective`.
            # Update the community weights to account for the move: add the
            # moved node's weighted degree to its new community's weight, and
            # subtract it from its old community's. Add all the node's
            # neighbors that are not part of the node's new community to the
            # queue, to force us to revisit them. Use the bitset to skip adding
            # nodes that are already in the queue.
            if best_delta_objective > base_delta:
                any_moved = True
                communities[i] = new_community
                community_weights[new_community] += weighted_degree
                community_weights[community] -= weighted_degree
                start_index = indptr[i]
                end_index = indptr[i + 1]
                for j in range(start_index, end_index):
                    other = indices[j]
                    other_community = communities[other]
                    if other_community != new_community:
                        word_index = other >> 6
                        bit_index = other & 63
                        if not bitset[word_index] & 1LL << bit_index:
                            bitset[word_index] |= 1LL << bit_index
                            queue[queue_tail & queue_size_minus_1] = other
                            queue_tail += 1

            # Check for KeyboardInterrupts
            if k % 131072 == 131071:
                PyErr_CheckSignals()

        # If no nodes moved during this Leiden iteration's move phase, we have
        # globally converged
        if not any_moved:
            # If Leiden converged on the very first iteration, every cell ended
            # up in a cluster by itself. This is rarely what the user wants.
            if leiden_iteration == 0:
                error_message = 'every cell ended up in a cluster by itself'
                if resolution > 1:
                    error_message += '; consider decreasing resolution'
                raise ValueError(error_message)

            # Map each cell's refined community through `communities` (the
            # move-phase labels of the nodes in the supergraph) to get its
            # final cluster
            for i in range(num_cells):
                final_communities[i] = communities[final_communities[i]]

            # Merge cells in communities of size less than `min_cluster_size`
            # (which are often disconnected from the rest of the shared nearest
            # neighbor graph) into the communities of their nearest neighbor
            # that is in a cluster of size ≥ `min_cluster_size`. Relabel the
            # final communities so that they will be contiguous after excluding
            # the too-small communities.
            community_sizes[:num_cells] = 0
            for i in range(num_cells):
                community_sizes[final_communities[i]] += 1
            num_final_communities = 0
            for community in range(num_cells):
                community_size = community_sizes[community]
                if community_size >= min_cluster_size:
                    original_to_contiguous[community] = num_final_communities
                    contiguous_community_sizes[num_final_communities] = \
                        community_size
                    num_final_communities += 1
            for i in range(num_cells):
                community = final_communities[i]
                if community_sizes[community] < min_cluster_size:
                    # Cell's community is too small, reassign to one of its
                    # neighbors' communities
                    for neighbor in neighbors[i]:
                        if neighbor >= num_cells:
                            # Out-of-bounds neighbor index
                            return num_final_communities, True, False
                        elif neighbor == i:
                            # Self-neighbors
                            return num_final_communities, False, True
                        other_community = final_communities[neighbor]
                        if neighbor < i:
                            # `other_community` has been mapped through
                            # `original_to_contiguous` to a contiguous
                            # community label already
                            if contiguous_community_sizes[other_community] >= \
                                    min_cluster_size:
                                final_communities[i] = other_community
                                break
                        else:
                            # `other_community` has not been mapped to a
                            # contiguous community label yet
                            if community_sizes[other_community] >= \
                                    min_cluster_size:
                                final_communities[i] = \
                                    original_to_contiguous[other_community]
                                break
                    else:
                        error_message = (
                            f'cell {i} and all of its {neighbors.shape[1]:,} '
                            f'nearest neighbors are in clusters smaller than '
                            f'min_cluster_size ({min_cluster_size:,}), so it '
                            f'cannot be assigned a cluster label; consider '
                            f'decreasing min_cluster_size')
                        if resolution > 1:
                            error_message += ' and/or resolution'
                        raise ValueError(error_message)
                else:
                    # Cell's community is large enough, just remap its label
                    final_communities[i] = original_to_contiguous[community]
            return num_final_communities, False, False

        # Otherwise, keep moving nodes until the queue is empty and all nodes
        # are locally optimal
        while queue_head != queue_tail:
            # Perform the same prefetch chain as the initial pass, adapted for
            # the circular queue
            if queue_head + 24 < queue_tail:
                PREFETCH(&indptr[queue[
                    (queue_head + 24) & queue_size_minus_1]])
            if queue_head + 16 < queue_tail:
                PREFETCH(&indices[indptr[queue[
                    (queue_head + 16) & queue_size_minus_1]]])
            if queue_head + 8 < queue_tail:
                future_i = queue[(queue_head + 8) & queue_size_minus_1]
                PREFETCH(&communities[indices[indptr[future_i]]])
                PREFETCH(&communities[future_i])
                PREFETCH(&weighted_degrees[future_i])

            # Pop a node from the queue
            i = queue[queue_head & queue_size_minus_1]
            word_index = i >> 6
            bit_index = i & 63
            bitset[word_index] &= ~(1LL << bit_index)
            queue_head += 1

            # Get the node's community, weighted degree, and community weight
            community = communities[i]
            weighted_degree = weighted_degrees[i]
            community_weight = community_weights[community]

            # Tabulate `node_to_community_weights`, the total edge weight
            # connecting this node to each community it's connected to. Store
            # the value for the node's own community separately, in
            # `node_to_community_weight`.
            start_index = indptr[i]
            end_index = indptr[i + 1]
            j = start_index
            while j < end_index - 16:
                # `indices` is sequential and will be hardware-prefetched, but
                # `communities[indices[j]]` is a random access that hardware
                # prefetching may not predict
                PREFETCH(&communities[indices[j + 16]])
                other = indices[j]
                node_to_community_weights[communities[other]] += data[j]
                j += 1
            while j < end_index:
                other = indices[j]
                node_to_community_weights[communities[other]] += data[j]
                j += 1
            node_to_community_weight = node_to_community_weights[community]
            node_to_community_weights[community] = 0

            # Find `new_community`, the community that would lead to the
            # largest positive change in the objective (`delta_objective`) if
            # we moved this node to it. Reset `node_to_community_weights` so it
            # can be used for the next node. Loop-hoisting optimization:
            # initialize `best_delta_objective` to `base_delta` instead of 0,
            # to avoid subtracting `base_delta` from `delta_objective` within
            # the loop.
            scaled_resolution_times_weighted_degree = \
                scaled_resolution * weighted_degree
            base_delta = node_to_community_weight + \
                scaled_resolution_times_weighted_degree * \
                (weighted_degree - community_weight)
            best_delta_objective = base_delta
            start_index = indptr[i]
            end_index = indptr[i + 1]
            for j in range(start_index, end_index):
                other = indices[j]
                other_community = communities[other]
                other_node_to_community_weight = \
                    node_to_community_weights[other_community]
                if other_node_to_community_weight > 0:
                    # First time seeing this community in this pass; reset
                    # `node_to_community_weights` (so it's fresh for the next
                    # node and to avoid evaluating the same community twice)
                    # and evaluate whether to move the node to this community
                    node_to_community_weights[other_community] = 0
                    other_community_weight = \
                        community_weights[other_community]
                    delta_objective = other_node_to_community_weight - \
                        scaled_resolution_times_weighted_degree * \
                        other_community_weight
                    if delta_objective > best_delta_objective:
                        best_delta_objective = delta_objective
                        new_community = other_community

            # If this node's community assignment is suboptimal (i.e. at least
            # one community had `delta_objective` larger than `base_delta`),
            # move it to the community with the largest `delta_objective`.
            # Update the community weights to account for the move: add the
            # moved node's weighted degree to its new community's weight, and
            # subtract it from its old community's. Add all the node's
            # neighbors that are not part of the node's new community to the
            # queue, to force us to revisit them. Use the bitset to skip adding
            # nodes that are already in the queue.
            if best_delta_objective > base_delta:
                communities[i] = new_community
                community_weights[new_community] += weighted_degree
                community_weights[community] -= weighted_degree
                start_index = indptr[i]
                end_index = indptr[i + 1]
                for j in range(start_index, end_index):
                    other = indices[j]
                    other_community = communities[other]
                    if other_community != new_community:
                        word_index = other >> 6
                        bit_index = other & 63
                        if not bitset[word_index] & 1LL << bit_index:
                            bitset[word_index] |= 1LL << bit_index
                            queue[queue_tail & queue_size_minus_1] = other
                            queue_tail += 1

            # Check for KeyboardInterrupts
            if queue_head % 131072 == 131071:
                PyErr_CheckSignals()

        # 2) Refinement phase:
        #    Refine the communities found in the move phase by allowing them to
        #    be broken up into smaller communities. This avoids disconnected
        #    communities (i.e. communities that do not form a single
        #    connected component, where the only way to get from one part of
        #    the community to another is to take a path that goes outside the
        #    community), and helps avoid poorly connected communities.
        #
        #    This phase starts by placing each node in its own "refined
        #    community", then moves nodes from one "refined community" to
        #    another to improve the objective, subject to the constraint that
        #    nodes cannot be moved across the boundaries of the communities
        #    found in the move phase. In effect, we perform a second round of
        #    community detection within each of the original ("unrefined")
        #    communities.

        if verbose:
            print(f'Leiden iteration {leiden_iteration:,}, refinement')

        # Initialize `refined_communities` (each node's refined community) so
        # that each node is its own refined community. To reflect this,
        # initialize `refined_community_weights` (the total weighted degree of
        # all nodes in the refined community) to the weighted degree of the
        # node itself.
        for i in range(num_nodes):
            refined_communities[i] = i
            refined_community_weights[i] = weighted_degrees[i]

        # Do a single pass over nodes, moving certain nodes and not moving
        # others. Nodes that are not moved will end up with the same refined
        # community label they started with - equal to their node index - and
        # exactly one node per refined community will not be moved. So whenever
        # we do not move a node, increment the number of refined communities
        # and map that node's community index to the current number of refined
        # communities in the `original_to_contiguous` mapping. This lets us
        # remap the community indices to be contiguous at the end of the
        # refinement phase. Store the reverse mapping in
        # `contiguous_to_original`.

        num_refined_communities = 0
        for i in range(num_nodes):
            # Nodes can only be moved if they are in a refined community by
            # themselves. Once a node has been merged into a refined community,
            # it can no longer be moved. To check whether this is the case, see
            # if the node's refined community weight still equals its own
            # weighted degree.
            #
            # This condition guarantees the resulting refined communities will
            # never become disconnected, since disconnected communities can
            # only happen when a node gets moved that was forming a bridge
            # between two parts of its community, and a node that is in a
            # community by itself will never form such a bridge, as there are
            # no other nodes in its community for it to form a bridge between!
            community = refined_communities[i]
            weighted_degree = weighted_degrees[i]
            community_weight = refined_community_weights[community]
            if weighted_degree != community_weight:
                original_to_contiguous[community] = num_refined_communities
                contiguous_to_original[num_refined_communities] = community
                num_refined_communities += 1
                continue

            # Tabulate `node_to_community_weights`, the total edge weight
            # connecting this node to each refined community it's connected to.
            # Store the value for the node's own community separately, in
            # `node_to_community_weight`.
            unrefined_community = communities[i]
            start_index = indptr[i]
            end_index = indptr[i + 1]
            j = start_index
            while j < end_index - 16:
                # `indices` is sequential and will be hardware-prefetched, but
                # `refined_communities[indices[j]]` is a random access that
                # hardware prefetching may not predict
                PREFETCH(&refined_communities[indices[j + 16]])
                other = indices[j]
                node_to_community_weights[refined_communities[other]] += \
                    data[j]
                j += 1
            while j < end_index:
                other = indices[j]
                node_to_community_weights[refined_communities[other]] += \
                    data[j]
                j += 1
            node_to_community_weight = node_to_community_weights[community]
            node_to_community_weights[community] = 0

            # Find `new_community`, the community that would lead to the
            # largest positive change in the objective (`delta_objective`) if
            # we moved this node to it. Reset `node_to_community_weights` so it
            # can be used for the next node. Loop-hoisting optimization:
            # initialize `best_delta_objective` to `base_delta` instead of 0,
            # to avoid subtracting `base_delta` from `delta_objective` within
            # the loop. Only consider refined communities that are in the same
            # unrefined community as this node.
            scaled_resolution_times_weighted_degree = \
                scaled_resolution * weighted_degree
            base_delta = node_to_community_weight + \
                scaled_resolution_times_weighted_degree * \
                (weighted_degree - community_weight)
            best_delta_objective = base_delta
            start_index = indptr[i]
            end_index = indptr[i + 1]
            for j in range(start_index, end_index):
                other = indices[j]
                other_community = refined_communities[other]
                other_node_to_community_weight = \
                    node_to_community_weights[other_community]
                if other_node_to_community_weight > 0:
                    # First time seeing this community in this pass; reset
                    # `node_to_community_weights` (so it's fresh for the next
                    # node and to avoid evaluating the same community twice)
                    # and evaluate whether to move the node to this community
                    node_to_community_weights[other_community] = 0
                    if communities[other] == unrefined_community:
                        other_community_weight = \
                            refined_community_weights[other_community]
                        delta_objective = other_node_to_community_weight - \
                            scaled_resolution_times_weighted_degree * \
                            other_community_weight
                        if delta_objective > best_delta_objective:
                            best_delta_objective = delta_objective
                            new_community = other_community

            # If this node's community assignment is suboptimal (i.e. at least
            # one community had `delta_objective` larger than `base_delta`),
            # move it to the community with the largest `delta_objective`.
            # Update the community weights to account for the move: add the
            # moved node's weighted degree to its new community's weight.
            # (Don't bother subtracting it from its old community, which is
            # now empty and will not be visited again.) If the node's community
            # assignment is already optimal, do not move it.
            if best_delta_objective > base_delta:
                refined_communities[i] = new_community
                refined_community_weights[new_community] += weighted_degree
            else:
                original_to_contiguous[community] = num_refined_communities
                contiguous_to_original[num_refined_communities] = community
                num_refined_communities += 1

            # Check for KeyboardInterrupts
            if i % 131072 == 131071:
                PyErr_CheckSignals()

        # Relabel the refined communities to make them contiguous integers, by
        # mapping them through the `original_to_contiguous` mapping. For
        # instance, if the only non-empty communities after the refinement
        # phase are 2, 7, etc., renumber 2 to 0, 7 to 1, and so on. This makes
        # it easier to create a new CSR graph in the aggregation phase, where
        # each refined community becomes a single node.
        for i in range(num_nodes):
            refined_communities[i] = \
                original_to_contiguous[refined_communities[i]]

        # Reinitialize `communities` to the unrefined communities from the move
        # phase: if multiple refined communities were part of the same
        # unrefined community, their nodes in the supergraph will start off
        # with the same community label as each other. This can be done
        # in-place, since each community's contiguous index is less than its
        # original index, so we never write to entries of `communities` before
        # reading them. Also reinitialize `weighted_degrees` to the refined
        # community weights (since each refined community becomes a node in
        # the supergraph).
        #
        # In the code below, `original_index` is the original index of the
        # refined community before making the indices contiguous.
        # `communities[original_index]` is its unrefined community label, and
        # by extension the unrefined community label of every node in its
        # refined community. Meanwhile, `contiguous_index` is the new label of
        # the refined community, and by extension the index of that refined
        # community's node in the supergraph to be constructed in the
        # aggregation phase and used in the next Leiden iteration.
        for contiguous_index in range(num_refined_communities):
            original_index = contiguous_to_original[contiguous_index]
            communities[contiguous_index] = communities[original_index]
            weighted_degrees[contiguous_index] = \
                refined_community_weights[original_index]

        # 3) Aggregation phase:
        #    Aggregate the graph into a "supergraph" where each refined
        #    community becomes a node, and edge weights are summed across
        #    inter-community edges.
        if verbose:
            print(f'Leiden iteration {leiden_iteration:,}, aggregation')

        # If this is not the first Leiden iteration, update each cell's refined
        # community in `final_communities` by mapping it through
        # `refined_communities`, to account for the aggregation. In other
        # words, if a cell's community label was `x` before the aggregation,
        # its new community label will be `refined_communities[x]`.
        #
        # For instance, if this is the second iteration, then right before this
        # step, `final_communities` contained each cell's refined community
        # from the first iteration's move phase (because `refined_communities`
        # pointed to it on the first iteration). But during the second
        # iteration's move and refine phases, some of these refined communities
        # (which each became a node in the supergraph during the first
        # iteration's aggregation phase) changed labels. `refined_communities`
        # contains a mapping from nodes (i.e. the previous iteration's refined
        # communities) to the current iteration's refined communities. So by
        # mapping through `refined_communities`, we update the constituent
        # cells of these refined communities from the previous iteration's
        # refined communities to the current iteration's refined communities.
        if leiden_iteration > 0:
            for i in range(num_cells):
                final_communities[i] = \
                    refined_communities[final_communities[i]]
        # If this is the first Leiden iteration, allocate
        # `community_node_indptr_buffer`
        else:
            community_node_indptr_buffer.resize(num_refined_communities + 1)
            community_node_indptr = \
                <unsigned[:num_refined_communities + 1]> \
                community_node_indptr_buffer.data()

        # Group nodes by their refined community using counting sort
        for i in range(num_refined_communities + 1):
            community_node_indptr[i] = 0
        for i in range(num_nodes):
            community_node_indptr[refined_communities[i]] += 1

        current_sum = 0
        for i in range(num_refined_communities):
            count = community_node_indptr[i]
            community_node_indptr[i] = current_sum
            current_sum += count
        community_node_indptr[num_refined_communities] = current_sum

        for i in range(num_nodes):
            community = refined_communities[i]
            community_nodes[community_node_indptr[community]] = i
            community_node_indptr[community] += 1

        i = num_refined_communities
        while i > 0:
            community_node_indptr[i] = community_node_indptr[i - 1]
            i -= 1
        community_node_indptr[0] = 0

        # Set up ping-pong pointers for the supergraph. On the first two
        # iterations, allocate the buffers underlying these pointers.
        # A supergraph with `num_refined_communities` nodes has at most
        # `num_refined_communities ** 2` edges. Thus, the new
        # edge count must be at most `num_refined_communities ** 2` or
        # the current graph's number of edges, whichever is less.
        if leiden_iteration % 2 == 0:
            if leiden_iteration == 0:
                num_edges = min(<unsigned long long> num_refined_communities *
                                num_refined_communities,
                                <unsigned long long> indptr[num_nodes])
                data_buffer.resize(num_edges)
                indices_buffer.resize(num_edges)
                indptr_buffer.resize(num_refined_communities + 1)
            next_data = data_buffer.data()
            next_indices = indices_buffer.data()
            next_indptr = indptr_buffer.data()
        else:
            if leiden_iteration == 1:
                num_edges = min(<unsigned long long> num_refined_communities *
                                num_refined_communities,
                                <unsigned long long> indptr[num_nodes])
                data_buffer_alt.resize(num_edges)
                indices_buffer_alt.resize(num_edges)
                indptr_buffer_alt.resize(num_refined_communities + 1)
            next_data = data_buffer_alt.data()
            next_indices = indices_buffer_alt.data()
            next_indptr = indptr_buffer_alt.data()

        # Aggregate the graph into the new buffers, summing the inter-community
        # weights for each pair of communities
        edges_written = 0
        next_indptr[0] = 0
        for community in range(num_refined_communities):
            num_touched = 0
            for k in range(community_node_indptr[community],
                           community_node_indptr[community + 1]):
                i = community_nodes[k]
                start_index = indptr[i]
                end_index = indptr[i + 1]
                j = start_index
                while j < end_index - 16:
                    PREFETCH(&refined_communities[indices[j + 16]])
                    other = indices[j]
                    other_community = refined_communities[other]
                    edge_weight = data[j]
                    if node_to_community_weights[other_community] == 0:
                        touched_communities[num_touched] = other_community
                        num_touched += 1
                    node_to_community_weights[other_community] += \
                        2 * edge_weight if other == i else edge_weight
                    j += 1
                while j < end_index:
                    other = indices[j]
                    other_community = refined_communities[other]
                    edge_weight = data[j]
                    if node_to_community_weights[other_community] == 0:
                        touched_communities[num_touched] = other_community
                        num_touched += 1
                    node_to_community_weights[other_community] += \
                        2 * edge_weight if other == i else edge_weight
                    j += 1

            # Halve the intra-community edges, which will become self-edges in
            # the supergraph. This is necessary because the input graph is
            # symmetric, so if nodes `i` and `j` are in the same community
            # `community`, edges i -> j and j -> i will both count towards the
            # self-edge. Non-self-edges are not a problem because edge i -> j
            # and edge j -> i will appear separately in the supergraph with the
            # same weight: thus, the supergraph will be symmetric, like the
            # input graph.
            #
            # However, do not halve self-loops if this is the supernode graph.
            # This is why we do `2 * edge_weight if other == i` above, so it
            # cancels out the halving here.
            node_to_community_weights[community] *= 0.5

            # Write out edges for this community
            for k in range(num_touched):
                other_community = touched_communities[k]
                next_data[edges_written] = \
                    node_to_community_weights[other_community]
                next_indices[edges_written] = other_community
                edges_written += 1
                node_to_community_weights[other_community] = 0

            next_indptr[community + 1] = edges_written

        # If this is the first iteration, allocate a buffer for the new refined
        # communities to avoid overwriting the original (which we need to
        # determine the final clusters), and point `contiguous_community_sizes`
        # to this new buffer
        if leiden_iteration == 0:
            refined_communities_buffer.resize(num_cells)
            refined_communities = \
                <unsigned[:num_cells]> refined_communities_buffer.data()
            contiguous_community_sizes = refined_communities

        # Swap memoryviews so the next iteration uses the new graph
        data = <float[:edges_written]> next_data
        indices = <signed_integer[:edges_written]> next_indices
        indptr = <signed_integer[:num_refined_communities + 1]> next_indptr

        # Update `num_nodes` to reflect that the number of nodes in the
        # supergraph is the number of refined communities
        num_nodes = num_refined_communities

        # Now we are done this Leiden iteration
        leiden_iteration += 1


cdef inline unsigned leiden_nogil(
        const float[::1] data_original,
        const signed_integer[::1] indices_original,
        const signed_integer[::1] indptr_original,
        const unsigned[:, ::1] neighbors,
        unsigned[::1] final_communities,
        const float resolution,
        const unsigned min_cluster_size,
        const unsigned long long seed,
        bint& too_large,
        bint& self_neighbors,
        const int read_fd,
        bint& interrupted) nogil:

    # This code almost exactly duplicates `leiden()`, but is designed to run
    # for each thread when clustering across multiple resolutions at once.
    # It differs by:
    # - being `nogil`
    # - using raw vectors instead of memoryviews over vectors for local
    #   variables (since pointing memoryviews to vectors requires the GIL)
    # - using `recv()` instead of `PyErr_CheckSignals()` to perform
    #   KeyboardInterrupt checks, since the latter does not work inside prange
    # - not including `verbose` or print statements
    # - including the resolution in error messages
    # - using hardware-level atomics to thread-safely flag out-of-bounds
    #   neighbors

    cdef unsigned i, k, leiden_iteration, queue_head, queue_tail, community, \
        other, other_community, new_community, num_refined_communities, \
        num_final_communities, community_size, neighbor, future_i, \
        unrefined_community, contiguous_index, original_index, current_sum, \
        count, num_touched, edges_written
    cdef unsigned long long num_nodes, word_index, bit_index, num_edges, \
        num_cells = indptr_original.shape[0] - 1, \
        queue_size_minus_1 = next_pow2_minus_1(num_cells), \
        queue_size = queue_size_minus_1 + 1, state = srand(seed)
    cdef signed_integer start_index, end_index, j
    cdef double weighted_degree, total_weighted_degree, scaled_resolution, \
        community_weight, node_to_community_weight, best_delta_objective, \
        scaled_resolution_times_weighted_degree, base_delta, \
        other_node_to_community_weight, other_community_weight, \
        delta_objective, edge_weight
    cdef bint any_moved
    cdef char signal_byte
    cdef uninitialized_vector[unsigned] communities, node_order, queue, \
        refined_communities_buffer, community_node_indptr
    cdef uninitialized_vector[double] weighted_degrees, community_weights, \
        refined_community_weights
    cdef uninitialized_vector[float] data_buffer, data_buffer_alt
    cdef uninitialized_vector[signed_integer] indices_buffer, \
        indices_buffer_alt, indptr_buffer, indptr_buffer_alt
    cdef vector[double] node_to_community_weights
    cdef vector[unsigned long long] bitset

    communities.resize(num_cells)
    node_order.resize(num_cells)
    queue.resize(queue_size)
    weighted_degrees.resize(num_cells)
    community_weights.resize(num_cells)
    refined_community_weights.resize(num_cells)
    node_to_community_weights.resize(num_cells)
    bitset.resize((num_cells + 63) / 64)

    # For the first Leiden iteration, `refined_communities` points to
    # `final_communities`, but on subsequent iterations it will point to a
    # separate buffer, `refined_communities_buffer`. This separation allows
    # `final_communities` to track each cell's refined community, while
    # `refined_communities` tracks each node's refined community. (On the
    # first iteration, nodes are cells, but on subsequent iterations they are
    # groups of cells that formed a refined community on the previous
    # iteration.) Also, share memory between arrays used in mutually exclusive
    # phases: `node_order` and `queue` are only used in the move phase,
    # `original_to_contiguous` only in the refinement phase and at the
    # very end, `contiguous_to_original` only in the refinement phase,
    # `community_nodes` and `touched_communities` only in the aggregation
    # phase, and `community_sizes` only at the very end. Another sharing
    # happens below between `refined_communities_buffer` and
    # `contiguous_community_sizes`.
    cdef unsigned* contiguous_community_sizes
    cdef unsigned* refined_communities = &final_communities[0]
    cdef unsigned* original_to_contiguous = queue.data()
    cdef unsigned* contiguous_to_original = node_order.data()
    cdef unsigned* community_nodes = node_order.data()
    cdef unsigned* touched_communities = queue.data()
    cdef unsigned* community_sizes = node_order.data()

    cdef float* data = <float*> &data_original[0]
    cdef signed_integer* indices = <signed_integer*> &indices_original[0]
    cdef signed_integer* indptr = <signed_integer*> &indptr_original[0]
    cdef float* next_data
    cdef signed_integer* next_indices
    cdef signed_integer* next_indptr

    # Loop over cells and:
    total_weighted_degree = 0
    for i in range(num_cells):
        # 1) Get the weighted degree (total edge weight) of each cell in the
        #    SNN graph.
        weighted_degree = 0
        start_index = indptr[i]
        end_index = indptr[i + 1]
        for j in range(start_index, end_index):
            edge_weight = data[j]
            weighted_degree += edge_weight
        weighted_degrees[i] = weighted_degree
        # 2) Get `2m`, twice the total edge weight of the SNN graph. Since the
        #    graph is symmetric, `2m` is simply the sum of the weighted
        #    degrees: we deliberately double-count i -> j and j -> i even
        #    though they are the same edge. Note that the total edge weight
        #    remains constant across Leiden iterations, since aggregation is
        #    just a matter of summing the edges.
        total_weighted_degree += weighted_degree
        # 3) Initialize `communities` (the mapping from nodes to communities)
        #    and `community_weights` (the total weighted degree of all nodes in
        #    the community) so that each cell is its own community.
        communities[i] = i
        community_weights[i] = weighted_degree

    # Get the 'scaled' resolution (resolution divided by `2m`), used in the
    # move and refinement phases
    scaled_resolution = resolution / total_weighted_degree

    # Start the Leiden iterations
    leiden_iteration = 0
    num_nodes = num_cells
    while True:
        # 1) Move phase:
        #    Starting from each nodes as its own community (on the first
        #    iteration), or the previous iteration's clustering (on subsequent
        #    iterations), greedily move nodes from one community to another to
        #    maximize the objective, until no more nodes can be moved.

        # Define the random order to iterate over nodes in, via the
        # "inside-out" variant of the Fisher-Yates shuffle. The uninitialized
        # variable is intentional!
        for k in range(num_nodes):
            i = randint(k + 1, &state)
            node_order[k] = node_order[i]
            node_order[i] = k

        # For the first move iteration, visit all nodes in the order specified
        # by `node_order`. Add nodes to the queue if they need to be revisited.
        # Use a bitset to track which nodes are already in the queue, to avoid
        # adding them twice.

        queue_head = queue_tail = 0
        any_moved = False
        for k in range(num_nodes):
            # Nodes are visited in random order (`node_order`), so hardware
            # prefetchers may not predict it. Instead, manually request
            # `indptr[node_order[k]]` 24 nodes beforehand; by 16 nodes
            # beforehand it should be available, so request
            # `indices[indptr[node_order[k]]]`; by 8 nodes beforehand it
            # should be available, so request
            # `communities[indices[indptr[node_order[k]]]]`, as well as the
            # node's own community, weighted degree, and community weight.
            if k + 24 < num_nodes:
                PREFETCH(&indptr[node_order[k + 24]])
            if k + 16 < num_nodes:
                PREFETCH(&indices[indptr[node_order[k + 16]]])
            if k + 8 < num_nodes:
                PREFETCH(&communities[indices[indptr[node_order[k + 8]]]])
                PREFETCH(&communities[node_order[k + 8]])
                PREFETCH(&weighted_degrees[node_order[k + 8]])
                PREFETCH(&community_weights[node_order[k + 8]])

            # Get the next node
            i = node_order[k]

            # Get the node's community, weighted degree, and community weight
            community = communities[i]
            weighted_degree = weighted_degrees[i]
            community_weight = community_weights[community]

            # Tabulate `node_to_community_weights`, the total edge weight
            # connecting this node to each community it's connected to. Store
            # the value for the node's own community separately, in
            # `node_to_community_weight`.
            start_index = indptr[i]
            end_index = indptr[i + 1]
            j = start_index
            while j < end_index - 16:
                # `indices` is sequential and will be hardware-prefetched, but
                # `communities[indices[j]]` is a random access that hardware
                # prefetching may not predict
                PREFETCH(&communities[indices[j + 16]])
                other = indices[j]
                node_to_community_weights[communities[other]] += data[j]
                j += 1
            while j < end_index:
                other = indices[j]
                node_to_community_weights[communities[other]] += data[j]
                j += 1
            node_to_community_weight = node_to_community_weights[community]
            node_to_community_weights[community] = 0

            # Find `new_community`, the community that would lead to the
            # largest positive change in the objective (`delta_objective`) if
            # we moved this node to it. Reset `node_to_community_weights` so it
            # can be used for the next node. Loop-hoisting optimization:
            # initialize `best_delta_objective` to `base_delta` instead of 0,
            # to avoid subtracting `base_delta` from `delta_objective` within
            # the loop.
            scaled_resolution_times_weighted_degree = \
                scaled_resolution * weighted_degree
            base_delta = node_to_community_weight + \
                scaled_resolution_times_weighted_degree * \
                (weighted_degree - community_weight)
            best_delta_objective = base_delta
            start_index = indptr[i]
            end_index = indptr[i + 1]
            for j in range(start_index, end_index):
                other = indices[j]
                other_community = communities[other]
                other_node_to_community_weight = \
                    node_to_community_weights[other_community]
                if other_node_to_community_weight > 0:
                    # First time seeing this community in this pass; reset
                    # `node_to_community_weights` (so it's fresh for the next
                    # node and to avoid evaluating the same community twice)
                    # and evaluate whether to move the node to this community
                    node_to_community_weights[other_community] = 0
                    other_community_weight = \
                        community_weights[other_community]
                    delta_objective = other_node_to_community_weight - \
                        scaled_resolution_times_weighted_degree * \
                        other_community_weight
                    if delta_objective > best_delta_objective:
                        best_delta_objective = delta_objective
                        new_community = other_community

            # If this node's community assignment is suboptimal (i.e. at least
            # one community had `delta_objective` larger than `base_delta`),
            # move it to the community with the largest `delta_objective`.
            # Update the community weights to account for the move: add the
            # moved node's weighted degree to its new community's weight, and
            # subtract it from its old community's. Add all the node's
            # neighbors that are not part of the node's new community to the
            # queue, to force us to revisit them. Use the bitset to skip adding
            # nodes that are already in the queue.
            if best_delta_objective > base_delta:
                any_moved = True
                communities[i] = new_community
                community_weights[new_community] += weighted_degree
                community_weights[community] -= weighted_degree
                start_index = indptr[i]
                end_index = indptr[i + 1]
                for j in range(start_index, end_index):
                    other = indices[j]
                    other_community = communities[other]
                    if other_community != new_community:
                        word_index = other >> 6
                        bit_index = other & 63
                        if not bitset[word_index] & 1LL << bit_index:
                            bitset[word_index] |= 1LL << bit_index
                            queue[queue_tail & queue_size_minus_1] = other
                            queue_tail += 1

            # Check for KeyboardInterrupts
            if k % 131072 == 131071:
                if recv(read_fd, &signal_byte, 1, 0) > 0:
                    atomic_or(interrupted, True)
                if interrupted:
                    with gil:
                        raise KeyboardInterrupt

        # If no nodes moved during this Leiden iteration's move phase, we have
        # globally converged
        if not any_moved:
            # If Leiden converged on the very first iteration, every cell ended
            # up in a cluster by itself. This is rarely what the user wants.
            if leiden_iteration == 0:
                with gil:
                    error_message = \
                        'every cell ended up in a cluster by itself'
                    if resolution > 1:
                        error_message += '; consider decreasing resolution'
                    error_message += f' (resolution = {resolution})'
                    raise ValueError(error_message)

            # Map each cell's refined community through `communities` (the
            # move-phase labels of the nodes in the supergraph) to get its
            # final cluster
            for i in range(num_cells):
                final_communities[i] = communities[final_communities[i]]

            # Merge cells in communities of size less than `min_cluster_size`
            # (which are often disconnected from the rest of the shared nearest
            # neighbor graph) into the communities of their nearest neighbor
            # that is in a cluster of size ≥ `min_cluster_size`. Relabel the
            # final communities so that they will be contiguous after excluding
            # the too-small communities.
            fill(community_sizes, community_sizes + num_cells, 0)
            for i in range(num_cells):
                community_sizes[final_communities[i]] += 1
            num_final_communities = 0
            for community in range(num_cells):
                community_size = community_sizes[community]
                if community_size >= min_cluster_size:
                    original_to_contiguous[community] = num_final_communities
                    contiguous_community_sizes[num_final_communities] = \
                        community_size
                    num_final_communities += 1
            for i in range(num_cells):
                community = final_communities[i]
                if community_sizes[community] < min_cluster_size:
                    # Cell's community is too small, reassign to one of its
                    # neighbors' communities
                    for neighbor in neighbors[i]:
                        if neighbor >= num_cells:
                            atomic_or(too_large, True)
                            return num_final_communities
                        elif neighbor == i:
                            atomic_or(self_neighbors, True)
                            return num_final_communities
                        other_community = final_communities[neighbor]
                        if neighbor < i:
                            # `other_community` has been mapped through
                            # `original_to_contiguous` to a contiguous
                            # community label already
                            if contiguous_community_sizes[other_community] >= \
                                    min_cluster_size:
                                final_communities[i] = other_community
                                break
                        else:
                            # `other_community` has not been mapped to a
                            # contiguous community label yet
                            if community_sizes[other_community] >= \
                                    min_cluster_size:
                                final_communities[i] = \
                                    original_to_contiguous[other_community]
                                break
                    else:
                        with gil:
                            error_message = (
                                f'cell {i} and all of its '
                                f'{neighbors.shape[1]:,} nearest neighbors '
                                f'are in clusters smaller than '
                                f'min_cluster_size ({min_cluster_size:,}), so '
                                f'it cannot be assigned a cluster label; '
                                f'consider decreasing min_cluster_size')
                            if resolution > 1:
                                error_message += ' and/or resolution'
                            error_message += f' (resolution = {resolution})'
                            raise ValueError(error_message)
                else:
                    # Cell's community is large enough, just remap its label
                    final_communities[i] = original_to_contiguous[community]
            return num_final_communities

        # Otherwise, keep moving nodes until the queue is empty and all nodes
        # are locally optimal
        while queue_head != queue_tail:
            # Perform the same prefetch chain as the initial pass, adapted for
            # the circular queue
            if queue_head + 24 < queue_tail:
                PREFETCH(&indptr[queue[
                    (queue_head + 24) & queue_size_minus_1]])
            if queue_head + 16 < queue_tail:
                PREFETCH(&indices[indptr[queue[
                    (queue_head + 16) & queue_size_minus_1]]])
            if queue_head + 8 < queue_tail:
                future_i = queue[(queue_head + 8) & queue_size_minus_1]
                PREFETCH(&communities[indices[indptr[future_i]]])
                PREFETCH(&communities[future_i])
                PREFETCH(&weighted_degrees[future_i])

            # Pop a node from the queue
            i = queue[queue_head & queue_size_minus_1]
            word_index = i >> 6
            bit_index = i & 63
            bitset[word_index] &= ~(1LL << bit_index)
            queue_head += 1

            # Get the node's community, weighted degree, and community weight
            community = communities[i]
            weighted_degree = weighted_degrees[i]
            community_weight = community_weights[community]

            # Tabulate `node_to_community_weights`, the total edge weight
            # connecting this node to each community it's connected to. Store
            # the value for the node's own community separately, in
            # `node_to_community_weight`.
            start_index = indptr[i]
            end_index = indptr[i + 1]
            j = start_index
            while j < end_index - 16:
                # `indices` is sequential and will be hardware-prefetched, but
                # `communities[indices[j]]` is a random access that hardware
                # prefetching may not predict
                PREFETCH(&communities[indices[j + 16]])
                other = indices[j]
                node_to_community_weights[communities[other]] += data[j]
                j += 1
            while j < end_index:
                other = indices[j]
                node_to_community_weights[communities[other]] += data[j]
                j += 1
            node_to_community_weight = node_to_community_weights[community]
            node_to_community_weights[community] = 0

            # Find `new_community`, the community that would lead to the
            # largest positive change in the objective (`delta_objective`) if
            # we moved this node to it. Reset `node_to_community_weights` so it
            # can be used for the next node. Loop-hoisting optimization:
            # initialize `best_delta_objective` to `base_delta` instead of 0,
            # to avoid subtracting `base_delta` from `delta_objective` within
            # the loop.
            scaled_resolution_times_weighted_degree = \
                scaled_resolution * weighted_degree
            base_delta = node_to_community_weight + \
                scaled_resolution_times_weighted_degree * \
                (weighted_degree - community_weight)
            best_delta_objective = base_delta
            start_index = indptr[i]
            end_index = indptr[i + 1]
            for j in range(start_index, end_index):
                other = indices[j]
                other_community = communities[other]
                other_node_to_community_weight = \
                    node_to_community_weights[other_community]
                if other_node_to_community_weight > 0:
                    # First time seeing this community in this pass; reset
                    # `node_to_community_weights` (so it's fresh for the next
                    # node and to avoid evaluating the same community twice)
                    # and evaluate whether to move the node to this community
                    node_to_community_weights[other_community] = 0
                    other_community_weight = \
                        community_weights[other_community]
                    delta_objective = other_node_to_community_weight - \
                        scaled_resolution_times_weighted_degree * \
                        other_community_weight
                    if delta_objective > best_delta_objective:
                        best_delta_objective = delta_objective
                        new_community = other_community

            # If this node's community assignment is suboptimal (i.e. at least
            # one community had `delta_objective` larger than `base_delta`),
            # move it to the community with the largest `delta_objective`.
            # Update the community weights to account for the move: add the
            # moved node's weighted degree to its new community's weight, and
            # subtract it from its old community's. Add all the node's
            # neighbors that are not part of the node's new community to the
            # queue, to force us to revisit them. Use the bitset to skip adding
            # nodes that are already in the queue.
            if best_delta_objective > base_delta:
                communities[i] = new_community
                community_weights[new_community] += weighted_degree
                community_weights[community] -= weighted_degree
                start_index = indptr[i]
                end_index = indptr[i + 1]
                for j in range(start_index, end_index):
                    other = indices[j]
                    other_community = communities[other]
                    if other_community != new_community:
                        word_index = other >> 6
                        bit_index = other & 63
                        if not bitset[word_index] & 1LL << bit_index:
                            bitset[word_index] |= 1LL << bit_index
                            queue[queue_tail & queue_size_minus_1] = other
                            queue_tail += 1

            # Check for KeyboardInterrupts
            if queue_head % 131072 == 131071:
                if recv(read_fd, &signal_byte, 1, 0) > 0:
                    atomic_or(interrupted, True)
                if interrupted:
                    with gil:
                        raise KeyboardInterrupt

        # 2) Refinement phase:
        #    Refine the communities found in the move phase by allowing them to
        #    be broken up into smaller communities. This avoids disconnected
        #    communities (i.e. communities that do not form a single
        #    connected component, where the only way to get from one part of
        #    the community to another is to take a path that goes outside the
        #    community), and helps avoid poorly connected communities.
        #
        #    This phase starts by placing each node in its own "refined
        #    community", then moves nodes from one "refined community" to
        #    another to improve the objective, subject to the constraint that
        #    nodes cannot be moved across the boundaries of the communities
        #    found in the move phase. In effect, we perform a second round of
        #    community detection within each of the original ("unrefined")
        #    communities.

        # Initialize `refined_communities` (each node's refined community) so
        # that each node is its own refined community. To reflect this,
        # initialize `refined_community_weights` (the total weighted degree of
        # all nodes in the refined community) to the weighted degree of the
        # node itself.
        for i in range(num_nodes):
            refined_communities[i] = i
            refined_community_weights[i] = weighted_degrees[i]

        # Do a single pass over nodes, moving certain nodes and not moving
        # others. Nodes that are not moved will end up with the same refined
        # community label they started with - equal to their node index - and
        # exactly one node per refined community will not be moved. So whenever
        # we do not move a node, increment the number of refined communities
        # and map that node's community index to the current number of refined
        # communities in the `original_to_contiguous` mapping. This lets us
        # remap the community indices to be contiguous at the end of the
        # refinement phase. Store the reverse mapping in
        # `contiguous_to_original`.

        num_refined_communities = 0
        for i in range(num_nodes):
            # Nodes can only be moved if they are in a refined community by
            # themselves. Once a node has been merged into a refined community,
            # it can no longer be moved. To check whether this is the case, see
            # if the node's refined community weight still equals its own
            # weighted degree.
            #
            # This condition guarantees the resulting refined communities will
            # never become disconnected, since disconnected communities can
            # only happen when a node gets moved that was forming a bridge
            # between two parts of its community, and a node that is in a
            # community by itself will never form such a bridge, as there are
            # no other nodes in its community for it to form a bridge between!
            community = refined_communities[i]
            weighted_degree = weighted_degrees[i]
            community_weight = refined_community_weights[community]
            if weighted_degree != community_weight:
                original_to_contiguous[community] = num_refined_communities
                contiguous_to_original[num_refined_communities] = community
                num_refined_communities += 1
                continue

            # Tabulate `node_to_community_weights`, the total edge weight
            # connecting this node to each refined community it's connected to.
            # Store the value for the node's own community separately, in
            # `node_to_community_weight`.
            unrefined_community = communities[i]
            start_index = indptr[i]
            end_index = indptr[i + 1]
            j = start_index
            while j < end_index - 16:
                # `indices` is sequential and will be hardware-prefetched, but
                # `refined_communities[indices[j]]` is a random access that
                # hardware prefetching may not predict
                PREFETCH(&refined_communities[indices[j + 16]])
                other = indices[j]
                node_to_community_weights[refined_communities[other]] += \
                    data[j]
                j += 1
            while j < end_index:
                other = indices[j]
                node_to_community_weights[refined_communities[other]] += \
                    data[j]
                j += 1
            node_to_community_weight = node_to_community_weights[community]
            node_to_community_weights[community] = 0

            # Find `new_community`, the community that would lead to the
            # largest positive change in the objective (`delta_objective`) if
            # we moved this node to it. Reset `node_to_community_weights` so it
            # can be used for the next node. Loop-hoisting optimization:
            # initialize `best_delta_objective` to `base_delta` instead of 0,
            # to avoid subtracting `base_delta` from `delta_objective` within
            # the loop. Only consider refined communities that are in the same
            # unrefined community as this node.
            scaled_resolution_times_weighted_degree = \
                scaled_resolution * weighted_degree
            base_delta = node_to_community_weight + \
                scaled_resolution_times_weighted_degree * \
                (weighted_degree - community_weight)
            best_delta_objective = base_delta
            start_index = indptr[i]
            end_index = indptr[i + 1]
            for j in range(start_index, end_index):
                other = indices[j]
                other_community = refined_communities[other]
                other_node_to_community_weight = \
                    node_to_community_weights[other_community]
                if other_node_to_community_weight > 0:
                    # First time seeing this community in this pass; reset
                    # `node_to_community_weights` (so it's fresh for the next
                    # node and to avoid evaluating the same community twice)
                    # and evaluate whether to move the node to this community
                    node_to_community_weights[other_community] = 0
                    if communities[other] == unrefined_community:
                        other_community_weight = \
                            refined_community_weights[other_community]
                        delta_objective = other_node_to_community_weight - \
                            scaled_resolution_times_weighted_degree * \
                            other_community_weight
                        if delta_objective > best_delta_objective:
                            best_delta_objective = delta_objective
                            new_community = other_community

            # If this node's community assignment is suboptimal (i.e. at least
            # one community had `delta_objective` larger than `base_delta`),
            # move it to the community with the largest `delta_objective`.
            # Update the community weights to account for the move: add the
            # moved node's weighted degree to its new community's weight.
            # (Don't bother subtracting it from its old community, which is
            # now empty and will not be visited again.) If the node's community
            # assignment is already optimal, do not move it.
            if best_delta_objective > base_delta:
                refined_communities[i] = new_community
                refined_community_weights[new_community] += weighted_degree
            else:
                original_to_contiguous[community] = num_refined_communities
                contiguous_to_original[num_refined_communities] = community
                num_refined_communities += 1

            # Check for KeyboardInterrupts
            if i % 131072 == 131071:
                if recv(read_fd, &signal_byte, 1, 0) > 0:
                    atomic_or(interrupted, True)
                if interrupted:
                    with gil:
                        raise KeyboardInterrupt

        # Relabel the refined communities to make them contiguous integers, by
        # mapping them through the `original_to_contiguous` mapping. For
        # instance, if the only non-empty communities after the refinement
        # phase are 2, 7, etc., renumber 2 to 0, 7 to 1, and so on. This makes
        # it easier to create a new CSR graph in the aggregation phase, where
        # each refined community becomes a single node.
        for i in range(num_nodes):
            refined_communities[i] = \
                original_to_contiguous[refined_communities[i]]

        # Reinitialize `communities` to the unrefined communities from the move
        # phase: if multiple refined communities were part of the same
        # unrefined community, their nodes in the supergraph will start off
        # with the same community label as each other. This can be done
        # in-place, since each community's contiguous index is less than its
        # original index, so we never write to entries of `communities` before
        # reading them. Also reinitialize `weighted_degrees` to the refined
        # community weights (since each refined community becomes a node in
        # the supergraph).
        #
        # In the code below, `original_index` is the original index of the
        # refined community before making the indices contiguous.
        # `communities[original_index]` is its unrefined community label, and
        # by extension the unrefined community label of every node in its
        # refined community. Meanwhile, `contiguous_index` is the new label of
        # the refined community, and by extension the index of that refined
        # community's node in the supergraph to be constructed in the
        # aggregation phase and used in the next Leiden iteration.
        for contiguous_index in range(num_refined_communities):
            original_index = contiguous_to_original[contiguous_index]
            communities[contiguous_index] = communities[original_index]
            weighted_degrees[contiguous_index] = \
                refined_community_weights[original_index]

        # 3) Aggregation phase:
        #    Aggregate the graph into a "supergraph" where each refined
        #    community becomes a node, and edge weights are summed across
        #    inter-community edges.

        # If this is not the first Leiden iteration, update each cell's refined
        # community in `final_communities` by mapping it through
        # `refined_communities`, to account for the aggregation. In other
        # words, if a cell's community label was `x` before the aggregation,
        # its new community label will be `refined_communities[x]`.
        if leiden_iteration > 0:
            for i in range(num_cells):
                final_communities[i] = \
                    refined_communities[final_communities[i]]
        # If this is the first Leiden iteration, allocate
        # `community_node_indptr`
        else:
            community_node_indptr.resize(num_refined_communities + 1)

        # Group nodes by their refined community using counting sort
        for i in range(num_refined_communities + 1):
            community_node_indptr[i] = 0
        for i in range(num_nodes):
            community_node_indptr[refined_communities[i]] += 1

        current_sum = 0
        for i in range(num_refined_communities):
            count = community_node_indptr[i]
            community_node_indptr[i] = current_sum
            current_sum += count
        community_node_indptr[num_refined_communities] = current_sum

        for i in range(num_nodes):
            community = refined_communities[i]
            community_nodes[community_node_indptr[community]] = i
            community_node_indptr[community] += 1

        i = num_refined_communities
        while i > 0:
            community_node_indptr[i] = community_node_indptr[i - 1]
            i -= 1
        community_node_indptr[0] = 0

        # Set up ping-pong pointers for the supergraph. On the first two
        # iterations, allocate the buffers underlying these pointers.
        # A supergraph with `num_refined_communities` nodes has at most
        # `num_refined_communities ** 2` edges. Thus, the new
        # edge count must be at most `num_refined_communities ** 2` or
        # the current graph's number of edges, whichever is less.
        if leiden_iteration % 2 == 0:
            if leiden_iteration == 0:
                num_edges = min(<unsigned long long> num_refined_communities *
                                num_refined_communities,
                                <unsigned long long> indptr[num_nodes])
                data_buffer.resize(num_edges)
                indices_buffer.resize(num_edges)
                indptr_buffer.resize(num_refined_communities + 1)
            next_data = data_buffer.data()
            next_indices = indices_buffer.data()
            next_indptr = indptr_buffer.data()
        else:
            if leiden_iteration == 1:
                num_edges = min(<unsigned long long> num_refined_communities *
                                num_refined_communities,
                                <unsigned long long> indptr[num_nodes])
                data_buffer_alt.resize(num_edges)
                indices_buffer_alt.resize(num_edges)
                indptr_buffer_alt.resize(num_refined_communities + 1)
            next_data = data_buffer_alt.data()
            next_indices = indices_buffer_alt.data()
            next_indptr = indptr_buffer_alt.data()

        # Aggregate the graph into the new buffers, summing the inter-community
        # weights for each pair of communities
        edges_written = 0
        next_indptr[0] = 0
        for community in range(num_refined_communities):
            num_touched = 0
            for k in range(community_node_indptr[community],
                           community_node_indptr[community + 1]):
                i = community_nodes[k]
                start_index = indptr[i]
                end_index = indptr[i + 1]
                j = start_index
                while j < end_index - 16:
                    PREFETCH(&refined_communities[indices[j + 16]])
                    other = indices[j]
                    other_community = refined_communities[other]
                    edge_weight = data[j]
                    if node_to_community_weights[other_community] == 0:
                        touched_communities[num_touched] = other_community
                        num_touched += 1
                    node_to_community_weights[other_community] += \
                        2 * edge_weight if other == i else edge_weight
                    j += 1
                while j < end_index:
                    other = indices[j]
                    other_community = refined_communities[other]
                    edge_weight = data[j]
                    if node_to_community_weights[other_community] == 0:
                        touched_communities[num_touched] = other_community
                        num_touched += 1
                    node_to_community_weights[other_community] += \
                        2 * edge_weight if other == i else edge_weight
                    j += 1

            # Halve the intra-community edges, which will become self-edges in
            # the supergraph. This is necessary because the input graph is
            # symmetric, so if nodes `i` and `j` are in the same community
            # `community`, edges i -> j and j -> i will both count towards the
            # self-edge. Non-self-edges are not a problem because edge i -> j
            # and edge j -> i will appear separately in the supergraph with the
            # same weight: thus, the supergraph will be symmetric, like the
            # input graph.
            #
            # However, do not halve self-loops if this is the supernode graph.
            # This is why we do `2 * edge_weight if other == i` above, so it
            # cancels out the halving here.
            node_to_community_weights[community] *= 0.5

            # Write out edges for this community
            for k in range(num_touched):
                other_community = touched_communities[k]
                next_data[edges_written] = \
                    node_to_community_weights[other_community]
                next_indices[edges_written] = other_community
                edges_written += 1
                node_to_community_weights[other_community] = 0

            next_indptr[community + 1] = edges_written

        # If this is the first iteration, allocate a buffer for the new refined
        # communities to avoid overwriting the original (which we need to
        # determine the final clusters), and point `contiguous_community_sizes`
        # to this new buffer
        if leiden_iteration == 0:
            refined_communities_buffer.resize(num_cells)
            refined_communities = refined_communities_buffer.data()
            contiguous_community_sizes = refined_communities

        # Swap pointers so the next iteration uses the new graph
        data = next_data
        indices = next_indices
        indptr = next_indptr

        # Update `num_nodes` to reflect that the number of nodes in the
        # supergraph is the number of refined communities
        num_nodes = num_refined_communities

        # Now we are done this Leiden iteration
        leiden_iteration += 1


def leiden_multiresolution(float[::1] data,
                           signed_integer[::1] indices,
                           signed_integer[::1] indptr,
                           const unsigned[:, ::1] neighbors,
                           unsigned[:, ::1] final_communities,
                           unsigned[::1] num_final_communities,
                           const float[::1] resolutions,
                           const unsigned min_cluster_size,
                           const unsigned long long seed,
                           const unsigned num_threads):
    cdef unsigned i, num_resolutions = resolutions.shape[0]
    cdef bint too_large = False, self_neighbors = False, interrupted = False, \
        main_thread = threading.current_thread() is threading.main_thread()
    cdef int read_fd, old_fd

    # Trick to get Ctrl + C to return inside `prange()`, since
    # `PyErr_CheckSignals()` is a no-op when not on the main thread (and in
    # many parallel runtimes, none of the worker threads are the main thread)
    r, w = socketpair()
    r.setblocking(False)
    w.setblocking(False)
    read_fd = r.fileno()
    if main_thread:
        old_fd = set_wakeup_fd(w.fileno())

    try:
        for i in prange(num_resolutions, nogil=True, num_threads=num_threads):
            num_final_communities[i] = leiden_nogil(
                data, indices, indptr, neighbors, final_communities[i],
                resolutions[i], min_cluster_size, seed,
                too_large, self_neighbors, read_fd, interrupted)
    finally:
        if main_thread:
            set_wakeup_fd(old_fd)
        r.close()
        w.close()
    return too_large, self_neighbors
