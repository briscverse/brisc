from cython.parallel cimport parallel, threadid
from libc.limits cimport UINT_MAX
from libc.math cimport M_PI
from libcpp.algorithm cimport lower_bound
from libcpp.cmath cimport cos, log, sqrt
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from scipy.linalg.cython_blas cimport sgemm as sgemm_, sgemv as sgemv_


ctypedef fused bit_width:
    unsigned
    unsigned long long


ctypedef fused integer:
    int
    unsigned
    long long
    unsigned long long


ctypedef fused numeric:
    int
    unsigned
    long long
    unsigned long long
    float
    double


ctypedef fused signed_integer:
    int
    long long


cdef extern from * nogil:
    """
    #if defined(_WIN32)
        static inline int dummy_recv(int, void *, size_t, int) {
            return -1;
        }
        #define CYTHON_RECV dummy_recv
    #else
        #include <sys/socket.h>
        #define CYTHON_RECV recv
    #endif
    """
    # Cython sees "recv", but writes "CYTHON_RECV" into the generated C++ file
    int recv "CYTHON_RECV"(int sockfd, void *buf, size_t len, int flags)


cdef extern from * nogil:
    """
    #define atomic_and(x, y) _Pragma("omp atomic") x &= y
    #define atomic_or(x, y) _Pragma("omp atomic") x |= y
    """
    void atomic_and(bint &x, bint y)
    void atomic_or(bint &x, bint y)


cdef extern from * nogil:
    """
    #if defined(__GNUC__) || defined(__clang__)
    #define PREFETCH __builtin_prefetch
    #else
    #define PREFETCH
    #endif
    """
    void PREFETCH(const void *addr)


# A drop-in replacement for Cython's `vector` that does not automatically
# zero-initialize

cdef extern from * nogil:
    """
    // A simplified standalone version of boost::noinit_adaptor
    template<class A>
    struct noinit_adaptor : A {
        template<class U>
        struct rebind {
            typedef noinit_adaptor<typename std::allocator_traits<A>::template
                                   rebind_alloc<U>> other;
        };

        template<class U>
        void construct(U* p) {
            ::new(p) U;
        }

        template<class U, class V>
        void construct(U* p, const V& v) {
            ::new(p) U(v);
        }

        template<class U>
        void destroy(U* p) {
            p->~U();
        }
    };

    #include <vector>
    template<class T>
    using uninitialized_vector = \
        std::vector<T, noinit_adaptor<std::allocator<T>>>;
    """
    cdef cppclass uninitialized_vector[T]:
        ctypedef T value_type
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        cppclass const_iterator
        cppclass iterator:
            iterator() except +
            iterator(iterator&) except +
            T& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator++(int)
            iterator operator--(int)
            iterator operator+(size_type)
            iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)
        cppclass const_iterator:
            const_iterator() except +
            const_iterator(iterator&) except +
            const_iterator(const_iterator&) except +
            operator=(iterator&) except +
            const T& operator*()
            const_iterator operator++()
            const_iterator operator--()
            const_iterator operator++(int)
            const_iterator operator--(int)
            const_iterator operator+(size_type)
            const_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)

        cppclass const_reverse_iterator
        cppclass reverse_iterator:
            reverse_iterator() except +
            reverse_iterator(reverse_iterator&) except +
            T& operator*()
            reverse_iterator operator++()
            reverse_iterator operator--()
            reverse_iterator operator++(int)
            reverse_iterator operator--(int)
            reverse_iterator operator+(size_type)
            reverse_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(reverse_iterator)
            bint operator==(const_reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator!=(const_reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator<(const_reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator>(const_reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator<=(const_reverse_iterator)
            bint operator>=(reverse_iterator)
            bint operator>=(const_reverse_iterator)
        cppclass const_reverse_iterator:
            const_reverse_iterator() except +
            const_reverse_iterator(reverse_iterator&) except +
            operator=(reverse_iterator&) except +
            const T& operator*()
            const_reverse_iterator operator++()
            const_reverse_iterator operator--()
            const_reverse_iterator operator++(int)
            const_reverse_iterator operator--(int)
            const_reverse_iterator operator+(size_type)
            const_reverse_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(reverse_iterator)
            bint operator==(const_reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator!=(const_reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator<(const_reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator>(const_reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator<=(const_reverse_iterator)
            bint operator>=(reverse_iterator)
            bint operator>=(const_reverse_iterator)

        uninitialized_vector() except +
        uninitialized_vector(uninitialized_vector&) except +
        uninitialized_vector(size_type) except +
        uninitialized_vector(size_type, T&) except +
        T& operator[](size_type)
        bint operator==(uninitialized_vector&, uninitialized_vector&)
        bint operator!=(uninitialized_vector&, uninitialized_vector&)
        bint operator<(uninitialized_vector&, uninitialized_vector&)
        bint operator>(uninitialized_vector&, uninitialized_vector&)
        bint operator<=(uninitialized_vector&, uninitialized_vector&)
        bint operator>=(uninitialized_vector&, uninitialized_vector&)
        void assign(size_type, const T&)
        void assign[InputIt](InputIt, InputIt) except +
        T& at(size_type) except +
        T& back()
        iterator begin()
        const_iterator const_begin "begin"()
        const_iterator cbegin()
        size_type capacity()
        void clear()
        bint empty()
        iterator end()
        const_iterator const_end "end"()
        const_iterator cend()
        iterator erase(iterator)
        iterator erase(iterator, iterator)
        T& front()
        iterator insert(iterator, const T&) except +
        iterator insert(iterator, size_type, const T&) except +
        iterator insert[InputIt](iterator, InputIt, InputIt) except +
        size_type max_size()
        void pop_back()
        void push_back(T&) except +
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        const_reverse_iterator crbegin()
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()
        const_reverse_iterator crend()
        void reserve(size_type) except +
        void resize(size_type) except +
        void resize(size_type, T&) except +
        size_type size()
        void swap(uninitialized_vector&)
        T* data()
        const T* const_data "data"()
        void shrink_to_fit() except +
        iterator emplace(const_iterator, ...) except +
        T& emplace_back(...) except +


ctypedef unsigned unsigned_  # hack to get templates to work with unsigned


cdef inline void bin_count_nogil(const integer[::1] arr,
                                 unsigned[::1] counts,
                                 unsigned num_threads) noexcept nogil:
    cdef unsigned long long start, end, i, num_bins, chunk_size, \
        num_elements = arr.shape[0]
    cdef unsigned thread_index
    cdef vector[vector[unsigned]] thread_counts
    cdef unsigned* counts_pointer

    num_threads = min(num_threads, num_elements)
    if num_threads <= 1:
        counts[:] = 0
        for i in range(num_elements):
            counts[arr[i]] += 1
    else:
        # Store counts for each thread in a temporary buffer, then aggregate at
        # the end. As an optimization, put the counts for the last thread
        # (`thread_index == num_threads - 1`) directly into the final `counts`
        # array.
        thread_counts.resize(num_threads - 1)
        num_bins = counts.shape[0]
        chunk_size = (num_elements + num_threads - 1) / num_threads
        with parallel(num_threads=num_threads):
            thread_index = threadid()
            start = thread_index * chunk_size
            if thread_index == num_threads - 1:
                end = num_elements
                counts[:] = 0
                for i in range(start, end):
                    counts[arr[i]] += 1
            else:
                thread_counts[thread_index].resize(num_bins)
                counts_pointer = thread_counts[thread_index].data()
                end = min(start + chunk_size, num_elements)
                for i in range(start, end):
                    counts_pointer[arr[i]] += 1

        # Aggregate counts from all threads except the last
        for thread_index in range(num_threads - 1):
            counts_pointer = thread_counts[thread_index].data()
            for i in range(num_bins):
                counts[i] += counts_pointer[i]


cdef inline pair[unsigned, unsigned] get_thread_offset(
        const signed_integer[::1] indptr,
        const unsigned thread_index,
        const unsigned num_threads) noexcept nogil:
    # A helper function for load-balancing parallel iteration over sparse
    # arrays. Gets the start and end row (for CSR) or column (for CSC) index
    # that each thread will work on, to ensure each thread works on about the
    # same number of sparse matrix elements.

    cdef unsigned start, end, num_major = indptr.shape[0] - 1
    cdef unsigned long long num_elements_per_thread = \
        indptr[num_major] / num_threads, \
        num_elements_start = thread_index * num_elements_per_thread, \
        num_elements_end = (thread_index + 1) * num_elements_per_thread

    start = 0 if thread_index == 0 else min(
        <unsigned>(lower_bound(&indptr[0], &indptr[0] + indptr.shape[0],
                               num_elements_start) - &indptr[0]),
        num_major)
    end = num_major if thread_index == num_threads - 1 else min(
        <unsigned>(lower_bound(&indptr[0], &indptr[0] + indptr.shape[0],
                               num_elements_end) - &indptr[0]),
        num_major)
    return pair[unsigned_, unsigned_](start, end)


cdef inline void get_thread_offsets(const signed_integer[::1] indptr,
                                    unsigned* thread_offsets,
                                    const unsigned num_threads) noexcept nogil:
    # A variant of `get_thread_offset()` that gets the start and end row (for
    # CSR) or column (for CSC) index that each thread will work on, for all
    # threads simultaneously. For each thread, `thread_offsets[thread_index]`
    # will contain the start index, and `thread_offsets[thread_index + 1]` will
    # contain the end index.

    cdef unsigned thread_index, num_major = indptr.shape[0] - 1
    cdef unsigned long long num_elements_start, \
        num_elements_per_thread = indptr[num_major] / num_threads

    thread_offsets[0] = 0
    for thread_index in range(1, num_threads):
        num_elements_start = thread_index * num_elements_per_thread
        thread_offsets[thread_index] = min(
            <unsigned>(lower_bound(&indptr[0], &indptr[0] + indptr.shape[0],
                                   num_elements_start) - &indptr[0]),
            num_major)
    thread_offsets[num_threads] = num_major


cdef inline void max_heap_pop(unsigned* labels_i,
                              float* distances_i,
                              const unsigned k) noexcept nogil:
    # Pops the top element from the max-heap defined by `distances_i[0..k-1]`
    # and `labels_i[0..k-1]`. On output the `k-1`th element is undefined.

    cdef unsigned label, j = 1, child
    cdef float distance
    distances_i -= 1  # use 1-based indexing for easier node->child translation
    labels_i -= 1
    distance = distances_i[k]
    label = labels_i[k]
    while True:
        child = j << 1
        if child > k:
            break
        child += child < k and distances_i[child] <= distances_i[child + 1]
        if distance > distances_i[child]:
            break
        distances_i[j] = distances_i[child]
        labels_i[j] = labels_i[child]
        j = child
    distances_i[j] = distance
    labels_i[j] = label


cdef inline void max_heap_replace_top(unsigned* labels_i,
                                      float* distances_i,
                                      const unsigned label,
                                      const float distance,
                                      const unsigned k) noexcept nogil:
    # Replaces the top element from the max-heap defined by
    # `distances_i[0..k-1]` and `labels_i[0..k-1]`. Equivalent to
    # `std::pop_heap` followed by `std::push_heap`, but done more efficiently
    # as a single operation.

    cdef unsigned j = 1, child
    distances_i -= 1  # use 1-based indexing for easier node->child translation
    labels_i -= 1
    while True:
        child = j << 1
        if child > k:
            break
        child += child < k and distances_i[child] <= distances_i[child + 1]
        if distance > distances_i[child]:
            break
        distances_i[j] = distances_i[child]
        labels_i[j] = labels_i[child]
        j = child
    distances_i[j] = distance
    labels_i[j] = label


cdef inline void max_heap_sort(unsigned* labels_i,
                               float* distances_i,
                               const unsigned k) noexcept nogil:
    cdef unsigned j, label
    cdef float distance
    for j in range(k):
        # Save the root (maximum element)
        distance = distances_i[0]
        label = labels_i[0]
        # Restore the heap property with reduced size `k - i`
        max_heap_pop(labels_i, distances_i, k - j)
        # Place the maximum element after the end of the heap
        distances_i[k - j - 1] = distance
        labels_i[k - j - 1] = label


cdef inline void min_heap_pop(unsigned* labels_i,
                              float* distances_i,
                              const unsigned k) noexcept nogil:
    # Pops the top element from the min-heap defined by `distances_i[0..k-1]`
    # and `labels_i[0..k-1]`. On output the `k-1`th element is undefined.

    cdef unsigned label, j = 1, child
    cdef float distance
    distances_i -= 1  # use 1-based indexing for easier node->child translation
    labels_i -= 1
    distance = distances_i[k]
    label = labels_i[k]
    while True:
        child = j << 1
        if child > k:
            break
        child += child < k and distances_i[child] >= distances_i[child + 1]
        if distance < distances_i[child]:
            break
        distances_i[j] = distances_i[child]
        labels_i[j] = labels_i[child]
        j = child
    distances_i[j] = distance
    labels_i[j] = label


cdef inline void min_heap_replace_top(unsigned* labels_i,
                                      float* distances_i,
                                      const unsigned label,
                                      const float distance,
                                      const unsigned k) noexcept nogil:
    # Replaces the top element from the min-heap defined by
    # `distances_i[0..k-1]` and `labels_i[0..k-1]`. Equivalent to
    # `std::pop_heap` followed by `std::push_heap`, but done more efficiently
    # as a single operation.

    cdef unsigned j = 1, child
    distances_i -= 1  # use 1-based indexing for easier node->child translation
    labels_i -= 1
    while True:
        child = j << 1
        if child > k:
            break
        child += child < k and distances_i[child] >= distances_i[child + 1]
        if distance < distances_i[child]:
            break
        distances_i[j] = distances_i[child]
        labels_i[j] = labels_i[child]
        j = child
    distances_i[j] = distance
    labels_i[j] = label


cdef inline void min_heap_sort(unsigned* labels_i,
                               float* distances_i,
                               const unsigned k) noexcept nogil:
    cdef unsigned j, label
    cdef float distance
    for j in range(k):
        # Save the root (minimum element)
        distance = distances_i[0]
        label = labels_i[0]
        # Restore the heap property with reduced size `k - i`
        min_heap_pop(labels_i, distances_i, k - j)
        # Place the minimum element after the end of the heap
        distances_i[k - j - 1] = distance
        labels_i[k - j - 1] = label


cdef inline float norm(float[::1] array) noexcept nogil:
    cdef unsigned i
    cdef float norm = 0
    for i in range(array.shape[0]):
        norm += array[i] * array[i]
    return sqrt(norm)


cdef inline void partial_distances(const float* A,
                                   const float* B,
                                   const float* B_norms,
                                   float* distances,
                                   const unsigned num_A,
                                   const unsigned num_B,
                                   const unsigned ldc,
                                   const unsigned num_dimensions) \
        noexcept nogil:
    # Calculate the "partial" distance from each row of A (of shape
    # `num_A × num_dimensions`) to each row of B (of shape
    # `num_B × num_dimensions`). Use the identity:
    # ||A - B||² = ||A||² - 2 * A.dot(B.T) + ||B||²,
    # but skip calculating the ||A||² term since the closest row of `B` for a
    # given row of `A` does not depend on ||A||². This is why we call it a
    # "partial" distance. `ldc` is the row stride of `distances`; pass
    # `ldc == num_B` for the untiled case.

    cdef char transA = b'T', transB = b'N'
    cdef float alpha = -2, beta = 1
    cdef unsigned i, j

    for i in range(num_A):
        for j in range(num_B):
            # distances = ||B||²
            distances[i * ldc + j] = B_norms[j]

    # distances -= 2 * A.dot(B.T)
    sgemm(transA, transB, num_B, num_A, num_dimensions, alpha, B,
          num_dimensions, A, num_dimensions, beta, distances, ldc)


cdef inline unsigned rand(unsigned long long* state) noexcept nogil:
    cdef unsigned long long x = state[0]
    state[0] = x * 6364136223846793005ULL + 1442695040888963407ULL
    cdef unsigned s = (x ^ (x >> 18)) >> 27
    cdef unsigned rot = x >> 59
    return (s >> rot) | (s << ((-rot) & 31))


cdef inline unsigned randint(const unsigned bound, unsigned long long* state) \
        noexcept nogil:
    # Lemire's method
    cdef unsigned long long m = \
        <unsigned long long> rand(state) * <unsigned long long> bound
    cdef unsigned threshold, l = <unsigned> m
    if l < bound:
        threshold = -bound % bound
        while l < threshold:
            m = <unsigned long long> rand(state) * <unsigned long long> bound
            l = <unsigned> m
    return <unsigned>(m >> 32)


cdef inline float random_uniform(unsigned long long* state) noexcept nogil:
    # Returns a random number in U(0, 1)
    return <float> rand(state) / UINT_MAX


cdef inline float random_normal(unsigned long long* state) noexcept nogil:
    # Samples a random number from the standard normal distribution
    # using the Box-Muller transform
    cdef float u1, u2, r, theta
    while True:
        u1 = random_uniform(state)
        if u1:
            break
    u2 = random_uniform(state)
    r = sqrt(<float> -2 * log(u1))
    theta = <float> 2 * <float> M_PI * u2
    return r * cos(theta)


cdef inline void sgemv(const char trans,
                       const int m,
                       const int n,
                       const float alpha,
                       const float* a,
                       const int lda,
                       const float* x,
                       const int incx,
                       const float beta,
                       float* y,
                       const int incy) noexcept nogil:
    sgemv_(<char*> &trans, <int*> &m, <int*> &n, <float*> &alpha, <float*> a,
           <int*> &lda, <float*> x, <int*> &incx, <float*> &beta, y,
           <int*> &incy)


cdef inline void sgemm(const char transa,
                       const char transb,
                       const int m,
                       const int n,
                       const int k,
                       const float alpha,
                       const float* a,
                       const int lda,
                       const float* b,
                       const int ldb,
                       const float beta,
                       float* c,
                       const int ldc) noexcept nogil:
    sgemm_(<char*> &transa, <char*> &transb, <int*> &m, <int*> &n, <int*> &k,
           <float*> &alpha, <float*> a, <int*> &lda, <float*> b, <int*> &ldb,
           <float*> &beta, c, <int*> &ldc)


cdef inline unsigned long long srand(const unsigned long long seed) \
        noexcept nogil:
    cdef unsigned long long state = seed + 1442695040888963407ULL
    rand(&state)
    return state
