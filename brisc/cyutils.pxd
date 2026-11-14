from cython.parallel cimport parallel, threadid
from libc.limits cimport UINT_MAX
from libc.math cimport M_PI
from libcpp.algorithm cimport lower_bound
from libcpp.cmath cimport cos, log, sqrt
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


# NUMA-aware allocation of zeroed memory, underlying `numa_zeros()` in
# utils.py. Large allocations get fresh pages from `mmap()` on Linux, bypassing
# glibc's malloc arena so that NUMA first-touch placement is guaranteed, with
# Transparent Huge Pages (THP) requested where they won't cause NUMA skew.
# Smaller allocations, and all allocations on other platforms, use `calloc()`.
# Either way, the memory is zero-initialized.
#
# `num_threads` is the number of threads that will first touch the memory
# (i.e. write to it before anything else does), each touching a contiguous
# chunk. When it is 1, or there is only one NUMA node, the memory cannot be
# split across NUMA nodes, so it always gets huge pages. Otherwise, it gets
# huge pages only if each thread's chunk is at least 16 MB: below that, a
# thread's chunk fits within the TLB's reach even with 4 KB pages, so huge
# pages provide little benefit, while the huge page straddling each boundary
# between NUMA nodes is placed entirely on one node, potentially making some
# threads' entire chunks remote.
#
# System information is read once (per extension module, since each module
# compiles its own copy of this code) on the first allocation.

cdef extern from * nogil:
    """
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <new>
    #ifdef __linux__
    #include <sched.h>
    #include <sys/mman.h>
    #endif

    // Allocations of at least this many bytes use `mmap()` on Linux
    constexpr size_t NUMA_MMAP_MIN_BYTES = 1ULL << 20;

    // Memory first touched by multiple threads on a multi-NUMA-node system
    // only gets huge pages if each thread's chunk is at least this large
    constexpr size_t NUMA_MIN_HUGE_PAGE_BYTES_PER_THREAD = 16ULL << 20;

    #ifdef __linux__
    struct numa_system_info {
        size_t huge_page_size = 2ULL << 20;  // fallback if sysfs is unreadable
        bool advise_hugepage = false;
        unsigned num_numa_nodes = 1;
    };

    // Read a sysfs file (at most `size - 1` bytes) into `buffer`; return
    // false if the file can't be read
    inline bool numa_read_file(const char* path, char* buffer, size_t size) {
        FILE* file = std::fopen(path, "r");
        if (!file) return false;
        size_t length = std::fread(buffer, 1, size - 1, file);
        std::fclose(file);
        buffer[length] = '\\0';
        return length > 0;
    }

    // Call `f(i)` for each number in a sysfs list like "0-47,96-143"
    template<class F>
    inline void numa_for_each_in_list(const char* list, F f) {
        char* end;
        for (;;) {
            long first = std::strtol(list, &end, 10);
            if (end == list) return;
            long last = *end == '-' ? std::strtol(end + 1, &end, 10) : first;
            for (long i = first; i <= last; i++) f(i);
            if (*end != ',') return;
            list = end + 1;
        }
    }

    inline numa_system_info numa_read_system_info() {
        numa_system_info info;
        char buffer[4096];

        // Huge page size: 2 MB on x86, but e.g. 512 MB on ARM with 64 KB
        // base pages
        if (numa_read_file(
                "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size",
                buffer, sizeof(buffer))) {
            unsigned long long size = std::strtoull(buffer, nullptr, 10);
            if (size > 0) info.huge_page_size = size;
        }

        // Request huge pages only when THP is in "madvise" mode (in "always"
        // mode the kernel uses them anyway, and in "never" mode it never
        // does), and only when the kernel won't compact memory synchronously
        // to satisfy the request (~10x slowdowns on fragmented nodes). The
        // selected option in each file is in brackets, e.g. "[madvise]".
        info.advise_hugepage =
            numa_read_file("/sys/kernel/mm/transparent_hugepage/enabled",
                           buffer, sizeof(buffer)) &&
            std::strstr(buffer, "[madvise]") &&
            numa_read_file("/sys/kernel/mm/transparent_hugepage/defrag",
                           buffer, sizeof(buffer)) &&
            (std::strstr(buffer, "[defer]") || std::strstr(buffer, "[never]"));

        // Count the NUMA nodes containing at least one CPU this process may
        // run on, which excludes memory-only nodes (e.g. HBM or CXL memory)
        // and nodes outside the process's cpuset (e.g. under Slurm)
        cpu_set_t cpus;
        char cpu_list[4096];
        if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0 &&
                numa_read_file("/sys/devices/system/node/has_cpu", buffer,
                               sizeof(buffer))) {
            unsigned num_nodes = 0;
            numa_for_each_in_list(buffer, [&](long node) {
                char path[64];
                std::snprintf(path, sizeof(path),
                              "/sys/devices/system/node/node%ld/cpulist",
                              node);
                bool usable = false;
                if (numa_read_file(path, cpu_list, sizeof(cpu_list)))
                    numa_for_each_in_list(cpu_list, [&](long cpu) {
                        usable |= cpu < CPU_SETSIZE && CPU_ISSET(cpu, &cpus);
                    });
                num_nodes += usable;
            });
            if (num_nodes > 0) info.num_numa_nodes = num_nodes;
        }
        return info;
    }

    inline const numa_system_info& numa_get_system_info() {
        static const numa_system_info info = numa_read_system_info();
        return info;
    }

    // Round the length of large mappings up to a multiple of the huge page
    // size, so that recent kernels (6.7+) align them to huge page boundaries.
    // Don't round up mappings smaller than one huge page: they can't contain
    // a huge page anyway, and rounding them up could let a single write
    // commit an entire huge page (e.g. 512 MB with 64 KB base pages).
    inline size_t numa_mapping_length(size_t num_bytes) {
        size_t huge_page_size = numa_get_system_info().huge_page_size;
        if (num_bytes < huge_page_size) return num_bytes;
        return (num_bytes + huge_page_size - 1) / huge_page_size *
            huge_page_size;
    }
    #endif

    inline void* numa_allocate(size_t num_bytes, unsigned num_threads) {
    #ifdef __linux__
        if (num_bytes >= NUMA_MMAP_MIN_BYTES) {
            const numa_system_info& info = numa_get_system_info();
            size_t length = numa_mapping_length(num_bytes);
            void* pointer = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (pointer == MAP_FAILED) throw std::bad_alloc();
            if (num_bytes >= info.huge_page_size) {
                bool huge = info.num_numa_nodes == 1 || num_threads <= 1 ||
                    num_bytes / num_threads >=
                        NUMA_MIN_HUGE_PAGE_BYTES_PER_THREAD;
                if (!huge)
                    madvise(pointer, length, MADV_NOHUGEPAGE);
                else if (info.advise_hugepage)
                    madvise(pointer, length, MADV_HUGEPAGE);
            }
            return pointer;
        }
    #endif
        (void) num_threads;
        void* pointer = std::calloc(num_bytes > 0 ? num_bytes : 1, 1);
        if (!pointer) throw std::bad_alloc();
        return pointer;
    }

    inline void numa_free(void* pointer, size_t num_bytes) noexcept {
    #ifdef __linux__
        if (num_bytes >= NUMA_MMAP_MIN_BYTES) {
            munmap(pointer, numa_mapping_length(num_bytes));
            return;
        }
    #endif
        (void) num_bytes;
        std::free(pointer);
    }
    """
    void* numa_allocate(size_t num_bytes, unsigned num_threads) except +
    void numa_free(void* pointer, size_t num_bytes) noexcept


cdef inline (unsigned, unsigned) get_thread_offset(
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
    return start, end


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
