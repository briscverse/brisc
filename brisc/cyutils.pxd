from libcpp.pair cimport pair


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


cdef void bin_count_nogil(const integer[::1] arr,
                          unsigned[::1] counts,
                          unsigned num_threads) noexcept nogil

cdef pair[unsigned, unsigned] get_thread_offset(
    const signed_integer[::1] indptr,
    const unsigned thread_index,
    const unsigned num_threads) noexcept nogil

cdef void get_thread_offsets(const signed_integer[::1] indptr,
                             unsigned* thread_offsets,
                             const unsigned num_threads) noexcept nogil

cdef void max_heap_pop(unsigned* labels_i,
                       float* distances_i,
                       const unsigned k) noexcept nogil

cdef void max_heap_replace_top(unsigned* labels_i,
                               float* distances_i,
                               const unsigned label,
                               const float distance,
                               const unsigned k) noexcept nogil

cdef void max_heap_sort(unsigned* labels_i,
                        float* distances_i,
                        const unsigned k) noexcept nogil

cdef void min_heap_pop(unsigned* labels_i,
                       float* distances_i,
                       const unsigned k) noexcept nogil

cdef void min_heap_replace_top(unsigned* labels_i,
                               float* distances_i,
                               const unsigned label,
                               const float distance,
                               const unsigned k) noexcept nogil

cdef void min_heap_sort(unsigned* labels_i,
                        float* distances_i,
                        const unsigned k) noexcept nogil

cdef float norm(float[::1] array) noexcept nogil

cdef void partial_distances(const float* A,
                            const float* B,
                            const float* B_norms,
                            float* distances,
                            const unsigned num_A,
                            const unsigned num_B,
                            const unsigned ldc,
                            const unsigned num_dimensions) noexcept nogil

cdef unsigned rand(unsigned long long* state) noexcept nogil

cdef unsigned randint(const unsigned bound, unsigned long long* state) \
    noexcept nogil

cdef float random_uniform(unsigned long long* state) noexcept nogil

cdef float random_normal(unsigned long long* state) noexcept nogil

cdef void sgemv(const char trans,
                const int m,
                const int n,
                const float alpha,
                const float* a,
                const int lda,
                const float* x,
                const int incx,
                const float beta,
                float* y,
                const int incy) noexcept nogil

cdef void sgemm(const char transa,
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
                const int ldc) noexcept nogil

cdef unsigned long long srand(const unsigned long long seed) noexcept nogil