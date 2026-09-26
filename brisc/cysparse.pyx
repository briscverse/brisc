# Cython code underlying sparse.py

from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libc.string cimport memcpy
from libcpp.algorithm cimport lower_bound
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from .cyutils cimport atomic_and, atomic_or, bit_width, get_thread_offset, \
    integer, numeric, signed_integer


ctypedef fused integer2:
    int
    unsigned
    long long
    unsigned long long


ctypedef fused numeric2:
    int
    unsigned
    long long
    unsigned long long
    float
    double


ctypedef fused unsigned_integer:
    unsigned
    unsigned long long


# Code to sort two arrays, one with indices and one with corresponding data, by
# index

cdef extern from * nogil:
    """
    #include <algorithm>
    #include <iterator>

    template<typename I, typename T>
    struct RefTuple;

    template<typename I, typename T>
    struct Tuple {
        I i;
        T x;
        Tuple(const RefTuple<I, T>& t) noexcept : i(t.i), x(t.x) {}
        bool operator<(const Tuple<I, T>& t) const noexcept {
            return i < t.i; };
        bool operator<(const RefTuple<I, T>& t) const noexcept {
            return i < t.i; };
    };

    template<typename I, typename T>
    struct RefTuple {
        I& i;
        T& x;
        RefTuple(I& i, T& x) noexcept : i(i), x(x) {}
        void operator=(const Tuple<I, T>& t) noexcept { i = t.i; x = t.x; };
        void operator=(const RefTuple<I, T>& t) noexcept { i = t.i; x = t.x; };
        bool operator<(const Tuple<I, T>& t) const noexcept {
            return i < t.i; };
    };

    template<typename I, typename T>
    inline void swap(RefTuple<I, T>&& t1, RefTuple<I, T>&& t2) noexcept {
        std::swap(t1.i, t2.i);
        std::swap(t1.x, t2.x);
    }

    template<typename I, typename T>
    class IterTuple {
        I* i;
        T* x;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = Tuple<I, T>;
        using difference_type = std::ptrdiff_t;
        using pointer = Tuple<I, T>*;
        using reference = RefTuple<I, T>;
        IterTuple(I* i, T* t) noexcept : i(i), x(t) {}
        IterTuple(const IterTuple&) noexcept = default;
        IterTuple& operator=(const IterTuple&) noexcept = default;
        RefTuple<I, T> operator*() const noexcept {
            return RefTuple<I, T>(*i, *x); }
        IterTuple& operator++() noexcept { i++; x++; return *this; }
        IterTuple& operator--() noexcept { i--; x--; return *this; }
        IterTuple operator++(int) noexcept {
            IterTuple tmp(*this); i++; x++; return tmp; }
        IterTuple operator--(int) noexcept {
            IterTuple tmp(*this); i--; x--; return tmp; }
        difference_type operator-(const IterTuple& rhs) const noexcept {
            return i - rhs.i; }
        IterTuple operator+(difference_type n) const noexcept {
            IterTuple tmp(*this); tmp.i += n; tmp.x += n; return tmp; }
        IterTuple operator-(difference_type n) const noexcept {
            IterTuple tmp(*this); tmp.i -= n; tmp.x -= n; return tmp; }
        IterTuple& operator+=(difference_type n) noexcept {
            i += n; x += n; return *this; }
        IterTuple& operator-=(difference_type n) noexcept {
            i -= n; x -= n; return *this; }
        bool operator==(const IterTuple& rhs) const noexcept {
            return i == rhs.i; }
        bool operator!=(const IterTuple& rhs) const noexcept {
            return i != rhs.i; }
        bool operator<(const IterTuple& rhs) const noexcept { return i < rhs.i; }
        bool operator>=(const IterTuple& rhs) const noexcept { return i >= rhs.i; }
    };

    template<typename I, typename T>
    void sort_both_by_indices(I* Aj_start, I* Aj_end,
                              T* Ax_start, T* Ax_end) noexcept {
        IterTuple<I, T> begin(Aj_start, Ax_start);
        IterTuple<I, T> end(Aj_end, Ax_end);
        std::sort(begin, end);
    }
    """
    cdef void sort_both_by_indices[I, T](I* Aj_start, I* Aj_end,
                                         T* Ax_start, T* Ax_end) noexcept


def cast(const numeric[::1] input,
         numeric2[::1] output,
         const unsigned num_threads):
    cdef unsigned long long i

    if num_threads == 1:
        for i in range(<unsigned long long> input.shape[0]):
            output[i] = <numeric2> input[i]
    else:
        for i in prange(<unsigned long long> input.shape[0], nogil=True,
                        num_threads=num_threads):
            output[i] = <numeric2> input[i]


def check_bounds_only(const unsigned_integer[::1] x,
                      const unsigned length,
                      unsigned num_threads):
    # For unsigned integer arrays

    cdef unsigned i, array_length = x.shape[0]
    cdef bint out_of_bounds = False

    num_threads = min(num_threads, array_length)
    if num_threads <= 1:
        for i in range(array_length):
            if x[i] >= length:
                out_of_bounds = True
                return out_of_bounds
    else:
        # Use hardware-level atomics to thread-safely flag
        # out-of-bounds indices
        for i in prange(array_length, nogil=True, num_threads=num_threads):
            if x[i] >= length:
                atomic_or(out_of_bounds, True)
                with gil:
                    return out_of_bounds
    return out_of_bounds


def check_bounds_and_negative_indices(const signed_integer[::1] x,
                                      const signed_integer length,
                                      unsigned num_threads):
    # First pass for signed integer arrays that do not own their own data, to
    # check out-of-bounds indices and decide whether to copy

    cdef unsigned i, array_length = x.shape[0]
    cdef bint out_of_bounds = False, negative_indices = False

    num_threads = min(num_threads, array_length)
    if num_threads <= 1:
        for i in range(array_length):
            if x[i] < -length or x[i] >= length:
                out_of_bounds = True
                return out_of_bounds, negative_indices
            if x[i] < 0:
                negative_indices = True
    else:
        # Use hardware-level atomics to thread-safely flag negative and
        # out-of-bounds indices
        for i in prange(array_length, nogil=True, num_threads=num_threads):
            if x[i] < -length or x[i] >= length:
                atomic_or(out_of_bounds, True)
                with gil:
                    return out_of_bounds, negative_indices
            if x[i] < 0:
                atomic_or(negative_indices, True)
    return out_of_bounds, negative_indices


def check_bounds_and_wrap_around(signed_integer[::1] x,
                                 const signed_integer length,
                                 unsigned num_threads):
    # For signed integer arrays that own their own data

    cdef unsigned i, array_length = x.shape[0]
    cdef bint out_of_bounds = False

    num_threads = min(num_threads, array_length)
    if num_threads <= 1:
        for i in range(array_length):
            if x[i] < -length or x[i] >= length:
                out_of_bounds = True
                return out_of_bounds
            if x[i] < 0:
                x[i] += length
    else:
        # Use hardware-level atomics to thread-safely flag
        # out-of-bounds indices
        for i in prange(array_length, nogil=True, num_threads=num_threads):
            if x[i] < -length or x[i] >= length:
                atomic_or(out_of_bounds, True)
                with gil:
                    return out_of_bounds
            if x[i] < 0:
                x[i] += length
    return out_of_bounds


def csr_has_canonical_format(const unsigned n_row,
                             const signed_integer[::1] Ap,
                             const signed_integer[::1] Aj,
                             unsigned num_threads):
    cdef unsigned i
    cdef signed_integer jj
    cdef bint has_canonical_format = True

    num_threads = min(num_threads, n_row)
    if num_threads <= 1:
        for i in range(n_row):
            if Ap[i] > Ap[i + 1]:
                return False
            for jj in range(Ap[i] + 1, Ap[i + 1]):
                if not Aj[jj - 1] < Aj[jj]:
                    return False
    else:
        for i in prange(n_row, nogil=True, num_threads=num_threads):
            if Ap[i] > Ap[i + 1]:
                atomic_and(has_canonical_format, False)
                with gil:
                    return False
            for jj in range(Ap[i] + 1, Ap[i + 1]):
                if not Aj[jj - 1] < Aj[jj]:
                    atomic_and(has_canonical_format, False)
                    with gil:
                        return False
    return has_canonical_format


def csr_has_sorted_indices(const unsigned n_row,
                           const signed_integer[::1] Ap,
                           const signed_integer[::1] Aj,
                           unsigned num_threads):
    cdef unsigned i
    cdef signed_integer jj
    cdef bint has_sorted_indices = True

    num_threads = min(num_threads, n_row)
    if num_threads <= 1:
        for i in range(n_row):
            for jj in range(Ap[i], Ap[i + 1] - 1):
                if Aj[jj] > Aj[jj + 1]:
                    return False
    else:
        for i in prange(n_row, nogil=True, num_threads=num_threads):
            for jj in range(Ap[i], Ap[i + 1] - 1):
                if Aj[jj] > Aj[jj + 1]:
                    atomic_and(has_sorted_indices, False)
                    with gil:
                        return False

    return has_sorted_indices


def csr_sort_indices(const unsigned n_row,
                     const signed_integer[::1] Ap,
                     signed_integer[::1] Aj,
                     bit_width[::1] Ax,
                     unsigned num_threads):
    cdef unsigned i, thread_index
    cdef signed_integer row_start, row_end
    cdef pair[unsigned, unsigned] row_range

    num_threads = min(num_threads, n_row)
    if num_threads <= 1:
        for i in range(n_row):
            row_start = Ap[i]
            row_end = Ap[i + 1]
            sort_both_by_indices(&Aj[0] + row_start, &Aj[0] + row_end,
                                 &Ax[0] + row_start, &Ax[0] + row_end)
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(Ap, thread_index, num_threads)
            for i in range(row_range.first, row_range.second):
                row_start = Ap[i]
                row_end = Ap[i + 1]
                sort_both_by_indices(&Aj[0] + row_start, &Aj[0] + row_end,
                                     &Ax[0] + row_start, &Ax[0] + row_end)


def get_csr_submatrix1(const unsigned n_row,
                       const unsigned n_col,
                       const signed_integer[::1] Ap,
                       const signed_integer[::1] Aj,
                       const signed_integer ir0,
                       const signed_integer ir1,
                       const signed_integer ic0,
                       const signed_integer ic1,
                       signed_integer[::1] Bp,
                       unsigned num_threads):
    cdef unsigned i, thread_index, new_n_row = ir1 - ir0
    cdef signed_integer new_nnz, jj, row_start, row_end
    cdef pair[unsigned, unsigned] row_range

    Bp[0] = 0
    num_threads = min(num_threads, new_n_row)
    if num_threads <= 1:
        new_nnz = 0
        for i in range(new_n_row):
            row_start = Ap[ir0 + i]
            row_end = Ap[ir0 + i + 1]
            for jj in range(row_start, row_end):
                if Aj[jj] >= ic0 and Aj[jj] < ic1:
                    new_nnz += 1
            Bp[i + 1] = new_nnz

            # Check for KeyboardInterrupts
            if i % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(Ap[ir0:ir1 + 1], thread_index,
                                          num_threads)
            for i in range(row_range.first, row_range.second):
                new_nnz = 0
                row_start = Ap[ir0 + i]
                row_end = Ap[ir0 + i + 1]
                for jj in range(row_start, row_end):
                    if Aj[jj] >= ic0 and Aj[jj] < ic1:
                        new_nnz = new_nnz + 1
                Bp[i + 1] = new_nnz

                # Check for KeyboardInterrupts
                if i % 8192 == 8191:
                    with gil:
                        PyErr_CheckSignals()

        for i in range(1, new_n_row):
            Bp[i + 1] += Bp[i]  # cumsum


def get_csr_submatrix1_check(const unsigned n_row,
                             const unsigned n_col,
                             const signed_integer[::1] Ap,
                             const signed_integer[::1] Aj,
                             const signed_integer ir0,
                             const signed_integer ir1,
                             const signed_integer ic0,
                             const signed_integer ic1,
                             signed_integer[::1] Bp,
                             unsigned num_threads):
    cdef unsigned i, thread_index, new_n_row = ir1 - ir0
    cdef signed_integer new_nnz, jj, row_start, row_end
    cdef pair[unsigned, unsigned] row_range
    cdef bint has_canonical_format = True, has_sorted_indices = True

    Bp[0] = 0
    num_threads = min(num_threads, new_n_row)
    if num_threads <= 1:
        new_nnz = 0
        for i in range(new_n_row):
            row_start = Ap[ir0 + i]
            row_end = Ap[ir0 + i + 1]
            if row_start < row_end:
                if Aj[row_start] >= ic0 and Aj[row_start] < ic1:
                    new_nnz += 1
                for jj in range(row_start + 1, row_end):
                    if Aj[jj - 1] >= Aj[jj]:
                        has_canonical_format = False
                        if Aj[jj - 1] > Aj[jj]:
                            has_sorted_indices = False
                    if Aj[jj] >= ic0 and Aj[jj] < ic1:
                        new_nnz += 1
            Bp[i + 1] = new_nnz

            if i % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(Ap[ir0:ir1 + 1], thread_index,
                                          num_threads)
            for i in range(row_range.first, row_range.second):
                new_nnz = 0
                row_start = Ap[ir0 + i]
                row_end = Ap[ir0 + i + 1]
                if row_start < row_end:
                    if Aj[row_start] >= ic0 and Aj[row_start] < ic1:
                        new_nnz = new_nnz + 1
                    for jj in range(row_start + 1, row_end):
                        if Aj[jj - 1] >= Aj[jj]:
                            atomic_and(has_canonical_format, False)
                            if Aj[jj - 1] > Aj[jj]:
                                atomic_and(has_sorted_indices, False)
                        if Aj[jj] >= ic0 and Aj[jj] < ic1:
                            new_nnz = new_nnz + 1
                Bp[i + 1] = new_nnz

                if i % 8192 == 8191:
                    with gil:
                        PyErr_CheckSignals()

        for i in range(1, new_n_row):
            Bp[i + 1] += Bp[i]  # cumsum
    return has_canonical_format, has_sorted_indices


def get_csr_submatrix1_sorted(const unsigned n_row,
                              const unsigned n_col,
                              const signed_integer[::1] Ap,
                              const signed_integer[::1] Aj,
                              const signed_integer ir0,
                              const signed_integer ir1,
                              const signed_integer ic0,
                              const signed_integer ic1,
                              signed_integer[::1] Bp,
                              unsigned num_threads):
    cdef unsigned i, thread_index, new_n_row = ir1 - ir0
    cdef signed_integer row_start, row_end, start_idx, end_idx
    cdef pair[unsigned, unsigned] row_range

    Bp[0] = 0
    num_threads = min(num_threads, new_n_row)
    if num_threads <= 1:
        for i in range(new_n_row):
            row_start = Ap[ir0 + i]
            row_end = Ap[ir0 + i + 1]
            if row_start < row_end:
                start_idx = lower_bound(
                    &Aj[0] + row_start, &Aj[0] + row_end, ic0) - &Aj[0]
                end_idx = lower_bound(
                    &Aj[0] + start_idx, &Aj[0] + row_end, ic1) - &Aj[0]
                Bp[i + 1] = Bp[i] + end_idx - start_idx
            else:
                Bp[i + 1] = Bp[i]
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(Ap[ir0:ir1 + 1], thread_index,
                                          num_threads)
            for i in range(row_range.first, row_range.second):
                row_start = Ap[ir0 + i]
                row_end = Ap[ir0 + i + 1]
                if row_start < row_end:
                    start_idx = lower_bound(
                        &Aj[0] + row_start, &Aj[0] + row_end,
                        ic0) - &Aj[0]
                    end_idx = lower_bound(
                        &Aj[0] + start_idx, &Aj[0] + row_end,
                        ic1) - &Aj[0]
                    Bp[i + 1] = end_idx - start_idx
                else:
                    Bp[i + 1] = 0

        for i in range(1, new_n_row):
            Bp[i + 1] += Bp[i]  # cumsum


def get_csr_submatrix2(const unsigned n_row,
                       const unsigned n_col,
                       const signed_integer[::1] Ap,
                       const signed_integer[::1] Aj,
                       const bit_width[::1] Ax,
                       const signed_integer ir0,
                       const signed_integer ir1,
                       const signed_integer ic0,
                       const signed_integer ic1,
                       signed_integer[::1] Bp,
                       signed_integer[::1] Bj,
                       bit_width[::1] Bx,
                       unsigned num_threads):
    cdef unsigned i, thread_index, new_n_row = ir1 - ir0
    cdef signed_integer jj, kk
    cdef pair[unsigned, unsigned] row_range

    num_threads = min(num_threads, new_n_row)
    if num_threads <= 1:
        kk = 0
        for i in range(new_n_row):
            for jj in range(Ap[ir0 + i], Ap[ir0 + i + 1]):
                if Aj[jj] >= ic0 and Aj[jj] < ic1:
                    Bj[kk] = Aj[jj] - ic0
                    Bx[kk] = Ax[jj]
                    kk = kk + 1

            # Check for KeyboardInterrupts
            if i % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(Ap[ir0:ir1 + 1], thread_index,
                                          num_threads)
            for i in range(row_range.first, row_range.second):
                kk = Bp[i]
                for jj in range(Ap[ir0 + i], Ap[ir0 + i + 1]):
                    if Aj[jj] >= ic0 and Aj[jj] < ic1:
                        Bj[kk] = Aj[jj] - ic0
                        Bx[kk] = Ax[jj]
                        kk = kk + 1

                # Check for KeyboardInterrupts
                if i % 8192 == 8191:
                    with gil:
                        PyErr_CheckSignals()


def get_csr_submatrix2_sorted(const unsigned n_row,
                              const unsigned n_col,
                              const signed_integer[::1] Ap,
                              const signed_integer[::1] Aj,
                              const bit_width[::1] Ax,
                              const signed_integer ir0,
                              const signed_integer ir1,
                              const signed_integer ic0,
                              const signed_integer ic1,
                              signed_integer[::1] Bp,
                              signed_integer[::1] Bj,
                              bit_width[::1] Bx,
                              unsigned num_threads):
    cdef unsigned i, thread_index, new_n_row = ir1 - ir0
    cdef signed_integer jj, kk, start_idx, end_idx, row_start, row_end
    cdef pair[unsigned, unsigned] row_range

    num_threads = min(num_threads, new_n_row)
    if num_threads <= 1:
        kk = 0
        for i in range(new_n_row):
            row_start = Ap[ir0 + i]
            row_end = Ap[ir0 + i + 1]
            if row_start < row_end:
                start_idx = lower_bound(
                    &Aj[0] + row_start, &Aj[0] + row_end, ic0) - &Aj[0]
                end_idx = lower_bound(
                    &Aj[0] + start_idx, &Aj[0] + row_end, ic1) - &Aj[0]
                for jj in range(start_idx, end_idx):
                    Bj[kk] = Aj[jj] - ic0
                    Bx[kk] = Ax[jj]
                    kk += 1
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(Ap[ir0:ir1 + 1], thread_index,
                                         num_threads)
            for i in range(row_range.first, row_range.second):
                kk = Bp[i]
                row_start = Ap[ir0 + i]
                row_end = Ap[ir0 + i + 1]
                if row_start < row_end:
                    start_idx = lower_bound(
                        &Aj[0] + row_start, &Aj[0] + row_end, ic0) - &Aj[0]
                    end_idx = lower_bound(
                        &Aj[0] + start_idx, &Aj[0] + row_end, ic1) - &Aj[0]
                    for jj in range(start_idx, end_idx):
                        Bj[kk] = Aj[jj] - ic0
                        Bx[kk] = Ax[jj]
                        kk = kk + 1


def csr_row_index(const unsigned n_row_idx,
                  const integer[::1] rows,
                  const signed_integer[::1] Ap,
                  const signed_integer[::1] Aj,
                  const bit_width[::1] Ax,
                  signed_integer[::1] Bp,
                  signed_integer[::1] Bj,
                  bit_width[::1] Bx,
                  unsigned num_threads):

    cdef unsigned i
    cdef signed_integer row, row_start, row_end, dest_row_start, dest_row_end

    num_threads = min(num_threads, n_row_idx)
    if num_threads <= 1:
        dest_row_start = 0
        for i in range(n_row_idx):
            row = rows[i]
            row_start = Ap[row]
            row_end = Ap[row + 1]
            dest_row_end = dest_row_start + row_end - row_start
            # Bj[dest_row_start:dest_row_end] = Aj[row_start:row_end]
            # Bx[dest_row_start:dest_row_end] = Ax[row_start:row_end]
            memcpy(&Bj[dest_row_start], &Aj[row_start],
                   (row_end - row_start) * sizeof(signed_integer))
            memcpy(&Bx[dest_row_start], &Ax[row_start],
                   (row_end - row_start) * sizeof(bit_width))
            dest_row_start = dest_row_end

            # Check for KeyboardInterrupts
            if i % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        for i in prange(n_row_idx, nogil=True, num_threads=num_threads):
            row = rows[i]
            row_start = Ap[row]
            row_end = Ap[row + 1]
            dest_row_start = Bp[i]
            dest_row_end = Bp[i + 1]
            # Bj[dest_row_start:dest_row_end] = Aj[row_start:row_end]
            # Bx[dest_row_start:dest_row_end] = Ax[row_start:row_end]
            memcpy(&Bj[dest_row_start], &Aj[row_start],
                   (row_end - row_start) * sizeof(signed_integer))
            memcpy(&Bx[dest_row_start], &Ax[row_start],
                   (row_end - row_start) * sizeof(bit_width))

            # Check for KeyboardInterrupts
            if i % 8192 == 8191:
                with gil:
                    PyErr_CheckSignals()


def csr_row_index_check(const unsigned n_row_idx,
                        const integer[::1] rows,
                        const signed_integer[::1] Ap,
                        const signed_integer[::1] Aj,
                        const bit_width[::1] Ax,
                        signed_integer[::1] Bp,
                        signed_integer[::1] Bj,
                        bit_width[::1] Bx,
                        unsigned num_threads):
    cdef unsigned i
    cdef signed_integer row, row_start, row_end, dest_row_start, \
        dest_row_end, jj
    cdef bint has_canonical_format = True, has_sorted_indices = True

    num_threads = min(num_threads, n_row_idx)
    if num_threads <= 1:
        dest_row_start = 0
        for i in range(n_row_idx):
            row = rows[i]
            row_start = Ap[row]
            row_end = Ap[row + 1]
            for jj in range(row_start + 1, row_end):
                if Aj[jj - 1] >= Aj[jj]:
                    has_canonical_format = False
                    if Aj[jj - 1] > Aj[jj]:
                        has_sorted_indices = False
            dest_row_end = dest_row_start + row_end - row_start
            # Bj[dest_row_start:dest_row_end] = Aj[row_start:row_end]
            # Bx[dest_row_start:dest_row_end] = Ax[row_start:row_end]
            memcpy(&Bj[dest_row_start], &Aj[row_start],
                   (row_end - row_start) * sizeof(signed_integer))
            memcpy(&Bx[dest_row_start], &Ax[row_start],
                   (row_end - row_start) * sizeof(bit_width))
            dest_row_start = dest_row_end

            if i % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        for i in prange(n_row_idx, nogil=True, num_threads=num_threads):
            row = rows[i]
            row_start = Ap[row]
            row_end = Ap[row + 1]
            for jj in range(row_start + 1, row_end):
                if Aj[jj - 1] >= Aj[jj]:
                    atomic_and(has_canonical_format, False)
                    if Aj[jj - 1] > Aj[jj]:
                        atomic_and(has_sorted_indices, False)
            dest_row_start = Bp[i]
            dest_row_end = Bp[i + 1]
            # Bj[dest_row_start:dest_row_end] = Aj[row_start:row_end]
            # Bx[dest_row_start:dest_row_end] = Ax[row_start:row_end]
            memcpy(&Bj[dest_row_start], &Aj[row_start],
                   (row_end - row_start) * sizeof(signed_integer))
            memcpy(&Bx[dest_row_start], &Ax[row_start],
                   (row_end - row_start) * sizeof(bit_width))

            if i % 8192 == 8191:
                with gil:
                    PyErr_CheckSignals()

    return has_canonical_format, has_sorted_indices


def csr_row_slice(const int start,
                  const int stop,
                  const int step,
                  const signed_integer[::1] Ap,
                  const signed_integer[::1] Aj,
                  const bit_width[::1] Ax,
                  signed_integer[::1] Bp,
                  signed_integer[::1] Bj,
                  bit_width[::1] Bx,
                  unsigned num_threads):

    cdef unsigned num_iterations, i, row
    cdef signed_integer row_start, row_end, dest_row_start, dest_row_end

    if step > 0 and start < stop:
        num_iterations = (stop - start + step - 1) // step
    elif step < 0 and start > stop:
        num_iterations = (start - stop - step - 1) // -step
    else:
        num_iterations = 0

    num_threads = min(num_threads, num_iterations)
    if num_threads <= 1:
        dest_row_start = 0
        for i in range(start, stop, step):
            row_start = Ap[i]
            row_end = Ap[i + 1]
            dest_row_end = dest_row_start + row_end - row_start
            # Bj[dest_row_start:dest_row_end] = Aj[row_start:row_end]
            # Bx[dest_row_start:dest_row_end] = Ax[row_start:row_end]
            memcpy(&Bj[dest_row_start], &Aj[row_start],
                   (row_end - row_start) * sizeof(signed_integer))
            memcpy(&Bx[dest_row_start], &Ax[row_start],
                   (row_end - row_start) * sizeof(bit_width))
            dest_row_start = dest_row_end

            # Check for KeyboardInterrupts
            if i % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        for i in prange(num_iterations, nogil=True,
                        num_threads=num_threads):
            row = start + i * step
            row_start = Ap[row]
            row_end = Ap[row + 1]
            dest_row_start = Bp[i]
            dest_row_end = Bp[i + 1]
            # Bj[dest_row_start:dest_row_end] = Aj[row_start:row_end]
            # Bx[dest_row_start:dest_row_end] = Ax[row_start:row_end]
            memcpy(&Bj[dest_row_start], &Aj[row_start],
                   (row_end - row_start) * sizeof(signed_integer))
            memcpy(&Bx[dest_row_start], &Ax[row_start],
                   (row_end - row_start) * sizeof(bit_width))

            # Check for KeyboardInterrupts
            if i % 8192 == 8191:
                with gil:
                    PyErr_CheckSignals()


def csr_row_slice_check(const int start,
                        const int stop,
                        const int step,
                        const signed_integer[::1] Ap,
                        const signed_integer[::1] Aj,
                        const bit_width[::1] Ax,
                        signed_integer[::1] Bp,
                        signed_integer[::1] Bj,
                        bit_width[::1] Bx,
                        unsigned num_threads):

    cdef unsigned num_iterations, i, row
    cdef signed_integer row_start, row_end, dest_row_start, dest_row_end, jj
    cdef bint has_canonical_format = True, has_sorted_indices = True

    if step > 0 and start < stop:
        num_iterations = (stop - start + step - 1) // step
    elif step < 0 and start > stop:
        num_iterations = (start - stop - step - 1) // -step
    else:
        num_iterations = 0

    num_threads = min(num_threads, num_iterations)
    if num_threads <= 1:
        dest_row_start = 0
        for i in range(num_iterations):
            row = start + i * step
            row_start = Ap[row]
            row_end = Ap[row + 1]
            for jj in range(row_start + 1, row_end):
                if Aj[jj - 1] >= Aj[jj]:
                    has_canonical_format = False
                    if Aj[jj - 1] > Aj[jj]:
                        has_sorted_indices = False
            dest_row_end = dest_row_start + row_end - row_start
            # Bj[dest_row_start:dest_row_end] = Aj[row_start:row_end]
            # Bx[dest_row_start:dest_row_end] = Ax[row_start:row_end]
            memcpy(&Bj[dest_row_start], &Aj[row_start],
                   (row_end - row_start) * sizeof(signed_integer))
            memcpy(&Bx[dest_row_start], &Ax[row_start],
                   (row_end - row_start) * sizeof(bit_width))
            dest_row_start = dest_row_end

            if i % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        for i in prange(num_iterations, nogil=True, num_threads=num_threads):
            row = start + i * step
            row_start = Ap[row]
            row_end = Ap[row + 1]
            for jj in range(row_start + 1, row_end):
                if Aj[jj - 1] >= Aj[jj]:
                    atomic_and(has_canonical_format, False)
                    if Aj[jj - 1] > Aj[jj]:
                        atomic_and(has_sorted_indices, False)
            dest_row_start = Bp[i]
            dest_row_end = Bp[i + 1]
            # Bj[dest_row_start:dest_row_end] = Aj[row_start:row_end]
            # Bx[dest_row_start:dest_row_end] = Ax[row_start:row_end]
            memcpy(&Bj[dest_row_start], &Aj[row_start],
                   (row_end - row_start) * sizeof(signed_integer))
            memcpy(&Bx[dest_row_start], &Ax[row_start],
                   (row_end - row_start) * sizeof(bit_width))

            if i % 8192 == 8191:
                with gil:
                    PyErr_CheckSignals()

    return has_canonical_format, has_sorted_indices


def csr_column_index1(const unsigned n_idx,
                      const integer[::1] col_idxs,
                      const unsigned n_row,
                      const unsigned n_col,
                      const signed_integer[::1] Ap,
                      const signed_integer[::1] Aj,
                      signed_integer[::1] col_offsets,
                      signed_integer[::1] Bp,
                      unsigned num_threads):
    cdef unsigned i, j, thread_index
    cdef signed_integer new_nnz, jj
    cdef pair[unsigned, unsigned] row_range

    # bincount(col_idxs)
    num_threads = min(num_threads, n_row)
    if num_threads <= 1:
        col_offsets[:] = 0
    else:
        for j in prange(n_col, nogil=True, num_threads=num_threads):
            col_offsets[j] = 0  # NUMA-aware
    for i in range(n_idx):
        j = col_idxs[i]
        col_offsets[j] += 1

    # Compute new indptr
    Bp[0] = 0
    if num_threads <= 1:
        new_nnz = 0
        for i in range(n_row):
            for jj in range(Ap[i], Ap[i + 1]):
                new_nnz += col_offsets[Aj[jj]]
            Bp[i + 1] = new_nnz
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(Ap, thread_index, num_threads)
            for i in range(row_range.first, row_range.second):
                new_nnz = 0
                for jj in range(Ap[i], Ap[i + 1]):
                    new_nnz = new_nnz + col_offsets[Aj[jj]]
                Bp[i + 1] = new_nnz
        for i in range(1, n_row):
            Bp[i + 1] += Bp[i]

    # cumsum in-place
    for j in range(1, n_col):
        col_offsets[j] += col_offsets[j - 1]


def csr_column_index2(const long long[::1] col_order,
                      const signed_integer[::1] col_offsets,
                      const signed_integer nnz,
                      const signed_integer[::1] Ap,
                      const signed_integer[::1] Aj,
                      const bit_width[::1] Ax,
                      signed_integer[::1] Bp,
                      signed_integer[::1] Bj,
                      bit_width[::1] Bx,
                      unsigned num_threads):

    cdef unsigned row, thread_index
    cdef signed_integer n, jj, j, offset, prev_offset, k
    cdef bit_width v
    cdef pair[unsigned, unsigned] row_range

    num_threads = min(num_threads, Ap.shape[0] - 1)
    if num_threads <= 1:
        n = 0
        for jj in range(nnz):
            j = Aj[jj]
            offset = col_offsets[j]
            prev_offset = 0 if j == 0 else col_offsets[j - 1]
            if offset != prev_offset:
                v = Ax[jj]
                for k in range(prev_offset, offset):
                    Bj[n] = col_order[k]
                    Bx[n] = v
                    n += 1

            # Check for KeyboardInterrupts
            if jj % 1048576 == 1048575:
                PyErr_CheckSignals()

    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(Ap, thread_index, num_threads)
            for row in range(row_range.first, row_range.second):
                n = Bp[row]
                for jj in range(Ap[row], Ap[row + 1]):
                    j = Aj[jj]
                    offset = col_offsets[j]
                    prev_offset = 0 if j == 0 else col_offsets[j - 1]
                    if offset != prev_offset:
                        v = Ax[jj]
                        for k in range(prev_offset, offset):
                            Bj[n] = col_order[k]
                            Bx[n] = v
                            n = n + 1

                # Check for KeyboardInterrupts
                if row % 8192 == 8191:
                    with gil:
                        PyErr_CheckSignals()


def csr_outer_index1(const integer[::1] row_indices,
                     const integer2[::1] col_indices,
                     const unsigned num_columns,
                     const signed_integer[::1] Ap,
                     const signed_integer[::1] Aj,
                     signed_integer[::1] col_offsets,
                     signed_integer[::1] Bp,
                     unsigned num_threads):
    cdef unsigned i, j, row, thread_index, \
        num_row_indices = row_indices.shape[0]
    cdef signed_integer new_nnz, jj
    cdef pair[unsigned, unsigned] row_range

    # bincount(col_indices)
    num_threads = min(num_threads, num_row_indices)
    if num_threads <= 1:
        col_offsets[:] = 0
    else:
        for j in prange(num_columns, nogil=True, num_threads=num_threads):
            col_offsets[j] = 0  # NUMA-aware
    for i in range(col_indices.shape[0]):
        j = col_indices[i]
        col_offsets[j] += 1

    # Compute new indptr
    Bp[0] = 0
    if num_threads <= 1:
        new_nnz = 0
        for i in range(num_row_indices):
            row = row_indices[i]
            for jj in range(Ap[row], Ap[row + 1]):
                new_nnz += col_offsets[Aj[jj]]
            Bp[i + 1] = new_nnz
    else:
        for i in prange(num_row_indices, nogil=True, num_threads=num_threads):
            new_nnz = 0
            row = row_indices[i]
            for jj in range(Ap[row], Ap[row + 1]):
                new_nnz = new_nnz + col_offsets[Aj[jj]]
            Bp[i + 1] = new_nnz
        for i in range(1, num_row_indices):
            Bp[i + 1] += Bp[i]

    # cumsum in-place
    for j in range(1, num_columns):
        col_offsets[j] += col_offsets[j - 1]


def csr_outer_index2(const integer[::1] row_indices,
                     const long long[::1] col_order,
                     const signed_integer[::1] col_offsets,
                     const signed_integer nnz,
                     const signed_integer[::1] Ap,
                     const signed_integer[::1] Aj,
                     const bit_width[::1] Ax,
                     signed_integer[::1] Bp,
                     signed_integer[::1] Bj,
                     bit_width[::1] Bx,
                     unsigned num_threads):

    cdef unsigned i, row, num_row_indices = row_indices.shape[0]
    cdef signed_integer n, jj, j, offset, prev_offset, k
    cdef bit_width v
    cdef pair[unsigned, unsigned] row_range

    num_threads = min(num_threads, Ap.shape[0] - 1)
    if num_threads <= 1:
        for i in range(num_row_indices):
            row = row_indices[i]
            n = Bp[i]
            for jj in range(Ap[row], Ap[row + 1]):
                j = Aj[jj]
                offset = col_offsets[j]
                prev_offset = 0 if j == 0 else col_offsets[j - 1]
                if offset != prev_offset:
                    v = Ax[jj]
                    for k in range(prev_offset, offset):
                        Bj[n] = col_order[k]
                        Bx[n] = v
                        n = n + 1

            # Check for KeyboardInterrupts
            if i % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        for i in prange(num_row_indices, nogil=True, num_threads=num_threads):
            row = row_indices[i]
            n = Bp[i]
            for jj in range(Ap[row], Ap[row + 1]):
                j = Aj[jj]
                offset = col_offsets[j]
                prev_offset = 0 if j == 0 else col_offsets[j - 1]
                if offset != prev_offset:
                    v = Ax[jj]
                    for k in range(prev_offset, offset):
                        Bj[n] = col_order[k]
                        Bx[n] = v
                        n = n + 1

            # Check for KeyboardInterrupts
            if i % 8192 == 8191:
                with gil:
                    PyErr_CheckSignals()


def csr_sample_values(const unsigned n_row,
                      const signed_integer[::1] Ap,
                      const signed_integer[::1] Aj,
                      const numeric[::1] Ax,
                      const signed_integer n_samples,
                      const integer[::1] Bi,
                      const integer2[::1] Bj,
                      numeric[::1] Bx,
                      unsigned num_threads):
    cdef signed_integer n, i, j, row_start, row_end, offset, jj
    cdef numeric x

    num_threads = min(num_threads, <unsigned> n_samples)
    if num_threads <= 1:
        for n in range(n_samples):
            i = Bi[n]  # sample row
            j = Bj[n]  # sample column

            row_start = Ap[i]
            row_end = Ap[i + 1]

            x = 0
            for jj in range(row_start, row_end):
                if Aj[jj] == j:
                    x += Ax[jj]
            Bx[n] = x

            # Check for KeyboardInterrupts
            if n % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        for n in prange(n_samples, nogil=True,
                        num_threads=num_threads):
            i = Bi[n]  # sample row
            j = Bj[n]  # sample column

            row_start = Ap[i]
            row_end = Ap[i + 1]

            x = 0
            for jj in range(row_start, row_end):
                if Aj[jj] == j:
                    x = x + Ax[jj]
            Bx[n] = x

            # Check for KeyboardInterrupts
            if n % 8192 == 8191:
                with gil:
                    PyErr_CheckSignals()


def csr_sample_values_canonical(const unsigned n_row,
                                const signed_integer[::1] Ap,
                                const signed_integer[::1] Aj,
                                const bit_width[::1] Ax,
                                const signed_integer n_samples,
                                const integer[::1] Bi,
                                const integer2[::1] Bj,
                                bit_width[::1] Bx,
                                unsigned num_threads):
    cdef signed_integer n, i, j, row_start, row_end, offset

    num_threads = min(num_threads, <unsigned> n_samples)
    if num_threads <= 1:
        for n in range(n_samples):
            i = Bi[n]
            j = Bj[n]
            row_start = Ap[i]
            row_end = Ap[i + 1]
            if row_start < row_end:
                offset = lower_bound(
                    &Aj[0] + row_start, &Aj[0] + row_end, j) - &Aj[0]
                if offset < row_end and Aj[offset] == j:
                    Bx[n] = Ax[offset]
                else:
                    Bx[n] = 0
            else:
                Bx[n] = 0
    else:
        for n in prange(n_samples, nogil=True, num_threads=num_threads):
            i = Bi[n]
            j = Bj[n]
            row_start = Ap[i]
            row_end = Ap[i + 1]
            if row_start < row_end:
                offset = lower_bound(
                    &Aj[0] + row_start, &Aj[0] + row_end, j) - &Aj[0]
                if offset < row_end and Aj[offset] == j:
                    Bx[n] = Ax[offset]
                else:
                    Bx[n] = 0
            else:
                Bx[n] = 0


def csr_sample_values_sorted(const unsigned n_row,
                             const signed_integer[::1] Ap,
                             const signed_integer[::1] Aj,
                             const numeric[::1] Ax,
                             const signed_integer n_samples,
                             const integer[::1] Bi,
                             const integer2[::1] Bj,
                             numeric[::1] Bx,
                             unsigned num_threads):
    cdef signed_integer n, i, j, row_start, row_end, offset
    cdef numeric x

    num_threads = min(num_threads, <unsigned> n_samples)
    if num_threads <= 1:
        for n in range(n_samples):
            i = Bi[n]
            j = Bj[n]
            row_start = Ap[i]
            row_end = Ap[i + 1]
            x = 0
            if row_start < row_end:
                offset = lower_bound(
                    &Aj[0] + row_start, &Aj[0] + row_end, j) - &Aj[0]
                while offset < row_end and Aj[offset] == j:
                    x = x + Ax[offset]
                    offset = offset + 1
            Bx[n] = x

            # Check for KeyboardInterrupts
            if n % 8192 == 8191:
                PyErr_CheckSignals()
    else:
        for n in prange(n_samples, nogil=True,
                        num_threads=num_threads):
            i = Bi[n]
            j = Bj[n]
            row_start = Ap[i]
            row_end = Ap[i + 1]
            x = 0
            if row_start < row_end:
                offset = lower_bound(
                    &Aj[0] + row_start, &Aj[0] + row_end, j) - &Aj[0]
                while offset < row_end and Aj[offset] == j:
                    x = x + Ax[offset]
                    offset = offset + 1
            Bx[n] = x

            # Check for KeyboardInterrupts
            if n % 8192 == 8191:
                with gil:
                    PyErr_CheckSignals()


def csr_tocsc(const unsigned n_row,
              const unsigned n_col,
              const signed_integer[::1] Ap,
              const signed_integer[::1] Aj,
              const bit_width[::1] Ax,
              signed_integer[::1] Bp,
              signed_integer[::1] Bi,
              bit_width[::1] Bx,
              unsigned num_threads):
    cdef unsigned col, row, thread_index, preceding_thread_index
    cdef signed_integer n, cumsum, count, jj, dest, preceding_count, i, \
        Bp_col, nnz = Ap[n_row]
    cdef pair[unsigned, unsigned] row_range
    cdef vector[vector[unsigned]] thread_col_counts
    cdef vector[vector[signed_integer]] thread_col_offsets
    cdef unsigned* col_counts
    cdef signed_integer* col_offsets

    num_threads = min(num_threads, n_row)
    if num_threads <= 1:
        # Count the number of non-zero entries per column of `A`
        Bp[:n_col] = 0
        for n in range(nnz):
            Bp[Aj[n]] += 1

        # Cumsum the nnz per column, shifting right by 1, to get `Bp`
        cumsum = 0
        for col in range(n_col):
            count = Bp[col]
            Bp[col] = cumsum
            cumsum += count
        Bp[n_col] = nnz

        # Fill in `Bi` and `Bx`, using `Bp[col]` to keep track of the current
        # insertion index for each column
        for row in range(n_row):
            for jj in range(Ap[row], Ap[row + 1]):
                col = Aj[jj]
                dest = Bp[col]
                Bi[dest] = row
                Bx[dest] = Ax[jj]
                Bp[col] += 1

            # Check for KeyboardInterrupts
            if row % 8192 == 8191:
                PyErr_CheckSignals()

        # After the previous step, `Bp[col]` now points to the start of
        # `col + 1`'s column. Reset it to point to the start of `col`'s column.
        col = n_col
        while col > 0:
            Bp[col] = Bp[col - 1]
            col -= 1
        Bp[0] = 0
    else:
        thread_col_counts.resize(num_threads)
        thread_col_offsets.resize(num_threads)

        with nogil:
            # Count the number of non-zero entries per column of `A`
            # per thread
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                thread_col_counts[thread_index].resize(n_col)
                thread_col_offsets[thread_index].resize(n_col)
                row_range = get_thread_offset(Ap, thread_index, num_threads)
                col_counts = thread_col_counts[thread_index].data()
                for row in range(row_range.first, row_range.second):
                    for jj in range(Ap[row], Ap[row + 1]):
                        col = Aj[jj]
                        col_counts[col] += 1

            # Sum these numbers of non-zero entries across threads
            for col in prange(n_col, num_threads=num_threads):
                Bp_col = 0
                for thread_index in range(num_threads):
                    Bp_col = Bp_col + thread_col_counts[thread_index][col]
                Bp[col] = Bp_col

            # Cumsum the nnz per column, shifting right by 1, to get `Bp`
            cumsum = 0
            for col in range(n_col):
                count = Bp[col]
                Bp[col] = cumsum
                cumsum += count
            Bp[n_col] = nnz

            # Fill in `Bi` and `Bx`, using `thread_col_offsets[col]` to keep
            # track of the current insertion index for each column. Initialize
            # each thread's `thread_col_offsets[col]` by summing
            # `thread_col_counts` for that column across all threads preceding
            # `thread_index`, and adding `Bp[col]`, the start index of the
            # column.
            with parallel(num_threads=num_threads):
                thread_index = threadid()
                col_offsets = thread_col_offsets[thread_index].data()
                for col in range(n_col):
                    preceding_count = Bp[col]
                    for preceding_thread_index in range(thread_index):
                        preceding_count = preceding_count + \
                            thread_col_counts[preceding_thread_index][col]
                    col_offsets[col] = preceding_count

                row_range = get_thread_offset(Ap, thread_index, num_threads)
                for row in range(row_range.first, row_range.second):
                    for jj in range(Ap[row], Ap[row + 1]):
                        col = Aj[jj]
                        dest = col_offsets[col]
                        Bi[dest] = row
                        Bx[dest] = Ax[jj]
                        col_offsets[col] += 1

                    # Check for KeyboardInterrupts
                    if row % 8192 == 8191:
                        with gil:
                            PyErr_CheckSignals()


def csr_eliminate_zeros(const unsigned n_row,
                        const unsigned n_col,
                        signed_integer[::1] Ap,
                        signed_integer[::1] Aj,
                        numeric[::1] Ax):
    cdef unsigned i
    cdef signed_integer row_end = 0, nnz = 0, jj, j
    cdef numeric x

    for i in range(n_row):
        jj = row_end
        row_end = Ap[i+1]
        while jj < row_end:
            j = Aj[jj]
            x = Ax[jj]
            if x:
                Aj[nnz] = j
                Ax[nnz] = x
                nnz += 1
            jj += 1
        Ap[i + 1] = nnz


def wrap_around(signed_integer[::1] x,
                const signed_integer length,
                unsigned num_threads):
    # Second pass for signed integer arrays that do not own their own data, to
    # wrap around negative indices

    cdef unsigned i, array_length = x.shape[0]

    num_threads = min(num_threads, array_length)
    if num_threads <= 1:
        for i in range(array_length):
            if x[i] < 0:
                x[i] += length
    else:
        for i in prange(array_length, nogil=True, num_threads=num_threads):
            if x[i] < 0:
                x[i] += length