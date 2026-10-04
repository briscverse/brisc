# Locally weighted regression (LOESS), as used by `SingleCell.hvg()`.
#
# This is a line-by-line Cython port of scikit-misc's `skmisc.loess` module
# (https://github.com/has2k1/scikit-misc, version 0.5.2), which in turn wraps
# the netlib `dloess` C and Fortran library by Cleveland, Grosse and Shyu
# (http://www.netlib.org/a/dloess). It exposes the same public API as
# `skmisc.loess` (`loess`, `loess_inputs`, `loess_model`, `loess_control`,
# `loess_outputs`, `loess_prediction`, `loess_confidence_intervals` and
# `loess_anova`). The linear algebra uses scipy's cython_blas and
# cython_lapack in place of the reference BLAS and the LINPACK routines that
# `dloess` uses: the local least-squares fits use a Householder QR
# decomposition (LAPACK's dgeqrf and dormqr instead of LINPACK's dqrdc and
# dqrsl) followed by a singular value decomposition of the triangular factor
# (dgesvd instead of dsvdc). The results therefore match scikit-misc to
# within floating-point error rather than bit for bit, and depend slightly on
# which BLAS and LAPACK scipy is linked to. (Before this substitution, results
# were bit-for-bit identical to scikit-misc on x86-64 Linux when compiling
# without fast-math semantics or fused multiply-adds.)
#
# Behavior that intentionally differs from scikit-misc, all in cases where
# scikit-misc reads or writes out of bounds:
# - fitting with a `span` so small that neighborhoods contain no points (or
#   needing a workspace of over INT_MAX elements) raises ValueError, where
#   scikit-misc dereferences a stale or null pointer and usually segfaults
# - fitting more than 8 predictors, or local models with more than 15 terms
#   (5+ predictors with `degree=2`), raises ValueError, where scikit-misc
#   overflows fixed-size arrays and usually segfaults
# - k-d trees deeper than 20 levels, or with more than 256 cells touching a
#   vertex, still raise ValueError, but no longer overflow fixed-size arrays
# - predicting with `surface='interpolate'` from a model whose k-d tree was
#   never built raises ValueError, where scikit-misc reads uninitialized
#   memory
# - `loess_outputs.divisor` has length p, not n (scikit-misc reads past the
#   end of the length-p array), and `loess_model.parametric` and
#   `loess_model.drop_square` return the values that were set (scikit-misc
#   reinterprets their int storage as doubles, so e.g. `[False, True]` reads
#   back as `[True, False]`)
# - confidence intervals are NaN in the cases where scikit-misc loops forever
#   computing the t quantile, which happens for some negative or infinite
#   degrees of freedom (e.g. for small fits with `surface='direct'`)
# - warning messages are always intact; scikit-misc builds them in a stack
#   buffer that is no longer live when Python reads it, so they are
#   occasionally garbage
# - outputs that scikit-misc leaves uninitialized are zero, e.g.
#   `outputs.diagonal` when the trace of the hat matrix is approximated, or
#   `outputs.fitted_values` when the model specification is invalid
#
# Behavior that is intentionally preserved, even though it looks like a bug:
# - for p > 1 predictors, `x.ravel()` (C order) is treated as column-major, so
#   an (n, p) array is interpreted as if its values were stored column by
#   column; likewise for `newdata` when predicting
# - every warning from the Fortran code (e.g. "pseudoinverse used at ...")
#   makes `fit()` and `predict()` raise ValueError, with the last warning or
#   error (as bytes, without a trailing newline) as the message, after the
#   computation has run to completion
# - `predict()` multiplies `outputs.robust` by the prior weights in place each
#   time it is called, which affects later predictions with `surface='direct'`
# - `loess_confidence_intervals` rounds `alpha` to single precision
#
# Original copyright notices:
#
# The authors of dloess are Cleveland, Grosse, and Shyu.
# Copyright (c) 1989, 1992 by AT&T.
# Permission to use, copy, modify, and distribute this software for any
# purpose without fee is hereby granted, provided that this entire notice
# is included in all copies of any software which is or includes a copy
# or modification of this software and in all copies of the supporting
# documentation for such software.
# THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
# WARRANTY. IN PARTICULAR, NEITHER THE AUTHORS NOR AT&T MAKE ANY
# REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
# OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
#
# scikit-misc: Copyright (c) 2016, Hassan Kibirige. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of  nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Conventions: Fortran arrays are column-major and one-based. Each Fortran
# routine keeps its original name and one-based loop variables, and an
# element `a(i, j)` of an array with leading dimension `lda` is accessed as
# `a[(i - 1) + (j - 1) * lda]`. Scalar arguments that the Fortran routine
# modifies are passed as pointers. Fortran's `x**2` and `x**3` are written as
# `x * x` and `x * x * x`, which is how gfortran evaluates them.

cimport cython
from libc.float cimport DBL_EPSILON, DBL_MAX, DBL_MIN
from libc.limits cimport INT_MAX
from libc.math cimport NAN, ceil, copysign, exp, fabs, floor, lgamma, log, \
    pow, sqrt
from libc.stdio cimport snprintf
from libc.stdlib cimport calloc, free, malloc, qsort
from scipy.linalg cimport cython_blas, cython_lapack

import numpy as np

###############################################################################
# Error handling (`ehg182`, `ehg183a` and `ehg184a` from loessc.c)
###############################################################################

# Like the original, errors and warnings are recorded in module-level state
# but do not stop the computation; `fit()` and `predict()` check the state
# afterwards and raise ValueError with the most recent message. `fatal` marks
# the cases where the original would go on to read or write out of bounds;
# these stop the computation immediately.

cdef int error_status = 0
cdef bint fatal = False
cdef const char* error_message = NULL
cdef char error_buffer[4000]


cdef inline void reset_error() noexcept nogil:
    global error_status, error_message, fatal
    error_status = 0
    error_message = NULL
    fatal = False


cdef inline void set_error(const char* message) noexcept nogil:
    global error_status, error_message
    error_status = 1
    error_message = message


cdef inline void set_fatal_error(const char* message) noexcept nogil:
    global fatal
    set_error(message)
    fatal = True


cdef void ehg182(int i) noexcept nogil:
    cdef const char* mess
    if i == 100:
        mess = "Wrong version number in lowesd.  Probably typo in caller."
    elif i == 101:
        mess = "d>dMAX in ehg131.  Need to recompile with increased " \
               "dimensions."
    elif i == 102:
        mess = "liv too small. (Discovered by lowesd)"
    elif i == 103:
        mess = "lv too small. (Discovered by lowesd)"
    elif i == 104:
        mess = "Span too small. Fewer data values than degrees of freedom."
    elif i == 105:
        mess = "k>d2MAX in ehg136.  Need to recompile with increased " \
               "dimensions."
    elif i == 106:
        mess = "lwork too small"
    elif i == 107:
        mess = "Invalid value for kernel"
    elif i == 108:
        mess = "Invalid value for ideg"
    elif i == 109:
        mess = "lowstt only applies when kernel=1."
    elif i == 110:
        mess = "Not enough extra workspace for robustness calculation"
    elif i == 120:
        mess = "Zero-width neighborhood. Make span bigger"
    elif i == 121:
        mess = "All data on boundary of neighborhood. make span bigger"
    elif i == 122:
        mess = "Extrapolation not allowed with blending"
    elif i == 123:
        mess = "ihat=1 (diag L) in l2fit only makes sense if z=x (eval=data)."
    elif i == 171:
        mess = "lowesd must be called first."
    elif i == 172:
        mess = "lowesf must not come between lowesb and lowese, lowesr, or " \
               "lowesl."
    elif i == 173:
        mess = "lowesb must come before lowese, lowesr, or lowesl."
    elif i == 174:
        mess = "lowesb need not be called twice."
    elif i == 175:
        mess = "Need setLf=.true. for lowesl."
    elif i == 180:
        mess = "nv>nvmax in cpvert."
    elif i == 181:
        mess = "nt>20 in eval."
    elif i == 182:
        mess = "svddc failed in l2fit."
    elif i == 183:
        mess = "Did not find edge in vleaf."
    elif i == 184:
        mess = "Zero-width cell found in vleaf."
    elif i == 185:
        mess = "Trouble descending to leaf in vleaf."
    elif i == 186:
        mess = "Insufficient workspace for lowesf."
    elif i == 187:
        mess = "Insufficient stack space."
    elif i == 188:
        mess = "lv too small for computing explicit L."
    elif i == 191:
        mess = "Computed trace L was negative; something is wrong!"
    elif i == 192:
        mess = "Computed delta was negative; something is wrong!"
    elif i == 193:
        mess = "Workspace in loread appears to be corrupted."
    elif i == 194:
        mess = "Trouble in l2fit/l2tr"
    elif i == 195:
        mess = "Only constant, linear, or quadratic local models allowed"
    elif i == 196:
        mess = "degree must be at least 1 for vertex influence matrix"
    elif i == 999:
        mess = "not yet implemented"
    else:
        snprintf(error_buffer, sizeof(error_buffer),
                 "Assert failed; error code %d\n", i)
        mess = error_buffer
    set_error(mess)


# `ehg183a` and `ehg184a` build their message in a stack buffer, append "\n"
# and store a pointer to the buffer. The compiler drops the final append
# (since the buffer's lifetime ends), so scikit-misc's messages have no
# trailing newline; these ports match that.

cdef void ehg183(const char* s, const int* i, int n, int inc) noexcept nogil:
    cdef int j, length
    length = snprintf(error_buffer, sizeof(error_buffer), "%s", s)
    for j in range(n):
        length += snprintf(error_buffer + length,
                           sizeof(error_buffer) - <size_t> length, " %d",
                           i[j * inc])
    set_error(error_buffer)


cdef void ehg184(const char* s, const double* x, int n,
                 int inc) noexcept nogil:
    cdef int j, length
    length = snprintf(error_buffer, sizeof(error_buffer), "%s", s)
    for j in range(n):
        length += snprintf(error_buffer + length,
                           sizeof(error_buffer) - <size_t> length, " %.5g",
                           x[j * inc])
    set_error(error_buffer)

###############################################################################
# Fortran intrinsics
###############################################################################

# gfortran evaluates `min(a, b)` as `b < a ? b : a` and `max(a, b)` as
# `b > a ? b : a`; these only differ from other definitions when an argument
# is NaN

cdef inline double f77_min(double a, double b) noexcept nogil:
    return b if b < a else a


cdef inline double f77_max(double a, double b) noexcept nogil:
    return b if b > a else a


cdef inline int ifloor(double x) noexcept nogil:
    cdef int result = <int> x
    if result > x:
        result -= 1
    return result

###############################################################################
# BLAS and LAPACK (scipy.linalg.cython_blas and cython_lapack)
###############################################################################

# scipy's BLAS and LAPACK take every argument by pointer, Fortran-style; these
# wrappers take scalars by value (and read-only arrays as const pointers)

cdef inline double ddot(int n, const double* dx, int incx, const double* dy,
                        int incy) noexcept nogil:
    return cython_blas.ddot(&n, <double*> dx, &incx, <double*> dy, &incy)


cdef inline double dnrm2(int n, const double* x, int incx) noexcept nogil:
    return cython_blas.dnrm2(&n, <double*> x, &incx)


cdef inline int idamax(int n, const double* dx, int incx) noexcept nogil:
    # Returns a one-based index
    return cython_blas.idamax(&n, <double*> dx, &incx)


# LAPACK workspace size; ample for the at most 15-column local design
# matrices (dgeqrf and dormqr need at least 15 and 1 elements, dgesvd 75)
cdef enum:
    LWORK = 1024


cdef void qr_decompose(double* b, int nf, int k, double* tau) noexcept nogil:
    # Householder QR decomposition of the nf x k matrix b (replaces LINPACK's
    # dqrdc with job = 0). R is left in the upper triangle of b, and the
    # reflectors in its lower triangle and in `tau`, which must hold min(nf, k)
    # elements.
    cdef int info, lwork = LWORK
    cdef double work[LWORK]
    cython_lapack.dgeqrf(&nf, &k, b, &nf, tau, work, &lwork, &info)


cdef void qr_multiply(bint transpose, double* b, int nf, int k,
                      const double* tau, double* y) noexcept nogil:
    # y <- Q^T y if transpose, else y <- Q y, for the Q of `qr_decompose` and
    # a length-nf vector y (replaces LINPACK's dqrsl with job = 1000 or 10000)
    cdef char side = b'L'
    cdef char trans = b'T' if transpose else b'N'
    cdef int one = 1, num_reflectors = min(nf, k), info, lwork = LWORK
    cdef double work[LWORK]
    if num_reflectors == 0:
        return
    cython_lapack.dormqr(&side, &trans, &nf, &one, &num_reflectors, b, &nf,
                         <double*> tau, y, &nf, work, &lwork, &info)


cdef int svd(double* u, double* sigma, double* v, int k) noexcept nogil:
    # Singular value decomposition of the k x k matrix u (leading dimension
    # 15), replacing LINPACK's dsvdc with job = 21: u is overwritten with the
    # left singular vectors, sigma receives the singular values in decreasing
    # order, and v (leading dimension 15) the right singular vectors, as
    # columns. Returns LAPACK's `info`, which is nonzero if it failed.
    cdef char jobu = b'O', jobvt = b'A'
    cdef int ld = 15, info, lwork = LWORK, i, j
    cdef double unused[1]
    cdef double vt[15 * 15]
    cdef double work[LWORK]
    cython_lapack.dgesvd(&jobu, &jobvt, &k, &k, u, &ld, sigma, unused, &ld,
                         vt, &ld, work, &lwork, &info)
    for i in range(k):
        for j in range(k):
            v[i + j * 15] = vt[j + i * 15]
    return info

###############################################################################
# dloess Fortran routines (loessf.f)
###############################################################################

cdef void ehg126(int d, int n, int vc, const double* x, double* v,
                 int nvmax) noexcept nogil:
    # Fill in the vertices of the bounding box of x
    cdef int i, j, k
    cdef double machin, alpha, beta, mu, t
    machin = DBL_MAX
    # Lower left, upper right
    for k in range(1, d + 1):
        alpha = machin
        beta = -machin
        for i in range(1, n + 1):
            t = x[(i - 1) + (k - 1) * n]
            alpha = f77_min(alpha, t)
            beta = f77_max(beta, t)
        # Expand the box a little
        mu = 0.005 * f77_max(beta - alpha, 1e-10 * f77_max(
            fabs(alpha), fabs(beta)) + 1e-30)
        alpha = alpha - mu
        beta = beta + mu
        v[(k - 1) * nvmax] = alpha
        v[(vc - 1) + (k - 1) * nvmax] = beta
    # Remaining vertices
    for i in range(2, vc):
        j = i - 1
        for k in range(1, d + 1):
            v[(i - 1) + (k - 1) * nvmax] = \
                v[(j % 2) * (vc - 1) + (k - 1) * nvmax]
            j = j / 2


cdef void ehg125(int p, int* nv, double* v, int* vhit, int nvmax, int d,
                 int k, double t, int r, int s, const int* f, int* l,
                 int* u) noexcept nogil:
    # Add the vertices created by splitting cell p at x_k = t; f, l and u
    # are the vertex lists of the cell and of its two children, each
    # dimensioned (r, 0:1, s)
    cdef int h, i, i3, j, m, mm
    cdef bint match
    h = nv[0]
    for i in range(1, r + 1):
        for j in range(1, s + 1):
            h = h + 1
            for i3 in range(1, d + 1):
                v[(h - 1) + (i3 - 1) * nvmax] = \
                    v[(f[(i - 1) + (j - 1) * 2 * r] - 1) + (i3 - 1) * nvmax]
            v[(h - 1) + (k - 1) * nvmax] = t
            # Check for redundant vertex
            match = False
            m = 1
            while not match and m <= nv[0]:
                match = v[m - 1] == v[h - 1]
                mm = 2
                while match and mm <= d:
                    match = v[(m - 1) + (mm - 1) * nvmax] == \
                        v[(h - 1) + (mm - 1) * nvmax]
                    mm = mm + 1
                m = m + 1
            m = m - 1
            if match:
                h = h - 1
            else:
                m = h
                if vhit[0] >= 0:
                    vhit[m - 1] = p
            l[(i - 1) + (j - 1) * 2 * r] = f[(i - 1) + (j - 1) * 2 * r]
            l[(i - 1) + r + (j - 1) * 2 * r] = m
            u[(i - 1) + (j - 1) * 2 * r] = m
            u[(i - 1) + r + (j - 1) * 2 * r] = \
                f[(i - 1) + r + (j - 1) * 2 * r]
    nv[0] = h
    if not nv[0] <= nvmax:
        ehg182(180)


cdef void ehg106(int il, int ir, int k, int nk, const double* p, int* pi,
                 int n) noexcept nogil:
    # Partial sort of p(1, il:ir) (Floyd and Rivest, CACM Mar '75, Algorithm
    # 489), returning the sort indices pi only, such that p(1, pi(k)) is the
    # k-th smallest element
    cdef double t
    cdef int i, ii, j, l, r
    l = il
    r = ir
    while l < r:
        # To avoid recursion, sophisticated partition deleted; partition
        # x[l..r] about t
        t = p[(pi[k - 1] - 1) * nk]
        i = l
        j = r
        ii = pi[l - 1]
        pi[l - 1] = pi[k - 1]
        pi[k - 1] = ii
        if t < p[(pi[r - 1] - 1) * nk]:
            ii = pi[l - 1]
            pi[l - 1] = pi[r - 1]
            pi[r - 1] = ii
        while i < j:
            ii = pi[i - 1]
            pi[i - 1] = pi[j - 1]
            pi[j - 1] = ii
            i = i + 1
            j = j - 1
            while p[(pi[i - 1] - 1) * nk] < t:
                i = i + 1
            while t < p[(pi[j - 1] - 1) * nk]:
                j = j - 1
        if p[(pi[l - 1] - 1) * nk] == t:
            ii = pi[l - 1]
            pi[l - 1] = pi[j - 1]
            pi[j - 1] = ii
        else:
            j = j + 1
            ii = pi[r - 1]
            pi[r - 1] = pi[j - 1]
            pi[j - 1] = ii
        if j <= k:
            l = j + 1
        if k <= j:
            r = j - 1


cdef void ehg127(const double* q, int n, int d, int nf, double f,
                 const double* x, int* psi, const double* y,
                 const double* rw, int* k, double* dist, double* eta,
                 double* b, int od, double* w, double* rcond, int* sing,
                 double* sigma, double* u, double* e, double* dgamma,
                 double* qraux, double* tol, int dd, int tdeg,
                 const int* cdeg, double* s) noexcept nogil:
    # Fit the local regression at q: find the nf nearest neighbors, compute
    # the tricube weights, and solve the weighted least-squares problem via
    # QR and SVD. `qraux` receives the QR decomposition's Householder scalars,
    # `u` and `e` the left and right singular vectors of R (the latter scaled
    # to undo the column equilibration), and `sigma` its singular values. The
    # neighborhood kernel is always tricube (kernel = 1) in dloess, so the
    # original's unused boxcar branch (kernel = 2) is omitted.
    cdef int column, i, i3, i9, info, inorm2, j, jj
    cdef double machep, i2, i4, i5, i6, i7, i8, i10, rho, scal, tmp, tmp3, \
        sqrt_rho
    cdef double colnor[15]
    machep = DBL_EPSILON

    # Sort by distance
    for i3 in range(1, n + 1):
        dist[i3 - 1] = 0
    for j in range(1, dd + 1):
        i4 = q[j - 1]
        for i3 in range(1, n + 1):
            tmp = x[(i3 - 1) + (j - 1) * n] - i4
            dist[i3 - 1] = dist[i3 - 1] + tmp * tmp
    ehg106(1, n, nf, 1, dist, psi, n)
    rho = dist[psi[nf - 1] - 1] * f77_max(1.0, f)
    if rho <= 0:
        ehg182(120)

    # Compute neighborhood weights
    for i3 in range(1, nf + 1):
        w[i3 - 1] = sqrt(dist[psi[i3 - 1] - 1] / rho)
    for i3 in range(1, nf + 1):
        tmp = w[i3 - 1]
        tmp3 = 1 - tmp * tmp * tmp
        w[i3 - 1] = sqrt(rw[psi[i3 - 1] - 1] * (tmp3 * tmp3 * tmp3))
    if fabs(w[idamax(nf, w, 1) - 1]) == 0:
        ehg184("at ", q, dd, 1)
        ehg184("radius ", &rho, 1, 1)
        ehg182(121)

    # Fill design matrix
    column = 1
    for i3 in range(1, nf + 1):
        b[(i3 - 1) + (column - 1) * nf] = w[i3 - 1]
    if tdeg >= 1:
        for j in range(1, d + 1):
            if cdeg[j - 1] >= 1:
                column = column + 1
                i5 = q[j - 1]
                for i3 in range(1, nf + 1):
                    b[(i3 - 1) + (column - 1) * nf] = w[i3 - 1] * (
                        x[(psi[i3 - 1] - 1) + (j - 1) * n] - i5)
    if tdeg >= 2:
        for j in range(1, d + 1):
            if cdeg[j - 1] >= 1:
                if cdeg[j - 1] >= 2:
                    column = column + 1
                    i6 = q[j - 1]
                    for i3 in range(1, nf + 1):
                        tmp = x[(psi[i3 - 1] - 1) + (j - 1) * n] - i6
                        b[(i3 - 1) + (column - 1) * nf] = \
                            w[i3 - 1] * (tmp * tmp)
                for jj in range(j + 1, d + 1):
                    if cdeg[jj - 1] >= 1:
                        column = column + 1
                        i7 = q[j - 1]
                        i8 = q[jj - 1]
                        for i3 in range(1, nf + 1):
                            b[(i3 - 1) + (column - 1) * nf] = w[i3 - 1] * (
                                x[(psi[i3 - 1] - 1) + (j - 1) * n] - i7) * (
                                x[(psi[i3 - 1] - 1) + (jj - 1) * n] - i8)
        k[0] = column
    for i3 in range(1, nf + 1):
        eta[i3 - 1] = w[i3 - 1] * y[psi[i3 - 1] - 1]

    # Equilibrate columns
    for j in range(1, k[0] + 1):
        scal = 0
        for inorm2 in range(1, nf + 1):
            tmp = b[(inorm2 - 1) + (j - 1) * nf]
            scal = scal + tmp * tmp
        scal = sqrt(scal)
        if 0 < scal:
            for i3 in range(1, nf + 1):
                b[(i3 - 1) + (j - 1) * nf] = b[(i3 - 1) + (j - 1) * nf] / scal
            colnor[j - 1] = scal
        else:
            colnor[j - 1] = 1

    # Singular value decomposition
    qr_decompose(b, nf, k[0], qraux)
    qr_multiply(True, b, nf, k[0], qraux, eta)
    for i9 in range(1, k[0] + 1):
        for i3 in range(1, k[0] + 1):
            u[(i3 - 1) + (i9 - 1) * 15] = 0
    for i in range(1, min(k[0], nf) + 1):
        for j in range(i, k[0] + 1):
            # When nf < k, rows past nf of R are zero (the original reads
            # past the end of b's columns instead)
            u[(i - 1) + (j - 1) * 15] = b[(i - 1) + (j - 1) * nf]
    info = svd(u, sigma, e, k[0])
    if not info == 0:
        ehg182(182)
    tol[0] = sigma[0] * (100 * machep)
    rcond[0] = f77_min(rcond[0], sigma[k[0] - 1] / sigma[0])
    if sigma[k[0] - 1] <= tol[0]:
        sing[0] = sing[0] + 1
        if sing[0] == 1:
            ehg184("pseudoinverse used at", q, d, 1)
            sqrt_rho = sqrt(rho)
            ehg184("neighborhood radius", &sqrt_rho, 1, 1)
            ehg184("reciprocal condition number ", rcond, 1, 1)
        elif sing[0] == 2:
            ehg184("There are other near singularities as well.", &rho, 1,
                   1)

    # Compensate for equilibration
    for j in range(1, k[0] + 1):
        i10 = colnor[j - 1]
        for i3 in range(1, k[0] + 1):
            e[(j - 1) + (i3 - 1) * 15] = e[(j - 1) + (i3 - 1) * 15] / i10

    # Solve least squares problem
    for j in range(1, k[0] + 1):
        if tol[0] < sigma[j - 1]:
            i2 = ddot(k[0], &u[(j - 1) * 15], 1, eta, 1) / sigma[j - 1]
        else:
            i2 = 0.0
        dgamma[j - 1] = i2
    for j in range(0, od + 1):
        # Bug fix 2006-07-04 for k = 1, od > 1 (thanks btyner@gmail.com)
        if j < k[0]:
            s[j] = ddot(k[0], &e[j], 15, dgamma, 1)
        else:
            s[j] = 0.0


cdef void ehg131(const double* x, const double* y, const double* rw,
                 double* trl, double* diagl, int* k, int n, int d, int* nc,
                 int ncmax, int vc, int* nv, int nvmax, int nf, double f,
                 int* a, int* c, int* hi, int* lo, int* pi, int* psi,
                 double* v, int* vhit, double* vval, double* xi, double* dist,
                 double* eta, double* b, int ntol, double* fd, double* w,
                 double* vval2, double* rcond, int* sing, int dd, int tdeg,
                 const int* cdeg, int* lq, double* lf,
                 bint setlf) noexcept nogil:
    cdef int i1, i2, j, identi
    cdef double delta[8]
    if not d <= 8:
        ehg182(101)
        set_fatal_error(error_message)
        return

    # Build k-d tree
    ehg126(d, n, vc, x, v, nvmax)
    nv[0] = vc
    nc[0] = 1
    for j in range(1, vc + 1):
        c[(j - 1) + (nc[0] - 1) * vc] = j
        vhit[j - 1] = 0
    for i1 in range(1, d + 1):
        delta[i1 - 1] = v[(vc - 1) + (i1 - 1) * nvmax] - v[(i1 - 1) * nvmax]
    fd[0] = fd[0] * dnrm2(d, delta, 1)
    for identi in range(1, n + 1):
        pi[identi - 1] = identi
    ehg124(1, n, d, n, nv, nc, ncmax, vc, x, pi, a, xi, lo, hi, c, v, vhit,
           nvmax, ntol, fd[0], dd)

    # Smooth
    if trl[0] != 0:
        for i2 in range(1, nv[0] + 1):
            for i1 in range(0, d + 1):
                vval2[i1 + (i2 - 1) * (d + 1)] = 0
    # `dist` is passed as both `dist` and `phi`, like the original
    ehg139(v, nvmax, nv[0], n, d, nf, f, x, pi, psi, y, rw, trl, k, dist,
           dist, eta, b, d, w, diagl, vval2, nc[0], vc, a, xi, lo, hi, c,
           rcond, sing, dd, tdeg, cdeg, lq, lf, setlf, vval)


cdef void ehg133(int d, int vc, int nvmax, int ncmax, const int* a,
                 const int* c, const int* hi, const int* lo, const double* v,
                 const double* vval, const double* xi, int m,
                 const double* z, double* s) noexcept nogil:
    # Evaluate the interpolated surface at the m points z(m, d)
    cdef int i, i1
    cdef double delta[8]
    for i in range(1, m + 1):
        for i1 in range(1, d + 1):
            delta[i1 - 1] = z[(i - 1) + <Py_ssize_t> (i1 - 1) * m]
        s[i - 1] = ehg128(delta, d, ncmax, vc, a, xi, lo, hi, c, v, nvmax,
                          vval)
        if fatal:
            return


cdef void ehg141(double trl, int n, int deg, int k, int d, int nsing,
                 int* dk, double* delta1, double* delta2) noexcept nogil:
    # Approximate delta1 and delta2 from trace(L) via a fitted lookup table
    cdef double c1, c2, c3, c4, corx, z
    cdef double zz[1]
    cdef int i
    # coef, d, deg, del
    cdef double c[48]
    c[:] = [.2971620, .3802660, .5886043, .4263766, .3346498, .6271053,
            .5241198, .3484836, .6687687, .6338795, .4076457, .7207693,
            .1611761, .3091323, .4401023, .2939609, .3580278, .5555741,
            .3972390, .4171278, .6293196, .4675173, .4699070, .6674802,
            .2848308, .2254512, .2914126, .5393624, .2517230, .3898970,
            .7603231, .2969113, .4740130, .9664956, .3629838, .5348889,
            .2075670, .2822574, .2369957, .3911566, .2981154, .3623232,
            .5508869, .3501989, .4371032, .7002667, .4291632, .4930370]
    if deg == 0:
        dk[0] = 1
    if deg == 1:
        dk[0] = d + 1
    if deg == 2:
        dk[0] = ((d + 2) * (d + 1)) / 2
    corx = sqrt(k / <double> n)
    z = (sqrt(k / trl) - corx) / (1 - corx)
    if nsing == 0 and 1 < z:
        ehg184("Chernobyl! trL<k", &trl, 1, 1)
    elif z < 0:
        ehg184("Chernobyl! trL>n", &trl, 1, 1)
    z = f77_min(1.0, f77_max(0.0, z))
    zz[0] = z
    c4 = exp(ehg176(zz))
    i = 1 + 3 * (min(d, 4) - 1 + 4 * (deg - 1))
    if d <= 4:
        c1 = c[i - 1]
        c2 = c[i]
        c3 = c[i + 1]
    else:
        c1 = c[i - 1] + (d - 4) * (c[i - 1] - c[i - 4])
        c2 = c[i] + (d - 4) * (c[i] - c[i - 3])
        c3 = c[i + 1] + (d - 4) * (c[i + 1] - c[i - 2])
    delta1[0] = n - trl * exp(c1 * pow(z, c2) * pow(1 - z, c3) * c4)
    i = i + 24
    if d <= 4:
        c1 = c[i - 1]
        c2 = c[i]
        c3 = c[i + 1]
    else:
        c1 = c[i - 1] + (d - 4) * (c[i - 1] - c[i - 4])
        c2 = c[i] + (d - 4) * (c[i] - c[i - 3])
        c3 = c[i + 1] + (d - 4) * (c[i + 1] - c[i - 2])
    delta2[0] = n - trl * exp(c1 * pow(z, c2) * pow(1 - z, c3) * c4)


cdef void lowesc(int n, double* l, double* ll, double* trl, double* delta1,
                 double* delta2) noexcept nogil:
    # Compute LL = (I - L)(I - L)' and the first two traces
    cdef int i, j
    cdef Py_ssize_t N = n
    for i in range(1, n + 1):
        l[(i - 1) + (i - 1) * N] = l[(i - 1) + (i - 1) * N] - 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            ll[(i - 1) + (j - 1) * N] = ddot(n, &l[i - 1], n, &l[j - 1], n)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            ll[(i - 1) + (j - 1) * N] = ll[(j - 1) + (i - 1) * N]
    for i in range(1, n + 1):
        l[(i - 1) + (i - 1) * N] = l[(i - 1) + (i - 1) * N] + 1
    trl[0] = 0
    delta1[0] = 0
    for i in range(1, n + 1):
        trl[0] = trl[0] + l[(i - 1) + (i - 1) * N]
        delta1[0] = delta1[0] + ll[(i - 1) + (i - 1) * N]
    # delta2 = trace(LL^2)
    delta2[0] = 0
    for i in range(1, n + 1):
        delta2[0] = delta2[0] + ddot(n, &ll[i - 1], n, &ll[(i - 1) * N], 1)


cdef void ehg169(int d, int vc, int nc, int ncmax, int nv, int nvmax,
                 double* v, int* a, double* xi, int* c, int* hi,
                 int* lo) noexcept nogil:
    # Rebuild the k-d tree's vertices and cells from its splits
    cdef int i, j, k, mc, mv, p
    cdef int novhit[1]
    # As in bbox: remaining vertices
    for i in range(2, vc):
        j = i - 1
        for k in range(1, d + 1):
            v[(i - 1) + (k - 1) * nvmax] = \
                v[(j % 2) * (vc - 1) + (k - 1) * nvmax]
            j = ifloor(j / 2.0)
    # As in ehg131
    mc = 1
    mv = vc
    novhit[0] = -1
    for j in range(1, vc + 1):
        c[(j - 1) + (mc - 1) * vc] = j
    # As in rbuild
    p = 1
    while p <= nc:
        if a[p - 1] != 0:
            k = a[p - 1]
            # Left son
            mc = mc + 1
            lo[p - 1] = mc
            # Right son
            mc = mc + 1
            hi[p - 1] = mc
            ehg125(p, &mv, v, novhit, nvmax, d, k, xi[p - 1], 1 << (k - 1),
                   1 << (d - k), &c[(p - 1) * vc], &c[(lo[p - 1] - 1) * vc],
                   &c[(hi[p - 1] - 1) * vc])
        p = p + 1
    if not mc == nc:
        ehg182(193)
    if not mv == nv:
        ehg182(193)


cdef double ehg176(const double* z) noexcept nogil:
    # A fixed one-dimensional k-d tree interpolant used by ehg141
    cdef int d = 1, vc = 2, nv = 10, nc = 17
    cdef int a[17]
    cdef int c[34]
    cdef int hi[17]
    cdef int lo[17]
    cdef double v[10]
    cdef double vval[20]
    cdef double xi[17]
    a[:] = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    hi[:] = [3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0]
    lo[:] = [2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0]
    xi[:] = [0.3705, 0.2017, 0.5591, 0.1204, 0.2815, 0.4536, 0.7132, 0, 0,
             0, 0, 0, 0, 0, 0.8751, 0, 0]
    c[:] = [1, 2, 1, 3, 3, 2, 1, 4, 4, 3, 3, 5, 5, 2, 1, 6, 6, 4, 4, 7, 7, 3,
            3, 8, 8, 5, 5, 9, 9, 2, 9, 10, 10, 2]
    v[:] = [-5e-3, 1.005, 0.3705, 0.2017, 0.5591, 0.1204, 0.2815, 0.4536,
            0.7132, 0.8751]
    vval[:] = [-9.0572e-2, 4.4844, -1.0856e-2, -0.7736, -5.3718e-2, -0.3495,
               2.6152e-2, -0.7286, -5.8387e-2, 0.1611, 9.5807e-2, -0.7978,
               -3.1926e-2, -0.4457, -6.4170e-2, 3.2813e-2, -2.0636e-2,
               0.3350, 4.0172e-2, -4.1032e-2]
    return ehg128(z, d, nc, vc, a, xi, lo, hi, c, v, nv, vval)


cdef void lowesa(double trl, int n, int d, int tau, int nsing,
                 double* delta1, double* delta2) noexcept nogil:
    cdef int dka, dkb
    cdef double alpha, d1a, d1b, d2a, d2b
    ehg141(trl, n, 1, tau, d, nsing, &dka, &d1a, &d2a)
    ehg141(trl, n, 2, tau, d, nsing, &dkb, &d1b, &d2b)
    alpha = <double> (tau - dka) / <double> (dkb - dka)
    delta1[0] = (1 - alpha) * d1a + alpha * d1b
    delta2[0] = (1 - alpha) * d2a + alpha * d2b


cdef void ehg191(int m, const double* z, double* l, int d, int n, int nf,
                 int nv, int ncmax, int vc, const int* a, const double* xi,
                 const int* lo, const int* hi, const int* c, const double* v,
                 int nvmax, double* vval2, const double* lf,
                 int* lq) noexcept nogil:
    # Compute the operator matrix L(m, n) of the interpolated fit
    cdef int lq1, i, i1, i2, j, p
    cdef double zi[8]
    for j in range(1, n + 1):
        for i2 in range(1, nv + 1):
            for i1 in range(0, d + 1):
                vval2[i1 + (i2 - 1) * (d + 1)] = 0
        for i in range(1, nv + 1):
            # Linear search for i in Lq
            lq1 = lq[i - 1]
            lq[i - 1] = j
            p = nf
            while lq[(i - 1) + (p - 1) * nvmax] != j:
                p = p - 1
            lq[i - 1] = lq1
            if lq[(i - 1) + (p - 1) * nvmax] == j:
                for i1 in range(0, d + 1):
                    vval2[i1 + (i - 1) * (d + 1)] = lf[
                        i1 + (i - 1) * (d + 1) +
                        <Py_ssize_t> (p - 1) * (d + 1) * nvmax]
        for i in range(1, m + 1):
            for i1 in range(1, d + 1):
                zi[i1 - 1] = z[(i - 1) + <Py_ssize_t> (i1 - 1) * m]
            l[(i - 1) + <Py_ssize_t> (j - 1) * m] = ehg128(
                zi, d, ncmax, vc, a, xi, lo, hi, c, v, nvmax, vval2)
            if fatal:
                return


cdef void ehg196(int tau, int d, double f, double* trl) noexcept nogil:
    # Approximate trace(L) from the span
    cdef int dka, dkb
    cdef double alpha, trla, trlb
    ehg197(1, d, f, &dka, &trla)
    ehg197(2, d, f, &dkb, &trlb)
    alpha = <double> (tau - dka) / <double> (dkb - dka)
    trl[0] = (1 - alpha) * trla + alpha * trlb


cdef void ehg197(int deg, int d, double f, int* dk,
                 double* trl) noexcept nogil:
    cdef double g1
    dk[0] = 0
    if deg == 1:
        dk[0] = d + 1
    if deg == 2:
        dk[0] = ((d + 2) * (d + 1)) / 2
    g1 = (-0.08125 * d + 0.13) * d + 1.05
    trl[0] = dk[0] * (1 + f77_max(0.0, (g1 - f) / f))


cdef inline void hermite(double h, double* phi0, double* phi1, double* psi0,
                         double* psi1) noexcept nogil:
    # Hermite basis
    phi0[0] = (1 - h) * (1 - h) * (1 + 2 * h)
    phi1[0] = h * h * (3 - 2 * h)
    psi0[0] = h * ((1 - h) * (1 - h))
    psi1[0] = h * h * (h - 1)


# Maximum k-d tree depth for `ehg128`; the original's limit is 20, past which
# it reports an error and then overflows its stack buffer
cdef enum:
    MAX_DEPTH = 1024

cdef double ehg128(const double* z, int d, int ncmax, int vc, const int* a,
                   const double* xi, const int* lo, const int* hi,
                   const int* c, const double* v, int nvmax,
                   const double* vval) noexcept nogil:
    # Evaluate the k-d tree's blended cubic Hermite interpolant at z
    cdef bint i2, i3
    cdef int i, i1, i11, i12, ig, ii, j, lg, ll, m, nt, ur, side
    cdef int t[MAX_DEPTH]
    cdef double ge, gn, gs, gw, gpe, gpn, gps, gpw, h, phi0, phi1, psi0, \
        psi1, s, sew, sns, v0, v1, xibar
    cdef double g[9 * 256]
    cdef double g0[9]
    cdef double g1[9]

    # Locate enclosing cell
    nt = 1
    t[nt - 1] = 1
    j = 1
    while a[j - 1] != 0:
        nt = nt + 1
        if z[a[j - 1] - 1] <= xi[j - 1]:
            i1 = lo[j - 1]
        else:
            i1 = hi[j - 1]
        if nt > MAX_DEPTH:
            ehg182(181)
            set_fatal_error(error_message)
            return 0
        t[nt - 1] = i1
        if not nt < 20:
            ehg182(181)
        j = t[nt - 1]

    # Tensor
    for i12 in range(1, vc + 1):
        for i11 in range(0, d + 1):
            g[i11 + (i12 - 1) * 9] = \
                vval[i11 + (c[(i12 - 1) + (j - 1) * vc] - 1) * (d + 1)]
    lg = vc
    ll = c[(j - 1) * vc]
    ur = c[(vc - 1) + (j - 1) * vc]
    for i in range(d, 0, -1):
        h = (z[i - 1] - v[(ll - 1) + (i - 1) * nvmax]) / (
            v[(ur - 1) + (i - 1) * nvmax] - v[(ll - 1) + (i - 1) * nvmax])
        if h < -.001:
            ehg184("eval ", z, d, 1)
            ehg184("lowerlimit ", &v[ll - 1], d, nvmax)
        elif 1.001 < h:
            ehg184("eval ", z, d, 1)
            ehg184("upperlimit ", &v[ur - 1], d, nvmax)
        if -.001 <= h:
            i2 = h <= 1.001
        else:
            i2 = False
        if not i2:
            ehg182(122)
        lg = lg / 2
        hermite(h, &phi0, &phi1, &psi0, &psi1)
        for ig in range(1, lg + 1):
            g[(ig - 1) * 9] = phi0 * g[(ig - 1) * 9] + \
                phi1 * g[(ig + lg - 1) * 9] + \
                (psi0 * g[i + (ig - 1) * 9] +
                 psi1 * g[i + (ig + lg - 1) * 9]) * (
                    v[(ur - 1) + (i - 1) * nvmax] -
                    v[(ll - 1) + (i - 1) * nvmax])
            for ii in range(1, i):
                g[ii + (ig - 1) * 9] = phi0 * g[ii + (ig - 1) * 9] + \
                    phi1 * g[ii + (ig + lg - 1) * 9]
    s = g[0]

    # Blending
    if d == 2:
        # ----- North -----
        v0 = v[ll - 1]
        v1 = v[ur - 1]
        for i11 in range(0, d + 1):
            g0[i11] = vval[i11 + (c[2 + (j - 1) * vc] - 1) * (d + 1)]
        for i11 in range(0, d + 1):
            g1[i11] = vval[i11 + (c[3 + (j - 1) * vc] - 1) * (d + 1)]
        xibar = v[(ur - 1) + nvmax]
        m = nt - 1
        while True:
            if m == 0:
                i3 = True
            elif a[t[m - 1] - 1] == 2:
                i3 = xi[t[m - 1] - 1] == xibar
            else:
                i3 = False
            if i3:
                break
            m = m - 1
        if m >= 1:
            m = hi[t[m - 1] - 1]
            while a[m - 1] != 0:
                if z[a[m - 1] - 1] <= xi[m - 1]:
                    m = lo[m - 1]
                else:
                    m = hi[m - 1]
            if v0 < v[c[(m - 1) * vc] - 1]:
                v0 = v[c[(m - 1) * vc] - 1]
                for i11 in range(0, d + 1):
                    g0[i11] = vval[i11 + (c[(m - 1) * vc] - 1) * (d + 1)]
            if v[c[1 + (m - 1) * vc] - 1] < v1:
                v1 = v[c[1 + (m - 1) * vc] - 1]
                for i11 in range(0, d + 1):
                    g1[i11] = vval[i11 + (c[1 + (m - 1) * vc] - 1) * (d + 1)]
        h = (z[0] - v0) / (v1 - v0)
        hermite(h, &phi0, &phi1, &psi0, &psi1)
        gn = phi0 * g0[0] + phi1 * g1[0] + \
            (psi0 * g0[1] + psi1 * g1[1]) * (v1 - v0)
        gpn = phi0 * g0[2] + phi1 * g1[2]

        # ----- South -----
        v0 = v[ll - 1]
        v1 = v[ur - 1]
        for i11 in range(0, d + 1):
            g0[i11] = vval[i11 + (c[(j - 1) * vc] - 1) * (d + 1)]
        for i11 in range(0, d + 1):
            g1[i11] = vval[i11 + (c[1 + (j - 1) * vc] - 1) * (d + 1)]
        xibar = v[(ll - 1) + nvmax]
        m = nt - 1
        while True:
            if m == 0:
                i3 = True
            elif a[t[m - 1] - 1] == 2:
                i3 = xi[t[m - 1] - 1] == xibar
            else:
                i3 = False
            if i3:
                break
            m = m - 1
        if m >= 1:
            m = lo[t[m - 1] - 1]
            while a[m - 1] != 0:
                if z[a[m - 1] - 1] <= xi[m - 1]:
                    m = lo[m - 1]
                else:
                    m = hi[m - 1]
            if v0 < v[c[2 + (m - 1) * vc] - 1]:
                v0 = v[c[2 + (m - 1) * vc] - 1]
                for i11 in range(0, d + 1):
                    g0[i11] = vval[i11 + (c[2 + (m - 1) * vc] - 1) * (d + 1)]
            if v[c[3 + (m - 1) * vc] - 1] < v1:
                v1 = v[c[3 + (m - 1) * vc] - 1]
                for i11 in range(0, d + 1):
                    g1[i11] = vval[i11 + (c[3 + (m - 1) * vc] - 1) * (d + 1)]
        h = (z[0] - v0) / (v1 - v0)
        hermite(h, &phi0, &phi1, &psi0, &psi1)
        gs = phi0 * g0[0] + phi1 * g1[0] + \
            (psi0 * g0[1] + psi1 * g1[1]) * (v1 - v0)
        gps = phi0 * g0[2] + phi1 * g1[2]

        # ----- East -----
        v0 = v[(ll - 1) + nvmax]
        v1 = v[(ur - 1) + nvmax]
        for i11 in range(0, d + 1):
            g0[i11] = vval[i11 + (c[1 + (j - 1) * vc] - 1) * (d + 1)]
        for i11 in range(0, d + 1):
            g1[i11] = vval[i11 + (c[3 + (j - 1) * vc] - 1) * (d + 1)]
        xibar = v[ur - 1]
        m = nt - 1
        while True:
            if m == 0:
                i3 = True
            elif a[t[m - 1] - 1] == 1:
                i3 = xi[t[m - 1] - 1] == xibar
            else:
                i3 = False
            if i3:
                break
            m = m - 1
        if m >= 1:
            m = hi[t[m - 1] - 1]
            while a[m - 1] != 0:
                if z[a[m - 1] - 1] <= xi[m - 1]:
                    m = lo[m - 1]
                else:
                    m = hi[m - 1]
            if v0 < v[(c[(m - 1) * vc] - 1) + nvmax]:
                v0 = v[(c[(m - 1) * vc] - 1) + nvmax]
                for i11 in range(0, d + 1):
                    g0[i11] = vval[i11 + (c[(m - 1) * vc] - 1) * (d + 1)]
            if v[(c[2 + (m - 1) * vc] - 1) + nvmax] < v1:
                v1 = v[(c[2 + (m - 1) * vc] - 1) + nvmax]
                for i11 in range(0, d + 1):
                    g1[i11] = vval[i11 + (c[2 + (m - 1) * vc] - 1) * (d + 1)]
        h = (z[1] - v0) / (v1 - v0)
        hermite(h, &phi0, &phi1, &psi0, &psi1)
        ge = phi0 * g0[0] + phi1 * g1[0] + \
            (psi0 * g0[2] + psi1 * g1[2]) * (v1 - v0)
        gpe = phi0 * g0[1] + phi1 * g1[1]

        # ----- West -----
        v0 = v[(ll - 1) + nvmax]
        v1 = v[(ur - 1) + nvmax]
        for i11 in range(0, d + 1):
            g0[i11] = vval[i11 + (c[(j - 1) * vc] - 1) * (d + 1)]
        for i11 in range(0, d + 1):
            g1[i11] = vval[i11 + (c[2 + (j - 1) * vc] - 1) * (d + 1)]
        xibar = v[ll - 1]
        m = nt - 1
        while True:
            if m == 0:
                i3 = True
            elif a[t[m - 1] - 1] == 1:
                i3 = xi[t[m - 1] - 1] == xibar
            else:
                i3 = False
            if i3:
                break
            m = m - 1
        if m >= 1:
            m = lo[t[m - 1] - 1]
            while a[m - 1] != 0:
                if z[a[m - 1] - 1] <= xi[m - 1]:
                    m = lo[m - 1]
                else:
                    m = hi[m - 1]
            if v0 < v[(c[1 + (m - 1) * vc] - 1) + nvmax]:
                v0 = v[(c[1 + (m - 1) * vc] - 1) + nvmax]
                for i11 in range(0, d + 1):
                    g0[i11] = vval[i11 + (c[1 + (m - 1) * vc] - 1) * (d + 1)]
            if v[(c[3 + (m - 1) * vc] - 1) + nvmax] < v1:
                v1 = v[(c[3 + (m - 1) * vc] - 1) + nvmax]
                for i11 in range(0, d + 1):
                    g1[i11] = vval[i11 + (c[3 + (m - 1) * vc] - 1) * (d + 1)]
        h = (z[1] - v0) / (v1 - v0)
        hermite(h, &phi0, &phi1, &psi0, &psi1)
        gw = phi0 * g0[0] + phi1 * g1[0] + \
            (psi0 * g0[2] + psi1 * g1[2]) * (v1 - v0)
        gpw = phi0 * g0[1] + phi1 * g1[1]

        # NS
        h = (z[1] - v[(ll - 1) + nvmax]) / (
            v[(ur - 1) + nvmax] - v[(ll - 1) + nvmax])
        hermite(h, &phi0, &phi1, &psi0, &psi1)
        sns = phi0 * gs + phi1 * gn + (psi0 * gps + psi1 * gpn) * (
            v[(ur - 1) + nvmax] - v[(ll - 1) + nvmax])

        # EW
        h = (z[0] - v[ll - 1]) / (v[ur - 1] - v[ll - 1])
        hermite(h, &phi0, &phi1, &psi0, &psi1)
        sew = phi0 * gw + phi1 * ge + (psi0 * gpw + psi1 * gpe) * (
            v[ur - 1] - v[ll - 1])
        s = (sns + sew) - s
    return s


cdef void ehg136(const double* u, int lm, int m, int n, int d, int nf,
                 double f, const double* x, int* psi, const double* y,
                 const double* rw, int* k, double* dist, double* eta,
                 double* b, int od, double* o, int ihat, double* w,
                 double* rcond, int* sing, int dd, int tdeg, const int* cdeg,
                 double* s) noexcept nogil:
    # The workhorse of lowesf: fit directly at each of the m points u(lm, d)
    # and, if ihat is 1 or 2, compute the diagonal or all of the operator
    # matrix o(m, n)
    cdef int identi, i, i1, j, l
    cdef double i2, scale, tol
    cdef double sigma[15]
    cdef double e[15 * 15]
    cdef double g[15 * 15]
    cdef double dgamma[15]
    cdef double q[8]
    cdef double qraux[15]
    if k[0] > nf - 1:
        ehg182(104)
    if k[0] > 15:
        ehg182(105)
        set_fatal_error(error_message)
        return
    for identi in range(1, n + 1):
        psi[identi - 1] = identi
    for l in range(1, m + 1):
        for i1 in range(1, d + 1):
            q[i1 - 1] = u[(l - 1) + <Py_ssize_t> (i1 - 1) * lm]
        ehg127(q, n, d, nf, f, x, psi, y, rw, k, dist, eta, b, od, w, rcond,
               sing, sigma, e, g, dgamma, qraux, &tol, dd, tdeg, cdeg,
               &s[(l - 1) * (od + 1)])
        if ihat == 1:
            # L(l, l) = V(1, :) SIGMA^+ U^T (Q^T W e_i)
            if not m == n:
                ehg182(123)
            # Find i such that l = psi(i)
            i = 1
            while l != psi[i - 1]:
                i = i + 1
                if not i < nf:
                    ehg182(123)
                    break
            for i1 in range(1, nf + 1):
                eta[i1 - 1] = 0
            eta[i - 1] = w[i - 1]
            # eta = Q^T W e_i
            qr_multiply(True, b, nf, k[0], qraux, eta)
            # gamma = U^T eta(1:k)
            for i1 in range(1, k[0] + 1):
                dgamma[i1 - 1] = 0
            for j in range(1, k[0] + 1):
                i2 = eta[j - 1]
                for i1 in range(1, k[0] + 1):
                    dgamma[i1 - 1] = dgamma[i1 - 1] + \
                        i2 * e[(j - 1) + (i1 - 1) * 15]
            # gamma = SIGMA^+ gamma
            for j in range(1, k[0] + 1):
                if tol < sigma[j - 1]:
                    dgamma[j - 1] = dgamma[j - 1] / sigma[j - 1]
                else:
                    dgamma[j - 1] = 0.0
            o[l - 1] = ddot(k[0], g, 15, dgamma, 1)
        elif ihat == 2:
            # L(l, :) = V(1, :) SIGMA^+ (U^T Q^T) W
            for i1 in range(1, n + 1):
                o[(l - 1) + <Py_ssize_t> (i1 - 1) * m] = 0
            for j in range(1, k[0] + 1):
                for i1 in range(1, nf + 1):
                    eta[i1 - 1] = 0
                for i1 in range(1, k[0] + 1):
                    eta[i1 - 1] = e[(i1 - 1) + (j - 1) * 15]
                qr_multiply(False, b, nf, k[0], qraux, eta)
                if tol < sigma[j - 1]:
                    scale = 1.0 / sigma[j - 1]
                else:
                    scale = 0.0
                for i1 in range(1, nf + 1):
                    eta[i1 - 1] = eta[i1 - 1] * (scale * w[i1 - 1])
                for i in range(1, nf + 1):
                    o[(l - 1) + <Py_ssize_t> (psi[i - 1] - 1) * m] = \
                        o[(l - 1) + <Py_ssize_t> (psi[i - 1] - 1) * m] + \
                        g[(j - 1) * 15] * eta[i - 1]


cdef void ehg139(const double* v, int nvmax, int nv, int n, int d, int nf,
                 double f, const double* x, const int* pi, int* psi,
                 const double* y, const double* rw, double* trl, int* k,
                 double* dist, double* phi, double* eta, double* b, int od,
                 double* w, double* diagl, double* vval2, int ncmax, int vc,
                 const int* a, const double* xi, const int* lo, const int* hi,
                 const int* c, double* rcond, int* sing, int dd, int tdeg,
                 const int* cdeg, int* lq, double* lf, bint setlf,
                 double* s) noexcept nogil:
    # Called from lowesb: fit at each vertex of the k-d tree, and optionally
    # compute the diagonal of the operator matrix (trl != 0) and the vertex
    # influence matrices Lf (setlf)
    cdef int identi, i, i2, i5, i6, ii, ileaf, j, l, nleaf
    cdef double i1, i4, i7, scale, term, tol
    cdef double sigma[15]
    cdef double u[15 * 15]
    cdef double e[15 * 15]
    cdef double dgamma[15]
    cdef double q[8]
    cdef double qraux[15]
    cdef double z[8]
    # The original's `leaf` holds 256 cells and its `ehg137` stack 20; size
    # them by the number of cells so they cannot overflow
    cdef int* leaf = NULL
    cdef int* pstack = NULL

    # l2fit with trace(L)
    if k[0] > nf - 1:
        ehg182(104)
    if k[0] > 15:
        ehg182(105)
        set_fatal_error(error_message)
        return
    if trl[0] != 0:
        for i5 in range(1, n + 1):
            diagl[i5 - 1] = 0
        for i6 in range(1, nv + 1):
            for i5 in range(0, d + 1):
                vval2[i5 + (i6 - 1) * (d + 1)] = 0
        leaf = <int*> malloc(ncmax * sizeof(int))
        pstack = <int*> malloc(ncmax * sizeof(int))
        if leaf == NULL or pstack == NULL:
            free(leaf)
            free(pstack)
            set_fatal_error("out of memory")
            return
    for identi in range(1, n + 1):
        psi[identi - 1] = identi
    for l in range(1, nv + 1):
        for i5 in range(1, d + 1):
            q[i5 - 1] = v[(l - 1) + (i5 - 1) * nvmax]
        ehg127(q, n, d, nf, f, x, psi, y, rw, k, dist, eta, b, od, w, rcond,
               sing, sigma, u, e, dgamma, qraux, &tol, dd, tdeg, cdeg,
               &s[(l - 1) * (od + 1)])
        if trl[0] != 0:
            # Invert psi
            for i5 in range(1, n + 1):
                phi[i5 - 1] = 0
            for i in range(1, nf + 1):
                phi[psi[i - 1] - 1] = i
            for i5 in range(1, d + 1):
                z[i5 - 1] = v[(l - 1) + (i5 - 1) * nvmax]
            nleaf = ehg137(z, leaf, pstack, d, ncmax, a, xi, lo, hi)
            for ileaf in range(1, nleaf + 1):
                for ii in range(lo[leaf[ileaf - 1] - 1],
                                hi[leaf[ileaf - 1] - 1] + 1):
                    i = <int> phi[pi[ii - 1] - 1]
                    if i != 0:
                        if not psi[i - 1] == pi[ii - 1]:
                            ehg182(194)
                        for i5 in range(1, nf + 1):
                            eta[i5 - 1] = 0
                        eta[i - 1] = w[i - 1]
                        # eta = Q^T W e_i
                        qr_multiply(True, b, nf, k[0], qraux, eta)
                        for j in range(1, k[0] + 1):
                            if tol < sigma[j - 1]:
                                i4 = ddot(k[0], &u[(j - 1) * 15], 1, eta,
                                          1) / sigma[j - 1]
                            else:
                                i4 = 0.0
                            dgamma[j - 1] = i4
                        for j in range(1, d + 2):
                            # Bug fix 2006-07-15 for k = 1, od > 1 (thanks
                            # btyner@gmail.com)
                            if j <= k[0]:
                                vval2[(j - 1) + (l - 1) * (d + 1)] = ddot(
                                    k[0], &e[j - 1], 15, dgamma, 1)
                            else:
                                vval2[(j - 1) + (l - 1) * (d + 1)] = 0.0
                        for i5 in range(1, d + 1):
                            z[i5 - 1] = x[(pi[ii - 1] - 1) + (i5 - 1) * n]
                        term = ehg128(z, d, ncmax, vc, a, xi, lo, hi, c, v,
                                      nvmax, vval2)
                        if fatal:
                            free(leaf)
                            free(pstack)
                            return
                        diagl[pi[ii - 1] - 1] = diagl[pi[ii - 1] - 1] + term
                        for i5 in range(0, d + 1):
                            vval2[i5 + (l - 1) * (d + 1)] = 0
        if setlf:
            # Lf(:, l, :) = V SIGMA^+ U^T Q^T W
            if not k[0] >= d + 1:
                ehg182(196)
            for i5 in range(1, nf + 1):
                lq[(l - 1) + (i5 - 1) * nvmax] = psi[i5 - 1]
            for i6 in range(1, nf + 1):
                for i5 in range(0, d + 1):
                    lf[i5 + (l - 1) * (d + 1) +
                       <Py_ssize_t> (i6 - 1) * (d + 1) * nvmax] = 0
            for j in range(1, k[0] + 1):
                for i5 in range(1, nf + 1):
                    eta[i5 - 1] = 0
                for i5 in range(1, k[0] + 1):
                    eta[i5 - 1] = u[(i5 - 1) + (j - 1) * 15]
                qr_multiply(False, b, nf, k[0], qraux, eta)
                if tol < sigma[j - 1]:
                    scale = 1.0 / sigma[j - 1]
                else:
                    scale = 0.0
                for i5 in range(1, nf + 1):
                    eta[i5 - 1] = eta[i5 - 1] * (scale * w[i5 - 1])
                for i in range(1, nf + 1):
                    i7 = eta[i - 1]
                    for i5 in range(0, d + 1):
                        if i5 < k[0]:
                            lf[i5 + (l - 1) * (d + 1) +
                               <Py_ssize_t> (i - 1) * (d + 1) * nvmax] = \
                                lf[i5 + (l - 1) * (d + 1) +
                                   <Py_ssize_t> (i - 1) * (d + 1) * nvmax] + \
                                e[i5 + (j - 1) * 15] * i7
                        else:
                            lf[i5 + (l - 1) * (d + 1) +
                               <Py_ssize_t> (i - 1) * (d + 1) * nvmax] = 0
    free(leaf)
    free(pstack)
    if trl[0] != 0:
        if n <= 0:
            trl[0] = 0.0
        else:
            i1 = diagl[n - 1]
            for i2 in range(n - 1, 0, -1):
                i1 = diagl[i2 - 1] + i1
            trl[0] = i1


cdef void lowesb(const double* xx, const double* yy, const double* ww,
                 double* diagl, int infl, int* iv, double* wv) noexcept nogil:
    # Build the k-d tree and fit at its vertices
    cdef double trl
    cdef bint setlf
    if not iv[27] != 173:
        ehg182(174)
    if iv[27] != 172:
        if not iv[27] == 171:
            ehg182(171)
    iv[27] = 173
    if infl != 0:
        trl = 1.0
    else:
        trl = 0.0
    setlf = iv[26] != iv[24]
    ehg131(xx, yy, ww, &trl, diagl, &iv[28], iv[2], iv[1], &iv[4], iv[16],
           iv[3], &iv[5], iv[13], iv[18], wv[0], &iv[iv[6] - 1],
           &iv[iv[7] - 1], &iv[iv[8] - 1], &iv[iv[9] - 1], &iv[iv[21] - 1],
           &iv[iv[26] - 1], &wv[iv[10] - 1], &iv[iv[22] - 1],
           &wv[iv[12] - 1], &wv[iv[11] - 1], &wv[iv[14] - 1],
           &wv[iv[15] - 1], &wv[iv[17] - 1], ifloor(iv[2] * wv[1]), &wv[2],
           &wv[iv[25] - 1], &wv[iv[23] - 1], &wv[3], &iv[29], iv[32], iv[31],
           &iv[40], &iv[iv[24] - 1], &wv[iv[33] - 1], setlf)
    if fatal:
        return
    if iv[13] < iv[5] + <double> iv[3] / 2.0:
        ehg183("k-d tree limited by memory; nvmax=", &iv[13], 1, 1)
    elif iv[16] < iv[4] + 2:
        ehg183("k-d tree limited by memory. ncmax=", &iv[16], 1, 1)


cdef void lowesd(int* iv, int liv, int lv, double* v, int d, int n, double f,
                 int ideg, int nf, int nvmax, int setlf) noexcept nogil:
    # Initialize iv and v(1:4)
    cdef int bound, i, i1, i2, j, ncmax, vc
    i1 = 0
    iv[27] = 171
    iv[1] = d
    iv[2] = n
    vc = 1 << d
    iv[3] = vc
    if not 0 < f:
        ehg182(120)
    iv[18] = nf
    iv[19] = 1
    if ideg == 0:
        i1 = 1
    elif ideg == 1:
        i1 = d + 1
    elif ideg == 2:
        i1 = ((d + 2) * (d + 1)) / 2
    iv[28] = i1
    iv[20] = 1
    iv[13] = nvmax
    ncmax = nvmax
    iv[16] = ncmax
    iv[29] = 0
    iv[31] = ideg
    if not ideg >= 0:
        ehg182(195)
    if not ideg <= 2:
        ehg182(195)
    iv[32] = d
    for i2 in range(41, 50):
        iv[i2 - 1] = ideg
    iv[6] = 50
    iv[7] = iv[6] + ncmax
    iv[8] = iv[7] + vc * ncmax
    iv[9] = iv[8] + ncmax
    iv[21] = iv[9] + ncmax
    # Initialize permutation
    j = iv[21] - 1
    for i in range(1, n + 1):
        iv[j + i - 1] = i
    iv[22] = iv[21] + n
    iv[24] = iv[22] + nvmax
    if setlf != 0:
        iv[26] = iv[24] + nvmax * nf
    else:
        iv[26] = iv[24]
    bound = iv[26] + n
    if not bound - 1 <= liv:
        ehg182(102)
    iv[10] = 50
    iv[12] = iv[10] + nvmax * d
    iv[11] = iv[12] + (d + 1) * nvmax
    iv[14] = iv[11] + ncmax
    iv[15] = iv[14] + n
    iv[17] = iv[15] + nf
    iv[23] = iv[17] + iv[28] * nf
    iv[33] = iv[23] + (d + 1) * nvmax
    if setlf != 0:
        iv[25] = iv[33] + (d + 1) * nvmax * nf
    else:
        iv[25] = iv[33]
    bound = iv[25] + nf
    if not bound - 1 <= lv:
        ehg182(103)
    v[0] = f
    v[1] = 0.05
    v[2] = 0.0
    v[3] = 1.0


cdef void lowese(int* iv, const double* wv, int m, const double* z,
                 double* s) noexcept nogil:
    # Evaluate the interpolated surface at the m points z(m, d)
    if not iv[27] != 172:
        ehg182(172)
    if not iv[27] == 173:
        ehg182(173)
    ehg133(iv[1], iv[3], iv[13], iv[16], &iv[iv[6] - 1], &iv[iv[7] - 1],
           &iv[iv[8] - 1], &iv[iv[9] - 1], &wv[iv[10] - 1], &wv[iv[12] - 1],
           &wv[iv[11] - 1], m, z, s)


cdef void lowesf(const double* xx, const double* yy, const double* ww,
                 int* iv, double* wv, int m, const double* z, double* l,
                 int ihat, double* s) noexcept nogil:
    # "direct" (non-interpolated) fit at the m points z(m, d)
    cdef bint i1
    if 171 <= iv[27]:
        i1 = iv[27] <= 174
    else:
        i1 = False
    if not i1:
        ehg182(171)
    iv[27] = 172
    if not iv[13] >= iv[18]:
        ehg182(186)
    ehg136(z, m, m, iv[2], iv[1], iv[18], wv[0], xx, &iv[iv[21] - 1], yy, ww,
           &iv[28], &wv[iv[14] - 1], &wv[iv[15] - 1], &wv[iv[17] - 1], 0, l,
           ihat, &wv[iv[25] - 1], &wv[3], &iv[29], iv[32], iv[31], &iv[40],
           s)


cdef void lowesl(int* iv, double* wv, int m, const double* z,
                 double* l) noexcept nogil:
    # Compute the operator matrix L(m, n) of the interpolated fit at z(m, d)
    if not iv[27] != 172:
        ehg182(172)
    if not iv[27] == 173:
        ehg182(173)
    if not iv[25] != iv[33]:
        ehg182(175)
        set_fatal_error(error_message)
        return
    ehg191(m, z, l, iv[1], iv[2], iv[18], iv[5], iv[16], iv[3],
           &iv[iv[6] - 1], &wv[iv[11] - 1], &iv[iv[9] - 1], &iv[iv[8] - 1],
           &iv[iv[7] - 1], &wv[iv[10] - 1], iv[13], &wv[iv[23] - 1],
           &wv[iv[33] - 1], &iv[iv[24] - 1])


cdef void lowesw(const double* res, int n, double* rw,
                 int* pi) noexcept nogil:
    # Compute bisquare robustness weights from the residuals (transliterated
    # from Devlin's ratfor)
    cdef int identi, i, i1, nh
    cdef double cmad, rsmall, ratio, tmp
    # Find median of absolute residuals
    for i1 in range(1, n + 1):
        rw[i1 - 1] = fabs(res[i1 - 1])
    for identi in range(1, n + 1):
        pi[identi - 1] = identi
    nh = ifloor(<double> n / 2.0) + 1
    # Partial sort to find 6 * mad
    ehg106(1, n, nh, 1, rw, pi, n)
    if (n - nh) + 1 < nh:
        ehg106(1, nh - 1, nh - 1, 1, rw, pi, n)
        cmad = 3 * (rw[pi[nh - 1] - 1] + rw[pi[nh - 2] - 1])
    else:
        cmad = 6 * rw[pi[nh - 1] - 1]
    rsmall = DBL_MIN
    if cmad < rsmall:
        for i1 in range(1, n + 1):
            rw[i1 - 1] = 1
    else:
        for i in range(1, n + 1):
            if cmad * 0.999 < rw[i - 1]:
                rw[i - 1] = 0
            elif cmad * 0.001 < rw[i - 1]:
                ratio = rw[i - 1] / cmad
                tmp = 1 - ratio * ratio
                rw[i - 1] = tmp * tmp
            else:
                rw[i - 1] = 1


cdef void lowesp(int n, const double* y, const double* yhat,
                 const double* pwgts, const double* rwgts, int* pi,
                 double* ytilde) noexcept nogil:
    # Compute pseudovalues for the robust fit
    cdef double c, i1, i4, mad, tmp
    cdef int i2, i3, i, m
    # Median absolute deviation (using partial sort)
    for i in range(1, n + 1):
        ytilde[i - 1] = fabs(y[i - 1] - yhat[i - 1]) * sqrt(pwgts[i - 1])
        pi[i - 1] = i
    m = ifloor(<double> n / 2.0) + 1
    ehg106(1, n, m, 1, ytilde, pi, n)
    if (n - m) + 1 < m:
        ehg106(1, m - 1, m - 1, 1, ytilde, pi, n)
        mad = (ytilde[pi[m - 2] - 1] + ytilde[pi[m - 1] - 1]) / 2
    else:
        mad = ytilde[pi[m - 1] - 1]
    # Magic constant
    c = (6 * mad) * (6 * mad) / 5
    for i in range(1, n + 1):
        tmp = y[i - 1] - yhat[i - 1]
        ytilde[i - 1] = 1 - ((tmp * tmp) * pwgts[i - 1]) / c
    for i in range(1, n + 1):
        ytilde[i - 1] = ytilde[i - 1] * sqrt(rwgts[i - 1])
    if n <= 0:
        i4 = 0.0
    else:
        i3 = n
        i1 = ytilde[i3 - 1]
        for i2 in range(i3 - 1, 0, -1):
            i1 = ytilde[i2 - 1] + i1
        i4 = i1
    c = n / i4
    # Pseudovalues
    for i in range(1, n + 1):
        ytilde[i - 1] = yhat[i - 1] + \
            (c * rwgts[i - 1]) * (y[i - 1] - yhat[i - 1])


cdef void ehg124(int ll, int uu, int d, int n, int* nv, int* nc, int ncmax,
                 int vc, const double* x, int* pi, int* a, double* xi,
                 int* lo, int* hi, int* c, double* v, int* vhit, int nvmax,
                 int fc, double fd, int dd) noexcept nogil:
    # Build the k-d tree
    cdef bint i1, i2, leaf
    cdef int i4, inorm2, k, l, m, p, u, upper, lower, check, offset
    cdef double diam
    cdef double diag[8]
    cdef double sigma[8]
    p = 1
    l = ll
    u = uu
    lo[p - 1] = l
    hi[p - 1] = u
    while p <= nc[0]:
        for i4 in range(1, dd + 1):
            diag[i4 - 1] = v[(c[(vc - 1) + (p - 1) * vc] - 1) +
                             (i4 - 1) * nvmax] - \
                v[(c[(p - 1) * vc] - 1) + (i4 - 1) * nvmax]
        diam = 0
        for inorm2 in range(1, dd + 1):
            diam = diam + diag[inorm2 - 1] * diag[inorm2 - 1]
        diam = sqrt(diam)
        if (u - l) + 1 <= fc:
            i1 = True
        else:
            i1 = diam <= fd
        if i1:
            leaf = True
        else:
            if ncmax < nc[0] + 2:
                i2 = True
            else:
                i2 = nvmax < nv[0] + <double> vc / 2.0
            leaf = i2
        if not leaf:
            ehg129(l, u, dd, x, pi, n, sigma)
            k = idamax(dd, sigma, 1)
            m = (l + u) / 2
            ehg106(l, u, m, 1, &x[(k - 1) * n], pi, n)

            # All ties go with hi son (bug fix from btyner@gmail.com
            # 2006-07-20)
            offset = 0
            while not (m + offset >= u or m + offset < l):
                if offset < 0:
                    lower = l
                    check = m + offset
                    upper = check
                else:
                    lower = m + offset + 1
                    check = lower
                    upper = u
                ehg106(lower, upper, check, 1, &x[(k - 1) * n], pi, n)
                if x[(pi[m + offset - 1] - 1) + (k - 1) * n] == \
                        x[(pi[m + offset] - 1) + (k - 1) * n]:
                    offset = -offset
                    if offset >= 0:
                        offset = offset + 1
                else:
                    m = m + offset
                    break

            if v[(c[(p - 1) * vc] - 1) + (k - 1) * nvmax] == \
                    x[(pi[m - 1] - 1) + (k - 1) * n]:
                leaf = True
            else:
                leaf = v[(c[(vc - 1) + (p - 1) * vc] - 1) +
                         (k - 1) * nvmax] == x[(pi[m - 1] - 1) + (k - 1) * n]
        if leaf:
            a[p - 1] = 0
        else:
            a[p - 1] = k
            xi[p - 1] = x[(pi[m - 1] - 1) + (k - 1) * n]
            # Left son
            nc[0] = nc[0] + 1
            lo[p - 1] = nc[0]
            lo[nc[0] - 1] = l
            hi[nc[0] - 1] = m
            # Right son
            nc[0] = nc[0] + 1
            hi[p - 1] = nc[0]
            lo[nc[0] - 1] = m + 1
            hi[nc[0] - 1] = u
            ehg125(p, nv, v, vhit, nvmax, d, k, xi[p - 1], 1 << (k - 1),
                   1 << (d - k), &c[(p - 1) * vc], &c[(lo[p - 1] - 1) * vc],
                   &c[(hi[p - 1] - 1) * vc])
        p = p + 1
        if p <= nc[0]:
            l = lo[p - 1]
            u = hi[p - 1]


cdef void ehg129(int l, int u, int d, const double* x, const int* pi, int n,
                 double* sigma) noexcept nogil:
    # Compute the range of each coordinate of x(pi(l:u), :)
    cdef int i, k
    cdef double machin, alpha, beta, t
    machin = DBL_MAX
    for k in range(1, d + 1):
        alpha = machin
        beta = -machin
        for i in range(l, u + 1):
            t = x[(pi[i - 1] - 1) + (k - 1) * n]
            alpha = f77_min(alpha, x[(pi[i - 1] - 1) + (k - 1) * n])
            beta = f77_max(beta, t)
        sigma[k - 1] = beta - alpha


cdef int ehg137(const double* z, int* leaf, int* pstack, int d, int ncmax,
                const int* a, const double* xi, const int* lo,
                const int* hi) noexcept nogil:
    # Find leaf cells affected by z; returns the number of leaves
    cdef int p, stackt, nleaf
    stackt = 0
    p = 1
    nleaf = 0
    while 0 < p:
        if a[p - 1] == 0:
            # Leaf
            nleaf = nleaf + 1
            leaf[nleaf - 1] = p
            # Pop
            if stackt >= 1:
                p = pstack[stackt - 1]
            else:
                p = 0
            stackt = max(0, stackt - 1)
        elif z[a[p - 1] - 1] == xi[p - 1]:
            # Push
            stackt = stackt + 1
            if not stackt <= 20:
                ehg182(187)
            pstack[stackt - 1] = hi[p - 1]
            p = lo[p - 1]
        elif z[a[p - 1] - 1] <= xi[p - 1]:
            p = lo[p - 1]
        else:
            p = hi[p - 1]
    if not nleaf <= 256:
        ehg182(185)
    return nleaf

###############################################################################
# dloess C driver routines (loessc.c)
###############################################################################

# The original keeps the Fortran workspace (`iv`, `v`, `liv`, `lv`, `tau`) in
# static globals; this port passes it around explicitly
cdef struct Workspace:
    int* iv
    double* v
    int liv
    int lv
    int tau


# The surface/statistics combinations from `condition()` in loess.c
cdef enum SurfStat:
    INTERPOLATE_NONE
    INTERPOLATE_1_APPROX
    INTERPOLATE_2_APPROX
    INTERPOLATE_EXACT
    DIRECT_NONE
    DIRECT_APPROXIMATE
    DIRECT_EXACT


cdef void loess_free(Workspace* ws) noexcept nogil:
    free(ws.v)
    free(ws.iv)
    ws.v = NULL
    ws.iv = NULL


cdef bint loess_workspace(Workspace* ws, int d, int n, double span,
                          int degree, int nonparametric,
                          const int* drop_square, int sum_drop_sqr,
                          int setLf) noexcept nogil:
    # Set tau, lv and liv, and allocate and initialize iv and v. Returns
    # whether it succeeded.
    cdef int D, N, tau0, nvmax, nf, i
    cdef double dlv, dliv
    ws.iv = NULL
    ws.v = NULL
    D = d
    N = n
    nvmax = max(200, N)
    nf = min(N, <int> floor(N * span + 1e-5))
    if nf <= 0:
        set_fatal_error("span is too small")
        return False
    tau0 = ((D + 2) * (D + 1) / 2) if degree > 1 else (D + 1)
    ws.tau = tau0 - sum_drop_sqr
    dlv = 50 + (3 * D + 3) * nvmax + N + (tau0 + 2) * nf
    dliv = 50 + (pow(2.0, <double> D) + 4.0) * nvmax + 2.0 * N
    if setLf:
        dlv = dlv + (D + 1.0) * nf * <double> nvmax
        dliv = dliv + nf * <double> nvmax
    if dlv < INT_MAX and dliv < INT_MAX:
        ws.lv = <int> dlv
        ws.liv = <int> dliv
    else:
        set_fatal_error("workspace required is too large")
        return False
    ws.iv = <int*> calloc(ws.liv, sizeof(int))
    ws.v = <double*> calloc(ws.lv, sizeof(double))
    if ws.iv == NULL or ws.v == NULL:
        loess_free(ws)
        set_fatal_error("out of memory")
        return False
    lowesd(ws.iv, ws.liv, ws.lv, ws.v, d, n, span, degree, nf, nvmax, setLf)
    ws.iv[32] = nonparametric
    for i in range(D):
        ws.iv[i + 40] = drop_square[i]
    return True


cdef void loess_prune(Workspace* ws, int* parameter, int* a, double* xi,
                      double* vert, double* vval) noexcept nogil:
    # Copy the k-d tree out of the workspace
    cdef int d, vc, a1, v1, xi1, vv1, nc, nv, nvmax, i, k
    cdef int* iv = ws.iv
    cdef double* v = ws.v
    d = iv[1]
    vc = iv[3] - 1
    nc = iv[4]
    nv = iv[5]
    a1 = iv[6] - 1
    v1 = iv[10] - 1
    xi1 = iv[11] - 1
    vv1 = iv[12] - 1
    nvmax = iv[13]
    for i in range(5):
        parameter[i] = iv[i + 1]
    parameter[5] = iv[21] - 1
    parameter[6] = iv[14] - 1
    for i in range(d):
        k = nvmax * i
        vert[i] = v[v1 + k]
        vert[i + d] = v[v1 + vc + k]
    for i in range(nc):
        xi[i] = v[xi1 + i]
        a[i] = iv[a1 + i]
    k = (d + 1) * nv
    for i in range(k):
        vval[i] = v[vv1 + i]


cdef bint loess_grow(Workspace* ws, int D, const int* parameter,
                     const int* a, const double* xi, const double* vert,
                     const double* vval) noexcept nogil:
    # Rebuild a workspace containing the k-d tree. Returns whether it
    # succeeded.
    cdef int d, vc, nc, nv, a1, v1, xi1, vv1, i, k
    cdef int* iv
    cdef double* v
    ws.iv = NULL
    ws.v = NULL
    d = parameter[0]
    vc = parameter[2]
    nc = parameter[3]
    nv = parameter[4]
    ws.liv = parameter[5]
    ws.lv = parameter[6]
    if d != D or ws.liv < 50 or ws.lv < 50:
        # The k-d tree was never built (e.g. the model was fit with
        # surface='direct'); the original reads uninitialized memory here
        ehg182(173)
        set_fatal_error(error_message)
        return False
    ws.iv = <int*> calloc(ws.liv, sizeof(int))
    ws.v = <double*> calloc(ws.lv, sizeof(double))
    if ws.iv == NULL or ws.v == NULL:
        loess_free(ws)
        set_fatal_error("out of memory")
        return False
    iv = ws.iv
    v = ws.v

    iv[1] = d
    iv[2] = parameter[1]
    iv[3] = vc
    iv[13] = nv
    iv[5] = nv
    iv[16] = nc
    iv[4] = nc
    iv[6] = 50
    iv[7] = iv[6] + nc
    iv[8] = iv[7] + vc * nc
    iv[9] = iv[8] + nc
    iv[10] = 50
    iv[12] = iv[10] + nv * d
    iv[11] = iv[12] + (d + 1) * nv
    iv[27] = 173

    v1 = iv[10] - 1
    xi1 = iv[11] - 1
    a1 = iv[6] - 1
    vv1 = iv[12] - 1

    for i in range(d):
        k = nv * i
        v[v1 + k] = vert[i]
        v[v1 + vc - 1 + k] = vert[i + d]
    for i in range(nc):
        v[xi1 + i] = xi[i]
        iv[a1 + i] = a[i]
    k = (d + 1) * nv
    for i in range(k):
        v[vv1 + i] = vval[i]

    ehg169(d, vc, nc, nc, nv, nv, v + v1, iv + a1, v + xi1, iv + iv[7] - 1,
           iv + iv[8] - 1, iv + iv[9] - 1)
    return True


cdef void loess_raw(const double* y, const double* x, const double* weights,
                    const double* robust, int d, int n, double span,
                    int degree, int nonparametric, const int* drop_square,
                    int sum_drop_sqr, double cell, SurfStat surf_stat,
                    double* surface, int* parameter, int* a, double* xi,
                    double* vert, double* vval, double* diagonal,
                    double* trL, double* one_delta, double* two_delta,
                    int setLf) noexcept nogil:
    cdef Workspace ws
    cdef int nsing, i
    cdef size_t k, nn
    cdef double* hat_matrix
    cdef double* LL
    cdef double dzero = 0

    trL[0] = 0
    if not loess_workspace(&ws, d, n, span, degree, nonparametric,
                           drop_square, sum_drop_sqr, setLf):
        return
    ws.v[1] = cell
    if surf_stat == INTERPOLATE_NONE:
        lowesb(x, y, robust, &dzero, 0, ws.iv, ws.v)
        if not fatal:
            lowese(ws.iv, ws.v, n, x, surface)
        if not fatal:
            loess_prune(&ws, parameter, a, xi, vert, vval)
    elif surf_stat == DIRECT_NONE:
        lowesf(x, y, robust, ws.iv, ws.v, n, x, &dzero, 0, surface)
    elif surf_stat == INTERPOLATE_1_APPROX:
        lowesb(x, y, weights, diagonal, 1, ws.iv, ws.v)
        if not fatal:
            lowese(ws.iv, ws.v, n, x, surface)
        if not fatal:
            nsing = ws.iv[29]
            for i in range(n):
                trL[0] = trL[0] + diagonal[i]
            lowesa(trL[0], n, d, ws.tau, nsing, one_delta, two_delta)
            loess_prune(&ws, parameter, a, xi, vert, vval)
    elif surf_stat == INTERPOLATE_2_APPROX:
        lowesb(x, y, weights, &dzero, 0, ws.iv, ws.v)
        if not fatal:
            lowese(ws.iv, ws.v, n, x, surface)
        if not fatal:
            nsing = ws.iv[29]
            ehg196(ws.tau, d, span, trL)
            lowesa(trL[0], n, d, ws.tau, nsing, one_delta, two_delta)
            loess_prune(&ws, parameter, a, xi, vert, vval)
    elif surf_stat == DIRECT_APPROXIMATE:
        lowesf(x, y, weights, ws.iv, ws.v, n, x, diagonal, 1, surface)
        if not fatal:
            nsing = ws.iv[29]
            for i in range(n):
                trL[0] = trL[0] + diagonal[i]
            lowesa(trL[0], n, d, ws.tau, nsing, one_delta, two_delta)
    elif surf_stat == INTERPOLATE_EXACT:
        nn = <size_t> n * <size_t> n
        hat_matrix = <double*> calloc(nn, sizeof(double))
        LL = <double*> calloc(nn, sizeof(double))
        if hat_matrix == NULL or LL == NULL:
            set_fatal_error("out of memory")
        else:
            lowesb(x, y, weights, diagonal, 1, ws.iv, ws.v)
            if not fatal:
                lowesl(ws.iv, ws.v, n, x, hat_matrix)
            if not fatal:
                lowesc(n, hat_matrix, LL, trL, one_delta, two_delta)
                lowese(ws.iv, ws.v, n, x, surface)
            if not fatal:
                loess_prune(&ws, parameter, a, xi, vert, vval)
        free(hat_matrix)
        free(LL)
    elif surf_stat == DIRECT_EXACT:
        nn = <size_t> n * <size_t> n
        hat_matrix = <double*> calloc(nn, sizeof(double))
        LL = <double*> calloc(nn, sizeof(double))
        if hat_matrix == NULL or LL == NULL:
            set_fatal_error("out of memory")
        else:
            lowesf(x, y, weights, ws.iv, ws.v, n, x, hat_matrix, 2, surface)
            if not fatal:
                lowesc(n, hat_matrix, LL, trL, one_delta, two_delta)
                k = <size_t> n + 1
                for i in range(n):
                    diagonal[i] = hat_matrix[<size_t> i * k]
        free(hat_matrix)
        free(LL)
    loess_free(&ws)


cdef void loess_dfit(const double* y, const double* x,
                     const double* x_evaluate, const double* weights,
                     double span, int degree, int nonparametric,
                     const int* drop_square, int sum_drop_sqr, int d, int n,
                     int m, double* fit) noexcept nogil:
    cdef Workspace ws
    cdef double dzero = 0.0
    if not loess_workspace(&ws, d, n, span, degree, nonparametric,
                           drop_square, sum_drop_sqr, 0):
        return
    lowesf(x, y, weights, ws.iv, ws.v, m, x_evaluate, &dzero, 0, fit)
    loess_free(&ws)


cdef void loess_dfitse(const double* y, const double* x,
                       const double* x_evaluate, const double* weights,
                       const double* robust, bint gaussian, double span,
                       int degree, int nonparametric, const int* drop_square,
                       int sum_drop_sqr, int d, int n, int m, double* fit,
                       double* L) noexcept nogil:
    cdef Workspace ws
    cdef double dzero = 0.0
    if not loess_workspace(&ws, d, n, span, degree, nonparametric,
                           drop_square, sum_drop_sqr, 0):
        return
    if gaussian:
        lowesf(x, y, weights, ws.iv, ws.v, m, x_evaluate, L, 2, fit)
    else:
        lowesf(x, y, weights, ws.iv, ws.v, m, x_evaluate, L, 2, fit)
        if not fatal:
            lowesf(x, y, robust, ws.iv, ws.v, m, x_evaluate, &dzero, 0, fit)
    loess_free(&ws)


cdef void loess_ifit(int D, const int* parameter, const int* a,
                     const double* xi, const double* vert,
                     const double* vval, int m, const double* x_evaluate,
                     double* fit) noexcept nogil:
    cdef Workspace ws
    if not loess_grow(&ws, D, parameter, a, xi, vert, vval):
        return
    lowese(ws.iv, ws.v, m, x_evaluate, fit)
    loess_free(&ws)


cdef void loess_ise(const double* y, const double* x,
                    const double* x_evaluate, const double* weights,
                    double span, int degree, int nonparametric,
                    const int* drop_square, int sum_drop_sqr, double cell,
                    int d, int n, int m, double* fit,
                    double* L) noexcept nogil:
    cdef Workspace ws
    cdef double dzero = 0.0
    if not loess_workspace(&ws, d, n, span, degree, nonparametric,
                           drop_square, sum_drop_sqr, 1):
        return
    ws.v[1] = cell
    lowesb(x, y, weights, &dzero, 0, ws.iv, ws.v)
    if not fatal:
        lowesl(ws.iv, ws.v, m, x_evaluate, L)
    loess_free(&ws)

###############################################################################
# dloess top-level fitting routine (`loess_` in loess.c)
###############################################################################

cdef int comp(const void* d1, const void* d2) noexcept nogil:
    cdef double a = (<const double*> d1)[0]
    cdef double b = (<const double*> d2)[0]
    if a < b:
        return -1
    elif a == b:
        return 0
    else:
        return 1


cdef SurfStat condition(bint surface_direct, bint statistics_none,
                        bint statistics_exact,
                        bint trace_hat_exact) noexcept nogil:
    if not surface_direct:
        if statistics_none:
            return INTERPOLATE_NONE
        elif statistics_exact:
            return INTERPOLATE_EXACT
        elif trace_hat_exact:
            return INTERPOLATE_1_APPROX
        else:
            return INTERPOLATE_2_APPROX
    else:
        if statistics_none:
            return DIRECT_NONE
        elif statistics_exact:
            return DIRECT_EXACT
        else:
            return DIRECT_APPROXIMATE


cdef void loess_(const double* y, const double* x_, int D, int N,
                 const double* weights, double span, int degree,
                 const int* parametric, const int* drop_square,
                 int normalize, bint statistics_exact, bint surface_direct,
                 double cell, bint trace_hat_exact, int iterations,
                 double* fitted_values, double* fitted_residuals, double* enp,
                 double* residual_scale, double* one_delta,
                 double* two_delta, double* pseudovalues,
                 double* trace_hat_out, double* diagonal, double* robust,
                 double* divisor, int* parameter, int* a, double* xi,
                 double* vert, double* vval) noexcept nogil:
    cdef double* x = NULL
    cdef double* x_tmp = NULL
    cdef double* temp = NULL
    cdef double* xi_tmp = NULL
    cdef double* vert_tmp = NULL
    cdef double* vval_tmp = NULL
    cdef double* diag_tmp = NULL
    cdef int* a_tmp = NULL
    cdef int* param_tmp = NULL
    cdef int* pi_tmp = NULL
    cdef int* order_parametric = NULL
    cdef int* order_drop_sqr = NULL
    cdef double new_cell, trL = 0, delta1 = 0, delta2 = 0, sum_squares = 0, \
        pseudo_resid, trL_tmp = 0, d1_tmp = 0, d2_tmp = 0, sum, mean
    cdef int i, j, k, p, sum_drop_sqr = 0, sum_parametric = 0, setLf, \
        nonparametric = 0, max_kd, cut
    cdef SurfStat surf_stat = INTERPOLATE_NONE
    cdef size_t q, ND = <size_t> N * <size_t> D

    max_kd = N if N > 200 else 200
    one_delta[0] = 0
    two_delta[0] = 0
    trace_hat_out[0] = 0

    x = <double*> malloc(ND * sizeof(double))
    x_tmp = <double*> malloc(ND * sizeof(double))
    temp = <double*> malloc(N * sizeof(double))
    a_tmp = <int*> malloc(max_kd * sizeof(int))
    xi_tmp = <double*> malloc(max_kd * sizeof(double))
    vert_tmp = <double*> malloc(D * 2 * sizeof(double))
    vval_tmp = <double*> malloc((D + 1) * max_kd * sizeof(double))
    diag_tmp = <double*> malloc(N * sizeof(double))
    # The original allocates N ints here, though loess_prune writes 7
    param_tmp = <int*> malloc(max(N, 7) * sizeof(int))
    # The original uses `temp` (a double array) as the integer permutation
    # for lowesw and lowesp
    pi_tmp = <int*> malloc(N * sizeof(int))
    order_parametric = <int*> malloc(D * sizeof(int))
    order_drop_sqr = <int*> malloc(D * sizeof(int))
    if (x == NULL or x_tmp == NULL or temp == NULL or a_tmp == NULL or
            xi_tmp == NULL or vert_tmp == NULL or vval_tmp == NULL or
            diag_tmp == NULL or param_tmp == NULL or pi_tmp == NULL or
            order_parametric == NULL or order_drop_sqr == NULL) and N > 0 \
            and D > 0:
        set_fatal_error("out of memory")
    else:
        new_cell = span * cell
        for i in range(N):
            robust[i] = 1

        for q in range(ND):
            x_tmp[q] = x_[q]

        if normalize and D > 1:
            cut = <int> ceil(0.100000000000000000001 * N)
            for i in range(D):
                k = i * N
                for j in range(N):
                    temp[j] = x_[k + j]
                qsort(temp, N, sizeof(double), comp)
                sum = 0
                for j in range(cut, N - cut):
                    sum = sum + temp[j]
                mean = sum / (N - 2 * cut)
                sum = 0
                for j in range(cut, N - cut):
                    temp[j] = temp[j] - mean
                    sum = sum + temp[j] * temp[j]
                divisor[i] = sqrt(sum / (N - 2 * cut - 1))
                for j in range(N):
                    p = k + j
                    x_tmp[p] = x_[p] / divisor[i]
        else:
            for i in range(D):
                divisor[i] = 1

        j = D - 1
        for i in range(D):
            sum_drop_sqr = sum_drop_sqr + drop_square[i]
            sum_parametric = sum_parametric + parametric[i]
            if parametric[i]:
                order_parametric[j] = i
                j -= 1
            else:
                order_parametric[nonparametric] = i
                nonparametric += 1
        # Reorder the predictors with the nonparametric ones first
        for i in range(D):
            order_drop_sqr[i] = 2 - drop_square[order_parametric[i]]
            k = i * N
            p = order_parametric[i] * N
            for j in range(N):
                x[k + j] = x_tmp[p + j]

        # Miscellaneous checks
        if degree == 1 and sum_drop_sqr:
            set_error("Specified the square of a factor predictor to be "
                      "dropped when degree = 1")
        elif D == 1 and sum_drop_sqr:
            set_error("Specified the square of a predictor to be dropped "
                      "with only one numeric predictor")
        elif sum_parametric == D:
            set_error("Specified parametric for all predictors")
        else:
            # Start the iterations
            for j in range(iterations + 1):
                for i in range(N):
                    robust[i] = weights[i] * robust[i]
                surf_stat = condition(surface_direct, j != 0,
                                      statistics_exact, trace_hat_exact)
                setLf = surf_stat == INTERPOLATE_EXACT
                loess_raw(y, x, weights, robust, D, N, span, degree,
                          nonparametric, order_drop_sqr, sum_drop_sqr,
                          new_cell, surf_stat, fitted_values, parameter, a,
                          xi, vert, vval, diagonal, &trL, &delta1, &delta2,
                          setLf)
                if fatal:
                    break
                if j == 0:
                    trace_hat_out[0] = trL
                    one_delta[0] = delta1
                    two_delta[0] = delta2
                for i in range(N):
                    fitted_residuals[i] = y[i] - fitted_values[i]
                if j < iterations:
                    lowesw(fitted_residuals, N, robust, pi_tmp)

            if not fatal:
                if iterations > 0:
                    lowesp(N, y, fitted_values, weights, robust, pi_tmp,
                           pseudovalues)
                    loess_raw(pseudovalues, x, weights, weights, D, N, span,
                              degree, nonparametric, order_drop_sqr,
                              sum_drop_sqr, new_cell, surf_stat, temp,
                              param_tmp, a_tmp, xi_tmp, vert_tmp, vval_tmp,
                              diag_tmp, &trL_tmp, &d1_tmp, &d2_tmp, 0)
                    for i in range(N):
                        pseudo_resid = pseudovalues[i] - temp[i]
                        sum_squares = sum_squares + \
                            weights[i] * pseudo_resid * pseudo_resid
                else:
                    for i in range(N):
                        sum_squares = sum_squares + weights[i] * \
                            fitted_residuals[i] * fitted_residuals[i]

                enp[0] = one_delta[0] + 2 * trace_hat_out[0] - N
                residual_scale[0] = sqrt(sum_squares / one_delta[0])

    free(x)
    free(x_tmp)
    free(temp)
    free(xi_tmp)
    free(vert_tmp)
    free(vval_tmp)
    free(diag_tmp)
    free(a_tmp)
    free(param_tmp)
    free(pi_tmp)
    free(order_parametric)
    free(order_drop_sqr)

###############################################################################
# dloess prediction (`pred_` in predict.c)
###############################################################################

cdef void pred_(const double* y, const double* x_, double* new_x, int D,
                int N, int M, double residual_scale, const double* weights,
                double* robust, double span, int degree,
                const int* parametric, const int* drop_square,
                bint direct_surface, double cell, bint gaussian_family,
                const int* parameter, const int* a, const double* xi,
                const double* vert, const double* vval,
                const double* divisor, int se, double* fit,
                double* se_fit) noexcept nogil:
    cdef double* x = NULL
    cdef double* x_tmp = NULL
    cdef double* x_evaluate = NULL
    cdef double* L = NULL
    cdef double* fit_tmp = NULL
    cdef int* order_parametric = NULL
    cdef int* order_drop_sqr = NULL
    cdef double new_cell, tmp
    cdef int sum_drop_sqr = 0, nonparametric = 0, i, j
    cdef size_t k, p
    cdef size_t ND = <size_t> N * <size_t> D
    cdef size_t MD = <size_t> M * <size_t> D
    cdef size_t NM = <size_t> N * <size_t> M

    x = <double*> malloc(ND * sizeof(double))
    x_tmp = <double*> malloc(ND * sizeof(double))
    x_evaluate = <double*> malloc(MD * sizeof(double))
    L = <double*> malloc(NM * sizeof(double))
    order_parametric = <int*> malloc(D * sizeof(int))
    order_drop_sqr = <int*> malloc(D * sizeof(int))
    if x == NULL or x_tmp == NULL or x_evaluate == NULL or L == NULL or \
            order_parametric == NULL or order_drop_sqr == NULL:
        set_fatal_error("out of memory")
    else:
        for k in range(ND):
            x_tmp[k] = x_[k]
        for i in range(D):
            k = <size_t> i * M
            for j in range(M):
                p = k + <size_t> j
                new_x[p] = new_x[p] / divisor[i]
        # The original tests `direct_surface || se` where `se` is a pointer,
        # so this always runs
        for i in range(D):
            k = <size_t> i * N
            for j in range(N):
                p = k + <size_t> j
                x_tmp[p] = x_[p] / divisor[i]
        j = D - 1
        for i in range(D):
            sum_drop_sqr = sum_drop_sqr + drop_square[i]
            if parametric[i]:
                order_parametric[j] = i
                j -= 1
            else:
                order_parametric[nonparametric] = i
                nonparametric += 1
        for i in range(D):
            order_drop_sqr[i] = 2 - drop_square[order_parametric[i]]
            k = <size_t> i * M
            p = <size_t> order_parametric[i] * M
            for j in range(M):
                x_evaluate[k + <size_t> j] = new_x[p + <size_t> j]
            k = <size_t> i * N
            p = <size_t> order_parametric[i] * N
            for j in range(N):
                x[k + <size_t> j] = x_tmp[p + <size_t> j]
        for i in range(N):
            robust[i] = weights[i] * robust[i]

        if direct_surface:
            if se:
                loess_dfitse(y, x, x_evaluate, weights, robust,
                             gaussian_family, span, degree, nonparametric,
                             order_drop_sqr, sum_drop_sqr, D, N, M, fit, L)
            else:
                loess_dfit(y, x, x_evaluate, robust, span, degree,
                           nonparametric, order_drop_sqr, sum_drop_sqr, D, N,
                           M, fit)
        else:
            loess_ifit(D, parameter, a, xi, vert, vval, M, x_evaluate, fit)
            if se and not fatal:
                new_cell = span * cell
                fit_tmp = <double*> malloc(M * sizeof(double))
                if fit_tmp == NULL:
                    set_fatal_error("out of memory")
                else:
                    loess_ise(y, x, x_evaluate, weights, span, degree,
                              nonparametric, order_drop_sqr, sum_drop_sqr,
                              new_cell, D, N, M, fit_tmp, L)
                    free(fit_tmp)
        if se and not fatal:
            for i in range(N):
                k = <size_t> i * M
                for j in range(M):
                    p = k + <size_t> j
                    L[p] = L[p] / weights[i]
                    L[p] = L[p] * L[p]
            for i in range(M):
                tmp = 0
                for j in range(N):
                    tmp = tmp + L[<size_t> i + <size_t> j * M]
                se_fit[i] = residual_scale * sqrt(tmp)
    free(x)
    free(x_tmp)
    free(x_evaluate)
    free(L)
    free(order_parametric)
    free(order_drop_sqr)

###############################################################################
# Statistical distributions (misc.c)
###############################################################################

cdef double invigauss_quick(double p) noexcept nogil:
    # Rational approximation to inverse Gaussian distribution. Absolute error
    # is bounded by 4.5e-4. Reference: Abramowitz and Stegun, page 933.
    # Assumption: 0 < p < 1.
    cdef bint lower
    cdef double t, n, d, q
    if p == 0.5:
        return 0
    lower = p < 0.5
    p = p if lower else 1 - p
    t = sqrt(-2 * log(p))
    n = (0.010328 * t + 0.802853) * t + 2.515517
    d = ((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.000000
    q = n / d - t if lower else t - n / d
    return q


cdef double invibeta_quick(double p, double a, double b) noexcept nogil:
    # Quick approximation to inverse incomplete beta function, by matching
    # first two moments with the Gaussian distribution. Assumption: 0 < p < 1,
    # a, b > 0.
    cdef double x, m, s, value
    x = a + b
    m = a / x
    s = sqrt((a * b) / (x * x * (x + 1)))
    value = invigauss_quick(p) * s + m
    value = 1.0 if 1.0 < value else value
    return 0.0 if 0.0 > value else value


cdef double invibeta(double p, double a, double b) noexcept nogil:
    # Inverse incomplete beta function. Assumption: 0 <= p <= 1, a, b > 0.
    cdef int i
    cdef double ql, qr, qm, qdiff, pl, pr, pm, pdiff
    qm = 0
    if p == 0 or p == 1:
        return p

    # Initialize [ql, qr] containing the root
    ql = qr = invibeta_quick(p, a, b)
    if ql != ql:
        # The original's bracketing loops below never terminate when the
        # initial guess is NaN, e.g. when the degrees of freedom are negative
        # (which can happen when fitting a few points with surface='direct')
        return ql
    pl = pr = ibeta(ql, a, b)
    if pl == p:
        return ql
    if pl < p:
        while True:
            qr += 0.05
            if qr >= 1:
                pr = qr = 1
                break
            pr = ibeta(qr, a, b)
            if pr == p:
                return pr
            if pr > p:
                break
    else:
        while True:
            ql -= 0.05
            if ql <= 0:
                pl = ql = 0
                break
            pl = ibeta(ql, a, b)
            if pl == p:
                return pl
            if pl < p:
                break

    # A few steps of bisection
    for i in range(5):
        qm = (ql + qr) / 2
        pm = ibeta(qm, a, b)
        qdiff = qr - ql
        pdiff = pm - p
        if fabs(qdiff) < DBL_EPSILON * qm or fabs(pdiff) < DBL_EPSILON:
            return qm
        if pdiff < 0:
            ql = qm
            pl = pm
        else:
            qr = qm
            pr = pm

    # A few steps of secant
    for i in range(40):
        qm = ql + (p - pl) * (qr - ql) / (pr - pl)
        pm = ibeta(qm, a, b)
        qdiff = qr - ql
        pdiff = pm - p
        if fabs(qdiff) < 2 * DBL_EPSILON * qm or \
                fabs(pdiff) < 2 * DBL_EPSILON:
            return qm
        if pdiff < 0:
            ql = qm
            pl = pm
        else:
            qr = qm
            pr = pm

    # No convergence
    return qm


cdef double qt(double p, double df) noexcept nogil:
    cdef double t
    t = invibeta(fabs(2 * p - 1), 0.5, df / 2)
    return (1 if p > 0.5 else -1) * sqrt(t * df / (1 - t))


cdef double pf(double q, double df1, double df2) noexcept nogil:
    return ibeta(q * df1 / (df2 + q * df1), df1 / 2, df2 / 2)


cdef double ibeta(double x, double a, double b) noexcept nogil:
    # Incomplete beta function. Reference: Abramowitz and Stegun, 26.5.8.
    # Assumptions: 0 <= x <= 1; a, b > 0.
    cdef bint flipped = False
    cdef int i, k, count
    cdef double I, temp, ak, bk, next, prev, factor, val
    cdef double pn[6]
    # The continued fraction below converges in a few hundred iterations at
    # most; give up long after that, rather than possibly looping forever
    cdef int max_count = 10000000

    if x <= 0:
        return 0
    if x >= 1:
        return 1

    # Use ibeta(x, a, b) = 1 - ibeta(1 - x, b, a)
    if (a + b + 1) * x > (a + 1):
        flipped = True
        temp = a
        a = b
        b = temp
        x = 1 - x

    pn[0] = 0.0
    pn[1] = 1.0
    pn[2] = 1.0
    pn[3] = 1.0
    count = 1
    val = x / (1.0 - x)
    bk = 1.0
    next = 1.0
    while True:
        count += 1
        k = count / 2
        prev = next
        if count % 2 == 0:
            ak = -((a + k - 1.0) * (b - k) * val) / \
                ((a + 2.0 * k - 2.0) * (a + 2.0 * k - 1.0))
        else:
            ak = ((a + b + k - 1.0) * k * val) / \
                ((a + 2.0 * k) * (a + 2.0 * k - 1.0))
        pn[4] = bk * pn[2] + ak * pn[0]
        pn[5] = bk * pn[3] + ak * pn[1]
        next = pn[4] / pn[5]
        for i in range(4):
            pn[i] = pn[i + 2]
        if fabs(pn[4]) >= DBL_MAX:
            for i in range(4):
                pn[i] /= DBL_MAX
        if fabs(pn[4]) <= DBL_MIN:
            for i in range(4):
                pn[i] /= DBL_MIN
        if not fabs(next - prev) > DBL_EPSILON * prev:
            break
        if count == max_count:
            return NAN
    factor = a * log(x) + (b - 1) * log(1 - x)
    factor -= lgamma(a + 1) + lgamma(b) - lgamma(a + b)
    I = exp(factor) * next
    return 1 - I if flipped else I

###############################################################################
# Python API (_loess.pyx)
###############################################################################

__all__ = ['loess', 'loess_model', 'loess_control', 'loess_inputs',
           'loess_outputs', 'loess_prediction', 'loess_confidence_intervals',
           'loess_anova']


cdef class loess_inputs:
    """
    Initialization class for loess data inputs

    Parameters
    ----------
    x : ndarray[n, p]
        n independent observations for p no. of variables
    y : ndarray[n]
        A (n,) ndarray of response observations
    weights : ndarray[n] or None
        Weights to be given to individual observations
        in the sum of squared residuals that forms the local fitting
        criterion. If not None, the weights should be non negative. If
        the different observations have non-equal variances, the weights
        should be inversely proportional to the variances. By default,
        an unweighted fit is carried out (all the weights are one).
    """
    cdef readonly allocated
    cdef long _n
    cdef long _p
    # Flattened copies of x (in C order, which the fitting routines treat as
    # column-major, like the original), y and weights
    cdef double[::1] _x
    cdef double[::1] _y
    cdef double[::1] _weights

    def __cinit__(self, x, y, weights=None):
        self.allocated = False

        x = np.asarray(x, dtype=np.float64, order='C')
        y = np.asarray(y, dtype=np.float64, order='C')
        n = len(x)

        # Check the dimensions
        if x.ndim == 1:
            p = 1
        elif x.ndim == 2:
            p = x.shape[1]
        else:
            raise ValueError("The array of indepedent varibales "
                             "should be 2D at most!")

        if y.ndim != 1:
            raise ValueError("The array of dependent variables "
                             "should be 1D.")
        elif n != len(y):
            raise ValueError("The independent and depedent varibales "
                             "should have the same number of "
                             "observations.")

        if weights is None:
            weights = np.ones((n,), dtype=np.float64)

        if weights.ndim > 1 or weights.size != n:
            raise ValueError("Invalid size of the 'weights' vector!")

        weights = np.asarray(weights, dtype=np.float64, order='C')

        self._x = np.array(x.ravel()[:n * p], dtype=np.float64)
        self._y = np.array(y[:n], dtype=np.float64)
        self._weights = np.array(weights[:n], dtype=np.float64)
        self._n = n
        self._p = p
        self.allocated = True

    def __init__(self, x, y, weights=None):
        # For documentation
        pass

    @property
    def n(self):
        """
        :class:`int` - Number of independent observations
        """
        return self._n

    @property
    def p(self):
        """
        :class:`int` - Number of variables
        """
        return self._p

    @property
    def x(self):
        """
        :class:`~numpy.ndarray` - Independent observations, shape ``(n, p)``
        """
        x = np.array(self._x)
        return x.reshape(self._n, self._p) if self._p > 1 else x

    @property
    def y(self):
        """
        :class:`~numpy.ndarray` - Response observations, shape ``(n,)``
        """
        return np.array(self._y)


cdef class loess_control:
    """
    Initialization class for loess control parameters

    Parameters
    ----------
    surface : str, optional
        One of ['interpolate', 'direct']
        Determines whether the fitted surface is computed directly
        at all points ('direct') or whether an interpolation method
        is used ('interpolate'). The default 'interpolate') is what
        most users should use unless special circumstances warrant.
    statistics : str, optional
        One of ['approximate', 'exact']
        Determines whether the statistical quantities are computed
        exactly ('exact') or approximately ('approximate'). 'exact'
        should only be used for testing the approximation in
        statistical development and is not meant for routine usage
        because computation time can be horrendous.
    trace_hat : str, optional
        One of ['wait.to.decide', 'exact', 'approximate']
        Determines how the trace of the hat matrix should be computed.
        The hat matrix is used in the computation of the statistical
        quantities. If 'exact', an exact computation is done; this
        could be slow when the number of observations n becomes large.
        If 'wait.to.decide' is selected, then a default is 'exact'
        for n < 500 and 'approximate' otherwise.
        This option is only useful when the fitted surface is
        interpolated. If surface is 'exact', an exact computation is
        always done for the trace. Setting trace_hat to 'approximate'
        for large dataset will substantially reduce the computation time.
    iterations : int, optional
        Number of iterations of the robust fitting method. If the family
        is 'gaussian', the number of iterations is set to 0.
    cell : float, optional
        Maximum cell size of the kd-tree. Suppose k = floor(n*cell*span),
        where n is the number of observations, and span the smoothing
        parameter. Then, a cell is further divided if the number of
        observations within it is greater than or equal to k. This
        option is only used if the surface is interpolated.
    """
    cdef str _surface
    cdef str _statistics
    cdef str _trace_hat
    cdef int _iterations
    cdef double _cell

    def __cinit__(self, *args, **kwargs):
        self._surface = 'interpolate'
        self._statistics = 'approximate'
        self._cell = 0.2
        self._trace_hat = 'wait.to.decide'
        self._iterations = 4

    def __init__(self, surface='interpolate', statistics='approximate',
                 trace_hat='wait.to.decide', iterations=4, cell=0.2):
        self.surface = surface
        self.statistics = statistics
        self.trace_hat = trace_hat
        self.iterations = iterations
        self.cell = cell

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, value):
        if value not in ('interpolate', 'direct'):
            raise ValueError(
                "Invalid value for the 'surface' argument: "
                "should be in ('interpolate', 'direct').")
        self._surface = value.encode('utf-8').decode('utf-8')

    @property
    def statistics(self):
        return self._statistics

    @statistics.setter
    def statistics(self, value):
        if value not in ('approximate', 'exact'):
            raise ValueError(
                "Invalid value for the 'statistics' argument: "
                "should be in ('approximate', 'exact').")
        self._statistics = value.encode('utf-8').decode('utf-8')

    @property
    def trace_hat(self):
        return self._trace_hat

    @trace_hat.setter
    def trace_hat(self, value):
        if value not in ('wait.to.decide', 'approximate', 'exact'):
            raise ValueError(
                "Invalid value for the 'trace_hat' argument: "
                "should be in ('approximate', 'exact').")
        self._trace_hat = value.encode('utf-8').decode('utf-8')

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        if value < 0:
            raise ValueError(
                "Invalid number of iterations: "
                "should be positive")
        self._iterations = value

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, value):
        if value <= 0:
            raise ValueError(
                "Invalid value for the cell argument: "
                " should be positive")
        self._cell = value

    def __str__(self):
        strg = ["Control",
                "-------",
                "Surface type     : %s" % self.surface,
                "Statistics       : %s" % self.statistics,
                "Trace estimation : %s" % self.trace_hat,
                "Cell size        : %s" % self.cell,
                "Nb iterations    : %s" % self.iterations,]
        return '\n'.join(strg)


cdef class loess_kd_tree:
    cdef int[::1] _parameter
    cdef int[::1] _a
    cdef double[::1] _xi
    cdef double[::1] _vert
    cdef double[::1] _vval

    def __cinit__(self, n, p):
        max_kd = n if n > 200 else 200
        self._parameter = np.zeros(7, dtype=np.intc)
        self._a = np.zeros(max_kd, dtype=np.intc)
        self._xi = np.zeros(max_kd, dtype=np.float64)
        self._vert = np.zeros(p * 2, dtype=np.float64)
        self._vval = np.zeros((p + 1) * max_kd, dtype=np.float64)


cdef class loess_model:
    """
    Initialization class for loess fitting parameters

    Parameters
    ----------
    p : int
        Number of variables
    family : str
        One of ('gaussian', 'symmetric')
        Determines the assumed distribution of the errors. If 'gaussian'
        the fit is performed with least-squares. If 'symmetric' is
        selected, the fit is performed robustly by redescending
        M-estimators.
    span : float
        Smoothing factor, as a fraction of the number of points to take
        into account. Should be in the range (0, 1]. Default is 0.75
    degree : int
        Overall degree of locally-fitted polynomial. 1 is locally-linear
        fitting and 2 is locally-quadratic fitting. Degree should be 2 at
        most. Default is 2.
    normalize : bool
        Determines whether the independent variables should be normalized.
        If True, the normalization is performed by setting the 10% trimmed
        standard deviation to one. If False, no normalization is carried
        out. This option is only useful for more than one variable. For
        spatial coordinates predictors or variables with a common scale,
        it should be set to False. Default is True.
    parametric : bool | list-of-bools of length p
        Indicates which independent variables should be
        conditionally-parametric (if there are two or more independent
        variables). If a sequence is given, the values should be ordered
        according to the predictor group in x.
    drop_square : bool | list-of-bools of length p
        Which squares to drop. When there are two or more independent
        variables and when a 2nd order polynomial(degree) is used,
        'drop_square' specifies those numeric predictors
        whose squares should be dropped from the set of fitting variables.
        If a sequence is given, the values should be ordered according to
        the predictor group in x.
    """
    cdef p
    cdef double _span
    cdef int _degree
    cdef int _normalize
    cdef str _family
    # The original stores these in fixed-size int[8] arrays
    cdef int[::1] _parametric
    cdef int[::1] _drop_square

    def __cinit__(self, *args, **kwargs):
        self._span = 0.75
        self._degree = 2
        self._normalize = True
        self._family = 'gaussian'

    def __init__(self, p, family='gaussian', span=0.75,
                 degree=2, normalize=True, parametric=False,
                 drop_square=False):
        self.p = p
        self._parametric = np.zeros(max(8, p), dtype=np.intc)
        self._drop_square = np.zeros(max(8, p), dtype=np.intc)
        self.family = family
        self.span = span
        self.degree = degree
        self.normalize = normalize
        self.parametric = parametric
        self.drop_square = drop_square

    @property
    def normalize(self):
        return bool(self._normalize)

    @normalize.setter
    def normalize(self, value):
        self._normalize = value

    @property
    def span(self):
        return self._span

    @span.setter
    def span(self, value):
        if value <= 0. or value > 1.:
            raise ValueError("Span should be between 0 and 1!")
        self._span = value

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value):
        if value < 0 or value > 2:
            raise ValueError("Degree should be be 0, 1 or 2!")
        self._degree = value

    @property
    def family(self):
        return self._family

    @family.setter
    def family(self, value):
        if value.lower()  not in ('symmetric', 'gaussian'):
            raise ValueError(
                "Invalid value for the 'family' argument: "
                "should be in ('symmetric', 'gaussian').")
        self._family = value.encode('utf-8').decode('utf-8')

    @property
    def parametric(self):
        return np.array(self._parametric[:self.p], dtype=bool)

    @parametric.setter
    def parametric(self, value):
        cdef int[::1] parametric = self._parametric
        cdef Py_ssize_t i

        if value in (True, False):
            value = [value] * self.p
        elif len(value) != self.p:
            raise ValueError(
                "'parametric' should be a boolean or a list "
                "of booleans with length equal to the number "
                "of independent variables")

        p_ndr = np.atleast_1d(np.asarray(value, dtype=bool))
        for i in range(self.p):
            parametric[i] = p_ndr[i]

    @property
    def drop_square(self):
        return np.array(self._drop_square[:self.p], dtype=bool)

    @drop_square.setter
    def drop_square(self, value):
        cdef int[::1] drop_square = self._drop_square
        cdef Py_ssize_t i

        if value in (True, False):
            value = [value] * self.p
        elif len(value) != self.p:
            raise ValueError(
                "'drop_square' should be a boolean or a list "
                "of booleans with length equal to the number "
                "of independent variables")

        d_ndr = np.atleast_1d(np.asarray(value, dtype=bool))
        for i in range(self.p):
            drop_square[i] = d_ndr[i]

    def __repr__(self):
        return "<loess object: model parameters>"

    def __str__(self):
        strg = ["Model parameters",
                "----------------",
                "Family          : %s" % self.family,
                "Span            : %s" % self.span,
                "Degree          : %s" % self.degree,
                "Normalized      : %s" % self.normalize,
                "Parametric      : %s" % self.parametric[:self.p],
                "Drop_square     : %s" % self.drop_square[:self.p],
                ]
        return '\n'.join(strg)


cdef class loess_outputs:
    """
    Class of a loess fit outputs

    This object is automatically created with empty values when a
    new loess object is instantiated. The object gets filled when the
    loess.fit() method is called.

    Parameters
    ----------
    n : int
        Number of independent observation
    p : int
        Number of variables
    """
    cdef double[::1] _fitted_values
    cdef double[::1] _fitted_residuals
    cdef double _enp
    cdef double _residual_scale
    cdef double _one_delta
    cdef double _two_delta
    cdef double[::1] _pseudovalues
    cdef double _trace_hat
    cdef double[::1] _diagonal
    cdef double[::1] _robust
    cdef double[::1] _divisor
    # private
    cdef readonly family
    cdef readonly n, p
    cdef readonly activated

    def __cinit__(self, n, p, family):
        self._fitted_values = np.zeros(n, dtype=np.float64)
        self._fitted_residuals = np.zeros(n, dtype=np.float64)
        self._diagonal = np.zeros(n, dtype=np.float64)
        self._robust = np.zeros(n, dtype=np.float64)
        self._divisor = np.zeros(p, dtype=np.float64)
        self._pseudovalues = np.zeros(n, dtype=np.float64)

    def __init__(self, n, p, family):
        self.n = n
        self.p = p
        self.family = family
        self.activated = False

    @property
    def fitted_values(self):
        """
        :class:`~numpy.ndarray` - Fitted values, shape ``(n,)``
        """
        return np.array(self._fitted_values)

    @property
    def fitted_residuals(self):
        """
        :class:`~numpy.ndarray` - Fitted residuals, shape ``(n,)``,
        (observations - fitted values)
        """
        return np.array(self._fitted_residuals)

    @property
    def pseudovalues(self):
        """
        :class:`~numpy.ndarray` - Adjusted values of the response
        when robust estimation is used, shape ``(n,)``
        """
        if self.family != 'symmetric':
            raise ValueError(
                "pseudovalues are available only when "
                "robust fitting. Use family='symmetric' "
                "for robust fitting")
        return np.array(self._pseudovalues)

    @property
    def diagonal(self):
        """
        :class:`~numpy.ndarray` - Diagonal of the operator hat matrix,
        shape ``(n,)``
        """
        return np.array(self._diagonal)

    @property
    def robust(self):
        """
        :class:`~numpy.ndarray` - Robustness weights for robust fitting,
        shape ``(n,)``
        """
        return np.array(self._robust)

    @property
    def divisor(self):
        """
        :class:`~numpy.ndarray` - Normalization divisors for numeric
        predictors, shape ``(p,)``
        """
        return np.array(self._divisor)

    @property
    def enp(self):
        """
        :class:`float` - Equivalent number of parameters
        """
        return self._enp

    @property
    def residual_scale(self):
        """
        :class:`float` - Estimate of the scale of residuals
        """
        return self._residual_scale

    @property
    def one_delta(self):
        """
        :class:`float` - Statistical parameter used in the computation
        of standard errors
        """
        return self._one_delta

    @property
    def two_delta(self):
        """
        :class:`float` - Statistical parameter used in the computation
        of standard errors
        """
        return self._two_delta

    @property
    def trace_hat(self):
        """
        :class:`float` - Trace of the operator hat matrix
        """
        return self._trace_hat

    def __str__(self):
        strg = ["Outputs",
                "-------",
                "Fitted values         : %s\n" % self.fitted_values,
                "Fitted residuals      : %s\n" % self.fitted_residuals,
                "Eqv. nb of parameters : %s" % self.enp,
                "Residual Scale        : %s" % self.residual_scale,
                "Deltas                : %s - %s" % (self.one_delta,
                                                     self.two_delta),
                "Normalization factors : %s" % self.divisor,]
        return '\n'.join(strg)


cdef class loess_confidence_intervals:
    """
    Pointwise confidence intervals of a loess-predicted object

    Parameters
    ----------
    pred : loess_prediction
        Prediction object
    alpha : float
        The alpha level for the confidence interval.
        It must be in the range (0, 1)
    """
    cdef double[::1] _fit
    cdef double[::1] _upper
    cdef double[::1] _lower
    cdef readonly m

    def __cinit__(loess_confidence_intervals self, loess_prediction pred,
                  float alpha):
        cdef double coverage, t_dist, limit, fit
        cdef int i
        cdef double[::1] pred_fit, pred_se_fit, ci_fit, ci_upper, ci_lower

        # Like the original, `alpha` is single precision (but `coverage` is
        # double precision)
        coverage = 1 - <double> alpha
        if coverage < .5:
            coverage = 1 - coverage

        if not 0 < <double> alpha < 1. :
            raise ValueError("The alpha value should be "
                             "between 0 and 1.")
        if not pred._se:
            raise ValueError("Cannot compute confidence intervals "
                             "without standard errors.")

        # `pointwise` in misc.c
        pred_fit = pred._fit
        pred_se_fit = pred._se_fit
        ci_fit = self._fit = np.empty(pred._m, dtype=np.float64)
        ci_upper = self._upper = np.empty(pred._m, dtype=np.float64)
        ci_lower = self._lower = np.empty(pred._m, dtype=np.float64)
        t_dist = qt(1 - (1 - coverage) / 2, pred._df)
        for i in range(pred._m):
            limit = pred_se_fit[i] * t_dist
            fit = pred_fit[i]
            ci_fit[i] = fit
            ci_upper[i] = fit + limit
            ci_lower[i] = fit - limit
        self.m = pred.m

    def __init__(self, pred, alpha):
        # For documentation
        pass

    @property
    def fit(self):
        """
        :class:`~numpy.ndarray` - Predicted values
        """
        return np.array(self._fit)

    @property
    def upper(self):
        """
        :class:`~numpy.ndarray` - Upper bounds of the confidence intervals
        """
        return np.array(self._upper)

    @property
    def lower(self):
        """
        :class:`~numpy.ndarray` - Lower bounds of the confidence intervals
        """
        return np.array(self._lower)


cdef class loess_prediction:
    """
    Class for loess prediction results

    Holds the predicted values and standard errors of a loess object

    Parameters
    ----------
    newdata : ndarray[m, p]
        Independent variables where the surface must be estimated,
        with m the number of new data points, and p the number of
        independent variables.
    loess : loess.loess
        Loess object that has been successfully fitted,
        i.e `loess.fit` has been called and it returned without
        any errors.
    stderror : boolean
        Whether the standard error should be computed
    """
    cdef double[::1] _fit
    cdef double[::1] _se_fit
    cdef int _se
    cdef int _m
    cdef double _residual_scale
    cdef double _df
    cdef readonly allocated

    def __cinit__(self, newdata, loess loess, stderror=False):
        cdef double[::1] p_dat
        cdef int se
        cdef loess_inputs inputs = loess.inputs
        cdef loess_model model = loess.model
        cdef loess_control control = loess.control
        cdef loess_outputs outputs = loess.outputs
        cdef loess_kd_tree kd_tree = loess.kd_tree
        cdef double[::1] fit, se_fit

        self.allocated = False

        # Note : we need a copy as we may have to normalize
        p_ndr = np.array(newdata, copy=True, subok=True, order='C')
        p_ndr = p_ndr.astype(float)

        # Dimensions should match those of the input
        if p_ndr.size == 0 or p_ndr.ndim == 0:
            raise ValueError("Can't predict without input data !")

        if p_ndr.ndim > 2:
            raise ValueError("New data has more than 2 dimensions.")

        _p = 1 if p_ndr.ndim == 1 else p_ndr.shape[1]
        if _p != loess.inputs.p:
            msg = ("Incompatible data size: there should be as many "
                   "columns as parameters. Got %d instead of %d "
                   "parameters" % (_p, loess.inputs.p))
            raise ValueError(msg)

        se = 1 if stderror else 0
        m = len(p_ndr)
        p_dat = np.ascontiguousarray(p_ndr.ravel(), dtype=np.float64)

        # `predict_setup` in predict.c
        self._m = m
        self._se = se
        fit = self._fit = np.zeros(m, dtype=np.float64)
        se_fit = self._se_fit = np.zeros(m if se else 0, dtype=np.float64)
        self._residual_scale = outputs._residual_scale
        self._df = (outputs._one_delta * outputs._one_delta) / \
            outputs._two_delta
        self.allocated = True

        # `predict` in predict.c
        reset_error()
        if inputs._p > 8:
            ehg182(101)
        else:
            pred_(&inputs._y[0], &inputs._x[0], &p_dat[0], inputs._p,
                  inputs._n, m, outputs._residual_scale, &inputs._weights[0],
                  &outputs._robust[0], model._span, model._degree,
                  &model._parametric[0], &model._drop_square[0],
                  control._surface == 'direct', control._cell,
                  model._family == 'gaussian', &kd_tree._parameter[0],
                  &kd_tree._a[0], &kd_tree._xi[0], &kd_tree._vert[0],
                  &kd_tree._vval[0], &outputs._divisor[0], se, &fit[0],
                  &se_fit[0])

        if error_status:
            raise ValueError(<bytes> error_message)

    def __init__(self, newdata, loess, stderror=False):
        # For documentation
        pass

    @property
    def values(self):
        """
        :class:`~numpy.ndarray` - loess values evaluated at newdata,
        shape ``(m,)``
        """
        return np.array(self._fit)

    @property
    def stderr(self):
        """
        :class:`~numpy.ndarray` - Estimates of the standard error on
        the estimated values, shape ``(m,)``

        Raises `ValueError` if the standard error was not computed.
        """
        if not self._se:
            raise ValueError("Standard error was not computed."
                             "Use 'stderror=True' when predicting.")
        return np.array(self._se_fit)

    @property
    def residual_scale(self):
        """
        :class:`float` - Estimate of the scale of the residuals
        """
        return self._residual_scale

    @property
    def df(self):
        """
        :class:`float` - Degrees of freedom of the loess fit.

        It is used with the t-distribution to compute pointwise
        confidence intervals for the evaluated surface. It is
        obtained using the formula ``(one_delta ** 2) / two_delta``.
        """
        return self._df

    @property
    def m(self):
        """
        :class:`int` - Number of observations in the new data points
        """
        return self._m

    def confidence(self, alpha=0.05):
        """
        Returns the pointwise confidence intervals

        Parameters
        ----------
        alpha : float
            The alpha level for the confidence interval. The
            default ``alpha=0.05`` returns a 95% confidence
            interval. Therefore it must be in the range (0, 1).

        Returns
        -------
        out : loess_confidence_intervals
            Confidence intervals object. It has attributes `fit`,
            `lower` and `upper`
        """
        return loess_confidence_intervals(self, alpha)

    def __str__(self):
        try:
            stderr = "Predicted std error   : %s\n" % self.stderr
        except ValueError:
            stderr = ""

        strg = ["Outputs",
                "-------",
                "Predicted values      : %s\n" % self.values,
                stderr,
                "Residual scale        : %s" % self.residual_scale,
                "Degrees of freedom    : %s" % self.df,
                ]
        return '\n'.join(strg)


def _new_loess(init_arguments, fit):
    """
    Create the loess object from initial arguments

    Used for pickling
    """
    # Generate the object from the initialising arguments
    # Run the fit method if it was run before
    l = loess(**init_arguments)
    if fit:
        l.fit()
    return l


cdef class loess:
    """
    Locally-weighted regression

    A loess object is initialized with the combined parameters of
    :class:`loess_inputs`, :class:`loess_model` and
    :class:`loess_control`. The parameters of :class:`loess_inputs`
    i.e ``x``, ``y`` and ``weights`` can be positional in that order.
    In the descriptions below, `n` is the number of observations,
    and `p` is the number of predictor variables.

    Parameters
    ----------
    x : ndarray[n, p]
        n independent observations for p no. of variables
    y : ndarray[n,]
        A (n,) ndarray of response observations
    weights : ndarray[n] or None
        Weights to be given to individual observations
        in the sum of squared residuals that forms the local fitting
        criterion. If not None, the weights should be non negative. If
        the different observations have non-equal variances, the weights
        should be inversely proportional to the variances. By default,
        an unweighted fit is carried out (all the weights are one).
    **options : dict
        The parameters of :class:`loess_model` and
        :class:`loess_control`.

    Attributes
    ----------
    inputs : :class:`loess_inputs`
        Object that handles the inputs
    model : :class:`loess_model`
        Object that handles the model
    control : :class:`loess_control`
        Object that holds the control parameters
    kd_tree : :class:`loess_kdtree`
        Object that holds the parameters and structures used
        internally by the regression algorithm.
    outputs : :class:`loess_outputs`
        Object that holds the output values and parameters.
        These should be read after :meth:`loess.fit` has been
        called.

    Note
    ----
    Loess smoothing creates large state, so pickling a fitted loess
    object does not save the state. It saves the `__init__` parameters
    from which the state is recreated (refitting) when unpickled. So
    pickling and unpickling does not save on computation time.
    """
    cdef readonly loess_inputs inputs
    cdef readonly loess_model model
    cdef readonly loess_control control
    cdef readonly loess_kd_tree kd_tree
    cdef readonly loess_outputs outputs

    # save all the arguments to enable pickling
    cdef object _init_arguments

    def __init__(self, object x, object y, object weights=None, **options):
        self._init_arguments = {
            "x": x, "y": y, "weights": weights, **options
        }

        # Process options
        model_options = {}
        control_options= {}
        for (k, v) in options.items():
            if k in ('family', 'span', 'degree', 'normalize',
                     'parametric', 'drop_square',):
                model_options[k] = v
            elif k in ('surface', 'statistics', 'trace_hat',
                       'iterations', 'cell'):
                control_options[k] = v

        # Initialize the inputs
        self.inputs = loess_inputs(x, y, weights)

        n = self.inputs.n
        p = self.inputs.p

        # Initialize the control parameters
        self.control = loess_control(**control_options)

        # Initialize the model parameters
        self.model = loess_model(p, **model_options)

        # Initialize the outputs
        self.outputs = loess_outputs(n, p, self.model.family)

        # Initialize the kd tree
        self.kd_tree = loess_kd_tree(n, p)

    def fit(self):
        """
        Computes the loess parameters on the current inputs and
        sets of parameters.
        """
        cdef loess_inputs inputs = self.inputs
        cdef loess_model model = self.model
        cdef loess_control control = self.control
        cdef loess_outputs outputs = self.outputs
        cdef loess_kd_tree kd_tree = self.kd_tree
        cdef int iterations

        # `loess_fit` in loess.c
        reset_error()
        iterations = 0 if model._family == 'gaussian' else \
            control._iterations
        if control._trace_hat == 'wait.to.decide':
            if control._surface == 'interpolate':
                control._trace_hat = 'exact' if inputs._n < 500 else \
                    'approximate'
            else:
                control._trace_hat = 'exact'
        if inputs._p > 8:
            # The original reports this error, then overflows fixed-size
            # arrays
            ehg182(101)
        else:
            loess_(&inputs._y[0], &inputs._x[0], inputs._p, inputs._n,
                   &inputs._weights[0], model._span, model._degree,
                   &model._parametric[0], &model._drop_square[0],
                   model._normalize, control._statistics == 'exact',
                   control._surface == 'direct', control._cell,
                   control._trace_hat == 'exact', iterations,
                   &outputs._fitted_values[0], &outputs._fitted_residuals[0],
                   &outputs._enp, &outputs._residual_scale,
                   &outputs._one_delta, &outputs._two_delta,
                   &outputs._pseudovalues[0], &outputs._trace_hat,
                   &outputs._diagonal[0], &outputs._robust[0],
                   &outputs._divisor[0], &kd_tree._parameter[0],
                   &kd_tree._a[0], &kd_tree._xi[0], &kd_tree._vert[0],
                   &kd_tree._vval[0])
        self.outputs.activated = True
        if error_status:
            raise ValueError(<bytes> error_message)
        return

    def input_summary(self):
        """
        Returns some generic information about the loess parameters.
        """
        toprint = [str(self.model), str(self.control)]
        return "\n\n".join(toprint)

    def output_summary(self):
        """Returns some generic information about the loess fit."""
        fit_flag = bool(self.outputs.activated)

        if self.model.family == "gaussian":
            rse = ("Residual Standard Error        : %.4f" %
                   self.outputs.residual_scale)
        else:
            rse = ("Residual Scale Estimate        : %.4f" %
                   self.outputs.residual_scale)

        strg = ["Output Summary",
                "--------------",
                "Number of Observations         : %d" % self.inputs.n,
                "Fit flag                       : %d" % fit_flag,
                "Equivalent Number of Parameters: %.1f" % self.outputs.enp,
                rse
                ]
        return '\n'.join(strg)

    def predict(self, newdata, stderror=False):
        """
        Compute loess estimates at the given new data points newdata.

        Parameters
        ----------
        newdata : ndarray[m, p]
            Independent variables where the surface must be estimated,
            with m the number of new data points, and p the number of
            independent variables.
        stderror : boolean
            Whether the standard error should be computed

        Returns
        -------
        A :class:`loess_prediction` object.
        """
        # Make sure there's been a fit earlier
        if self.outputs.activated == 0:
            self.fit()

        return loess_prediction(newdata, self, stderror)

    # Pickling support
    def __reduce__(self):
        return (_new_loess, (self._init_arguments, self.outputs.activated))


cdef class loess_anova:
    """
    Analysis of variance for two loess objects

    Parameters
    ----------
    loess_one : loess.loess
        First loess object
    loess_two : loess.loess
        Second loess object

    Attributes
    ----------
    F_value : float
        Value of the F-statistic
    Pr_F : float
        Probability of getting a value as large as the
        `F_value`. The is the p-value of the F-statistic.
    """
    cdef readonly double dfn, dfd, F_value, Pr_F

    # Like the original, raise ZeroDivisionError on division by zero
    @cython.cdivision(False)
    def __init__(self, loess_one, loess_two):
        cdef double one_d1, one_d2, one_s, two_d1, two_d2, two_s
        cdef double rssdiff, d1diff, tmp, df1, df2

        if (not isinstance(loess_one, loess) or
                not isinstance(loess_two, loess)):
            raise ValueError("Arguments should be valid loess objects!"
                             "got '%s' instead" % type(loess_one))

        out_one = loess_one.outputs
        out_two = loess_two.outputs

        one_d1 = out_one.one_delta
        one_d2 = out_one.two_delta
        one_s = out_one.residual_scale

        two_d1 = out_two.one_delta
        two_d2 = out_two.two_delta
        two_s = out_two.residual_scale

        rssdiff = abs(one_s * one_s * one_d1 - two_s * two_s * two_d1)
        d1diff = abs(one_d1 - two_d1)
        self.dfn = d1diff * d1diff / abs(one_d2 - two_d2)
        df1 = self.dfn

        if out_one.enp > out_two.enp:
            self.dfd = one_d1 * one_d1 / one_d2
            tmp = one_s
        else:
            self.dfd = two_d1 * two_d1 / two_d2
            tmp = two_s
        df2 = self.dfd
        self.F_value = (rssdiff / d1diff) / (tmp * tmp)
        self.Pr_F = 1. - pf(self.F_value, df1, df2)
