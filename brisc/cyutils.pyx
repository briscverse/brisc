# Cython utility functions, many of which underlie utils.py

import numpy as np
import threading
cimport numpy as np
np.import_array()
from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport parallel, prange, threadid
from libc.limits cimport UINT_MAX
from libc.math cimport M_PI
from libc.string cimport memcpy
from libcpp.algorithm cimport lower_bound, sort
from libcpp.cmath cimport sqrt, log, cos
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from scipy.linalg.cython_blas cimport sgemm as sgemm_, sgemv as sgemv_
from signal import set_wakeup_fd
from socket import socketpair
from .cyutils cimport atomic_or, bit_width, integer, numeric, recv, \
    signed_integer, uninitialized_vector


ctypedef unsigned unsigned_  # hack to get templates to work with unsigned


def bin_count(const integer[::1] arr,
              unsigned[::1] counts,
              unsigned num_threads):
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
        with nogil, parallel(num_threads=num_threads):
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


def concatenate_dense(list arrays, const unsigned num_threads):
    cdef unsigned array_index, thread_index, chunk_size = 64, \
        itemsize = arrays[0].itemsize, num_arrays = len(arrays)
    cdef unsigned long long bytes_copied, bytes_to_copy, num_bytes, \
        num_chunks, start_chunk, end_chunk, current_chunk, array_offset, \
        prev_chunk, total_bytes = 0, total_chunks = 0, total_rows = 0
    cdef uninitialized_vector[unsigned long long] byte_count_buffer, \
        chunk_count_buffer, byte_offset_buffer, chunk_offset_buffer
    cdef uninitialized_vector[char*] array_pointers
    cdef unsigned long long[::1] byte_count, chunk_count, byte_offset, \
        chunk_offset
    cdef char* output_pointer
    cdef char* src
    cdef char* dest
    cdef np.ndarray array, output
    cdef int read_fd, old_fd
    cdef bint interrupted = False, \
        main_thread = threading.current_thread() is threading.main_thread()
    cdef char signal_byte

    if num_threads == 1:
        # Cache relevant information from each array:
        array_pointers.resize(num_arrays)
        byte_count_buffer.resize(num_arrays)
        byte_count = <unsigned long long[:num_arrays]> byte_count_buffer.data()

        for array_index in range(num_arrays):
            array = arrays[array_index]

            # 1) the pointer to each array's data
            array_pointers[array_index] = <char*> np.PyArray_DATA(array)

            # 2) the number of bytes per array
            byte_count[array_index] = array.size * itemsize

            # 3) the total number of rows across all arrays
            total_rows += array.shape[0]

        # Create the output array and get a pointer to its data
        if array.ndim == 1:
            output = np.empty(total_rows, dtype=array.dtype)
        else:
            output = np.empty((total_rows, array.shape[1]),
                              dtype=array.dtype)
        output_pointer = <char*> np.PyArray_DATA(output)

        # Copy the data from each input array to the output array, in 32-MB
        # chunks to allow interrupt checks
        for array_index in range(num_arrays):
            bytes_copied = 0
            while bytes_copied < byte_count[array_index]:
                bytes_to_copy = byte_count[array_index] - bytes_copied
                if bytes_to_copy > 33_554_432:  # 32 MB
                    bytes_to_copy = 33_554_432
                memcpy(output_pointer + bytes_copied,
                       array_pointers[array_index] + bytes_copied,
                       bytes_to_copy)
                bytes_copied += bytes_to_copy
                PyErr_CheckSignals()
            output_pointer += byte_count[array_index]
    else:
        # Cache relevant information from each array:
        array_pointers.resize(num_arrays)
        byte_count_buffer.resize(num_arrays)
        byte_count = \
            <unsigned long long[:num_arrays]> byte_count_buffer.data()
        chunk_count_buffer.resize(num_arrays)
        chunk_count = \
            <unsigned long long[:num_arrays]> chunk_count_buffer.data()
        byte_offset_buffer.resize(num_arrays)
        byte_offset = \
            <unsigned long long[:num_arrays]> byte_offset_buffer.data()
        chunk_offset_buffer.resize(num_arrays)
        chunk_offset = \
            <unsigned long long[:num_arrays]> chunk_offset_buffer.data()

        for array_index in range(num_arrays):
            array = arrays[array_index]

            # 1) the pointer to each array's data
            array_pointers[array_index] = <char*> np.PyArray_DATA(array)

            # 2) the number of bytes per array
            num_bytes = array.size * itemsize
            byte_count[array_index] = num_bytes

            # 3) the number of `chunk_size`-byte chunks per array, rounding up
            num_chunks = (num_bytes + chunk_size - 1) / chunk_size
            chunk_count[array_index] = num_chunks

            # 4) the total number of bytes and chunks up to the start of each
            # array
            byte_offset[array_index] = total_bytes
            chunk_offset[array_index] = total_chunks

            # 5) the total number of bytes, chunks, and rows across all arrays
            total_bytes += num_bytes
            total_chunks += num_chunks
            total_rows += array.shape[0]

        # Ensure each thread has at least one chunk to work on
        num_threads = min(num_threads, total_chunks)

        # Create the output array and get a pointer to its data
        if array.ndim == 1:
            output = np.empty(total_rows, dtype=array.dtype)
        else:
            output = np.empty((total_rows, array.shape[1]),
                              dtype=array.dtype)
        output_pointer = <char*> np.PyArray_DATA(output)

        # Distribute threads across arrays, proportional to their number of
        # chunks. Each thread operates on a contiguous range of chunks, which
        # may span multiple arrays.
        r, w = socketpair()
        r.setblocking(False)
        w.setblocking(False)
        read_fd = r.fileno()
        if main_thread:
            old_fd = set_wakeup_fd(w.fileno())
        try:
            with nogil, parallel(num_threads=num_threads):
                # Get the thread index
                thread_index = threadid()

                # Calculate the range of chunks this thread is responsible for
                start_chunk = (thread_index * total_chunks) / num_threads
                end_chunk = ((thread_index + 1) * total_chunks) / num_threads

                # Find the (first) array that corresponds to this thread's
                # chunk range, using a simple linear scan
                array_index = 0
                while chunk_offset[array_index] + \
                        chunk_count[array_index] <= start_chunk:
                    array_index = array_index + 1

                # Copy data from the input arrays to the output array
                current_chunk = start_chunk
                while current_chunk < end_chunk:
                    # Calculate the chunk offset within the current array
                    array_offset = current_chunk - chunk_offset[array_index]

                    # Calculate the source and destination pointers
                    src = \
                        array_pointers[array_index] + array_offset * chunk_size
                    dest = output_pointer + byte_offset[array_index] + \
                        array_offset * chunk_size

                    # Calculate the number of chunks to copy from the current
                    # array: min(remaining in array, remaining in thread)
                    num_chunks = min(chunk_count[array_index] - array_offset,
                                     end_chunk - current_chunk)

                    # Get the number of bytes to copy from the current array:
                    # min(num_chunks * chunk_size, remaining bytes in array)
                    num_bytes = min(num_chunks * chunk_size,
                                    byte_count[array_index] -
                                    array_offset * chunk_size)

                    # Copy the data
                    memcpy(dest, src, num_bytes)

                    # Update the position and array index
                    prev_chunk = current_chunk
                    current_chunk = current_chunk + num_chunks
                    array_index = array_index + 1

                    # Check for interrupts after every 32 MB copied
                    if (prev_chunk >> 19) != (current_chunk >> 19):
                        if recv(read_fd, &signal_byte, 1, 0) > 0:
                            atomic_or(interrupted, True)
                        if interrupted:
                            with gil:
                                return

        finally:
            if main_thread:
                set_wakeup_fd(old_fd)
            r.close()
            w.close()

    return output


def concatenate_indptrs_int32(list arrays, const unsigned num_threads):
    # The int32 version of `concatenate_indptrs_int64()`. We can safely use
    # `unsigned` rather than `unsigned long long` for everything.

    cdef unsigned array_index, num_elements, offset, num_chunks, \
        thread_index, start_chunk, end_chunk, current_chunk, array_offset, i, \
        chunk_size = 16, num_arrays = len(arrays), total_elements = 0, \
        total_chunks = 0, total_nnz = 0
    cdef uninitialized_vector[unsigned] element_count_buffer, \
        chunk_count_buffer, element_offset_buffer, chunk_offset_buffer, \
        nnz_offset_buffer
    cdef uninitialized_vector[int*] array_pointers
    cdef unsigned[::1] element_count, chunk_count, element_offset, \
        chunk_offset, nnz_offset
    cdef int* output_pointer
    cdef int* array_pointer
    cdef int* src
    cdef int* dest
    cdef np.ndarray array, output
    cdef np.npy_intp total_elements_plus_one

    if num_threads == 1:
        # Cache relevant information from each array:
        array_pointers.resize(num_arrays)
        element_count_buffer.resize(num_arrays)
        element_count = \
            <unsigned[:num_arrays]> element_count_buffer.data()
        element_offset_buffer.resize(num_arrays)
        element_offset = \
            <unsigned[:num_arrays]> element_offset_buffer.data()
        nnz_offset_buffer.resize(num_arrays)
        nnz_offset = <unsigned[:num_arrays]> nnz_offset_buffer.data()

        for array_index in range(num_arrays):
            array = arrays[array_index]

            # 1) the pointer to each array's data, after the first 0
            array_pointers[array_index] = (<int*> np.PyArray_DATA(array)) + 1

            # 2) the number of elements per array (for the entirety of this
            # function, by "elements", we mean elements after the first 0)
            num_elements = array.shape[0] - 1
            element_count[array_index] = num_elements

            # 3) the total number of nnz up to the start of each array
            nnz_offset[array_index] = total_nnz

            # 4) the total number of elements and nnz across all arrays
            total_elements += num_elements
            total_nnz += array[num_elements]

        # Create the output indptr and get a pointer to its data after the
        # first element, which we set to 0
        # output = np.empty(total_elements + 1, dtype=np.int32)
        total_elements_plus_one = total_elements + 1
        output = np.PyArray_EMPTY(1, &total_elements_plus_one, np.NPY_INT32, 0)
        output_pointer = <int*> np.PyArray_DATA(output)
        output_pointer[0] = 0
        output_pointer += 1

        # Copy the first indptr's data
        num_elements = element_count[0]
        memcpy(output_pointer, array_pointers[0],
               num_elements * sizeof(int))
        output_pointer += num_elements

        # Copy the remaining indptrs' data, adding the nnz offset
        for array_index in range(1, num_arrays):
            num_elements = element_count[array_index]
            offset = nnz_offset[array_index]
            array_pointer = array_pointers[array_index]
            for i in range(num_elements):
                output_pointer[i] = array_pointer[i] + offset
            output_pointer += num_elements
    else:
        # Cache relevant information from each array:
        array_pointers.resize(num_arrays)
        element_count_buffer.resize(num_arrays)
        element_count = \
            <unsigned[:num_arrays]> element_count_buffer.data()
        chunk_count_buffer.resize(num_arrays)
        chunk_count = <unsigned[:num_arrays]> chunk_count_buffer.data()
        element_offset_buffer.resize(num_arrays)
        element_offset = \
            <unsigned[:num_arrays]> element_offset_buffer.data()
        chunk_offset_buffer.resize(num_arrays)
        chunk_offset = <unsigned[:num_arrays]> chunk_offset_buffer.data()
        nnz_offset_buffer.resize(num_arrays)
        nnz_offset = <unsigned[:num_arrays]> nnz_offset_buffer.data()

        for array_index in range(num_arrays):
            array = arrays[array_index]

            # 1) the pointer to each array's data, after the first 0
            array_pointers[array_index] = (<int*> np.PyArray_DATA(array)) + 1

            # 2) the number of elements per array (for the entirety of this
            # function, by "elements", we mean elements after the first 0)
            num_elements = array.shape[0] - 1
            element_count[array_index] = num_elements

            # 3) the number of `chunk_size`-element chunks per array, rounding
            # up
            num_chunks = (num_elements + chunk_size - 1) / chunk_size
            chunk_count[array_index] = num_chunks

            # 4) the total number of elements, chunks, and nnz up to the start
            # of each array
            element_offset[array_index] = total_elements
            chunk_offset[array_index] = total_chunks
            nnz_offset[array_index] = total_nnz

            # 5) the total number of elements, chunks, and nnz across all
            # arrays
            total_elements += num_elements
            total_chunks += num_chunks
            total_nnz += array[num_elements]

        # Ensure each thread has at least one chunk to work on
        num_threads = min(num_threads, total_chunks)

        # Create the output indptr and get a pointer to its data after the
        # first element, which we set to 0
        # output = np.empty(total_elements + 1, dtype=np.int32)
        total_elements_plus_one = total_elements + 1
        output = np.PyArray_EMPTY(1, &total_elements_plus_one, np.NPY_INT32, 0)
        output_pointer = <int*> np.PyArray_DATA(output)
        output_pointer[0] = 0
        output_pointer += 1

        # Distribute threads across arrays, proportional to their number of
        # chunks. Each thread operates on a contiguous range of chunks, which
        # may span multiple arrays.
        with nogil, parallel(num_threads=num_threads):
            # Get the thread index
            thread_index = threadid()

            # Calculate the range of chunks this thread is responsible for
            start_chunk = (thread_index * total_chunks) / num_threads
            end_chunk = ((thread_index + 1) * total_chunks) / num_threads

            # Find the (first) array that corresponds to this thread's chunk
            # range, using a simple linear scan
            array_index = 0
            while chunk_offset[array_index] + \
                    chunk_count[array_index] <= start_chunk:
                array_index = array_index + 1

            # Copy data from the input indptrs to the output indptr
            current_chunk = start_chunk
            while current_chunk < end_chunk:
                # Calculate the chunk offset within the current array
                array_offset = current_chunk - chunk_offset[array_index]

                # Calculate the source and destination pointers
                src = array_pointers[array_index] + array_offset * chunk_size
                dest = output_pointer + element_offset[array_index] + \
                    array_offset * chunk_size

                # Calculate the number of chunks to copy from the current
                # array: min(remaining in array, remaining in thread)
                num_chunks = min(chunk_count[array_index] - array_offset,
                                 end_chunk - current_chunk)

                # Get the number of elements to copy from the current array:
                # min(num_chunks * chunk_size, remaining elements in array)
                num_elements = min(num_chunks * chunk_size,
                                   element_count[array_index] -
                                   array_offset * chunk_size)

                # Copy the data, adding the nnz offset for all indptrs but the
                # first
                if array_index == 0:
                    memcpy(dest, src, num_elements * sizeof(int))
                else:
                    offset = nnz_offset[array_index]
                    for i in range(num_elements):
                        dest[i] = src[i] + offset

                # Update the position and array index
                current_chunk = current_chunk + num_chunks
                array_index = array_index + 1

    return output


def concatenate_indptrs_int64(list arrays, const unsigned num_threads):
    # Similar to `concatenate()`, but instead of a straight copy, skip the
    # first element and add the cumulative nnz while copying, for every indptr
    # but the first. This means we can't just use bytes and generic `char*`
    # pointers, we need to use elements and `long long*` pointers. So our chunk
    # size is now 8 elements, not 64 bytes.

    cdef unsigned array_index, thread_index, chunk_size = 8, \
        num_arrays = len(arrays)
    cdef unsigned long long num_elements, offset, num_chunks, start_chunk, \
        end_chunk, current_chunk, array_offset, i, total_elements = 0, \
        total_chunks = 0, total_nnz = 0
    cdef uninitialized_vector[unsigned long long] element_count_buffer, \
        chunk_count_buffer, element_offset_buffer, chunk_offset_buffer, \
        nnz_offset_buffer
    cdef uninitialized_vector[long long*] array_pointers
    cdef unsigned long long[::1] element_count, chunk_count, element_offset, \
        chunk_offset, nnz_offset
    cdef long long* output_pointer
    cdef long long* array_pointer
    cdef long long* src
    cdef long long* dest
    cdef np.ndarray array, output
    cdef np.npy_intp total_elements_plus_one

    if num_threads == 1:
        # Cache relevant information from each array:
        array_pointers.resize(num_arrays)
        element_count_buffer.resize(num_arrays)
        element_count = \
            <unsigned long long[:num_arrays]> element_count_buffer.data()
        element_offset_buffer.resize(num_arrays)
        element_offset = \
            <unsigned long long[:num_arrays]> element_offset_buffer.data()
        nnz_offset_buffer.resize(num_arrays)
        nnz_offset = <unsigned long long[:num_arrays]> nnz_offset_buffer.data()

        for array_index in range(num_arrays):
            array = arrays[array_index]

            # 1) the pointer to each array's data, after the first 0
            array_pointers[array_index] = \
                (<long long*> np.PyArray_DATA(array)) + 1

            # 2) the number of elements per array (for the entirety of this
            # function, by "elements", we mean elements after the first 0)
            num_elements = array.shape[0] - 1
            element_count[array_index] = num_elements

            # 3) the total number of nnz up to the start of each array
            nnz_offset[array_index] = total_nnz

            # 4) the total number of elements and nnz across all arrays
            total_elements += num_elements
            total_nnz += array[num_elements]

        # Create the output indptr and get a pointer to its data after the
        # first element, which we set to 0
        # output = np.empty(total_elements + 1, dtype=np.int64)
        total_elements_plus_one = total_elements + 1
        output = np.PyArray_EMPTY(1, &total_elements_plus_one, np.NPY_INT64, 0)
        output_pointer = <long long*> np.PyArray_DATA(output)
        output_pointer[0] = 0
        output_pointer += 1

        # Copy the first indptr's data
        num_elements = element_count[0]
        memcpy(output_pointer, array_pointers[0],
               num_elements * sizeof(long long))
        output_pointer += num_elements

        # Copy the remaining indptrs' data, adding the nnz offset
        for array_index in range(1, num_arrays):
            num_elements = element_count[array_index]
            offset = nnz_offset[array_index]
            array_pointer = array_pointers[array_index]
            for i in range(num_elements):
                output_pointer[i] = array_pointer[i] + offset
            output_pointer += num_elements
    else:
        # Cache relevant information from each array:
        array_pointers.resize(num_arrays)
        element_count_buffer.resize(num_arrays)
        element_count = \
            <unsigned long long[:num_arrays]> element_count_buffer.data()
        chunk_count_buffer.resize(num_arrays)
        chunk_count = \
            <unsigned long long[:num_arrays]> chunk_count_buffer.data()
        element_offset_buffer.resize(num_arrays)
        element_offset = \
            <unsigned long long[:num_arrays]> element_offset_buffer.data()
        chunk_offset_buffer.resize(num_arrays)
        chunk_offset = \
            <unsigned long long[:num_arrays]> chunk_offset_buffer.data()
        nnz_offset_buffer.resize(num_arrays)
        nnz_offset = \
            <unsigned long long[:num_arrays]> nnz_offset_buffer.data()

        for array_index in range(num_arrays):
            array = arrays[array_index]

            # 1) the pointer to each array's data, after the first 0
            array_pointers[array_index] = \
                (<long long*> np.PyArray_DATA(array)) + 1

            # 2) the number of elements per array (for the entirety of this
            # function, by "elements", we mean elements after the first 0)
            num_elements = array.shape[0] - 1
            element_count[array_index] = num_elements

            # 3) the number of `chunk_size`-element chunks per array, rounding
            # up
            num_chunks = (num_elements + chunk_size - 1) / chunk_size
            chunk_count[array_index] = num_chunks

            # 4) the total number of elements, chunks, and nnz up to the start
            # of each array
            element_offset[array_index] = total_elements
            chunk_offset[array_index] = total_chunks
            nnz_offset[array_index] = total_nnz

            # 5) the total number of elements, chunks, and nnz across all
            # arrays
            total_elements += num_elements
            total_chunks += num_chunks
            total_nnz += array[num_elements]

        # Ensure each thread has at least one chunk to work on
        num_threads = min(num_threads, total_chunks)

        # Create the output indptr and get a pointer to its data after the
        # first
        # element, which we set to 0
        # output = np.empty(total_elements + 1, dtype=np.int64)
        total_elements_plus_one = total_elements + 1
        output = np.PyArray_EMPTY(1, &total_elements_plus_one, np.NPY_INT64, 0)
        output_pointer = <long long*> np.PyArray_DATA(output)
        output_pointer[0] = 0
        output_pointer += 1

        # Distribute threads across arrays, proportional to their number of
        # chunks. Each thread operates on a contiguous range of chunks, which
        # may span multiple arrays.
        with nogil, parallel(num_threads=num_threads):
            # Get the thread index
            thread_index = threadid()

            # Calculate the range of chunks this thread is responsible for
            start_chunk = (thread_index * total_chunks) / num_threads
            end_chunk = ((thread_index + 1) * total_chunks) / num_threads

            # Find the (first) array that corresponds to this thread's chunk
            # range, using a simple linear scan
            array_index = 0
            while chunk_offset[array_index] + \
                    chunk_count[array_index] <= start_chunk:
                array_index = array_index + 1

            # Copy data from the input indptrs to the output indptr
            current_chunk = start_chunk
            while current_chunk < end_chunk:
                # Calculate the chunk offset within the current array
                array_offset = current_chunk - chunk_offset[array_index]

                # Calculate the source and destination pointers
                src = array_pointers[array_index] + array_offset * chunk_size
                dest = output_pointer + element_offset[array_index] + \
                    array_offset * chunk_size

                # Calculate the number of chunks to copy from the current
                # array: min(remaining in array, remaining in thread)
                num_chunks = min(chunk_count[array_index] - array_offset,
                                 end_chunk - current_chunk)

                # Get the number of elements to copy from the current array:
                # min(num_chunks * chunk_size, remaining elements in array)
                num_elements = min(num_chunks * chunk_size,
                                   element_count[array_index] -
                                   array_offset * chunk_size)

                # Copy the data, adding the nnz offset for all indptrs but the
                # first
                if array_index == 0:
                    memcpy(dest, src, num_elements * sizeof(long long))
                else:
                    offset = nnz_offset[array_index]
                    for i in range(num_elements):
                        dest[i] = src[i] + offset

                # Update the position and array index
                current_chunk = current_chunk + num_chunks
                array_index = array_index + 1

    return output


def csr_hstack(list arrays,
               const signed_integer[::1] offsets,
               bit_width[::1] data,
               signed_integer[::1] indices,
               signed_integer[::1] indptr,
               const unsigned num_major,
               const unsigned num_threads):
    # Closely based on SciPy's `csr_hstack()` function from sparsetools/csr.h

    cdef unsigned array_index, i, num_arrays = len(arrays)
    cdef np.ndarray array_data, array_indices, array_indptr
    cdef signed_integer total_nnz, start, end, nnz, offset, output_offset, j
    cdef uninitialized_vector[bit_width*] data_pointers
    cdef uninitialized_vector[signed_integer*] indices_pointers, \
        indptr_pointers
    indptr_pointers.resize(num_arrays)
    indices_pointers.resize(num_arrays)
    data_pointers.resize(num_arrays)

    # Get pointers to the start of each sparse array's `data`,
    # `indices` and `indptr`
    for array_index in range(num_arrays):
        array_data = arrays[array_index].data
        array_indices = arrays[array_index].indices
        array_indptr = arrays[array_index].indptr
        data_pointers[array_index] = <bit_width*> np.PyArray_DATA(array_data)
        indices_pointers[array_index] = \
            <signed_integer*> np.PyArray_DATA(array_indices)
        indptr_pointers[array_index] = \
            <signed_integer*> np.PyArray_DATA(array_indptr)

    if num_threads == 1:
        # Populate `data`, `indices`, and `indptr`
        indptr[0] = 0
        total_nnz = 0
        for i in range(num_major):
            # First iteration (array_index == 0)
            start = indptr_pointers[0][i]
            end = indptr_pointers[0][i + 1]
            nnz = end - start

            # Copy data
            memcpy(&data[total_nnz], &data_pointers[0][start],
                   nnz * sizeof(bit_width))

            # Copy indices (no offset needed)
            memcpy(&indices[total_nnz], &indices_pointers[0][start],
                   nnz * sizeof(signed_integer))

            total_nnz += nnz

            # Remaining iterations (array_index > 0)
            for array_index in range(1, num_arrays):
                start = indptr_pointers[array_index][i]
                end = indptr_pointers[array_index][i + 1]
                nnz = end - start

                # Copy data
                memcpy(&data[total_nnz],
                       &data_pointers[array_index][start],
                       nnz * sizeof(bit_width))

                # Copy indices and apply offset
                offset = offsets[array_index]
                output_offset = total_nnz - start
                for j in range(start, end):
                    indices[output_offset + j] = \
                        indices_pointers[array_index][j] + offset

                total_nnz += nnz

            indptr[i + 1] = total_nnz
    else:
        # Same as the single-threaded version, but cumsum `indptr` in a
        # separate single-threaded loop, then populate `data` and `indices` in
        # a second parallel loop once the offsets from `indptr` are known.

        with nogil:
            # Populate `indptr`
            indptr[0] = 0
            for i in prange(num_major, num_threads=num_threads):
                total_nnz = 0
                for array_index in range(num_arrays):
                    start = indptr_pointers[array_index][i]
                    end = indptr_pointers[array_index][i + 1]
                    nnz = end - start
                    total_nnz = total_nnz + nnz
                indptr[i + 1] = total_nnz
            for i in range(1, num_major):
                indptr[i + 1] += indptr[i]

            # Populate `data` and `indices`
            for i in prange(num_major, num_threads=num_threads):
                total_nnz = indptr[i]

                # First iteration (array_index == 0)
                start = indptr_pointers[0][i]
                end = indptr_pointers[0][i + 1]
                nnz = end - start

                # Copy data
                memcpy(&data[total_nnz], &data_pointers[0][start],
                       nnz * sizeof(bit_width))

                # Copy indices (no offset needed)
                memcpy(&indices[total_nnz], &indices_pointers[0][start],
                       nnz * sizeof(signed_integer))

                total_nnz = total_nnz + nnz

                # Remaining iterations (array_index > 0)
                for array_index in range(1, num_arrays):
                    start = indptr_pointers[array_index][i]
                    end = indptr_pointers[array_index][i + 1]
                    nnz = end - start

                    # Copy data
                    memcpy(&data[total_nnz],
                           &data_pointers[array_index][start],
                           nnz * sizeof(bit_width))

                    # Copy indices and apply offset
                    offset = offsets[array_index]
                    for j in range(start, end):
                        output_offset = total_nnz - start
                        indices[output_offset + j] = \
                            indices_pointers[array_index][j] + offset

                    total_nnz = total_nnz + nnz


def get_count_at_least_threshold_csr(const numeric[::1] data,
                                     const signed_integer[::1] indptr,
                                     const unsigned threshold,
                                     char[::1] output,
                                     unsigned num_threads):
    cdef unsigned long long i, j, num_major = output.shape[0]
    cdef unsigned thread_index
    cdef numeric count
    cdef pair[unsigned, unsigned] row_range

    num_threads = min(num_threads, num_major)
    if num_threads <= 1:
        for i in range(num_major):
            count = 0
            for j in range(<unsigned long long> indptr[i],
                           <unsigned long long> indptr[i + 1]):
                count += data[j]
            output[i] = count >= <numeric> threshold
    else:
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            row_range = get_thread_offset(indptr, thread_index, num_threads)

            for i in range(row_range.first, row_range.second):
                count = 0
                for j in range(<unsigned long long> indptr[i],
                               <unsigned long long> indptr[i + 1]):
                    count = count + data[j]
                output[i] = count >= <numeric> threshold


def getnnz_csr(const signed_integer[::1] indptr,
               unsigned[::1] output,
               unsigned num_threads):
    cdef unsigned long long i, num_major = output.shape[0]

    num_threads = min(num_threads, num_major)
    if num_threads <= 1:
        for i in range(num_major):
            output[i] = indptr[i + 1] - indptr[i]
    else:
        for i in prange(num_major, nogil=True,
                        num_threads=num_threads):
            output[i] = indptr[i + 1] - indptr[i]


def getnnz_at_least_threshold_csr(
        const signed_integer[::1] indptr,
        const signed_integer threshold,
        char[::1] output,
        unsigned num_threads):
    cdef unsigned long long i, num_major = output.shape[0]

    num_threads = min(num_threads, num_major)
    if num_threads <= 1:
        for i in range(num_major):
            output[i] = (indptr[i + 1] - indptr[i]) >= threshold
    else:
        for i in prange(num_major, nogil=True,
                        num_threads=num_threads):
            output[i] = (indptr[i + 1] - indptr[i]) >= threshold


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


def greater_than_or_equal(
        const integer[::1] nnz,
        const integer threshold,
        char[::1] output,
        unsigned num_threads):
    cdef unsigned long long i, num_major = output.shape[0]

    num_threads = min(num_threads, num_major)
    if num_threads <= 1:
        for i in range(num_major):
            output[i] = nnz[i] >= threshold
    else:
        for i in prange(num_major, nogil=True,
                        num_threads=num_threads):
            output[i] = nnz[i] >= threshold


def has_all_zero_columns(const numeric[:, ::1] X):
    cdef unsigned i, j
    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            if X[i, j]:
                break
        else:
            return True
    return False


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


def parallel_subset_1d_cython(const bit_width[::1] input_arr,
                              const integer[::1] indices,
                              bit_width[::1] output_arr,
                              const unsigned num_threads):
    cdef unsigned i, j, input_row, num_rows = indices.shape[0]

    for i in prange(num_rows, num_threads=num_threads, nogil=True):
        input_row = indices[i]
        output_arr[i] = input_arr[input_row]


def parallel_subset_2d_cython(const bit_width[:, ::1] input_arr,
                              const integer[::1] indices,
                              bit_width[:, ::1] output_arr,
                              const unsigned num_threads):
    cdef unsigned i, j, input_row, num_rows = indices.shape[0], \
        num_cols = input_arr.shape[1]

    for i in prange(num_rows, num_threads=num_threads, nogil=True):
        input_row = indices[i]
        for j in range(num_cols):
            output_arr[i, j] = input_arr[input_row, j]


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


def weighted_bin_count(const integer[::1] arr,
                       const numeric[::1] weights,
                       numeric[::1] counts,
                       unsigned num_threads):
    cdef unsigned long long start, end, i, num_bins, chunk_size, \
        num_elements = arr.shape[0]
    cdef unsigned thread_index
    cdef vector[vector[numeric]] thread_counts
    cdef numeric* counts_pointer

    num_threads = min(num_threads, num_elements)
    if num_threads <= 1:
        counts[:] = 0
        for i in range(num_elements):
            counts[arr[i]] += weights[i]
    else:
        # Store sums for each thread in a temporary buffer, then aggregate at
        # the end. As an optimization, put the sums for the last thread
        # directly into the final `counts` array.
        thread_counts.resize(num_threads - 1)
        num_bins = counts.shape[0]
        chunk_size = (num_elements + num_threads - 1) / num_threads
        with nogil, parallel(num_threads=num_threads):
            thread_index = threadid()
            start = thread_index * chunk_size
            if thread_index == num_threads - 1:
                end = num_elements
                counts[:] = 0
                for i in range(start, end):
                    counts[arr[i]] += weights[i]
            else:
                thread_counts[thread_index].resize(num_bins)
                counts_pointer = thread_counts[thread_index].data()
                end = min(start + chunk_size, num_elements)
                for i in range(start, end):
                    counts_pointer[arr[i]] += weights[i]

        # Aggregate counts from all threads except the last
        for thread_index in range(num_threads - 1):
            counts_pointer = thread_counts[thread_index].data()
            for i in range(num_bins):
                counts[i] += counts_pointer[i]