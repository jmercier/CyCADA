# Copyright (c) 2010-2012 Jean-Pascal Mercier
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#
#
#
cimport cuda.libcuda as libcuda
cimport cycada.core as core

from . import core as cuda

cimport numpy as cnp

import itertools

# ------------------------------------------------------------------------------
cdef class ReadBufferNDArray(core.LLCommand):
    cdef:
        libcuda.CUdeviceptr    __raw_src__
        void *                 __raw_dst__
        unsigned int           __size__

        core.LLBuffer          __src__
        cnp.ndarray            __dst__

    property size:
        def __get__(self):
            return self.__size__

        def __set__(self, new_size):
            if ((self.__src__._size < new_size) or \
                (self.__dst__.dtype.itemsize  * self.__dst__.size < new_size)):
                raise RuntimeError("Invalid Size")
            self.__size__ = new_size

    def __init__(self, cnp.ndarray dst, src, int size = -1):
        """
        """
        cdef core.LLBuffer src_buf = src.__buffer__
        if (size == -1):
            size = src_buf._size

        self.__src__ = src_buf
        self.__dst__ = dst

        self.__raw_dst__ = self.__dst__.data
        self.__raw_src__ = self.__src__.device()

        self.size = size

    cdef libcuda.CUresult __exec__(self, core.LLStream stream):
        """
        """
        return libcuda.cuMemcpyDtoHAsync(self.__raw_dst__, self.__raw_src__, self.__size__, stream._handle)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class WriteBufferNDArray(core.LLCommand):
    cdef :
        libcuda.CUdeviceptr    __raw_dst__
        void *                 __raw_src__
        unsigned int           __size__

        core.LLBuffer           __dst__
        cnp.ndarray             __src__

    property size:
        def __get__(self):
            return self.__size__

        def __set__(self, new_size):
            if ((self.__dst__._size < new_size) or \
                (self.__src__.size * self.__src__.dtype.itemsize < new_size)):
                raise RuntimeError("Invalid Size")
            self.__size__ = new_size


    def __init__(self, dst, cnp.ndarray src, int size = -1):
        """
        """
        cdef core.LLBuffer dst_buf = dst.__buffer__
        if (size == -1):
            size = src.size * src.dtype.itemsize

        self.__src__  = src
        self.__dst__  = dst_buf
        self.__size__ = size

        self.__raw_dst__ = self.__dst__.device()
        self.__raw_src__ = self.__src__.data

        self.size = size

    cdef libcuda.CUresult __exec__(self, core.LLStream stream):
        """
        """
        return libcuda.cuMemcpyAsync(self.__raw_dst__, <libcuda.CUdeviceptr>self.__raw_src__, self.__size__, stream._handle)
# ------------------------------------------------------------------------------
