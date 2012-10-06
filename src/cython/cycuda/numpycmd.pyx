cimport cuda.libcuda as libcuda
cimport cycuda.core as core

from . import core as cuda

cimport numpy as cnp

# ------------------------------------------------------------------------------
cdef class CopyBuffer(core.LLCommand):
    cdef:
        libcuda.CUdeviceptr     __raw_src__
        libcuda.CUdeviceptr     __raw_dst__
        unsigned int            __size__

        core.LLBuffer           __src__
        core.LLBuffer           __dst__

    property size:
        def __get__(self):
            return self.__size__

        def __set__(self, new_size):
            if ((self.__src__._size < new_size) or \
                (self.__dst__._size < new_size)):
                raise RuntimeError("Invalid Size")
            self.__size__ = new_size

    def __init__(self, dst, src, int size = -1):
        """
        """
        cdef core.LLBuffer src_buf = src.__buffer__
        cdef core.LLBuffer dst_buf = dst.__buffer__
        if (size == -1):
            size = src_buf._size

        self.__src__ = src_buf
        self.__dst__ = dst_buf

        self.__raw_dst__ = self.__dst__.device()
        self.__raw_src__ = self.__src__.device()

        self.size = size

    cdef void __exec__(self, core.LLStream stream) except *:
        """
        """
        libcuda.cuMemcpyDtoDAsync(self.__raw_dst__, self.__raw_src__, self.__size__, stream._handle)
# ------------------------------------------------------------------------------


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

    cdef void __exec__(self, core.LLStream stream) except *:
        """
        """
        libcuda.cuMemcpyDtoHAsync(self.__raw_dst__, self.__raw_src__, self.__size__, stream._handle)
# ------------------------------------------------------------------------------
#

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

    cdef void __exec__(self, core.LLStream stream) except *:
        """
        """
        libcuda.cuMemcpyAsync(self.__raw_dst__, <libcuda.CUdeviceptr>self.__raw_src__, self.__size__, stream._handle)
# ------------------------------------------------------------------------------
