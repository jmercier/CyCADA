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



        self.__src__ = src.__buffer__
        self.__dst__ = dst.__buffer__

        if (size == -1):
            size = self.__src__._size

        self.__raw_dst__ = self.__dst__.device()
        self.__raw_src__ = self.__src__.device()

        self.size = size

    cdef libcuda.CUresult __exec__(self, core.LLStream stream):
        """
        """
        return libcuda.cuMemcpyDtoDAsync(self.__raw_dst__, self.__raw_src__, self.__size__, stream._handle)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef move_forward(char **current_ptr, core.LLParameter param,  object arg):
    current_ptr[0] = param.compact(current_ptr[0], arg)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class RangeKernel(core.LLCommand):
    cdef:
        unsigned int        grid_dim[3]
        unsigned int        block_dim[3]
        unsigned int        shared_size

        char                args[256]
        void*               arg_ptr[64]

        core.LLFunction     function
        object              arguments




    def __init__(self, function, tuple args = tuple(), tuple block_dim = (1, 1, 1), tuple grid_dim = (1, 1, 1), unsigned int shared_size = 0):
        """
        """
        for i in xrange(3):
            self.grid_dim[i] = grid_dim[i]
            self.block_dim[i] = block_dim[i]

        self.shared_size = shared_size

        cdef char *current_ptr = self.args

        self.function = function.__function__
        self.arguments = args

        cdef tuple parameters = function.parameters

        if len(parameters) != len(args):
            raise RuntimeError("Invalid Number of arguments")

        for i in xrange(len(parameters)):
            self.arg_ptr[i] = current_ptr;
            move_forward(&current_ptr, parameters[i], args[i])

    cdef libcuda.CUresult __exec__(self, core.LLStream stream):
        """
        """
        return libcuda.cuLaunchKernel( self.function._handle,
                                self.grid_dim[0],
                                self.grid_dim[1],
                                self.grid_dim[2],
                                self.block_dim[0],
                                self.block_dim[1],
                                self.block_dim[2],
                                self.shared_size,
                                stream._handle,
                                self.arg_ptr,
                                NULL)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class BaseMapUnmapGraphicsResources(core.LLCommand):
    cdef:
        libcuda.CUgraphicsResource  resources[256]
        unsigned int                size
        list                        arguments

    def __init__(self, *args):
        """
        """
        self.arguments = args
        self.size = len(args)
        for i in xrange(self.size):
            self.resources[i] = (<core.LLGraphicsResource>args[i])._handle
# ------------------------------------------------------------------------------
cdef class UnmapGraphicsResources(BaseMapUnmapGraphicsResources):
    def __init__(self, *args):
        BaseMapUnmapGraphicsResources(self, *args)

    cdef libcuda.CUresult __exec__(self, core.LLStream stream):
        """
        """
        return libcuda.cuGraphicsUnmapResources(self.size, self.resources, stream._handle)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class MapGraphicsResources(BaseMapUnmapGraphicsResources):
    cdef unsigned int flags

    def __init__(self, *args, flags = 0):
        BaseMapUnmapGraphicsResources(self, *args)
        self.flags = flags

    cdef libcuda.CUresult __exec__(self, core.LLStream stream):
        """
        """
        libcuda.cuGraphicsReesourceSetMapFlags(self._handle, self.flags)
        return libcuda.cuGraphicsMapResources(self.size, self.resources, stream._handle)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class CopyPeer(core.LLCommand):
    cdef:
        libcuda.CUdeviceptr     __raw_src__
        libcuda.CUdeviceptr     __raw_dst__
        core.LLBuffer           __src__
        core.LLBuffer           __dst__
        unsigned int            __size__

    def __init__(self, src, dst):
        self.__src__ = src
        self.__dst__ = dst
        self.__raw_src__ = self.__src__.device()
        self.__raw_dst__ = self.__dst__.device()
        self.__size__ = 0

    cdef libcuda.CUresult __exec__(self, core.LLStream stream):
        """
        """
        return libcuda.cuMemcpyPeerAsync(self.__raw_dst__,
                                                              self.__dst__._ctx._handle,
                                                              self.__raw_src__,
                                                              self.__src__._ctx._handle,
                                                              self.__size__,
                                                              stream._handle)
# ------------------------------------------------------------------------------

