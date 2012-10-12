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
cimport cuda.libcuda as libcuda

# ------------------------------------------------------------------------------
cdef class LLDevice(object):
    cdef:
        libcuda.CUdevice    _handle
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLContext(object):
    cdef:
        libcuda.CUcontext   _handle
        LLDevice            _dev
        bint                _opengl
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLStream(object):
    cdef:
        LLContext           _ctx
        libcuda.CUstream    _handle
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLCommand(object):
    cdef:
        void __exec__(self, LLStream stream) except *
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLBuffer(object):
    cdef:
        unsigned int        _size

        libcuda.CUdeviceptr device(self) except *
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLDeviceBuffer(LLBuffer):
    cdef:
        libcuda.CUdeviceptr _handle
        LLContext           _ctx
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLHostBuffer(LLBuffer):
    cdef:
        void *              _handle
        LLContext           _ctx
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLModule(object):
    cdef:
        libcuda.CUmodule    _handle
        LLContext           _ctx
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLFunction(object):
    cdef:
        libcuda.CUfunction  _handle
        LLModule            _mod
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLParameter:
    cdef char *compact(self, char *ptr, object value)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLTexRef:
    cdef:
        libcuda.CUtexref    _handle
        LLModule            _mod
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLSurfRef:
    cdef:
        libcuda.CUsurfref   _handle
        LLModule            _mod
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLGlobal(LLBuffer):
    cdef:
        libcuda.CUdeviceptr     _handle
        LLModule                _mod
# ------------------------------------------------------------------------------

cdef class LLGraphicsResource(LLBuffer):
    cdef libcuda.CUgraphicsResource _handle
    cdef libcuda.CUdeviceptr device(self) except *
    cdef libcuda.CUdeviceptr _device


# ------------------------------------------------------------------------------
cdef:
    LLStream _LLStream_factory(LLContext context, unsigned int flags)
    LLContext _LLContext_factory(LLDevice device, unsigned int flags, bint opengl)
    LLDeviceBuffer _LLDeviceBuffer_factory(LLContext context, unsigned int size)
    LLHostBuffer _LLHostBuffer_factory(LLContext context, unsigned int size)
    LLModule _LLModule_factory(LLContext context, char *content)
    LLFunction _LLFunction_factory(LLModule module, bytes name)
    LLTexRef _LLTexRef_factory(LLModule module, bytes name)
    LLSurfRef _LLSurfRef_factory(LLModule module, bytes name)
    LLGlobal _LLGlobal_factory(LLModule module, bytes name)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef:
    int get_device_attribute(libcuda.CUdevice dev, libcuda.CUdevice_attribute att) except *
    bytes get_device_name(libcuda.CUdevice dev)
    tuple get_device_capability(libcuda.CUdevice dev)
    unsigned int get_context_api_version(libcuda.CUcontext ctx) except *
    bint stream_query(libcuda.CUstream stream) except *
    void stream_synchronize(libcuda.CUstream stream) except *
# ------------------------------------------------------------------------------
