cimport cuda.libcuda as libcuda

cdef class LLDevice(object):
    cdef libcuda.CUdevice _handle

cdef class LLContext(object):
    cdef libcuda.CUcontext _handle
    cdef LLDevice _dev

cdef class LLStream(object):
    cdef LLContext _ctx
    cdef libcuda.CUstream _handle

cdef class LLCommand(object):
    cdef void __exec__(self, LLStream stream) except *

cdef class LLBuffer(object):
    cdef LLContext _ctx
    cdef unsigned int _size

    cdef libcuda.CUdeviceptr device(self) except *

cdef class LLDeviceBuffer(LLBuffer):
    cdef libcuda.CUdeviceptr _handle

cdef class LLHostBuffer(LLBuffer):
    cdef void * _handle

cdef int get_device_attribute(libcuda.CUdevice dev, libcuda.CUdevice_attribute att) except *
cdef bytes get_device_name(libcuda.CUdevice dev)
cdef tuple get_device_capability(libcuda.CUdevice dev)

cdef unsigned int get_context_api_version(libcuda.CUcontext ctx) except *

cdef bint stream_query(libcuda.CUstream stream) except *
cdef void stream_synchronize(libcuda.CUstream stream) except *
