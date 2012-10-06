cimport cuda.libcuda as libcuda
cimport cycuda.core as core

from . import core as cuda


cdef class ReadBufferNDArray(core.LLCommand):
    cdef void __exec__(self, core.LLStream stream) except *:
        pass


cdef class WriteBufferNDArray(core.LLCommand):
    cdef void __exec__(self, core.LLStream stream) except *:
        pass

