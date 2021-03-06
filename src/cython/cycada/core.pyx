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
from OpenGL.GL import GLuint

# ------------------------------------------------------------------------------
cdef dict error_translation_table     = \
    { libcuda.CUDA_ERROR_INVALID_VALUE                  : "INVALID_VALUE",
      libcuda.CUDA_ERROR_OUT_OF_MEMORY                  : "OUT_OF_MEMORY",
      libcuda.CUDA_ERROR_NOT_INITIALIZED                : "NOT_INITIALIZED",
      libcuda.CUDA_ERROR_DEINITIALIZED                  : "DEINITIALIZED",
      libcuda.CUDA_ERROR_PROFILER_DISABLED              : "PROFILER_DISABLED",
      libcuda.CUDA_ERROR_PROFILER_NOT_INITIALIZED       : "PROFILER_NOT_INITIALIZED",
      libcuda.CUDA_ERROR_PROFILER_ALREADY_STARTED       : "PROFILER_ALREADY_STARTED",
      libcuda.CUDA_ERROR_PROFILER_ALREADY_STOPPED       : "PROFILER_ALREADY_STOPPED",
      libcuda.CUDA_ERROR_NO_DEVICE                      : "NO_DEVICE",
      libcuda.CUDA_ERROR_INVALID_DEVICE                 : "INVALID_DEVICE",
      libcuda.CUDA_ERROR_INVALID_IMAGE                  : "INVALID_IMAGE",
      libcuda.CUDA_ERROR_INVALID_CONTEXT                : "INVALID_CONTEXT",
      libcuda.CUDA_ERROR_CONTEXT_ALREADY_CURRENT        : "CONTEXT_ALREADY_CURRENT",
      libcuda.CUDA_ERROR_MAP_FAILED                     : "MAP_FAILED",
      libcuda.CUDA_ERROR_UNMAP_FAILED                   : "UNMAP_FAILED",
      libcuda.CUDA_ERROR_ARRAY_IS_MAPPED                : "ARRAY_IS_MAPPED",
      libcuda.CUDA_ERROR_ALREADY_MAPPED                 : "ALREADY_MAPPED",
      libcuda.CUDA_ERROR_NO_BINARY_FOR_GPU              : "NO_BINARY_FOR_GPU",
      libcuda.CUDA_ERROR_ALREADY_ACQUIRED               : "ALREADY_ACQUIRED",
      libcuda.CUDA_ERROR_NOT_MAPPED                     : "NOT_MAPPED",
      libcuda.CUDA_ERROR_NOT_MAPPED_AS_ARRAY            : "NOT_MAPPED_AS_ARRAY",
      libcuda.CUDA_ERROR_NOT_MAPPED_AS_POINTER          : "NOT_MAPPED_AS_POINTER",
      libcuda.CUDA_ERROR_ECC_UNCORRECTABLE              : "ECC_UNCORRECTABLE",
      libcuda.CUDA_ERROR_UNSUPPORTED_LIMIT              : "UNSUPPORTED_LIMIT",
      libcuda.CUDA_ERROR_CONTEXT_ALREADY_IN_USE         : "CONTEXT_ALREADLY_IN_USE",
      libcuda.CUDA_ERROR_INVALID_SOURCE                 : "INVALID_SOURCE",
      libcuda.CUDA_ERROR_FILE_NOT_FOUND                 : "FILE_NOT_FOUND",
      libcuda.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      : "SHARED_OBJECT_INIT_FAILED",
      libcuda.CUDA_ERROR_OPERATING_SYSTEM               : "OPERATING_SYSTEM",
      libcuda.CUDA_ERROR_INVALID_HANDLE                 : "INVALID_HANDLE",
      libcuda.CUDA_ERROR_NOT_FOUND                      : "NOT_FOUND",
      libcuda.CUDA_ERROR_NOT_READY                      : "NOT_READY",
      libcuda.CUDA_ERROR_LAUNCH_FAILED                  : "LAUNCH_FAILED",
      libcuda.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        : "LAUNCH_OUT_OF_RESOURCES",
      libcuda.CUDA_ERROR_LAUNCH_TIMEOUT                 : "LAUNCH_TIMEOUT",
      libcuda.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  : "LAUNCH_INCOMPATIBLE_TEXTURING",
      libcuda.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    : "PEER_ACCESS_ALREADY_ENABLED",
      libcuda.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        : "PEER_ACCESS_NOT_ENABLED",
      libcuda.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         : "PRIMARY_CONTEXT_ACTIVE",
      libcuda.CUDA_ERROR_CONTEXT_IS_DESTROYED           : "CONTEXT_IS_DESCROYED",
      libcuda.CUDA_ERROR_UNKNOWN                        : "ERROR_UNKNOWN" }
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
class CudaError(Exception):
    translation_table = error_translation_table

    def __init__(self, errid):
        Exception.__init__(self, self.translation_table[errid])
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
cdef inline CudaSafeCall(libcuda.CUresult result):
    if result != libcuda.CUDA_SUCCESS:
        raise CudaError(result)
# ------------------------------------------------------------------------------

###############################################################################
#
# Factories Section
#
###############################################################################

# ------------------------------------------------------------------------------
cdef LLContext _LLContext_factory(LLDevice device, unsigned int flags, bint opengl):
    cdef libcuda.CUcontext handle
    if opengl:
        CudaSafeCall(libcuda.cuGLCtxCreate(&handle, flags, device._handle));
    else:
        CudaSafeCall(libcuda.cuCtxCreate(&handle, flags, device._handle));

    cdef LLContext ctx = LLContext.__new__(LLContext)
    ctx._handle = handle
    ctx._dev = device
    ctx._opengl = opengl

    return ctx
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLStream _LLStream_factory(LLContext context, unsigned int flags):
    cdef libcuda.CUstream handle
    CudaSafeCall(libcuda.cuStreamCreate(&handle, flags))

    cdef LLStream stream = LLStream.__new__(LLStream)
    stream._handle = handle
    stream._ctx = context

    return stream
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLDeviceBuffer _LLDeviceBuffer_factory(LLContext context, unsigned int size):
    cdef libcuda.CUdeviceptr handle
    CudaSafeCall(libcuda.cuMemAlloc(&handle, size))

    cdef LLDeviceBuffer buf = LLDeviceBuffer.__new__(LLDeviceBuffer)
    buf._handle = handle
    buf._ctx = context
    buf._size = size

    return buf
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLHostBuffer _LLHostBuffer_factory(LLContext context, unsigned int size):
    cdef void *handle
    CudaSafeCall(libcuda.cuMemAllocHost(&handle, size))

    cdef LLHostBuffer buf = LLHostBuffer.__new__(LLHostBuffer)
    buf._handle     = handle
    buf._ctx        = context
    buf._size       = size

    return buf
# ------------------------------------------------------------------------------
cdef LLGraphicsResource _LLGraphicsResource_factory(LLContext context, libcuda.GLuint buf, unsigned int flags):
    cdef libcuda.CUgraphicsResource handle
    CudaSafeCall(libcuda.cuGraphicsGLRegisterBuffer(&handle, buf, flags))

    cdef LLGraphicsResource res = LLGraphicsResource.__new__(LLGraphicsResource)
    res._handle     = handle
    res._ctx        = context
    return res


# ------------------------------------------------------------------------------
cpdef init(unsigned int flags = 0):
    global version, device_count

    cdef int v, d
    CudaSafeCall(libcuda.cuInit(flags))
    CudaSafeCall(libcuda.cuDriverGetVersion(&v))
    CudaSafeCall(libcuda.cuDeviceGetCount(&d))
    version         = v
    device_count    = d
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLModule _LLModule_factory(LLContext context, char *content):
    cdef:
        libcuda.CUmodule handle
        libcuda.CUjit_option options[32]
        void *optionValues[32]
        char info_buffer[1024]
        char error_buffer[1024]
        unsigned int error_buffer_size = 1024
        unsigned int info_buffer_size = 1024
        float wall_time = 0

    options[0] = libcuda.CU_JIT_INFO_LOG_BUFFER
    optionValues[0] = <void *>info_buffer;

    options[1] = libcuda.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
    optionValues[1] = <void *>&info_buffer_size

    options[2] = libcuda.CU_JIT_ERROR_LOG_BUFFER
    optionValues[2] = <void *>error_buffer;

    options[3] = libcuda.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    optionValues[3] = <void *>&error_buffer_size

    options[4] = libcuda.CU_JIT_WALL_TIME
    optionValues[4] = <void *>&wall_time

    options[5] = libcuda.CU_JIT_TARGET_FROM_CUCONTEXT
    optionValues[5] = <void *>NULL

    #options[6] = libcuda.CU_JIT_OPTIMIZATION_LEVEL
    #optionValues[6] = <void *>&optimization_level


    CudaSafeCall(libcuda.cuModuleLoadDataEx(&handle, content, 6, options, optionValues))

    cdef LLModule module = LLModule.__new__(LLModule)
    module._ctx     = context
    module._handle  = handle
    return module
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLFunction _LLFunction_factory(LLModule module, bytes name):
    cdef libcuda.CUfunction handle
    CudaSafeCall(libcuda.cuModuleGetFunction(&handle, module._handle, name))

    cdef LLFunction function = LLFunction.__new__(LLFunction)
    function._handle = handle
    function._mod = module
    return function
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLTexRef _LLTexRef_factory(LLModule module, bytes name):
    cdef libcuda.CUtexref handle
    CudaSafeCall(libcuda.cuModuleGetTexRef(&handle, module._handle, name))

    cdef LLTexRef texture = LLTexRef.__new__(LLTexRef)
    texture._handle = handle
    texture._mod = module
    return texture
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLSurfRef _LLSurfRef_factory(LLModule module, bytes name):
    cdef libcuda.CUsurfref handle
    CudaSafeCall(libcuda.cuModuleGetSurfRef(&handle, module._handle, name))

    cdef LLSurfRef surface = LLSurfRef.__new__(LLSurfRef)
    surface._handle = handle
    surface._mod = module
    return surface
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLGlobal _LLGlobal_factory(LLModule module, bytes name):
    cdef:
        libcuda.CUdeviceptr handle
        size_t size

    CudaSafeCall(libcuda.cuModuleGetGlobal(&handle, &size, module._handle, name))

    cdef LLGlobal g = LLGlobal.__new__(LLGlobal)
    g._handle = handle
    g._mod = module
    g._size = size

    return g
# ------------------------------------------------------------------------------
###############################################################################
#
# Context Section
#
###############################################################################

# ------------------------------------------------------------------------------
cdef unsigned int get_context_api_version(libcuda.CUcontext ctx) except *:
    cdef unsigned int version
    CudaSafeCall(libcuda.cuCtxGetApiVersion(ctx, &version))
    return version
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef assert_active_context(libcuda.CUcontext ctx):
    cdef libcuda.CUcontext current
    CudaSafeCall(libcuda.cuCtxGetCurrent(&current))
    if (current != ctx):
        raise RuntimeError("The Current CUDA Context is NOT active")
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLContext(object):
    def __assert_active__(self):
        """
        """
        cdef libcuda.CUcontext current
        CudaSafeCall(libcuda.cuCtxGetCurrent(&current))
        if (current != self._handle):
            raise RuntimeError("The Current CUDA Context is NOT active")

    def __dealloc__(self):
        """
        """
        libcuda.cuCtxDestroy(self._handle)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
class Context(object):
    def __init__(self, device, unsigned int flags, bint opengl):
        """
        """
        cdef LLContext ctx = _LLContext_factory(device, flags, opengl)
        self.__ctx__ = ctx

        self.api_version = get_context_api_version(ctx._handle)

    def create_stream(self, unsigned int flags = 0):
        """
        """
        self.__check_active__()
        return Stream(self.__ctx__, flags)

    def alloc(self, size):
        self.__check_active__()
        return DeviceBuffer(self.__ctx__, size)

    def alloc_host(self, size):
        self.__check_active__()
        return HostBuffer(self.__ctx__, size)

    def __get_opengl__(self):
        cdef LLContext ctx = self.__ctx__
        return ctx._opengl

    def load_module(self, data):
        self.__check_active__()
        return Module(self.__ctx__, data)

    def __check_active__(self):
        cdef LLContext ctx = self.__ctx__
        assert_active_context(ctx._handle)

    def register_buffer(self, bufid, flags = 0):
        return _LLGraphicsResource_factory(self.__ctx__, bufid, flags)

    opengl = property(__get_opengl__)
# ------------------------------------------------------------------------------



###############################################################################
#
# Device Section
#
###############################################################################
DEF MAXNAMELENGTH = 256

# ------------------------------------------------------------------------------
cdef int get_device_attribute(libcuda.CUdevice dev, libcuda.CUdevice_attribute att) except *:
    cdef int val
    CudaSafeCall(libcuda.cuDeviceGetAttribute(&val, att, dev))
    return val
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef int get_function_attribute(libcuda.CUfunction fun, libcuda.CUfunction_attribute att) except *:
    cdef int val
    CudaSafeCall(libcuda.cuFuncGetAttribute(&val, att, fun))
    return val
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef bytes get_device_name(libcuda.CUdevice dev):
    cdef char name[MAXNAMELENGTH]
    CudaSafeCall(libcuda.cuDeviceGetName(name, MAXNAMELENGTH, dev))
    return name
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef tuple get_device_capability(libcuda.CUdevice dev):
    cdef int cMajor, cMinor
    CudaSafeCall(libcuda.cuDeviceComputeCapability(&cMajor, &cMinor, dev))
    return (cMajor, cMinor)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLDevice(object):
    def __cinit__(self, unsigned int device_id):
        """
        """
        libcuda.cuDeviceGet(&self._handle, device_id)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class Device(object):
    properties = {}
    def __init__(self, device_id = 0):
        """
        """
        cdef LLDevice device    = LLDevice(device_id)
        self.__device__         = device

        cdef libcuda.CUdevice handle = device._handle
        for k, i in self.properties.iteritems():
            if len(i) == 1:
                value = get_device_attribute(handle, i[0])
            else:
                value = tuple([get_device_attribute(handle, p) for p in i])
            setattr(self, k, value)

        cdef size_t mem_size
        CudaSafeCall(libcuda.cuDeviceTotalMem(&mem_size, handle))

        self.total_memory    = mem_size
        self.name            = get_device_name(handle)
        self.capability      = get_device_capability(handle)
        self.device_id       = device_id

    def __repr__(self):
        """
        """
        return "<%s.%s %s [device %d]>" % (self.__class__.__name__, self.__class__.__module__, self.name, self.device_id)

    def create_context(self, flags = 0, opengl = False):
        """
        """
        return Context(self.__device__, flags, opengl)
# ------------------------------------------------------------------------------

#######################################
#
# Stream Section
#
#######################################

# ------------------------------------------------------------------------------
cdef bint stream_query(libcuda.CUstream stream) except *:
    return libcuda.cuStreamQuery(stream) == libcuda.CUDA_SUCCESS
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef void stream_synchronize(libcuda.CUstream stream) except *:
    CudaSafeCall(libcuda.cuStreamSynchronize(stream))
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLStream(object):
    def __dealloc__(self):
        """
        """
        libcuda.cuStreamDestroy(self._handle)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class Stream(object):
    cdef LLStream __stream__

    def __init__(self, LLContext ctx, unsigned int flags):
        """
        """
        self.__stream__ = _LLStream_factory(ctx, flags)

    property status:
        def __get__(self):
            return stream_query(self.__stream__._handle)

    def synchronize(self):
        """
        """
        CudaSafeCall(libcuda.cuStreamSynchronize(self.__stream__._handle))

    def enqueue(self, LLCommand cmd):
        """
        """
        cdef LLStream stream = self.__stream__
        CudaSafeCall(cmd.__exec__(self.__stream__))
# ------------------------------------------------------------------------------

#######################################
#
# Command Section
#
#######################################

# ------------------------------------------------------------------------------
cdef class LLCommand(object):
    cdef libcuda.CUresult __exec__(self, LLStream stream):
        """
        """
        return libcuda.CUDA_ERROR_UNKNOWN
# ------------------------------------------------------------------------------

#######################################
#
# Buffer Section
#
#######################################


# ------------------------------------------------------------------------------
cdef class LLBuffer(object):
    cdef libcuda.CUdeviceptr device(self) except *:
        """
        """
        raise RuntimeError("Empty Buffer")
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLDeviceBuffer(object):
    cdef libcuda.CUdeviceptr device(self) except *:
        """
        """
        return self._handle

    def __dealloc__(self):
        """
        """
        libcuda.cuMemFree(self._handle)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLGraphicsResource(LLBuffer):
    def __dealloc__(self):
        libcuda.cuGraphicsUnregisterResource(self._handle)

    cdef libcuda.CUdeviceptr device(self) except *:
        cdef:
            libcuda.CUdeviceptr device
            size_t size

        CudaSafeCall(libcuda.cuGraphicsResourceGetMappedPointer(&device, &size, self._handle))
        self._size = size
        return device
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLHostBuffer(object):
    cdef libcuda.CUdeviceptr device(self) except *:
        """
        """
        cdef libcuda.CUdeviceptr ptr
        CudaSafeCall(libcuda.cuMemHostGetDevicePointer(&ptr, self._handle, 0))
        return ptr

    def __dealloc__(self):
        libcuda.cuMemFreeHost(self._handle)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class DeviceBuffer(object):
    def __init__(self, LLContext ctx, unsigned int size):
        """
        """
        cdef LLDeviceBuffer buf = _LLDeviceBuffer_factory(ctx, size)
        self.__buffer__ = buf
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class HostBuffer(object):
    def __init__(self, LLContext ctx, unsigned int size):
        """
        """
        cdef LLHostBuffer buf = _LLHostBuffer_factory(ctx, size)
        self.__buffer__ = buf
# ------------------------------------------------------------------------------

#######################################
#
# Module Section
#
#######################################

# ------------------------------------------------------------------------------
cdef class LLModule(object):
    def __dealloc__(self):
        """
        """
        libcuda.cuModuleUnload(self._handle)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class Module(object):
    cdef:
        LLModule __module__
        dict __functions__

    def __init__(self, LLContext ctx, data):
        """
        """
        self.__module__ = _LLModule_factory(ctx, data)
        self.__functions__ = {}

    def get_function(self, name):
        """
        """
        if name not in self.__functions__:
            self.__functions__[name] = Function(self.__module__, name)
        return self.__functions__[name]

    def get_texref(self, bytes name):
        pass

    def get_surfref(self, bytes name):
        pass
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLFunction(object):
    pass
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
class Function(object):
    attributes = {}
    def __init__(self, module, name):
        """
        """
        cdef LLFunction function = _LLFunction_factory(module, name)
        self.__function__ = function
        cdef libcuda.CUfunction handle = function._handle
        for k, i in self.attributes.iteritems():
            if len(i) == 1:
                value = get_function_attribute(handle, i[0])
            else:
                value = tuple([get_function_attribute(handle, p) for p in i])
            setattr(self, k, value)
        self.__parameters__ = tuple()

    def __get_parameters__(self):
        return self.__parameters__

    def __set_parameters__(self, args):
        self.__parameters__ = args
    parameters = property(__get_parameters__, __set_parameters__)
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
cdef char *_compact_mem(char *ptr, object value):
    cdef LLBuffer buf = value
    (<libcuda.CUdeviceptr *>ptr)[0] = buf.device()
    return ptr + sizeof(libcuda.CUdeviceptr)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef char *_compact_int(char *ptr, object param):
    cdef int value = param
    (<int *>ptr)[0] = value
    return ptr + sizeof(int)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef char *_compact_float(char *ptr, object param):
    cdef float value = param
    (<float *>ptr)[0] = value
    return ptr + sizeof(float)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef char *_compact_uint(char *ptr, object param):
    cdef unsigned int value = param
    (<unsigned int *>ptr)[0] = value
    return ptr + sizeof(unsigned int)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef char *_compact_double(char *ptr, object param):
    cdef double value = param
    (<double *>ptr)[0] = value
    return ptr + sizeof(double)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef class LLParameter:
    cdef char *compact(self, char *ptr, object value):
        return self.__compact__(ptr, value)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLParameter _MEMParameter = LLParameter()
_MEMParameter.__compact__ = _compact_mem
MEMParameter = _MEMParameter
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLParameter _INTParameter = LLParameter()
_INTParameter.__compact__ = _compact_int
INTParameter = _INTParameter
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLParameter _FLOATParameter = LLParameter()
_FLOATParameter.__compact__ = _compact_float
FLOATParameter = _FLOATParameter
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLParameter _UINTParameter = LLParameter()
_UINTParameter.__compact__ = _compact_uint
UINTParameter = _UINTParameter
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef LLParameter _DOUBLEParameter = LLParameter()
_DOUBLEParameter.__compact__ = _compact_double
DOUBLEParameter = _DOUBLEParameter
# ------------------------------------------------------------------------------









#######################################
#
# Function Properties
#
#######################################

Function.attributes['max_thread_per_block']         = [libcuda.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
Function.attributes['shared_size']                  = [libcuda.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
Function.attributes['const_size']                   = [libcuda.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES]
Function.attributes['local_size']                   = [libcuda.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
Function.attributes['num_regs']                     = [libcuda.CU_FUNC_ATTRIBUTE_NUM_REGS]
Function.attributes['ptx_version']                  = [libcuda.CU_FUNC_ATTRIBUTE_PTX_VERSION]
Function.attributes['binary_version']               = [libcuda.CU_FUNC_ATTRIBUTE_BINARY_VERSION]


#######################################
#
# Device Properties
#
#######################################

Device.properties['max_grid_dim']                   = [libcuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z]
Device.properties['max_block_dim']                  = [libcuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z]
Device.properties['max_threads_per_block']          = [libcuda.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
Device.properties['max_shared_memory_per_block']    = [libcuda.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK]
Device.properties['shared_memory_per_block']        = [libcuda.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK]
Device.properties['total_constant_memory']          = [libcuda.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY]
Device.properties['warp_size']                      = [libcuda.CU_DEVICE_ATTRIBUTE_WARP_SIZE]
Device.properties['max_pitch']                      = [libcuda.CU_DEVICE_ATTRIBUTE_MAX_PITCH]
Device.properties['max_registers_per_block']        = [libcuda.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK]
Device.properties['register_per_block']             = [libcuda.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK]
Device.properties['clock_rate']                     = [libcuda.CU_DEVICE_ATTRIBUTE_CLOCK_RATE]
Device.properties['texture_alignment']              = [libcuda.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT]
Device.properties['gpu_overlap']                    = [libcuda.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP]
Device.properties['multiprocessor_count']           = [libcuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT]
Device.properties['kernel_exec_timeout']            = [libcuda.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT]
Device.properties['integrated']                     = [libcuda.CU_DEVICE_ATTRIBUTE_INTEGRATED]
Device.properties['can_map_host_memory']            = [libcuda.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY]
Device.properties['compute_mode']                   = [libcuda.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE]
Device.properties['maximum_texture_1d']             = [libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH]
Device.properties['maximum_texture_2d']             = [libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT]
Device.properties['maximum_texture_3d']             = [libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH]
Device.properties['maximum_texture_1d_layered']     = [libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS]
Device.properties['maximum_texture_2d_layered']     = [libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS]
Device.properties['maximum_texture_2d_array']       = [libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES]
Device.properties['surface_alignment']              = [libcuda.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT]
Device.properties['concurent_kernels']              = [libcuda.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS]
Device.properties['ECC']                            = [libcuda.CU_DEVICE_ATTRIBUTE_ECC_ENABLED]
Device.properties['pci']                            = [libcuda.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                                       libcuda.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID]
Device.properties['tcc_driver']                     = [libcuda.CU_DEVICE_ATTRIBUTE_TCC_DRIVER]
Device.properties['memory_clock_rate']              = [libcuda.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE]
Device.properties['global_memory_bus_width']        = [libcuda.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH]
Device.properties['l2_cache_size']                  = [libcuda.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE]
Device.properties['max_threads_per_multiprocessor'] = [libcuda.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR]
Device.properties['async_engine_count']             = [libcuda.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT]
Device.properties['unified_addressing']             = [libcuda.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING]



