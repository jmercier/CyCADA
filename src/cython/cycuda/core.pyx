cimport cuda.libcuda as libcuda

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


class CudaError(Exception):
    translation_table = error_translation_table

    def __init__(self, errid):
        Exception.__init__(self, self.translation_table[errid])


cdef inline CudaSafeCall(libcuda.CUresult result):
    if result != libcuda.CUDA_SUCCESS:
        raise CudaError(result)

###############################################################################
#
# Factories Section
#
###############################################################################

cdef LLContext _LLContext_factory(LLDevice device, unsigned int flags, bint opengl):
    cdef libcuda.CUcontext handle
    if opengl:
        CudaSafeCall(libcuda.cuGLCtxCreate(&handle, flags, device._handle));
    else:
        CudaSafeCall(libcuda.cuCtxCreate(&handle, flags, device._handle));

    cdef LLContext ctx = LLContext.__new__(LLContext)
    ctx._handle = handle
    ctx._dev = device

    return ctx

cdef LLStream _LLStream_factory(LLContext context, unsigned int flags):
    cdef libcuda.CUstream handle
    CudaSafeCall(libcuda.cuStreamCreate(&handle, flags))

    cdef LLStream stream = LLStream.__new__(LLStream)
    stream._handle = handle
    stream._ctx = context

    return stream

cpdef init(unsigned int flags = 0):
    libcuda.cuInit(flags)

###############################################################################
#
# Context Section
#
###############################################################################

cdef unsigned int get_context_api_version(libcuda.CUcontext ctx) except *:
    cdef unsigned int version
    CudaSafeCall(libcuda.cuCtxGetApiVersion(ctx, &version))
    return version

cdef assert_active_context(libcuda.CUcontext ctx):
    cdef libcuda.CUcontext current
    CudaSafeCall(libcuda.cuCtxGetCurrent(&current))
    if (current != ctx):
        raise RuntimeError("The Current CUDA Context is NOT active")

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
        return Stream(self, flags)

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

        self.name           = get_device_name(handle)
        self.capability     = get_device_capability(handle)
        self.device_id      = device_id

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

cdef bint stream_query(libcuda.CUstream stream) except *:
    return libcuda.cuStreamQuery(stream) == libcuda.CUDA_SUCCESS

cdef void stream_synchronize(libcuda.CUstream stream) except *:
    CudaSafeCall(libcuda.cuStreamSynchronize(stream))

cdef class LLStream(object):
    def __dealloc__(self):
        """
        """
        libcuda.cuStreamDestroy(self._handle)

class Stream(object):
    def __init__(self, context, unsigned int flags):
        """
        """
        cdef LLContext ctx = context.__ctx__
        assert_active_context(ctx._handle)
        cdef LLStream stream = _LLStream_factory(ctx, flags)
        self.__stream__ = stream

    def __get_status__(self):
        """
        """
        cdef LLStream stream = self.__stream__
        return stream.query()
    status = property(__get_status__)

    def synchronize(self):
        """
        """
        cdef LLStream stream = self.__stream__
        stream.synchronize()

    def enqueue(self, LLCommand cmd):
        """
        """
        cdef LLStream stream = self.__stream__
        cmd.__exec__(stream)

#######################################
#
# Command Section
#
#######################################

cdef class LLCommand(object):
    cdef void __exec__(self, LLStream stream) except *:
        raise RuntimeError("Empty Command")



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

