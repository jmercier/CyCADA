cdef extern from "curand.h":
    ctypedef enum curandStatus:
        CURAND_STATUS_SUCCESS
        CURAND_STATUS_VERSION_MISMATCH
        CURAND_STATUS_NOT_INITIALIZED
        CURAND_STATUS_ALLOCATION_FAILED
        CURAND_STATUS_TYPE_ERROR
        CURAND_STATUS_OUT_OF_RANGE
        CURAND_STATUS_LENGTH_NOT_MULTIPLE
        CURAND_STATUS_LAUNCH_FAILURE
        CURAND_STATUS_PREEXISTING_FAILURE
        CURAND_STATUS_INITIALIZATION_FAILED
        CURAND_STATUS_ARCH_MISMATCH
        CURAND_STATUS_INTERNAL_ERROR

    ctypedef enum curandRngType:
        CURAND_RNG_TEST
        CURAND_RNG_PSEUDO_DEFAULT
        CURAND_RNG_PSEUDO_XORWOW
        CURAND_RNG_PSEUDO_MRG32K3A
        CURAND_RNG_PSEUDO_MTGP32
        CURAND_RNG_QUASI_DEFAULT
        CURAND_RNG_QUASI_SOBOL32
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
        CURAND_RNG_QUASI_SOBOL64
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL64

    ctypedef enum curandOrdering:
        CURAND_ORDERING_PSEUDO_BEST
        CURAND_ORDERING_PSEUDO_DEFAULT
        CURAND_ORDERING_PSEUDO_SEEDED
        CURAND_ORDERING_QUASI_DEFAULT

    ctypedef enum curandDirectionVectorSet:
        CURAND_DIRECTION_VECTORS_32_JOEKUO6
        CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
        CURAND_DIRECTION_VECTORS_64_JOEKUO6
        CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6

    ctypedef enum curandMethod:
        CURAND_CHOOSE_BEST
        CURAND_ITR
        CURAND_KNUTH
        CURAND_HITR
        CURAND_M1
        CURAND_M2
        CURAND_BINARY_SEARCH
        CURAND_DISCRETE_GAUSS
        CURAND_REJECTION
        CURAND_DEVICE_API
        CURAND_FAST_REJECTION
        CURAND_3RD
        CURAND_DEFINITION
        CURAND_POISSON

    ctypedef struct curandGenerator_t:
        pass

    cdef:
        curandStatus curandCreateGenerator(curandGenerator_t *, curandRngType)
        curandStatus curandCreateGeneratorHost(curandGenerator_t *, curandRngType)
        curandStatus curandDestroyGenerator(curandGenerator_t generator)
        curandStatus curandGetVersion(int *version)
        curandStatus curandSetPseudoRandomGeneratorSeed(curandGenerator_t, unsigned long long)

    cdef:
        curandStatus curandGenerateUniform(curandGenerator_t , float *, size_t )
        curandStatus curandGenerate(curandGenerator_t , unsigned int *, size_t )
        curandStatus curandGenerateLongLong(curandGenerator_t , unsigned long long *, size_t )
        curandStatus curandGenerateUniform(curandGenerator_t , float *, size_t )
        curandStatus curandGenerateUniformDouble(curandGenerator_t , double *, size_t )
        curandStatus curandGenerateNormal(curandGenerator_t , float *, size_t , float , float )
        curandStatus curandGenerateNormalDouble(curandGenerator_t , double *, size_t , float , float )

        curandStatus curandGenerateLogNormal(curandGenerator_t , float *, size_t , float , float )
        curandStatus curandGenerateLogNormalDouble(curandGenerator_t , double *, size_t , float , float )
        curandStatus curandGeneratePoisson(curandGenerator_t , unsigned int *, size_t , double )
