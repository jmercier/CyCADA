cmake_minimum_required (VERSION 2.8)

project (CyCuda)

set (CyCuda_VERSION_MAJOR 0)
set (CyCuda_VERSION_MINOR 1)

set (CMAKE_MODULE_PATH ${CyCuda_SOURCE_DIR}/cmake)

set (CMAKE_C_FLAGS "-O3 -DNDEBUG")

FIND_PACKAGE(Cython REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

SET (CyCuda_MODULE_SRCS "${CyCuda_SOURCE_DIR}/src/cycuda/__init__.py")
SET (CMAKE_C_FLAGS  "-Wall -Wno-unused-value -Wno-unused-function")

set (CYTHON_FLAGS -I${PROJECT_SOURCE_DIR}/src/cython/)
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS} )

#CYTHON_ADD_MODULE(cycuda/core core  core.c)
#CYTHON_ADD_MODULE(cycuda/fft fft fft.c)
#CYTHON_ADD_MODULE(cycuda/random random random.c)

#add_custom_target(cycuda COMMAND cp -r ${PROJECT_SOURCE_DIR}/cycuda cycuda DEPENDS ${CyCuda_MODULE})



#TARGET_LINK_LIBRARIES(cycuda/garray "cuda")
#SET_TARGET_PROPERTIES(cycuda/garray PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
#ADD_DEPENDENCIES(cycuda/garray cycuda)

CYTHON_ADD_MODULE(cycuda/core ${PROJECT_SOURCE_DIR}/src/cython/cycuda/core.pyx)
TARGET_LINK_LIBRARIES(cycuda/core "cuda")
SET_TARGET_PROPERTIES(cycuda/core PROPERTIES INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")

CYTHON_ADD_MODULE(cycuda/numpycmd ${PROJECT_SOURCE_DIR}/src/cython/cycuda/numpycmd.pyx)
TARGET_LINK_LIBRARIES(cycuda/numpycmd "cuda")
SET_TARGET_PROPERTIES(cycuda/numpycmd PROPERTIES INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")

set(CyCuda_MODULE "")
FOREACH(infile ${CyCuda_MODULE_SRCS})
    GET_FILENAME_COMPONENT(filename ${infile} NAME)
    set(outfile ${CMAKE_CURRENT_BINARY_DIR}/cycuda/${filename})
    add_custom_command(
        OUTPUT ${outfile}
        COMMAND ${CMAKE_COMMAND} -E copy ${infile} ${outfile}
        MAIN_DEPENDENCY ${infile})
    LIST(APPEND CyCuda_MODULE ${outfile})
ENDFOREACH(infile ${CyCuda_MODULE})
ADD_CUSTOM_TARGET(cy DEPENDS ${CyCuda_MODULE})

ADD_DEPENDENCIES(cycuda/core cy)
ADD_DEPENDENCIES(cycuda/numpycmd cy)


#PYTHON_ADD_MODULE(cycuda ${PROJECT_SOURCE_DIR}/src/cycuda/__init__.py)


#TARGET_LINK_LIBRARIES(cycuda/fft "cuda" ${CUDA_CUFFT_LIBRARIES})
#SET_TARGET_PROPERTIES(cycuda/fft PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
#ADD_DEPENDENCIES(cycuda/fft cycuda)
#
#TARGET_LINK_LIBRARIES(cycuda/random "cuda")
#SET_TARGET_PROPERTIES(cycuda/random PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
#ADD_DEPENDENCIES(cycuda/random cycuda)





