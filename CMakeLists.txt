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
cmake_minimum_required (VERSION 2.8)

project (CyCADA)

set (CyCADA_VERSION_MAJOR 0)
set (CyCADA_VERSION_MINOR 1)

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set (CMAKE_C_FLAGS "-O3 -DNDEBUG")

FIND_PACKAGE(Cython REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(PythonInterp 2.7 COMPONENTS numpy REQUIRED)

SET (CyCADA_MODULE_SRCS  "${CMAKE_CURRENT_SOURCE_DIR}/src/cycada/__init__.py")
SET (CMAKE_C_FLAGS  "-Wall -Wno-unused-function -g")

set (CYTHON_FLAGS -I${PROJECT_SOURCE_DIR}/src/cython/)

# SHOULD FIND A BETTER WAY TO FIND THIS DIRECTORY
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}  /usr/lib/python2.7/site-packages/numpy/core/include/)

#add_custom_target(cycada COMMAND cp -r ${PROJECT_SOURCE_DIR}/cycada cycada DEPENDS ${CyCADA_MODULE})



#TARGET_LINK_LIBRARIES(cycada/garray "cuda")
#SET_TARGET_PROPERTIES(cycada/garray PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
#ADD_DEPENDENCIES(cycada/garray cycada)

CYTHON_ADD_MODULE(cycada/core ${PROJECT_SOURCE_DIR}/src/cython/cycada/core.pyx)
TARGET_LINK_LIBRARIES(cycada/core "cuda")
SET_TARGET_PROPERTIES(cycada/core PROPERTIES INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")

CYTHON_ADD_MODULE(cycada/numpycmd ${PROJECT_SOURCE_DIR}/src/cython/cycada/numpycmd.pyx)
TARGET_LINK_LIBRARIES(cycada/numpycmd "cuda")

CYTHON_ADD_MODULE(cycada/cmd ${PROJECT_SOURCE_DIR}/src/cython/cycada/cmd.pyx)
TARGET_LINK_LIBRARIES(cycada/cmd "cuda")

CYTHON_ADD_MODULE(cycada/numpy_ext ${PROJECT_SOURCE_DIR}/src/cython/cycada/numpy_ext.pyx)
TARGET_LINK_LIBRARIES(cycada/numpy_ext "cuda")

CYTHON_ADD_MODULE(cycada/opengl ${PROJECT_SOURCE_DIR}/src/cython/cycada/opengl.pyx)
TARGET_LINK_LIBRARIES(cycada/opengl "cuda")

CYTHON_ADD_MODULE(cycada/manager ${PROJECT_SOURCE_DIR}/src/cython/cycada/manager.pyx)
TARGET_LINK_LIBRARIES(cycada/manager "cuda")

set(CyCADA_MODULE "")
FOREACH(infile ${CyCADA_MODULE_SRCS})
    GET_FILENAME_COMPONENT(filename ${infile} NAME)
    set(outfile ${CMAKE_CURRENT_BINARY_DIR}/cycada/${filename})
    add_custom_command(
        OUTPUT ${outfile}
        COMMAND ${CMAKE_COMMAND} -E copy ${infile} ${outfile}
        MAIN_DEPENDENCY ${infile})
    LIST(APPEND CyCADA_MODULE ${outfile})
ENDFOREACH(infile ${CyCADA_MODULE})
ADD_CUSTOM_TARGET(cy DEPENDS ${CyCADA_MODULE})

ADD_DEPENDENCIES(cycada/core cy)
ADD_DEPENDENCIES(cycada/numpycmd cy)
ADD_DEPENDENCIES(cycada/cmd cy)
ADD_DEPENDENCIES(cycada/numpy_ext cy)
ADD_DEPENDENCIES(cycada/opengl cy)
ADD_DEPENDENCIES(cycada/manager cy)


#PYTHON_ADD_MODULE(cycada ${PROJECT_SOURCE_DIR}/src/cycada/__init__.py)


#TARGET_LINK_LIBRARIES(cycada/fft "cuda" ${CUDA_CUFFT_LIBRARIES})
#SET_TARGET_PROPERTIES(cycada/fft PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
#ADD_DEPENDENCIES(cycada/fft cycada)
#
#TARGET_LINK_LIBRARIES(cycada/random "cuda")
#SET_TARGET_PROPERTIES(cycada/random PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
#ADD_DEPENDENCIES(cycada/random cycada)





