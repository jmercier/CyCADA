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
cimport libc.stdlib as stdlib

cimport cuda.libcuda as libcuda
cimport cycuda.core as core

from . import core as cuda

cimport numpy as cnp

import functools
import operator
import types

"""
This import is MENDATORY in order to initialize the function table of numpy.
DO NOT REMOVE otherwise it will segfault.
"""
cnp.import_array()
"""
This import is MENDATORY in order to initialize the function table of numpy.
DO NOT REMOVE otherwise it will segfault.
"""


# ------------------------------------------------------------------------------
cdef class numpy_base:
    """
    """
    cdef:
        core.LLHostBuffer _buf
        cnp.npy_intp *_dims

    def __dealloc__(self):
        stdlib.free(self._dims)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
cdef cnp.ndarray _host_ndarray(core.LLContext ctx, tuple shape, object dtype = 'float'):
    """
    """

    cdef cnp.dtype dt           = cnp.dtype(dtype)

    cdef:
        size_t ndim             = len(shape)
        size_t nbytes           = functools.reduce(operator.mul, shape) * dt.itemsize
        core.LLHostBuffer llbuf = core._LLHostBuffer_factory(ctx, nbytes)

    cdef cnp.npy_intp *dims      = <cnp.npy_intp *>stdlib.malloc(sizeof(cnp.npy_intp) * ndim)

    for i from 0 <= i < ndim:
        dims[i] = shape[i]


    cdef numpy_base base        = numpy_base.__new__(numpy_base)
    base._buf                   = llbuf
    base._dims                  = dims


    cdef cnp.ndarray ary         = cnp.PyArray_SimpleNewFromData(ndim, dims, dt.type_num, llbuf._handle)

    cnp.set_array_base(ary, base)
    return ary
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def ndarray_host(context, shape, dtype = 'float'):
    return _host_ndarray(context.__ctx__, shape, dtype);
# ------------------------------------------------------------------------------


cuda.Context.ndarray_host = types.MethodType(ndarray_host, None, cuda.Context)
