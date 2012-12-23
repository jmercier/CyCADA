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
cdef extern from "nvToolExt.h":
    ctypedef enum nvtxColorType_t:
        NVTX_COLOR_UNKNOWN
        NVTX_COLOR_ARGB

    ctypedef enum nvtxPayloadType_t:
        NVTX_PAYLOAD_UNKNOWN
        NVTX_PAYLOAD_TYPE_UNSIGNED_INT64
        NVTX_PAYLOAD_TYPE_INT64
        NVTX_PAYLOAD_TYPE_DOUBLE

    ctypedef enum nvtxMessageType_t:
        NVTX_MESSAGE_UNKNOWN
        NVTX_MESSAGE_TYPE_ASCII
        NVTX_MESSAGE_TYPE_UNICODE

    ctypedef nvtxRangeId_t

    cdef nvtxRangeId_t nvtxRangeStartA(char *message)
    cdef nvtxRangeId_t nvtxRangeStartW(wchar *message)
    cdef nvtxRangeEnd(nvtxRangeId_t)

    cdef int nvtxRangePushA(char *message)
    cdef int nvtxRangePushW(wchar *message)
    cdef int nvtxRangePop()

    cdef void nvtxNamecategoryA(unsigned int category, char *name)
    cdef void nvtxNamecategoryW(unsigned int category, wchar *name)
    cdef void nvtxNameOsThreadA(unsigned int threadId, char *name)
    cdef void nvtxNameOsThreadW(unsigned int threadId, wchar *name)

