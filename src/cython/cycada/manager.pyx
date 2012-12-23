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

cimport cuda.libcuda as libcuda
cimport cycada.core as core

from . import core as cuda

import md5


cdef class ModuleDescription:
    def __init__(self, context, filename):
        data = open(filename, 'r').read()
        self._checksum = md5.new(data).hexdigest()
        self._functions = []
        self._module = context.load_module(data)

    cdef reload(self):
        data = open(self._filename, 'r').read()
        checksum = md5.new(data).hexdigest()
        if checksum != self._checksum:
            pass

    cdef:
        object _module
        list _functions
        bytes _checksum
        bytes _filename

cdef class DynamicModuleManager:
    cdef:
        dict __modules__

    def __reload__(self, name):
        pass

    def reload(self):
        for k, i in self.__modules__.iteritems():
            i.reload()


    def add_managed_module(self, context, name, filename):
        cdef ModuleDescription md = ModuleDescription(context, filename)
        self.__modules__[name] = md

    def function(self, module, name):
        if name not in self.__modules__:
            raise AttributeError("No module named %s loaded" % name)

        cdef ModuleDescription md = self.__modules__[name]
        function = md._module.get_function(name)

        return function
