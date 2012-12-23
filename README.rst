======
======
Example usage::

import cycada.core as cuda
import numpy as np
import cycada.numpycmd as ncmd
import cycada.cmd as cmd
import cycada.numpy_ext

cuda.init()
d = cuda.Device()
c = d.create_context(flags = 0, opengl = False)

s = c.create_stream()

dbuf = c.alloc(256 * 256 * 4)

hbuf = c.ndarray_host((256, 256), dtype = 'int32') 
#hbuf = np.zeros(256 * 256, "int32")

write_cmd = ncmd.WriteBufferNDArray(dbuf, hbuf)
read_cmd = ncmd.ReadBufferNDArray(hbuf, dbuf)

s.enqueue(write_cmd)
hbuf.fill(10)


print hbuf
s.enqueue(read_cmd)
s.synchronize()
print hbuf

r = "../kernel/test.ptx"
m = c.load_module(open(r).read())
f = m.get_function("fill")
f.parameters = (cuda.MEMParameter, cuda.UINTParameter, cuda.UINTParameter)

k_cmd = cmd.RangeKernel(f, (dbuf.__buffer__, 1001, 256 * 256), block_dim = (256, 1, 1), grid_dim = (256, 1, 1))

s.enqueue(k_cmd)
s.enqueue(read_cmd)
s.synchronize()
print hbuf


