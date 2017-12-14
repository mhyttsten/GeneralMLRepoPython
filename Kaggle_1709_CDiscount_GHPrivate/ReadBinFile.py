import numpy as np
import struct

f = open("/tmp2/cdiscount/file.bin", 'r')
bottleneck_float16 = np.fromfile(f, dtype=np.float16)
print("Type of read value: %s, length: %d" % (type(bottleneck_float16), len(bottleneck_float16)))

for idx, x in enumerate(bottleneck_float16):
    if idx < 10:
      print("Byte, type: %s, value: %f" % (type(x), float(x)))
