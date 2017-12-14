import numpy as np


a = ["1", "2"]
b = np.array(a, dtype=np.int32)

for x in b:
    print(type(x))

d = {"a": 1, "b": 2, "c": 3}
for key in d.keys():
    print("Key: %s, size: %d" % (key, len(d.items())))
    del d[key]
