import os


# with open("/tmp2/cdiscount/retrain_inception/bottleneck_dir/1000000237/0000012600_1000000237_01.jpg_inception_v3.txt", "r") as f:
with open("/tmp2/cdiscount/retrain_inception/bottleneck_dir/1000000271/0013526333_1000000271_04.jpg_inception_v3.txt", "r") as f:
    str = f.read()

split_str = str.split(",")
print("Length: %d" % len(split_str))

