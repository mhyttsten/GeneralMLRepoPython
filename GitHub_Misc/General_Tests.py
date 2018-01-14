import numpy as np



# --------------------------
gamma = 0.9

print(10 + (20*gamma) + (30*gamma**2) + (40*gamma**3))
print(20 + (30*gamma) + (40*gamma**2))

rp = [10, 20, 30, 40]
r = np.array(rp)

discounted_r = np.zeros_like(r)
running_add = 0
for t in reversed(range(0, r.size)):
    running_add = running_add * gamma+ r[t]
    discounted_r[t] = running_add
print(r)
print(discounted_r)

# --------------------------
# a = np.array([
#     [0.0, 0.1, 0.2, 0.3],
#     [1.0, 1.1, 1.2, 1.3],
#     [2.0, 2.1, 2.2, 2.3],
#     [3.0, 3.1, 3.2, 3.3],
# ])
# b = a[:,2]  # b is a reference into
# print(b)
# a[:,2] = [9.0, 9.1, 9.2, 9.3]  # so this changes the matrix and b
# print(a[:,2])
# print(b)
# print(np.vstack(a[:,2]))

# --------------------------
# from collections import Counter
# def rearrange_str(sample):
# 	str_arr = []
# 	c = Counter(sample)
# 	for k, v in c.most_common(len(c)).items():
# 		str_arr.append(k*v)
# 	return ''.join(str_arr)
#
# def rearrange_str1(sample):
# 	str_arr = []
# 	c = Counter(sample)
# 	for k, v in c.most_common(len(c)):
# 		str_arr.append(k*v)
# 	return ''.join(str_arr)
#
# print(rearrange_str1("This is a sample string"))

# def string_counter(input):
# 	counter = {}
# 	for let in input:
# 		if let not in counter:
# 			counter[let] = 1
# 		else:
# 			counter[let] += 1
# 	return counter
#
# def rearrange_str(str_counter):
# 	str_arr = []
# 	for k, v in sorted(str_counter.items(), key=lambda x: x[1], reverse=True):
# 		str_arr.append(k*v)
# 	return ''.join(str_arr)
#
# print(rearrange_str(string_counter("This is a sample string")))


# a = [["a1","a2"], ["b1","b2"], ["c1","c2"]]
# b = dict(a)
# print(b)


