from collections import Counter

from collections import Counter

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


def string_counter(input):
	counter = {}
	for let in input:
		if let not in counter:
			counter[let] = 1
		else:
			counter[let] += 1
	return counter

def rearrange_str(str_counter):
	str_arr = []
	for k, v in sorted(str_counter.items(), key=lambda x: x[1], reverse=True):
		str_arr.append(k*v)
	return ''.join(str_arr)

print(rearrange_str(string_counter("This is a sample string")))



# a = [["a1","a2"], ["b1","b2"], ["c1","c2"]]
# b = dict(a)
# print(b)


