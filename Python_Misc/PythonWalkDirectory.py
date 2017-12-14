import os


# directory = "/tmp2/cdiscount/raw"
directory = "/tmp2/cdiscount/input/train.unpacked"

print("Starting")

# os.chdir(directory)

a = "hello"
print a.find("/")

dir_entries = {}
count = 0
for x in os.walk(directory):
    count += 1
    if count == 1:
        continue
    if (count % 100) == 0:
        print("Now processed: {}".format(count))
        break

    io = x[0].rfind("/")
    if io != -1:
        sub_directory = x[0][io+1:]
    else:
        sub_directory = x[0]
    files = x[2]
    if sub_directory in dir_entries:
        assert False, "Directory already existed"
    dir_entries[sub_directory] = files
    # print("Directory: {}, has: {}, number of files".format(sub_directory, len(files)))

print("Done counting files, now sorting")
result = sorted(dir_entries.iteritems(), key=lambda (k,v): -len(v))
for i in range(15):
    print("[{}]: {} files".format(result[i][0], len(result[i][1])))

print("Finished")

