import argparse
from datetime import datetime
import hashlib
import os.path
import shutil
import random
import re
import sys
import tarfile

import datetime
import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

image_dir = "/tmp2/cdiscount/raw"
part_dir  = "/tmp2/cdiscount/partition_dir"
num_partitions = 32

if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    sys.exit()
if not gfile.Exists(part_dir):
    print("Partition directory '" + part_dir + "' not found.")
    sys.exit()
count = 0
for x in gfile.Walk(part_dir):
    # Root folder is ok
    if count >= 1:
        print("Partition directory not empty, e.g. first entry found was: %s" % str(x))
        sys.exit()
    count += 1


def get_files_count(partition_labels, dir_dict_orig):
    count = 0
    for l in partition_labels:
        count += len(dir_dict_orig[l])
    return count

def find_closest_label(dir_dict, count):
    closest_label = None
    for key in dir_dict:
        items = dir_dict[key]
        if closest_label == None and len(items) < count:
            closest_label = key
        else:
            curr_count = len(dir_dict[key])
            closest_count = len(dir_dict[key])
            if curr_count <= count and curr_count > closest_count:
                closest_label = key
    return closest_label

def get_files_dictionary(dir):
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        sys.exit()

    result = {}
    print("Now listing subdirs for: %s" % image_dir)
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    is_root = True
    total_files = 0
    subdir_count = 0
    print("Now starting analysis for each subdir")
    for sub_dir in sub_dirs:
        if is_root:
            is_root = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print("No files found in sub_dir: %s" % sub_dir)
            continue
        if len(file_list) < 20:
            print("Subdir has less than 20 images which may cause problems: %s" % sub_dir)
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        # print("Label name: %s" % label_name)
        # print("File list: %s" % file_list)
        # break
        result[label_name] = file_list
        total_files += len(file_list)
        subdir_count += 1
        # print("Processed subdir: %s, with: %d files. Total now at: %d" %(label_name, len(file_list), total_files))
        if (subdir_count % 10) == 0:
            print("Processed another subdir batch, now at: %d" % subdir_count)
            print("...This subdir: %s, with: %d files. Total now at: %d" %(label_name, len(file_list), total_files))
    return result

dir_dict = get_files_dictionary(image_dir)
total_files = 0
for key in dir_dict:
    total_files += len(dir_dict[key])
print("Total number of files in dataset: %d" % total_files)

files_per_partition = int(total_files / num_partitions)
print("Files per partition: %d" % files_per_partition)

dir_dict_orig = dir_dict.copy()
partition_labels = []
for partition in range(num_partitions-1):
    print("Now doing partition: %d" % partition)
    # If this is the last partition, take everything remaining
    current_partition_filecount = 0
    labels = []
    while len(dir_dict):
        label = find_closest_label(dir_dict, files_per_partition-current_partition_filecount)
        if label != None:
            # We found another label to consume
            labels.append(label)
            current_partition_filecount += len(dir_dict[label])
            del dir_dict[label]
        else:
            # We are full, close this one and start next
            partition_labels.append(labels)
            break
# Set last partition to everything remaining
partition_labels.append(dir_dict.keys())
print("Len of partition_labels: %d" % len(partition_labels))

max_count = None
max_partition = None
min_count = None
min_partition = None
total_elems = 0
for i in range(num_partitions):
    count = get_files_count(partition_labels[i], dir_dict_orig)
    total_elems += count

    if max_count == None or count > max_count:
        max_count = count
        max_partition = i
    if min_count == None or count < min_count:
        min_count = count
        min_partition =i

    print("Partition: %s, had count: %d" % (str(i), count))
    labels = partition_labels[i]
    # for i in labels:
    #     print("...label: %s, had: %d elems" % (i, len(dir_dict_orig[i])))

print("Total elements: %d" % total_elems)
print("Max partition was: %d, with: %d elements" % (max_partition, max_count))
print("Min partition was: %d, with: %d elements" % (min_partition, min_count))

# Move the stuff
for i in range(len(partition_labels)):
    target_dir = part_dir + os.sep + ("%02X" % i)
    os.mkdir(target_dir)
    labels = partition_labels[i]
    for label in labels:
        src_dir = image_dir + os.sep + label
        # print("Will now move: %s, to: %s" % (src_dir, part_dir))
        shutil.move(src_dir, target_dir)

print("Done")

# Input:  400000 files, total of 3GB
# Output: 400000 * 4096 = 2GB

