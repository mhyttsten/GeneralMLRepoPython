# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "/tmp2/cdiscount/input"]).decode("utf8"))
print("...Continuing...")

# Any results you write to the current directory are saved as output.

import os
import io
import bson                       # this is installed with the pymongo package
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.show(block=True)

from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data

# Simple data processing

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


# data = bson.decode_file_iter(open('/tmp2/cdiscount/input/train_example.bson', 'rb'))
# data = bson.decode_file_iter(open('/tmp2/cdiscount/input/test.bson', 'rb'))
data = bson.decode_file_iter(open('/tmp2/cdiscount/input/train.bson', 'rb'))
# print("bson decoded, type: ", type(data))
# print(data)

coder = ImageCoder()
prod_to_category = dict()

def printStuff(d):
    img = d['imgs']
    print("type(img): {}, len(img): {}".format(type(img), len(img)))
    print("type(img[0]): {}".format(type(img[0])))
    print img[0]  # .encode('utf-8')
    nimg = img[0]['picture']
    print("type(nimg[0]): {}".format(type(nimg)))
    print "len of image: ", len(nimg)

writer = tf.python_io.TFRecordWriter('/tmp2/cdiscount/output/tfrecordfile')

# def writeOneRecord(product_id, category_id, img_raw):
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
#         'product_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[product_id])),
#         'category_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[category_id]))
#     }))
#     writer.write(example.SerializeToString())

directory = "/tmp2/cdiscount/raw"
def writeOneRecord(product_id, category_id, img_count, img_raw):
    subdir = directory + os.sep + "%010d" % category_id
    fname = subdir + os.sep + "%010d_%010d_%02d.jpg" % (product_id, category_id, img_count)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    if os.path.exists(fname):
        raise AssertionError("ERROR, fname: %s already existed == duplicate product ids" % fname)
    with open(fname, "w") as f:
        f.write(img_raw)
        f.close()
    # print("Wrote file: %s" % fname)

print("Now iterating over input file")
coder = ImageCoder()
summary_file = open("/tmp2/cdiscount/test/summary.csv", "w")
summary_file.write("category, product, category_product, img_count\n")
product_ids = {}
category_ids = {}
cp_ids = {}
rcount = 0
total_img_count = 0
for i1, d in enumerate(data):
    if i1 < 1:
        printStuff(d)
    # if rcount >= 20:
    #     break

    rcount += 1
    if (rcount % 100000) == 0:
        print("We have now processed: %d records" % rcount)

    # if (rcount >= 100000):
    #     break

    has__id = False
    has_category_id = False
    has_imgs = False
    for key in d:
        if key == '_id':
            if has__id:
                raise AssertionError("Multiple _id keys")
            has__id = True
        elif key == 'category_id':
            if has_category_id:
                raise AssertionError("Multiple category_id keys")
            has_category_id = True
        elif key == 'imgs':
            if has_imgs:
                raise AssertionError("Multiple imgs keys")
            has_imgs = True
        else:
            raise AssertionError("Unexpected key: {}".format(key))
    if not has__id:
        raise AssertionError("No _id key found")
    if not has_category_id:
        raise AssertionError("No category_id key found")
    if not has_imgs:
        raise AssertionError("No imgs key found")

    imgs = d['imgs']
    product_id = d['_id']
    category_id = d['category_id']
    found_image = False
    # product_ids[product_id] = None
    category_ids[category_id] = None
    cp_key = "%d_%d" % (category_id, product_id)
    # print("product_id: %d, category_id: %d" % (product_id, category_id))
    if cp_key in cp_ids:
        cp_ids[cp_key] = cp_ids[cp_key] + 1
    else:
        cp_ids[cp_key] = 1

    # Process images
    img_count = 0;
    for i2, img_dict in enumerate(imgs):
        total_img_count += 1
        found_image = True
        img_count += 1
        for key in img_dict:
            if key != 'picture':
                raise AssertionError("Encountered non-picture key: {}".format(key))
        img_data = img_dict['picture']

        # This is kosher, checked against 100k images in test set
        # img_decode_result = coder.decode_jpeg(img_data)
        # if img_decode_result.shape[0] != 180:
        #     raise AssertionError("Height was not 180, it was: %d" % img_decode_result.shape[0])
        # if img_decode_result.shape[1] != 180:
        #     raise AssertionError("Width was not 180, it was: %d" % img_decode_result.shape[1])
        # if img_decode_result.shape[2] != 3:
        #     raise AssertionError("Channels was not 3, it was: %d" % img_decode_result.shape[2])

        # This code saves the image in its own .jpg file
        # for i in range(len(img_decode_result)):
        #     print("...[%d]: " % (i, img_decode_result[i]))
        # with open("/tmp2/cdiscount/test/cdiscount_%d_%d_%02d.jpg" % (category_id, product_id, count), "w") as f:
        #     f.write(img)
        #     f.close()
        # if product_id > lci:
        #     lci = product_id
        writeOneRecord(product_id, category_id, img_count, img_data)
    summary_file.write("%d, %d, %s, %d\n" % (category_id, product_id, cp_key, img_count))
#    writeOneRecord(product_id, category_id, img_data)

    if not found_image:
        raise AssertionError("No images found within imgs key")

summary_file.close()
writer.close()

print("\n")
print("Total records: %d" % rcount)
print("Total images:  %d" % total_img_count)
print("Number of category ids: %d" % len(category_ids.keys()))




