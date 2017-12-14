import tensorflow as tf
import os
import pickle
import datetime
from Python_Misc import DatasetShuffler


def print_files_structure(files):
    print("Data structure")
    print("...Type: {}".format(type(files)))
    print("...Length: {}".format(len(files)))
    for idx in range(5):
        elem = files[idx]
        print("...[{}]: {}, {}".format(idx, type(elem), elem))

def exec_dataset_iteration(files):
  print("Creating dataset: {}".format(datetime.datetime.now()))
  ds = (tf.data.Dataset.from_tensor_slices(files)
    .shuffle(len(files))
    .repeat(2)
    .batch(100))

  print("Creating iterator: {}".format(datetime.datetime.now()))
  iter = ds.make_one_shot_iterator()
  print("Creating batch tensor: {}".format(datetime.datetime.now()))
  batch = iter.get_next()

  with tf.Session() as sess:
    print("Will no execute batch 1: {}".format(datetime.datetime.now()))
    b1 = sess.run(batch)
    print("Will no execute batch 2: {}".format(datetime.datetime.now()))
    b2 = sess.run(batch)
  print("Done: {}".format(datetime.datetime.now()))

tf.reset_default_graph()

def test_pickle_file():
    print("\n--------------------------")
    print("Testing pickle file")
    files = []
    if os.path.exists('/tmp2/cdiscount/pickled_dataset_input.pkl'):
        print("Reading pickle file: {}".format(datetime.datetime.now()))
        with open('/tmp2/cdiscount/pickled_dataset_input.pkl', 'rb') as f:
            print("Successfully opened pickle file: {}".format(datetime.datetime.now()))
            files = pickle.load(f)
        files = files
    print_files_structure(files)
    exec_dataset_iteration(files)

def test_memory_generation():
    print("\n--------------------------")
    print("Testing memory generation")
    files = []
    for x in range(12371293):
        files.append(("100000{:04d}/0000{:04d}{:04d}_100027{:04d}_01.jpg").format(x, x, x, x))
    print_files_structure(files)
    exec_dataset_iteration(files)

def test_pickle_file_dataset_shuffler():
    print("\n--------------------------")
    print("Testing pickle file DatasetShuffler")
    files = []
    if os.path.exists('/tmp2/cdiscount/pickled_dataset_input.pkl'):
        print("Reading pickle file: {}".format(datetime.datetime.now()))
        with open('/tmp2/cdiscount/pickled_dataset_input.pkl', 'rb') as f:
            print("Successfully opened pickle file: {}".format(datetime.datetime.now()))
            files = pickle.load(f)
        files = files
    print("Total number of files: {}".format(len(files)))
    dss = DatasetShuffler.DatasetShuffler(files, 1, 1000000)
    batch_count = 0
    while dss.get_next():
        batch = dss.batch
        for idx, f in enumerate(batch):
            label_name = f[0:f.find('/')]
            print("Processing file from label: {}".format(label_name))
            if idx >= 4:
                break
        batch_count += 1
        print("Now processing batch: {}, had: {} number of elements".format(batch_count, len(dss.batch)))

test_pickle_file_dataset_shuffler()

# test_pickle_file()
# test_memory_generation()
