'''
tf.layers
---------
- Convolutional layers
- Pooling layers (max, average)
- Batch normalization layer
- Dropout layers
- Dense layers

tf.losses
---------
- Different loss functions



Dataset
-------
TFRecord:
- Binary file containing tf.train.Example protobuf instances
- You can write them using tf.python_io.TFRecordWriter
- Each tf.train.Example contains one or more features
- Input pipeline typically converts these to Tensors

- tf.parse_single_example( # Parses a single tf.train.Example proto
    proto, # the proto
    features) # dict of

Dataset:
- A Dataset comprises elements
  Each element contains 1+ Tensor objects called components
  Each component has a tf.DType (output_types), and a tf.TensorShape (output_shapes)
  The output_shape could be a partially specified static shape of each element
  First dimension of each element must be the same, e.g ([4], [4,10])
- ds.zip(ds1, ds2)
  For each next_elem it will return (ds1[i], ds2[i])

Source
- ds = Dataset.from_tensor_slices(tensor) # from memory
  argument can also be a string/value: from_tensor_slices({"a": ..., "b": ...})
  For example if they represent different featuers
- ds = TFRecordDataSet() # From disk, inherits DataSet I presume
- ds = Dataset.range(5, 1, -2) # == [5, 3]
- (NEW): `Dataset.from_generator()` constructs a dataset from a Python generator.

Transform
- ds.map(map_fn) # Apply function to each element
- ds.flat_map(map_fn) # Also flattens the result???
- ds.padded_batch(2, padded_shapes=[None]) # E.g. [[1], [2, 2]] will become [[1,0], [2,2]]
  (but observe, only way to get the that dataset is through .map)
- ds.batch(128) # set batch size
  1. Let's say your map(decode_fn) returns 10 CSV columns
     Not setting batch size, iterator.get_next will return shape (10,)
     Setting batch size, it will return (?, 10) (true for size 1 or larger)
     Setting a batch size will pick size random elements from the dataset
  2. Custom estimators, labels comes out of data set as single-dim list with batch_size
     You need to extend_dim, and transpose to use in loss calcs



Iterator: Main way to extract data from a DataSet

1. One-shot: Iterates through dataset once
  i = ds.make_one_shot_iterator() # ***
  next_elem = i.get_next()
  ... later, value = sess.run(next_elem)

2. Initializable: You need to run Iterator.initializer
  max_value = tf.placeholder(dtype=tf.int32, shape=[])
  ds = tf.contrib.data.Dataset.range(max_value)
  i = ds.make_initializable_iterator() # ***
  next_elem = i.get_next()
  sess.run(i.initializer, feed_dict={max_value:10})
  ... Now you can call sess.run(next_elem) 10 times

3. Reinitializable: Can be initialized from multiple dataset objects
  i = Iterator.from_structure(dst.output_types, dst.output_values) # ***
  next_elem = i.get_next()
  dst_init = i.make_initializer(dst)
  dsv_init = i.make_initializer(dsv)
  ... When you run dst/v_init you initialize the ds the iterator should use
  ... Both ds must have same structure (given in from_structure)
  ... When running dst/v_init, initializer is initialized from start/scratch

4. Feedable: Select which iterator to use in sess.run() through feed_dict
  Difference from Reinitializable is that switching does not start from scratch
  handle = tf.placeholder(tf.string, shape=[])
  i = Iterator.from_string_handle(handle, dst.output_types, dst.output_values)
  next_elem = i.get_next()
  dst_iter = dst.make_one_shot_iterator()
  dsv_iter = dst.make_initializable_iterator()
  dst_iter_handle = sess.run(dst_iter.string_handle)
  dsv_iter_handle = sess.run(dsv_iter.string_handle)
  ... Now you can select which iterator to use in sess.run() call
  sess.run(next_elem, feed_dict={handle:dst_iter_handle})

- Termination: tf.errors.OutOfRangeError is thrown
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    // At this point iterator must be initialized again
    break

*** Reading Data
1. Numpy as part of graph
  with np.load("/var/data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]
  ds = tf.contrib.data.Dataset.from_tensor_slices((features, labels)) # Makes data part of graph, bad!

2. Numpy as part of feed_dict (*** how is feed_dict passed to graph execution and why better than #1?)
  f_ph = tf.placeholders(features.dtype, features.shape)
  l_ph = tf.placeholders(features.dtype, features.shape)
  ds = tf.contrib.data.Dataset.from_tensor_slices((f_ph, l_ph))
  i = ds.make_initializable_iterator()
  sess.run(i.initializer, feed_dict={f_ph: features, l_ph: labels})

3a. TFRecordDataset (TFRecord is a record-oriented binary format)
  fnames = tf.placeholder(tf.string, shape=[None])
  ds = tf.contrib.data.Dataset(fnames)
  i = ds.make_initializable_iterator()
  ne = i.get_next()
  sess.run(i.initializer, feed_dict={fnames: ["f1", "f2"]})
  ... You can of course just provide real filenames for TFREcordDataset __init__

b. Parsing each tf.train.Example proto
  def _parse_fn(example_proto):
    features = { "image": tf.FixedLenFeature(), tf.string, default_value=""),
      "label": tf.FixedLenFeature(), tf.int32, default_value=0)
    }
    p_features = tf.parse_single_example(example_proto, features)
    return p_features["image"), p_features["label"]
  fnames = ["f1", "f2"]
  ds = tf.contrib.data.TFRecordDataset(fnames)
  ds.map(_parse_fn)

4. TFTextDataset: Extracts lines from a file
  fnames = ["f1.txt", "f2.txt"]
  ds = tf.contrib.data.Dataset.from_tensor_slices(fnames)
  ds = ds.flat_map(lambda filename:
    tf.contrib.data.TFTextDataset(filename)
      .skip(1) # Skip header row
      .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), '#')))) # Filter out lines starting w '#'

5. Read image, decode and resize it
  def _parse_fn(fname, label):
    image_str = tf.read(fname)
    imageOrig = tf.image.decode_image(image_str)
    image = tf.image.resize_images(imageOrig, [28,28])
    return image, label
  fnames = tf.constant(["f1", "f2", ...])
  labels = tf.constant([0, 37, ...]) label of image in file[i]
  ds = tf.contrib.data.Dataset.from_tensor_slices((fnames, labels)) *** iterates over 1 item each for _parse_fn?
  ds.map(_parse_fn)
  ... You could also use tf.py_func if you need to execute Python code from sess.run()




Goodies
tds = tf.contrib.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))





'''

import tensorflow as tf
import sys

ds1 = tf.contrib.data.Dataset.from_tensor_slices(
    (tf.ones([4,2]),
    tf.ones([4])*2))
ds2 = tf.contrib.data.Dataset.from_tensor_slices(
    tf.ones([4,2])*3)
# ds3 = tf.contrib.data.Dataset.zip((ds1, ds2))
ds3 = tf.contrib.data.Dataset.zip((ds1, ds2))
print(ds1.output_types)  # ==> "tf.float32"
print(ds1.output_shapes)  # ==> "(10,)"
# print(ds3.output_types)  # ==> "tf.float32"
# print(ds3.output_shapes)  # ==> "(10,)"

'''
- batch(v): Will batch v entries together when next_element is evaluated
  If at end, and v elements are not available, whatever number available will be extracted
- repeat(v): How many times to repeat data serving. If you leave out 'v', repeat is indefinately
  [if you want stats after each epoch, don't use repeat, catch tf.errors.OutOfRangeError instead,
  use an overarching loop, and reinit iterator at each start: sess.run(iterator.initializer) at each 
- shuffle(buffer_size): Read buffer size, and randomize within it
'''

# ds3 = tf.contrib.data.Dataset.range(4)
# ds3 = ds3.repeat()
# ds3 = ds3.batch(3)
# iter = ds3.make_one_shot_iterator()
# next_element = iter.get_next()
# with tf.Session() as sess:
#   count = 0;
#   while True:
#       try:
#           a = sess.run(next_element)
#           count += 1
#           print("At: {}, got element: {}".format(count, a))
#       except tf.errors.OutOfRangeError:
#           print("Caught OutOfRangeError")
#           break

# a = tf.constant([])
# b = tf.constant([2,2])
# dataset = tf.contrib.data.Dataset.from_tensor_slices([a, b])

# dataset = tf.contrib.data.Dataset.range(100)
# dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
# #dataset = dataset.padded_batch(1, padded_shapes=[None])
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#   print(sess.run(next_element))
#   print(sess.run(next_element))
#   print(sess.run(next_element))
#   print(sess.run(next_element))
#   # print(sess.run(next_element))

tf_version = tf.__version__
print tf_version
# dps_train = tf.constant([
#     # Sun tomorrow?, Sun today?, Temperature today (F)
#     # It will be sun tomorrow regardless of whether it rains today or not,
#     # as long as the temperature is 65 degrees (or above)
#     [0, 0, 45], [0, 0, 50], [0, 1, 55], [0, 1, 60],
#     [1, 0, 65], [1, 0, 70], [1, 1, 75], [1, 1, 80]])
# dps_infer = tf.constant([
#     [0, 1, 0, 20], [1, 0, 1, 70], [0, 1, 1, 30]])
#
#
# def my_input_fn(dps):
#    dataset = tf.contrib.data.Dataset.from_tensor_slices(dps)
#    iterator = dataset.make_one_shot_iterator()
#    next_batch = iterator.get_next()
#    label = tf.equal(next_batch[:1], 1)
#    features = next_batch[1:]
#    return { 'i1': features[:1], 'i2': features[1:] }, label

# dps_train = [((x%2), x) for x in range(1, 99)]
# dps_predict = [(99,1), (99, 32), (99, 33), (99, 34)]
dps_train = [((x<50), x) for x in range(1, 99)]
dps_predict = [(99,1), (99, 62), (99, 33), (99, 51)]

def my_input_fn(dps, is_predict=True):
    def parse(elem):
        return { "in_value": elem[1] }, elem[0]
    dataset = tf.contrib.data.Dataset.from_tensor_slices(dps)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(128)
    if not is_predict:
        dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    return next_batch

classifier = tf.estimator.DNNClassifier(
   feature_columns=[tf.feature_column.numeric_column('in_value')],
   hidden_units=[100, 100], # h1 and h2, each with 3 neurons
   n_classes=2, # Two output nodes. Probability of Sun / No Sun
   model_dir='/tmp2/My1stEstimator') # Checkpoints etc are stored here

classifier.train(
   input_fn=lambda: my_input_fn(dps_train, False),
   steps=2000) # Train for 2000 batches of data
#
# r = classifier.evaluate(
#     input_fn=lambda: my_input_fn(dps_train),
#     steps=2000)  # Evaluate using 100 batches of data

result = classifier.predict(input_fn=lambda: my_input_fn(dps_predict))
for idx,i in enumerate(result):
    print("Result, index: {}, value: {}".format(idx, i))

print("Done with evaluation")

# predictions = classifier.predict(input_fn=lambda: my_input_fn(dps_infer))
# for i, p in enumerate(predictions):
#     print("i:{}, p:{}".format(i, p))
# print("hello")



# f, l = my_input_fn()
# with tf.Session() as sess:
#   fr, lr = sess.run([f, l])
#   print("f: {}, l: {}".format(fr, lr))
#
# my_input_fn()
