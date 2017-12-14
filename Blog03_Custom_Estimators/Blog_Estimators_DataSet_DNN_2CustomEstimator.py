import tensorflow as tf
import collections
import os
import urllib
import sys
import six

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.3" <= tf_version, "TensorFlow r1.3 or later is needed"

# a = { "f1": [0,1,2,3], "f2": [4,5,6,7]}
# b = six.itervalues(a)
# print a.values()
# for i in b:
#     print i
# sys.exit()

# Windows users: You only need to change PATH, rest is platform independent
PATH = "/tmp/tf_custom_estimators"

# Load Training and Test dataset files
PATH_DATASET = PATH +os.sep + "dataset"
FILE_TRAIN =   PATH_DATASET + os.sep + "boston_train.csv"
FILE_TEST =    PATH_DATASET + os.sep + "boston_test.csv"
URL_TRAIN =   "http://download.tensorflow.org/data/boston_train.csv"
URL_TEST =    "http://download.tensorflow.org/data/boston_test.csv"
def downloadDataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)

    if not os.path.exists(file):
        data = urllib.urlopen(url).read()
        with open(file, "w") as f:
            f.write(data)
            f.close()

downloadDataset(URL_TRAIN, FILE_TRAIN)
downloadDataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)

# The CSV fields in the files
dataset_fields = collections.OrderedDict([
    ('CrimeRate',           [0.]),
    ('LargeLotsRate',       [0.]),
    ('NonRetailBusRate',    [0.]),
    ('NitricOxidesRate',    [0.]),
    ('RoomsPerHouse',       [0.]),
    ('Older1940Rate',       [0.]),
    ('Dist2EmployeeCntr',   [0.]),
    ('PropertyTax',         [0.]),
    ('StudentTeacherRatio', [0.]),
    ('MarketValueIn10k',    [0.])
])

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, repeat_count):
    def decode_csv(line):
        parsed = tf.decode_csv(line, list(dataset_fields.values()))
        # return parsed
        return dict(zip(dataset_fields.keys(), parsed))

    dataset = (
        tf.contrib.data.TextLineDataset(file_path) # Read text line file
            .skip(1) # Skip header row
            .map(decode_csv) # Transform each elem by applying decode_csv fn
            .shuffle(buffer_size=1000) # Obs: buffer_size is read into memory
            .repeat(repeat_count) #
            .batch(128)) # Batch size to use
    iterator = dataset.make_one_shot_iterator()

    batch_features = iterator.get_next()

    for k in batch_features.keys():
        print("key: ", k, ", shape: ", batch_features[k].shape)
    print(batch_features['PropertyTax'].shape)

    batch_labels = batch_features.pop('MarketValueIn10k')
    print("*** input_fn: ", batch_labels.shape)
    print(batch_features['PropertyTax'].shape)
    print(batch_labels.shape)

    # line_records = iterator.get_next()
    # print type(line_records)
    # print line_records.shape
    # batch_features, batch_labels = tf.split(line_records, [9, 1], 1)
    return batch_features,\
           batch_labels


# print("###############################")
# nf,nl = my_input_fn(FILE_TRAIN, 1)
# a = { "a": tf.constant(1.0), "b": tf.constant(2.0) }
# print("nf.type: ", type(nf))
# print("nl.type: ", type(nl))

a = tf.constant([0,1,2,3])
b = tf.expand_dims(a,0)
b = tf.transpose(b)

with tf.Session() as sess:
    print(sess.run(b))
#     rf, rl = sess.run([nf, nl])
#     print("### rl.type: ", type(rl))
#     print("### rl.shape: ", rl.shape)
#     print("### rf.type: ", type(rf))
#     print("### rf: ", rf)
#     print("### rl.0: ", rl[0])
#     print("### rl.1: ", rl[1])
#     print("### rl.2: ", rl[2])


def my_model_fn(features, labels, mode):
    print("*** my_model_fn called with mode:")
    if mode == tf.estimator.ModeKeys.TRAIN:
        print("    TRAIN")
    if mode == tf.estimator.ModeKeys.EVAL:
        print("    EVAL")
    if mode == tf.estimator.ModeKeys.PREDICT:
        print("    PREDICT")

    featuresAsList = features.values()
    features = tf.stack(featuresAsList, 1)

    # First hidden layer
    print("...about to create h1")
    h1 = tf.layers.dense(
        inputs=features,
        units=10,
        activation=tf.nn.relu,
        name='h1')

    # Second hidden layer
    print("...about to create h2")
    h2 = tf.layers.dense(
        inputs=h1,
        units=10,
        activation=tf.nn.relu,
        name='h2')

    # Output value
    MarketValueIn10k = tf.layers.dense(
        inputs=h2,
        units=1,
        activation=None,
        name='MarketValueIn10k')

    # If predicting, return predictions
    if (mode == tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=MarketValueIn10k)

    # Calculate loss
    MarketValueIn10k = tf.Print(MarketValueIn10k, [tf.shape(MarketValueIn10k)], message='***MK***')
    labels = tf.expand_dims(labels, 0)
    labels = tf.transpose(labels)
    loss = tf.losses.mean_squared_error(MarketValueIn10k, labels)

    # If evaluating, return loss
    if (mode == tf.estimator.ModeKeys.EVAL):
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss)

    # If we are not predicting, or evaluating, we must be doing training
    # Training requires us to return loss and training optimizer
    assert mode == tf.estimator.ModeKeys.TRAIN
    train_op = tf.train.AdamOptimizer().minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    print("### now returning estimator spec for training")
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op)


# Create our custom estimator, use my_model_fn to create our model
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=PATH)

# Train our model, use the previously function my_input_fn
# Input to training is a file with training example
# Stop training after 2000 batches have been processed
classifier.train(
    input_fn=lambda: my_input_fn(FILE_TRAIN, None),
    steps=2000)

# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
# evaluate_result = classifier.evaluate(
#     input_fn=lambda: my_input_fn(FILE_TEST, None),
#     steps=4)
# print("Evaluation results")
# for key in evaluate_result:
#     print("   {}, was: {}".format(key, evaluate_result[key]))

# Model evaluated, now use it to predict some house prices
# Let's predict the examples in FILE_TEST
# predict_result = classifier.predict(
#     input_fn=lambda: my_input_fn(FILE_TEST, 1))
# for x in predict_result:
#     print x
