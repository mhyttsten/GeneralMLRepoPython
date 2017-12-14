import tensorflow as tf
import numpy as np

from tensorflow.contrib.hooks import ProfilerHook


a = np.random.randn(1)
print a


# weights = tf.constant([1,2,3,4,5,6])
# action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
# responsible_weight = tf.slice(weights, action_holder, [2])
# with tf.Session() as sess:
#     a,_ = sess.run([responsible_weight, weights], feed_dict={ action_holder: [3]})
# print a
#
# # ph = ProfilerHook()
# # print("Hello")
#
# a = tf.feature_column.bucketsized_column(
#     tf.feature_column.numeric_colum(""))


a = tf.constant([2.,3.,4.,5.])
b = tf.constant([6.,7.,8.,9.])
c = a * b
d = tf.sigmoid(a)

with tf.Session() as sess:
    print sess.run(c)
    print sess.run(d)

