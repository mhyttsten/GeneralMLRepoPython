import tensorflow as tf
import numpy as np

# x = [[ 1,  2,  3,  5],
#      [ 7,  9, 11, 13],
#      [17, 19, 23, 27],
#      [29, 31, 37, 41]]
x = np.zeros([1, 4, 4, 1], dtype=np.float32)
# batch=1, example=4x4, channels=1
x[0,0,0,0] =  1.; x[0,0,1,0] =  2.; x[0,0,2,0] =  3.; x[0,0,3,0] =  5.
x[0,1,0,0] =  7.; x[0,1,1,0] =  9.; x[0,1,2,0] = 11.; x[0,1,3,0] = 13.
x[0,2,0,0] = 17.; x[0,2,1,0] = 19.; x[0,2,2,0] = 23.; x[0,2,3,0] = 27.
x[0,3,0,0] = 29.; x[0,3,1,0] = 31.; x[0,3,2,0] = 37.; x[0,3,3,0] = 41.
x = tf.pack(x)
xprint = tf.Print(x, [x], message='x\n', summarize=16)

# W = [[ 1, 0, 0],
#      [ 0, 1, 0],
#      [ 0, 0, 1]]
# filter=3x3, in_channels=1, out_channnels=1
W = np.zeros([3, 3, 1, 1], dtype=np.float32)
W[0,0,0,0] = 1.; W[1, 1, 0, 0] = 1.; W[2, 2, 0, 0] = 1.
W = tf.pack(W)
Wprint = tf.Print(W, [W], message='W\n', summarize=4)

conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
convPrint = tf.Print(conv, [conv], message='conv\n', summarize=16)

mpool = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
mpoolPrint = tf.Print(mpool, [mpool], message='pool\n', summarize=16)

# When you add a bias of size of last conv dimension, it is added to everything
conv1 = tf.zeros([1, 2, 2, 2])
bias = [-1,2]
convBiasSum = conv1 + bias
convBiasSumPrint = tf.Print(convBiasSum, [convBiasSum], message='conv_bias_sum\b', summarize=12)
convBiasReLU = tf.nn.relu(convBiasSum)
convBiasReLUPrint = tf.Print(convBiasReLU, [convBiasReLU], message='conv_bias_relu\b', summarize=12)

topk_ypred = [[.33, .67, 0], [.6, .4, 0], [0, .75, .25]]
topk_ytrue = [2, 0, 2]
topk_res = tf.nn.in_top_k(topk_ypred, topk_ytrue, 2)
topk_res = tf.Print(topk_res, [topk_res], message='topk: ', summarize=9)

with tf.Session() as sess:
    sess.run(convPrint)
    sess.run(mpoolPrint)
    sess.run(convBiasSumPrint)
    sess.run(convBiasReLUPrint)
    sess.run(topk_res)

# Input
#  0,  0,  0,  0,  0,  0
#  0,  1,  2,  3,  5,  0
#  0,  7,  9, 11, 13,  0
#  0, 17, 19, 23, 27,  0
#  0, 29, 31, 37, 41,  0
#  0,  0,  0,  0,  0,  0
#
# Filter
#  1 0 0
#  0 1 0
#  0 0 1
#
# After conv: Result
# 10, 13, 16,  5
# 26, 33, 40, 16
# 48, 63, 73, 38
# 29, 48, 56, 64
#

