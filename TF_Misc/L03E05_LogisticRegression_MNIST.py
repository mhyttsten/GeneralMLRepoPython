import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


alpha = 0.01
batch_size = 128
n_epochs = 30

mnist = input_data.read_data_sets('/tmp2/mnist', one_hot=True)

X = tf.placeholder(tf.float32, shape=[batch_size, 784])
Y = tf.placeholder(tf.float32, shape=[batch_size, 10])

w = tf.Variable(tf.random_normal(shape=[784,10], stddev=0.01))
b = tf.Variable(tf.zeros(shape=[1,10]))

logits = tf.matmul(X, w) + b
entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
loss = tf.reduce_mean(entropy)

opt = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss)

with tf.Session() as sess:
  writer = tf.summary.FileWriter("./my_graph/L03E05_MNIST_LogReg", sess.graph)
  sess.run(tf.global_variables_initializer())
  n_batches = int(mnist.train.num_examples/batch_size)
  for i in range(n_epochs):
      for _ in range(n_batches):
          X_batch, Y_batch = mnist.train.next_batch(batch_size)
          _, loss_batch = sess.run([opt, loss], feed_dict={X:X_batch, Y:Y_batch})
          total_loss += loss_batch



