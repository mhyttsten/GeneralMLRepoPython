import tensorflow as tf
import os
import sys
import numpy as np


# sess = tf.InteractiveSession()
x = tf.Variable([1], tf.int32)
OP = tf.assign(x, x + 1)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   print(sess.run(OP)) # 1
   print(sess.run(OP)) # 2
   print(sess.run(OP)) # 1
   print(sess.run(OP)) # 2
if True:
    sys.exit()



# print os.getpid()

# Operations that are in each others execution path are only evaluated once
t0 = tf.constant(0)
t1 = tf.Print(t0, [t0], "t1: ")
t2 = tf.Print(t1, [t1], "t2: ")
with tf.Session() as sess:
    sess.run([t2, t1]) # t1, t2 only printed once as part of t2 execution


q = tf.constant([[0,1,2,3], [4,5,6,7], [8,9,10,11],[12,13,14,15]])
# s = tf.constant([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
s = tf.constant([[1,0,0,0]])
w = tf.matmul(s, q)
pa = tf.argmax(w, 1)
with tf.Session() as sess:
    print(sess.run([w, pa]))

print(np.identity(4)[3:4])

print("Final")
np_v1 = np.identity(4)
print(np.matmul([0,1,0,0], np_v1))