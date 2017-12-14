import tensorflow as tf


# Run program
# $ tensorboard --logdir="./graphs" --port 6006
# Browse to http://localhost:6006

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(x)
writer.close()



