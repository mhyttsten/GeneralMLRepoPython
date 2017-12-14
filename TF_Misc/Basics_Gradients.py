import tensorflow as tf

x = tf.Variable(2.0)
y = 2.0 * (x**3)
z = 3.0 + (y**2)

grad_z = tf.gradients(y, [x])
with tf.Session() as sess:
    sess.run(x.initializer)
    print sess.run(grad_z)

# dz/dy = 2y = 2* (2* (2**3)) = 32
# dy/dx = 6 * x**2 --> 6*4 == 24
# dz/dx = dz/dy * dy/dx = 768

