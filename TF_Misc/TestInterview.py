import tensorflow as tf

sess = tf.Session()

X = tf.Variable(1)
Y = tf.Variable(1)
T = tf.Variable(1)
T_assign = tf.assign(T, Y)
X_assign = tf.assign(X ,T)
A = tf.assign(Y, T_assign + X_assign)
init = tf.global_variables_initializer()
sess.run(init)
while True:
	print sess.run(A)
