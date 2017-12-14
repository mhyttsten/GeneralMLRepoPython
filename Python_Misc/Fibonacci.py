import tensorflow as tf

Var1 = tf.Variable([1])
Var2 = tf.Variable([1])
Sum = tf.Variable([1])
Sum_assign = tf.assign(Sum, Var1 + Var2)
Var1_assign = tf.assign(Var1, Var2)
Var2_assign = tf.assign(Var2, Sum)

N = 10

with tf.Session() as sess:
	for _ in range(N):
		sess.run([Sum_assign, Var1_assign, Var2_assign])
