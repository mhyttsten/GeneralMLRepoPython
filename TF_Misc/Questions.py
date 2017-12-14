import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 10])


# ------
# Ok
xprint = tf.Print(x, [x[0]], message='x[0]: ', summarize=100)

# TypeError: Tensors in list passed to 'data' of 'Print' Op have types [<NOT CONVERTIBLE TO TENSOR>] that are invalid.
# xprint = tf.Print(xprint, [x.get_shape()], message='x.get_shape: ', summarize=10)
# *** QUESTION #1: How can I print the shape of a tensor?

# -----

# vartest = tf.Variable([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], name="vartest")
# vartest = tf.Print(vartest, [vartest], 'vartest: ', summarize=16)
# *** QUESTION #2: Attempting to use uninitialized value vartest

# ------
y_true = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], name="y_true", dtype=tf.float32)
y_pred = tf.constant([-0.49220103, #0
                   -0.74068147, #1
                   -1.6505798,  #2
                   0.084895976, #3
                   -0.39271244, #4
                   -1.2519366,  #5
                   -5.0827861,  #6
                   10.603678,   #7
                   0.27147627,  #8
                   2.3505583], name="y_pred")  #9
cesmcewl = tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true, name="ce")
cesmcewl = tf.Print(cesmcewl, [cesmcewl], message='ce_automatic: ', summarize=16) # ce: [0.00037579628]

# Alt #2
# cemanual = tf.ones([2,2])
cemanual = tf.nn.softmax(y_pred)
cemanual = tf.Print(cemanual, [cemanual], message='ce_manual_softmax: ', summarize=16) # ce: [0.00037579628]
cemanual = -(y_true * tf.log(cemanual))
# cemanual = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
cemanual = tf.Print(cemanual, [cemanual], message='ce_manual: ', summarize=16) # ce: [0.00037579628]

test = tf.constant([0.99962425], dtype=tf.float32)
test = -tf.log(test)
test = tf.Print(test, [test], message='test: ', summarize=4)

with tf.Session() as sess:
    sess.run([xprint, cesmcewl, cemanual, test], feed_dict={x:[[0,1,2,3,4,5,6,7,8,9]]})

