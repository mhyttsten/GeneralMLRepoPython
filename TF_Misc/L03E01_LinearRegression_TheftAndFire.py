import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xlrd

DATA_FILE="../DS_TheftAndFire/fire_theft.xls"

book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows-1

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

Y_predicted = X * w + b

loss = tf.square(Y - Y_predicted, name="loss")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/L03E01_TheftAndFire_LR', sess.graph)
    for i in range(100): # 100 Epochs
        for x, y in data:
            sess.run(optimizer, feed_dict={X:x, Y:y})
    writer.close()

    w_value, b_value = sess.run([w, b])


print "w: %s, b: %s" % (w_value, b_value)


# X, Y = data.T[0], data.T[1]
# plt.plot(X, Y, 'bo', label='Real data')
# plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')

print type(data)
print data.shape[0], ", ", data.shape[1]
data_fires, data_thefts = data[:,0], data[:,1] # = data.T[0], data.T[1] also possible
print "data_fires: {0}".format(str(data_fires.shape))
print "data_thefts: {0}".format(str(data_thefts.shape))
plt.title('Thefts (y) in relation to Fire (x)')
plt.xlabel('Fires')
plt.ylabel('Thefts')
plt.plot(data_fires, data_thefts, 'bo', label='Real data')
plt.plot(data_fires, data_fires*w_value+b_value, 'r', label='Predicted data')

plt.legend()
plt.show()
