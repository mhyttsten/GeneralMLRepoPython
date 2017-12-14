import tensorflow as tf
import numpy as np

def lstmTF():
  lstm_size = 1 # same as the output_size
  lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

  # In tf/python/ops/rnn_cell_impl.py
  # state = _zero_state_tensors(state_size, batch_size, dtype)
  # c, h = state # both filled with 0, and have shape: [lstm_size]
  state = lstm.zero_state(1, tf.float32)
  print type(state)


  c = tf.constant([[1, 2]], dtype=tf.float32)
  rnn = lstm(c, state)
  # print lstm.state_size
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r = sess.run(state)
    print "Initial value of c: ", r.c, ", h: " ,r.h
    r = sess.run(rnn)
    print type(r[0]), ", size: ", len(r[0]), ", e[0]: ", r[0]
    print type(r[1]), ", size: ", len(r[1]), ", e[0]: ", r[1][0], ", e[1]: ", r[1][1]
    print "c: ", r[1].c, "h: ", r[1].h

# Argument to BasicLSTMCell defines size of c & h arrays
# This means it also defines size of output (since that is h)
def lstmMagnus():

  # *** Now _RNNCell.zero_state (tf/python/ops/rnn_cell_impl.py)
  #     (_zero_state_tensors)
  output_size = 2
  c = np.zeros([output_size], dtype=np.float32)
  h = np.zeros([output_size], dtype=np.float32)

  inputs = np.array([1, 2, 3], dtype=np.float32) # input batch

  # *** Now __linear call (tf/contrib/rnn/python/ops/core_rnn_cell_impl.py)
  # concat = _linear([inputs, h], 4 * self._num_units, True)
  #   def _linear(args, output_size, bias, bias_start=0.0):
  args = [inputs, h] # Now in
  output_size = 4 * len(c) # i, j, f, o

  shapes = [a.shape for a in args]
  print "shapes: ", shapes
  # total_arg_size = num input params + num h params (can be different)
  #    since size of h is given by BasicLSTMCell(size_of_h)
  #    and size of input is what is fed in each iteration
  #    shape[0] is always the same == batch_size
  total_arg_size = 0
  for shape in shapes:
    total_arg_size += shape[0]
  print "total_arg_size: ", total_arg_size

  weights = np.empty([total_arg_size, output_size], dtype=np.float32)
  biases = np.zeros([output_size], dtype=np.float32)

  # args = [ [[ip1], [ip2], ..., [ipn]],
  #          [[hu1], [hu2], ..., [hun]] ]
  #    where n is batch_size, if batch_size == 1
  # args = [ [[ip1]],
  #          [[hu1]] ]
  # array_ops.concat(args, 1)
  #   t1 = [[1,2,3],[4,5,6]]
  #   t2 = [[7,8,9],[10,11,12]]
  #   array_ops.concat([t1,t2], 0) --> [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
  #   array_ops.concat([t1,t2], 1) --> [[1,2,3,7,8,9],[4,5,6,10,11,12]]
  concat_result = np.concatenate(args, axis=0) # = input + h
  print "concat_result: ", concat_result
  print "weights:\n", weights

  # res = math_ops.matmul(array_ops.concat(args, 1), weights)
  # return nn_ops.bias_add(res, biases)
  res = np.matmul(concat_result, weights)
  res = np.add(res, biases)
  print "res:\n", res

  # *** Now in BasicLSTMCell.__call__ (tf/contrib/rnn/python/ops/core_rnn_cell_impl.py)
  # split_test = np.split(np.array(
  #     [[ 1, 2 ,3, 4, 5, 6, 7, 8],
  #      [ 9,10,11,12,13,14,15,16],
  #      [17,18,19,20,21,22,23,24],
  #      [25,26,27,28,29,30,31,32]]), 4, 1)
  # print "Split test:"
  # for se in split_test:
  #     print se
  concat = res
  print "Concat:\n",concat
  splitted = np.split(concat, 4, axis=0)
  print "Splitted:\n",splitted
  i, j, f, o = splitted[0], splitted[1], splitted[2], splitted[3]
  print "i:", i, ", j:", j, ", f:", f, ", o:", o

  c_new = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
  h_new = tf.sigmoid(o) * tf.tanh(c_new)

  # BasicRNNCell
  #   concat = _linear([inputs, state], self._num_units, True, 0.0)
  #   output = tanh(concat)
  #   return output, output
  #
  # BasicLSTMCell
  #  concat = _linear([inputs, h], 4 * self._num_units, True, 0.0)
  #  i, j, f, o = array_ops.split(concat, num_or_size_splits=4)
  #  new_c = (c * sigmoid(f) + sigmoid(i) * tanh(j)
  #  new_h = tanh(new_c) * sigmoid(o)
  #  return new_h, (new_c,new_h)
  #
  # GRUCell
  #   concat = _linear([inputs, state], 2 * self._num_units, True, 1.0)
  #   value = sigmoid(concat)
  #   r, u = array_ops.split(value, num_or_size_splits=2)
  #   concat = _linear([inputs, r * state], self._num_units, True)
  #   c = tanh(concat)
  #   new_h = u * state + (1 - u) * c
  #   return new_h, new_h

lstmTF()
# lstmMagnus()

