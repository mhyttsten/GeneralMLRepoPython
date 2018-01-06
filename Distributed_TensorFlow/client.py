import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)
config = tf.ConfigProto(log_device_placement=True)
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

cluster = tf.train.ClusterSpec({"l1": ["localhost:2222", "localhost:2223"]})

# server1 = tf.train.Server.create_local_server()
# server2 = tf.train.Server.create_local_server()
# print "Server.create_local_server, s1: {}, s2: {}".format(server1.target, server2.target)

with tf.device("/job:l1/task:1"):
  h = tf.constant("Hello from TensorFlow", name='hello')
  a = tf.constant(1, name='a')
  b = tf.constant(2, name='b')
  c = tf.add(a, b, name='c')
print 'Now executing session'
# with tf.Session(server1.target, config=config) as sess:
with tf.Session("grpc://localhost:2222", config=config) as sess:
    result = sess.run(h, options=options)
print result
print 'Now finished'


