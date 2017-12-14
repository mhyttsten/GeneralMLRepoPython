import tensorflow as tf

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

    reader = tf.TextLineReader()
    key, record_string = reader.read(filename_queue)
    f1, f2, f3, label = tf.decode_csv(record_string, record_defaults=[["a"], ["b"], ["c"], ["d"]])
    features = tf.stack([f1, f2, f3])

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3*batch_size
    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, label],
        num_threads=16,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        allow_smaller_final_batch=True)
    return feature_batch, label_batch

get_batch = input_pipeline(
    filenames=[("csv_file%d.csv" % i) for i in range(4)],
    batch_size=3,
    num_epochs=2)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
coord = tf.train.Coordinator()

sess = tf.Session()
writer = tf.summary.FileWriter('tb_report', sess.graph)
sess.run(init_op)
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    while not coord.should_stop():
        (features, labels) = sess.run(get_batch)
        for i in range(features.shape[0]):
            print "f: " + str(features[i]) + ", l: " + labels[i]
        print 'features: {}, labels: {}'.format(str(features.shape), str(labels.shape))
except tf.errors.OutOfRangeError, e:
    print 'Done training -- epoch limit reached'
finally:
    coord.request_stop()
    coord.join(threads)
writer.close()