import tensorflow as tf

def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "product_id": tf.FixedLenFeature((), tf.int64, default_value=0),
                "category_id": tf.FixedLenFeature((), tf.int64, default_value=0),
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["image"], parsed_features["product_id"], parsed_features["category_id"]

dataset = tf.contrib.data.TFRecordDataset("/tmp2/cdiscount/output/tfrecordfile")
dataset = dataset.map(_parse_function)
iterator = dataset.make_one_shot_iterator()
nb = iterator.get_next()

count = 0
with tf.Session() as sess:
    while(True):
        count += 1
        i, p, c = sess.run(nb)
        # print("product_id: %d, category_id: %d" % (p, c))
        # product_id: 0, category_id: 1000010653
        with open("/tmp2/cdiscount/test/try_%02d.jpg" % count, "w") as f:
            f.write(i)
            f.close()
