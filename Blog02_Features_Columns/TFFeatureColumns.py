import tensorflow as tf

# Estimators create their stuff in a separate graph
# So When model_fn is called, it has changed the default graph and what you create in there becomes part of the
# estimator graph
# That's why

# *** Questions:
#    linear_model can take _CategoricalColumn, but input_layer requires a _DenseColumn

# Input to models:
# - numeric values or categories

# Columns taking another column instance (instead of a column name) as argument
#    bucketized_column
#    indicator_column
#    weighted_categorical_column


print("\n*** weighted_categorical_column")
f1 = tf.constant(["b", "b", "d", "d", "a"]) # Gives
f2 = tf.constant([1, 2, 4, 8, 16], dtype=tf.float32) # weight tensor must be float32
f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    "f1",
    vocabulary_list=["a", "b", "c", "d"])
f_c = tf.feature_column.weighted_categorical_column(
    categorical_column=f_c,
    weight_feature_key='f2')
f_c = tf.feature_column.indicator_column(f_c)
lm = tf.feature_column.input_layer(
    {"f1": f1, "f2": f2},
    [f_c])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(init)
    print sess.run(lm)

# *** categorical_column_with_vocabulary_list
print("\n*** categorical_column_with_vocabulary_list")
f1 = tf.constant([2, 2, 3]) # Gives
f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    "f1",
    vocabulary_list=[1, 2, 3, 4, 5, 6])
f_c = tf.feature_column.indicator_column(f_c)
lm = tf.feature_column.input_layer({"f1": f1}, [f_c])
print lm.shape
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print sess.run(lm)


# *** categorical_column_with_vocabulary_list
print("\n*** categorical_column_with_vocabulary_list")
f1 = tf.constant(["b", "c", "c"]) # Gives
f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    "f1",
    vocabulary_list=["a", "b", "c", "d"])
f_c = tf.feature_column.indicator_column(f_c)
lm = tf.feature_column.input_layer({"f1": f1}, [f_c])
print lm.shape
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print sess.run(lm)

# *** categorical_column_with_hash [string]
# Use this when your sparse features are in string or integer format,
# and you want to distribute your inputs into a finite number of buckets by hashing.
# output_id = Hash(input_feature_string) % bucket_size
# Example uses: keywords
print("\n*** categorical_column_with_hash_bucket (str)")
f1 = tf.constant(["aaaa", "kkkk"]) # Gives [2, 0], "llll" gives [1, 1]
f_c = tf.feature_column.categorical_column_with_hash_bucket("f1", hash_bucket_size=2)
f_c = tf.feature_column.indicator_column(f_c)
lm = tf.feature_column.input_layer({"f1": f1}, [f_c])
print lm.shape
with tf.Session() as sess:
    print sess.run(lm)

# *** categorical_column_with_hash_bucket [int]
# See above
print("\n*** categorical_column_with_hash_bucket (int)")
f1 = tf.constant([10001, 12500]) # Gives [1, 1], 10001 gives [2, 0]
f_c = tf.feature_column.categorical_column_with_hash_bucket("f1", hash_bucket_size=2, dtype=tf.int64)
f_c = tf.feature_column.indicator_column(f_c)
lm = tf.feature_column.input_layer({"f1": f1}, [f_c])
print lm.shape
with tf.Session() as sess:
    print sess.run(lm)


# *** categorical_column_with_identity
# Typically, this is used for contiguous ranges of integer indexes, but it doesn't have to be.
# This might be inefficient, however, if many of IDs are unused.
# Consider categorical_column_with_hash_bucket in that case.
# Example uses: video_id
print("\n*** categorical_column_with_identity")
f0 = tf.ones([5, 1]) * 0x0A
f1 = tf.constant([3,3,2,1,1])
f_c0 = tf.feature_column.numeric_column("f0")
f_c1 = tf.feature_column.categorical_column_with_identity("f1", num_buckets=4)
f_c1 = tf.feature_column.indicator_column(f_c1)
lm = tf.feature_column.input_layer({"f0": f0, "f1": f1}, [f_c0, f_c1])
# lm = tf.feature_column.input_layer({"f1": f1}, [f_c1])
print lm.shape
with tf.Session() as sess:
    print sess.run(lm)


# *** bucketized_column
print("\n*** bucketized_column")
# f1 = tf.constant([[1, 101], [15, 16], [99, 101]])
f1 = tf.constant([-1, 5, 50, 101])
# f1_c = tf.feature_column.numeric_column("f1", shape=(2,))
f1_c = tf.feature_column.numeric_column("f1", shape=(1,))
f1_b_c = tf.feature_column.bucketized_column(f1_c, boundaries=[0, 10, 100])
lm = tf.feature_column.input_layer({"f1": f1}, [f1_b_c])
print lm.shape
with tf.Session() as sess:
    print sess.run(lm)

# ******************************

# Elisabeth Reilly
# Jean Colio


# f1 = tf.constant(0)
# a = tf.constant(0)
# def input_fn():
#     return {'f1': [f1]}, [a]
#
# def model_fn(features, labels, mode):
#
#     if mode = tf.estimators.ModeSpec.
#
#     f1_c = tf.feature_column.numeric_column('f1')
#     lm = tf.feature_column.input_layer({'f1': [f1]}, [f1_c])
#     return tf.estimator.EstimatorSpec(
#         mode,
#         predictions=lm,
#         train_op=tf.constant(0),
#         loss=tf.constant(0)
#     )
#
#
# e = tf.estimator.Estimator(model_fn=model_fn)
# e.train(input_fn)
# p = e.predict(input_fn)
#
# for idx, prediction in enumerate(p):
#     print prediction

