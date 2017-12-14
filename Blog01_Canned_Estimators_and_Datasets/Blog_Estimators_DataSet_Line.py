import tensorflow as tf
import sys

print(tf.__version__)
assert tf.__version__.startswith("1.3"), "Code needs TensorFlow 1.3 (or higher...)"

input_train = tf.constant([[2*x+3,x] for x in range(1, 99)])
input_predict = tf.constant([[-1,x] for x in range(100, 400, 100)])

def my_input_fn(input, is_predict=True):
    def parse(elem):
        return { "in": elem[1] }, elem[0]

    dataset = tf.contrib.data.Dataset.from_tensor_slices(input)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(128)
    if not is_predict:
        dataset = dataset.repeat(10)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    return next_batch

classifier = tf.estimator.LinearRegressor(
    feature_columns=[tf.feature_column.numeric_column('in')],
    model_dir='/tmp2/My1stEstimator')
classifier.train(
   input_fn=lambda: my_input_fn(input_train, False),
   steps=2000) # Train for 2000 batches of data

result = classifier.predict(input_fn=lambda: my_input_fn(input_predict))
for idx,r in enumerate(result):
  print("i:{}, r:{}".format(idx, r))


# # r = classifier.evaluate(
# #     input_fn=lambda: my_input_fn2(dps_train),
# #     steps=2000)  # Evaluate using 100 batches of data
# predictions = classifier.predict(input_fn=lambda: my_input_fn2(dps_infer))
# for i, p in enumerate(predictions):
# print("hello")



print("Done with evaluation")

