import tensorflow as tf

filenames = ["/var/data/file00.txt",
             "/var/data/file01.txt",
             "/var/data/file02.txt",
             "/var/data/file03.txt",
             "/var/data/file04.txt",
             "/var/data/file05.txt",
             "/var/data/file06.txt",
             "/var/data/file07.txt",
             "/var/data/file08.txt",
             "/var/data/file09.txt",
             "/var/data/file0A.txt",
             "/var/data/file0B.txt",
             "/var/data/file0C.txt",
             "/var/data/file0D.txt",
             "/var/data/file0E.txt",
             "/var/data/file0F.txt"
             ]

# dataset = (tf.contrib.data.TextLineDataset(file_path)  # Read text file
#            .skip(1)  # Skip header row
#            .map(decode_csv, num_threads=4)  # Decode each line using decode_csv
#            .cache()  # Warning: Caches entire dataset, can cause out of memory

dataset = (tf.data.Dataset.from_tensor_slices(filenames)
           .shuffle(10)  # Randomize elems (1 == no operation)
           .repeat(1)  # Repeats dataset this # times
           .batch(3)
           .prefetch(1))  # Make sure you always have 1 batch ready to serve
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

with tf.Session() as sess:
    while True:
        first_batch = sess.run(next_batch)
        print(first_batch)

