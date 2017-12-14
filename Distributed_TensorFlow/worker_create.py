# Get task number from command line
import sys
task_number = int(sys.argv[1])

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)

cluster = tf.train.ClusterSpec({"l1": ["localhost:2222", "localhost:2223"]})

server = tf.train.Server(
    cluster,
    task_index=task_number)
    # job_name=None # job name to which server is a member
    # protocol="grpc"
    # config=None # Default ConfigDef for all sessions in server
    # start=True

print("Starting server #{}".format(task_number))

server.join()

