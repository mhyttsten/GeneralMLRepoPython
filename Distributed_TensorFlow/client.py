import tensorflow as tf

'''
Distributed TensorFlow series:
- Interview Derek Murray & XYZ, get them to talk about something specific
- Perhaps I have a problem, go to Derek & ask for help, and then explain it in studio

1. Distributed TensorFlow is Easy! (Beginner)
- Assume basic TF knowledge (what do I refer to, reference section getting started)
- Our ML example: 3 MNIST layers (as one function, accuracy not important)
- "When using Distributed_TensorFlow TF, it is generally recommended to use 'between-graph' execution (see later for in-graph)'"
- Problem: When describing this you inevitably need to write MonitoredSession in your code
  (not generic across TF core, Estimators, and Keras)
- Learn about Servers
- What is a PS, what is a WS, and how you start them?

2. Feeding The Cluster (With Training Data) (Beginner)
- Using DataSet (refer to DataSet blog?)
- Getting data from a central location for all WS
- What should WS work on - do we need to partitioning data for each WS?
- With better and better GPUs & TPUs: This become more & more important

3. What's Up With Concurrency
- [Funny: Dreaming. Getting all dizzy trying to figure out PS updates]
- WS updating PS - synchronous, asynchronous

4. Getting to Know 'The Chief' (Beginner)
- [Funny: We can do something funny here with the Chief, Indian clothing, deciding]
- What is the Chief, who is the Chief, MonitoredSession
- Saving to, and starting from a checkpoint
- Why is not Chief the PS? (data locality?) Is the Chief a Worker also?

5. Scaling Beyond All Reasonable Horizons (Intermediate)
- [Funny: A single jar is not capable of dealing with the load]
- [Funny: Describing my cluster: "One PS To Rule Them All" - Derek: It's not gonna work. Poor PS, It's Not Gonna Work]
- Partitioning data across multiple PSs (why would you need it?), WS -> PS segmenting

6. My TensorFlow Cluster: It's Full of Servers! (Intermediate)
- [Funny: There are so many of them, I've lost control. Sitting with many terminal windows open.]
- How do I manage a cluster of servers (starting and stopping their Python programs from central location)
- Growing your training servers (dynamic or do I need to restart with new ClusterSpec?)

7. Who Is Saying What Now? (Intermediate)
- [Funny: Getting all the ducks in a row]
- In-depth on how data and control flows
  WS <-> PS (when is PS data read/updated & refer back to sync/async updates)
- Checkpointing with Chief (what data is part of a checkpoint)
- Can Chief become congested. Who can become congested?

8. Similar But Different (Advanced)
- When/how to use TF Distributed_TensorFlow: TF core, TF Keras, TF Estimators

9. Failure Is Always An Option (Advanced)
- Failure modes, what if PS, WS, or Chief stops. Recovery procedure, multiple failures, etc

10. My Computer Is a Powerful Computer
- [Funny: I have so many GPUs, it's unbelievable]
- In-graph execution (across multiple GPUs in a single computer)

# Baseline Reference blogs
- Why are GPUs good for Machine Learning
- Why are TPUs good for Machine Learning
- DataSet - A Primer
- Different ways of storing your TF program: checkpoint, SavedModel
  + First describe the different representations (checkpoint, SavedModel)
  + Then describe how they are used by different TF: Core, Estimators, Keras

******
- Running client with no workers running just causes client to hang indefinitely

- Having a ClusterSpec with {w1, w2}. If only w1 is running, client distributes to it but w1 says
  master.cc:209] CreateSession still waiting for response from worker: /job:l1/replica:0/task:1
  Reporting correctly 'task:1' is not up', but why does 'task:1' need to be up in the first place?
  Starting w2 resolves the problem.
- Rule: To work, all workers (tasks) in the system must be running although the may not be used
  
- If you use: tf.Session() as sess:, it will assume distribution to /job:localhost/task:0
  This is the default localhost execution
- In such case trying: with tf.device("/job:l1/task:0"):, will fail because such device does not
  exist within "/job:localhost/task:0" even though "/job:l1/task:0" worker is running
- I.e. the: with tf.Session("...") sess:, binds all subsequent explicit device requests

'''

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


