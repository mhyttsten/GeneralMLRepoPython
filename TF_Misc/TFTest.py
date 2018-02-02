import tensorflow as tf
import os
import sys
import numpy as np

def exit():
    if True:
        sys.exit()


def test_timesplit_into_av():
    a = tf.constant([
            [
                ["000","001","002","003","004","005"],["010","011","012","013","014","015"],["020","021","022","023","024","025"],["030","031","032","033","034","035"]
            ],
            [
                ["100","101","102","103","104","105"],["110","111","112","113","114","115"],["120","121","122","123","124","125"],["130","131","132","133","134","135"]
            ],
            [
                ["200","201","202","203","204","205"],["210","211","212","213","214","215"],["220","221","222","223","224","225"],["230","231","232","233","234","235"]
            ],
            [
                ["300","301","302","303","304","305"],["310","311","312","313","314","315"],["320","321","322","323","324","325"],["330","331","332","333","334","335"]
            ]
        ])
    b = tf.reshape(a, shape=[-1, 2])
    c1, c2 = tf.split(b, 2, 1)
    with tf.Session() as sess:
        print(sess.run(tf.shape(a)))
        print(sess.run(tf.shape(b)))
        print(sess.run(tf.shape(c1)))
        print(sess.run(c1))

    maskA = tf.zeros([2, 4])
    maskB = tf.ones([2, 4])
    mask1 = tf.concat([maskA, maskB], 1)
    mask = tf.reshape(mask1, [-1])
    with tf.Session() as sess:
        print(sess.run(tf.shape(mask1)))
        print(sess.run(tf.shape(mask)))

test_timesplit_into_av()
exit()


def test_experience_buffer():
    episode_buffer = []
    a1 = np.array([[0,1,2,3],     0, 1, [4,5,6,7],     False])
    a2 = np.array([[8,9,10,11],   1, 0, [12,13,14,15], False])
    a3 = np.array([[16,17,18,19], 1, 0, [20,21,22,23], False])
    a4 = np.array([[24,25,26,27], 1, 0, [28,29,30,31], False])
    e1 = np.reshape(a1, [1, 5])
    e2 = np.reshape(a2, [1, 5])
    e3 = np.reshape(a3, [1, 5])
    e4 = np.reshape(a4, [1, 5])
    episode_buffer.append(e1)
    episode_buffer.append(e2)
    episode_buffer.append(e3)
    episode_buffer.append(e4)

    bufferArray = np.array(episode_buffer)
    episodeBuffer = list(zip(bufferArray))
    for a in episodeBuffer:
        print(a)
        for b in a:
            print("...{}".format(b))
            for c in b:
                print("......{}".format(c))
                for d in c:
                    print(".........{}".format(d))
test_experience_buffer()
exit()


def test_advantage():
    c = tf.constant([[1,3], [3,9], [4,12]])
    with tf.Session() as sess:
        d = sess.run(tf.subtract(c, tf.reduce_mean(c, axis=1, keep_dims=True)))
    print(d)
test_advantage()
exit()

def test_split():
    # a = [
    #     [["r0c0p0", "r0c0p1", "r0c0p2", "r0c0p3"], ["r0c1p0", "r0c1p1", "r0c1p2", "r0c1p3"], ["r0c2p0", "r0c2p1", "r0c2p2", "r0c2p3"], ["r0c3p0", "r0c3p1", "r0c3p2", "r0c3p3"]],
    #     [["r1c0p0", "r1c0p1", "r1c0p2", "r0c0p3"], ["r1c1p0", "r1c1p1", "r1c1p2", "r1c1p3"], ["r1c2p0", "r1c2p1", "r1c2p2", "r1c2p3"], ["r1c3p0", "r1c3p1", "r1c3p2", "r1c3p3"]],
    #     [["r2c0p0", "r2c0p1", "r2c0p2", "r0c0p3"], ["r2c1p0", "r2c1p1", "r2c1p2", "r2c1p3"], ["r2c2p0", "r2c2p1", "r2c2p2", "r2c2p3"], ["r2c3p0", "r2c3p1", "r2c3p2", "r2c3p3"]],
    #     [["r3c0p0", "r3c0p1", "r3c0p2", "r0c0p3"], ["r3c1p0", "r3c1p1", "r3c1p2", "r3c1p3"], ["r3c2p0", "r3c2p1", "r3c2p2", "r3c2p3"], ["r3c3p0", "r3c3p1", "r3c3p2", "r3c3p3"]],
    # ]
    a = [
        [[["r0c0p0", "r0c0p1", "r0c0p2", "r0c0p3"]]],
        [[["r1c0p0", "r1c0p1", "r1c0p2", "r1c0p3"]]],
    ]

    tf_a = tf.constant(a)
    tf_split = tf.split(a, 2, 3)
    with tf.Session() as sess:
        tf_a_split1, tf_a_split2 = sess.run(tf_split)
    print("1: {}".format(tf_a_split1))
    print("2: {}".format(tf_a_split2))
# test_split()


e02_a = tf.constant([[1,1,1],[2,2,2],[3,3,3]])
e02_rm_0 = tf.reduce_mean(e02_a, 0, keep_dims=True)
e02_rm_1 = tf.reduce_mean(e02_a, 1, keep_dims=True)
with tf.Session() as sess:
    e02_r0, e02_r1 = sess.run([e02_rm_0, e02_rm_1])
print(e02_r0)
print(e02_r1)


exit()




# sess = tf.InteractiveSession()
x = tf.Variable([1], tf.int32)
OP = tf.assign(x, x + 1)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   print(sess.run(OP)) # 1
   print(sess.run(OP)) # 2
   print(sess.run(OP)) # 1
   print(sess.run(OP)) # 2
if True:
    sys.exit()



# print os.getpid()

# Operations that are in each others execution path are only evaluated once
t0 = tf.constant(0)
t1 = tf.Print(t0, [t0], "t1: ")
t2 = tf.Print(t1, [t1], "t2: ")
with tf.Session() as sess:
    sess.run([t2, t1]) # t1, t2 only printed once as part of t2 execution


q = tf.constant([[0,1,2,3], [4,5,6,7], [8,9,10,11],[12,13,14,15]])
# s = tf.constant([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
s = tf.constant([[1,0,0,0]])
w = tf.matmul(s, q)
pa = tf.argmax(w, 1)
with tf.Session() as sess:
    print(sess.run([w, pa]))

print(np.identity(4)[3:4])

print("Final")
np_v1 = np.identity(4)
print(np.matmul([0,1,0,0], np_v1))