# In this iPython notebook I implement a Deep Q-Network using both Double DQN and Dueling DQN.
# The agent learn to solve a navigation task in a basic grid world.

from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
matplotlib.use("TkAgg")
import scipy.misc
import matplotlib.pyplot as plt
import os
# %matplotlib inline

# Load the game environment
# Feel free to adjust the size of the gridworld.
# Making it smaller provides an easier task for our DQN agent, while making the world larger increases the challenge.
from gridworld import gameEnv


def debug(tensor, message):
    # debug_flag = True
    debug_flag = False
    if debug_flag:
        tensor = tf.Print(tensor, [tensor], message=message, summarize=100)
    return tensor

env = gameEnv(partial=False,size=5)

# This gives an example of a starting environment in our simple game.
# The agent controls the blue square, and can move up, down, left, or right.
# The goal is to move to the green square (for +1 reward) and avoid the red square (for -1 reward).
# The position of the three blocks is randomized every episode.

class Qnetwork():
    def __init__(self, h_size):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(
            inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None)
        print("conv1: {}".format(self.conv1))
        self.conv2 = slim.conv2d(
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None)
        print("conv2: {}".format(self.conv2))
        self.conv3 = slim.conv2d(
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        print("conv3: {}".format(self.conv3))
        self.conv4 = slim.conv2d(
            inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        print("conv4: {}".format(self.conv4))

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)

        print("streamAC[0]: {}".format(self.streamAC[0]))

        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        print("streamA: {}".format(self.streamA))
        print("streamV: {}".format(self.streamV))
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

# This class allows us to store experies and sample then randomly to train the network.
class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

# This is a simple function to resize our game frames.
def processState(states):
    return np.reshape(states, [21168])

# These functions allow us to update the parameters of our target network with those of the main network
def updateTargetGraph(tfVars, tau):
    # total_vars = len(tfVars)
    # op_holder = []
    # # 1st half of vars == mainQN
    # # 2nd half of vars == targetQN
    # for idx, var_main in enumerate(tfVars[0:total_vars//2]):
    #     var_target = tfVars[idx+total_vars//2]
    #     op_holder.append(var_target.assign((var_main.value()*tau) + ((1-tau)*tfVars[var_target].value()))
    #     )
    # return op_holder
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
       sess.run(op)

# Training the network
# Setting all the training parameters
batch_size = 32  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 500      # How many episodes of game environment to train network with.
pre_train_steps = 10000   # How many steps of random actions before training begins.
max_epLength = 50  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path = "/tmp2/TF_RL_04_DQN"  # The path to save our model to.
h_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(num_episodes):
        episodeBuffer = experience_buffer()

        # Reset environment and get first new observation
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        # The Q-Network
        while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j += 1

            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save experience to buffer

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    # [32, 5] == [s, a, r, s1, d]]
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.

                    # Get predicted actions from mainQN [32] tf.int32 (because batch_size == 32]
                    predict_value = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})

                    # Get Q-values from targetQN [32, 4] tf.float32 (because batch_size == 32]
                    q_values = sess.run(targetQN.Qout,  feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})

                    # Get the targetQN value for the action we picked from mainQN [32] tf.float32
                    q_value = q_values[range(batch_size), predict_value]

                    # 1 if !D, 0 otherwise
                    end_multiplier = -(trainBatch[:, 4] - 1)

                    # targetQ = reward + (0.99 * q_value * 1/0[if done or !done])
                    target_q_value = trainBatch[:, 2] + (y * q_value * end_multiplier)

                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                            mainQN.targetQ: target_q_value,
                                            mainQN.actions: trainBatch[:, 1]})

                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.
            rAll += r
            s = s1

            if d == True:
                break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)

        # Periodically save the model.
        if i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(i, total_steps, np.mean(rList[-10:]), e)
    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

# Checking network learning. Mean reward over time
rMat = np.resize(np.array(rList), [len(rList)//100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)


