import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

env = gym.make('CartPole-v0')

gamma = 0.99

def debug(tensor, message):
    # debug_flag = True
    debug_flag = False
    if debug_flag:
        tensor = tf.Print(tensor, [tensor], message=message, summarize=100)
    return tensor

def discount_rewards(r):
    # Take 1D float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Agent():

    def __init__(self, lr, s_size, a_size, h_size):

        # Feed-forward part of the network.
        # The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name="state_in")
        hidden = slim.fully_connected(self.state_in,
                                      h_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,
                                           a_size,
                                           activation_fn=tf.nn.softmax,
                                           biases_initializer=None)
        # self.output = debug(self.output, "Output")
        self.output_argmax = tf.argmax(self.output, axis=1)
        self.output_reduce_max = tf.reduce_max(self.output)
        self.output_rolldice = tf.multinomial(self.output, 1)
        # self.output_rolldice_equal = tf.equal(self.output, self.output_rolldice)

        # Training procedure.
        # Feed the reward and chosen action to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.reward_holder2 = debug(self.reward_holder, "reward_holder2")

        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.action_holder2 = debug(self.action_holder, "action_holder2")

        # This experssion will become: [self.action_holder]
        # Since tf.shape(self.output)[0] == 1 gives tf.range result [0]
        #    Then multiplying with tf.shape(self.output)[1] gives result [0]
        #    So result will be [self.action_holder]
        self.output2 = debug(self.output, "Output")
        self.range1 = tf.range(0, tf.shape(self.output2)[0])
        self.range1 = debug(self.range1, "range1")
        self.range2 = tf.shape(self.output2)[1]
        self.range2 = debug(self.range2, "range2")
        self.range_mult = self.range1 * self.range2
        self.range_mult = debug(self.range_mult, "range_mult")
        self.indexes = self.range_mult + self.action_holder2
        self.indexes = debug(self.indexes, "indexes")

        self.responsible_outputs = tf.gather(tf.reshape(self.output2, [-1]), self.indexes)
        self.responsible_outputs = debug(self.responsible_outputs, "responsible_outputs")

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder2)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)
        self.gradients = tf.gradients(self.loss, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


myAgent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8)  # Load the agent.

total_episodes = 5000  # Set total number of episodes to train agent on.
episode_number = 0

max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    tf.logging.set_verbosity(tf.logging.ERROR)

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = 0

    s = env.reset()
    a_dist_list = []
    while episode_number < 5000:
        running_reward = 0
        ep_history = []
        steps_taken = 0

        # Feed state into network
        # Get a probability distribution over actions it wants to take
        # (output layer has softmax activation)
        # a_dist = [[action_1, action_2]]
        a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
        a_dist_list.append(a_dist)

        # print("-----------")
        # print("action_distribution: {}".format(a_dist))
        # print("...its argmax: {}".format(sess.run(myAgent.output_argmax, feed_dict={myAgent.state_in: [s]})))
        # print("...its reduce_max: {}".format(sess.run(myAgent.output_reduce_max, feed_dict={myAgent.state_in: [s]})))
        # print("multinomial: {}".format(sess.run(myAgent.output_rolldice, feed_dict={myAgent.state_in: [s]})))
        # print(sess.run(myAgent.output_rolldice_equal, feed_dict={myAgent.state_in: [s]}))

        a_sel = np.random.choice(a_dist[0], p=a_dist[0])  # Pick action value based on its probability
        a_cond = a_dist == a_sel  # Create vector [Bool, Bool] where element is True for selected action
        a = np.argmax(a_cond)  # Pick index of the True element, that's our action, either {0, 1}
        # print("a_dist: {}, a_sel: {}, a_cond: {}, a: {}".format(a_dist, a_sel, a_cond, a))

        if episode_number >= 3000:
            env.render()
        s1, r, d, _ = env.step(a)  # Get our reward for taking an action
        ep_history.append([s, a, r, s1])
        s = s1
        running_reward += r
        steps_taken += 1

        # It's game over.
        # Now collect all the training points and update network
        if d == True:
            ep_history = np.array(ep_history)
            # print("rewards: {}".format(ep_history[:,2]))
            ep_history[:, 2] = discount_rewards(ep_history[:,2])
            # print("discounted_rewards: {}".format(ep_history[:,2]))

            # print("Shape of ep_history: {}".format(ep_history.shape))
            grads = sess.run(myAgent.gradients,
                             feed_dict={myAgent.reward_holder: ep_history[:, 2],
                                        myAgent.action_holder: ep_history[:, 1],
                                        myAgent.state_in: np.vstack(ep_history[:, 0])})
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if episode_number != 0 and episode_number % 3 == 0:
                _ = sess.run(myAgent.update_batch,
                             feed_dict=dict(zip(myAgent.gradient_holders, gradBuffer)))

                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = 0


            # Update our running tally of scores.
            episode_number += 1
            if episode_number > 0 and episode_number % 100 == 0:
                # print(a_dist_list)
                # print("al: {}".format(len(a_dist_list)))
                # print("a0: {}".format(a_dist_list[0:1]))
                # print("a1: {}".format(a_dist_list[1:2]))
                action_0_average = np.mean(a_dist_list[0:1])
                print("...finished with episode: {}".format(episode_number))
                a_dist_list = []

            s = env.reset()

print("Done, episode number: {}".format(episode_number))