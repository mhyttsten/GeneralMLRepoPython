import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

env = gym.make('CartPole-v0')

gamma = 0.99

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
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,
                                      h_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,
                                           a_size,
                                           activation_fn=tf.nn.softmax,
                                           biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # Training proceedure.
        # Feed the reward and chosen action to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        # This experssion will become: [self.action_holder]
        # Since tf.shape(self.output)[0] == 1 gives tf.range result [0]
        #    Then multiplying with tf.shape(self.output)[1] gives result [0]
        #    So result will be [self.action_holder)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        # So this statement will select the self.action_holder index from the output
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

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
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        steps_taken = 0
        for j in range(max_ep):

            # Feed state into network
            # Get a probability distribution over actions it wants to take
            # (output layer has softmax activation)
            # a_dist = [[action_1, action_2]]
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a_sel = np.random.choice(a_dist[0], p=a_dist[0])  # Pick action value based on its probabiliy
            a_cond = a_dist == a_sel  # Create vector [Bool, Bool] where element is True for selected action
            a = np.argmax(a_cond)  # Pick index of the True element, that's our action, either {0, 1}
            # print("a_dist: {}, a_sel: {}, a_cond: {}, a: {}".format(a_dist, a_sel, a_cond, a))

            s1, r, d, _ = env.step(a)  # Get our reward for taking an action
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            steps_taken += 1

            # It's game over.
            # Now collect all the training points and update network
            if d == True:
                ep_history = np.array(ep_history)
                ep_reward_history = ep_history[:, 2]
                print("ep_reward_history.shape: {}, steps_taken: {}".format(ep_reward_history.shape, steps_taken))

                ep_history[:, 2] = discount_rewards(ep_reward_history)
                grads = sess.run(myAgent.gradients,
                                 feed_dict={myAgent.reward_holder: ep_history[:, 2],
                                            myAgent.action_holder: ep_history[:, 1],
                                            myAgent.state_in: np.vstack(ep_history[:, 0])})
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    _ = sess.run(myAgent.update_batch,
                                 feed_dict=dict(zip(myAgent.gradient_holders, gradBuffer)))

                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

            # Update our running tally of scores.
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1
