import tensorflow as tf
import numpy as np

'''
Here we define our bandits. For this example we are using a four-armed bandit.
The pullBandit function generates a random number from a normal distribution with a mean of 0.
The lower the bandit number, the more likely a positive reward will be returned.
We want our agent to learn to always choose the bandit that will give that positive reward.
'''
# List out our bandits.
# Currently bandit 4 (index#3) is set to most often provide a positive reward.
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)
def pullBandit(bandit):
    result = np.random.randn(1)  # Get a single random number from normal distribution, mean=0, variance=1
    if result > bandit:
        return 1   # Positive reward
    else:
        return -1  # Negative reward

'''
The code below established our simple neural agent.
It consists of a set of values for each of the bandits.
Each value is an estimate of the value of the return from choosing the bandit.
We use a policy gradient method to update the agent by moving the value for
the selected action toward the recieved reward.
'''

# These two lines established the feed-forward part of the network.
# This does the actual choosing.
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

# The next six lines establish the training proceedure.
# We feed the reward and chosen action into the network
# to compute the loss, and use it to update the network.
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -( tf.log(responsible_weight) * reward_holder )
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

'''
We will train our agent by taking actions in our environment, and recieving rewards.
Using the rewards and actions, we can know how to properly update our network in order
to more often choose actions that will yield the highest rewards over time.
'''
total_episodes = 1000  # Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits)  # Set scoreboard for bandits to 0.
e = 0.1  # Set the chance of taking a random action.

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:

        # Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)

        reward = pullBandit(bandits[action])  # Get our reward from picking one of the bandits.

        #  Update the network.
        _, ww = sess.run([train, weights], feed_dict={ reward_holder: [reward], action_holder: [action] })

        #  Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the {} bandits: {}".format(str(num_bandits), str(total_reward)))
        i+=1

print("The weights of the bandits: {}".format(ww))
print("The agent thinks bandit {} is the most promising".format(str(np.argmax(ww)+1)))
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")


