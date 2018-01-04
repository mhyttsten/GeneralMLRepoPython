import numpy as np
import tensorflow as tf

import gym

env = gym.make('CartPole-v0')


t2 = [0.,1.,2.,3.]
print(np.vstack(t2))
t3 = [y for y in t2][:-1]
print(t3)
print(t2[:-1])

# Hyperparameters
H = 8  # Number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99  # Discount factor for reward
decay_rate = 0.99  # Decay factor for RMSProp leaky sum of grad^2
resume = False  # Resume from previous checkpoint?

model_bs = 3  # Batch size when learning from model
real_bs = 3  # Batch size when learning from real environment

# Model initialization
D = 4  # Input dimensionality

'''
------------------------------------------------------------------------------
Policy Network
'''
observations_ph = tf.placeholder(tf.float32, [None,4] , name="input_x")
W1 = tf.get_variable("W1", shape=[4, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations_ph, W1))
W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
action_sigmoid = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None,1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
loglik = tf.log(input_y * (input_y-action_sigmoid) + (1-input_y) * (input_y+action_sigmoid))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


'''
------------------------------------------------------------------------------
Model Network
Here we implement a multi-layer neural network that predicts the next observation,
reward, and done state from a current state and action.
'''
mH = 256  # Model layer size

previous_state = tf.placeholder(tf.float32, [None,5] , name="previous_state")
W1M = tf.get_variable("W1M", shape=[5, mH], initializer=tf.contrib.layers.xavier_initializer())
B1M = tf.Variable(tf.zeros([mH]), name="B1M")
layer1M = tf.nn.relu(tf.matmul(previous_state, W1M) + B1M)

W2M = tf.get_variable("W2M", shape=[mH, mH], initializer=tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(tf.zeros([mH]), name="B2M")
layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)

wO = tf.get_variable("wO", shape=[mH, 4], initializer=tf.contrib.layers.xavier_initializer())
wR = tf.get_variable("wR", shape=[mH, 1], initializer=tf.contrib.layers.xavier_initializer())
wD = tf.get_variable("wD", shape=[mH, 1], initializer=tf.contrib.layers.xavier_initializer())

bO = tf.Variable(tf.zeros([4]), name="bO")
bR = tf.Variable(tf.zeros([1]), name="bR")
bD = tf.Variable(tf.ones([1]), name="bD")

predicted_observation = tf.matmul(layer2M, wO, name="predicted_observation") + bO
predicted_reward = tf.matmul(layer2M, wR, name="predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name="predicted_done") + bD)

true_observation = tf.placeholder(tf.float32,[None,4], name="true_observation")
true_reward = tf.placeholder(tf.float32,[None,1], name="true_reward")
true_done = tf.placeholder(tf.float32,[None,1], name="true_done")

predicted_state = tf.concat([predicted_observation, predicted_reward, predicted_done], 1)

observation_loss = tf.square(true_observation - predicted_observation)
reward_loss = tf.square(true_reward - predicted_reward)
done_loss = (predicted_done * true_done) + (1-predicted_done) * (1-true_done)
done_loss = -tf.log(done_loss)

model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

modelAdam = tf.train.AdamOptimizer(learning_rate=learning_rate)
updateModel = modelAdam.minimize(model_loss)


'''
------------------------------------------------------------------------------
Helper Functions
'''
def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer


def discount_rewards(r):
    # Take 1D float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# This function uses our model to produce a new state when given a previous state and action
def stepModel(sess, xs, action):
    toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
    myPredict = sess.run([predicted_state], feed_dict={previous_state: toFeed})
    reward = myPredict[0][:, 4]
    observation = myPredict[0][:, 0:4]
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    doneP = np.clip(myPredict[0][:, 5], 0, 1)
    if doneP > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
    return observation, reward, done


'''
------------------------------------------------------------------------------
Training the Policy and Model
'''
hist_observation, hist_action, hist_reward, hist_done = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
init = tf.global_variables_initializer()
batch_size = real_bs

drawFromModel = False   # When set to True, will use model for observations
trainTheModel = True    # Whether to train the model
trainThePolicy = False  # Whether to train the policy
switch_point = 1

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(tvars)
    gradBuffer = resetGradBuffer(gradBuffer)

    while episode_number <= 5000:

        # Start displaying environment once performance is acceptably high.
        if (reward_sum / batch_size > 150 and drawFromModel == False) or rendering == True:
            env.render()
            rendering = True

        observation = np.reshape(observation, [1, 4])
        hist_observation.append(observation)

        action_probability = sess.run(action_sigmoid, feed_dict={observations_ph: observation})
        action = 1 if np.random.uniform() < action_probability else 0
        action_invert = 1 if action == 0 else 0  # Invert value, otherwise we get grad problem
        hist_action.append(action)

        # Step the  model or real environment and get new measurements
        if drawFromModel == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = stepModel(sess, hist_observation, action)

        reward_sum += reward

        hist_done.append(int(done))  # True becomes 1, False becomes 0
        hist_reward.append(reward)  # Record reward (has to be done after we call step() to get reward for previous action)

        if done:
            if drawFromModel == False:
                real_episodes += 1

            episode_number += 1

            # Stack together all inputs, hidden states, action gradients, and rewards for this episode
            epo = np.vstack(hist_observation)
            epa = np.vstack(hist_action)
            epr = np.vstack(hist_reward)
            epd = np.vstack(hist_done)
            hist_observation, hist_action, hist_reward, hist_done = [], [], [], []  # Reset array memory

            if trainTheModel == True:
                actions = epa[:-1]
                state_prevs1 = epo[:-1, :]
                state_prevs = np.hstack([state_prevs1, actions])
                state_nexts = epo[1:, :]
                rewards = np.array(epr[1:, :])
                dones = np.array(epd[1:, :])
                state_nextsAll = np.hstack([state_nexts, rewards, dones])

                loss, pState, _ = sess.run([model_loss, predicted_state, updateModel],
                                           feed_dict={previous_state: state_prevs,
                                                      true_observation: state_nexts,
                                                      true_done: dones,
                                                      true_reward: rewards})

            if trainThePolicy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                actions = np.array([np.abs(value-1) for value in epa])
                tGrad = sess.run(newGrads,
                                 feed_dict={observations_ph: epo,
                                            input_y: actions,
                                            advantages: discounted_epr})

                if np.sum(tGrad[0] == tGrad[0]) == 0:
                    print("Terminating because of grad problem")
                    break
                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

            if switch_point + batch_size == episode_number:
                switch_point = episode_number
                if trainThePolicy == True:
                    sess.run(updateGrads,
                             feed_dict={W1Grad: gradBuffer[0],
                                        W2Grad: gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                if drawFromModel == False:
                    print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' %
                          (real_episodes, reward_sum / real_bs, action, running_reward / real_bs))
                    if reward_sum / batch_size > 200:
                        print("Reward sum became too large")
                        break
                reward_sum = 0

                # Once the model has been trained on 100 episodes, we start alternating between training the policy
                # from the model and training the model from the real environment.
                if episode_number > 100:
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy

            # We were done, so reset the environment and move to next epoch
            if drawFromModel == True:
                observation = np.random.uniform(-0.1, 0.1, [4])  # Generate reasonable starting point
                batch_size = model_bs
            else:
                observation = env.reset()
                batch_size = real_bs

print(real_episodes)

