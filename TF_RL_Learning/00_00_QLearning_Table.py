import gym
import numpy as np


env = gym.make('FrozenLake-v0')
env.render()
# SFFF
# FHFH
# FFFH
# HFFG

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Set learning parameters
lr = .8
y = .95
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):

    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    j = 0

    # The Q-Table learning algorithm
    actions_before_done = 0
    while j < 99:
        j += 1

        # Choose an action by greedily (with noise) picking from Q table
        # Noise dimension [1, 4] with values from normal distribution w mean=0, stddev=1
        # As epochs grow, lessen the importance of the noise
        a = np.argmax(Q[s,] + np.random.randn(1, env.action_space.n) * (1. / (i+1)))

        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        actions_before_done += 1

        # Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y * np.max(Q[s1,]) - Q[s,a])

        rAll += r
        s = s1
        if d == True:
            break

    # print("actions before done: {}".format(actions_before_done))
    rList.append(rAll)

print("--------")
print("Doing a round with trained table")
print("Starting state")
s = env.reset()
env.render()
while True:
    print("-----------------")
    print("Another round, now in state: {}".format(s))
    env.render()
    a = np.argmax(Q[s,])
    s, r, d, _ = env.step(a)
    print("...performed action: {}, ended up in state: {}".format(a, s))
    env.render()
    if d == True:
        break
print("-----------------")
print("...And we're done")

print("Score over time: {}".format(str(sum(rList)/num_episodes)))
print("Final Q-Table Values")
print(Q)

