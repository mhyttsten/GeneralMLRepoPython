# Install full gym environment
# - Create, activate, and cd to virtualenv
# - $ git clone https://github.com/openai/gym
# - $ cd gym
# - $ pip install -e .[all] # Installs everything

# Unsolved environment:
#   - Means it does not have specified reward threshold when it's considered solved
#   - MsPacman is an unsolved environment

# Agent chooses and action
# And environment returns an observation and a reward
# Observation
#   - Is specific to environment (pixels, joint angels, etc)
#   - MsPacman: RGB image of screen (pixels) (210, 160, 3)
# Reward (float)
# Info (dict)
#   - Diagnostic information for debugging, e.g probability behind env last state change
#   - Official agent is not allowed to use this information
# Spaces
#   - Action space (env.action_space): What actions can be taken
#   - Observation space (env.observation_space): What you will see
# Type of agents:
#   - Random agent
#   - CEM (generic cross-entropy agent)
#   - TabularQAgent: Implements tabular Q-learning
#   - DQN: Basic DQN (/sherjilozair/dqn)
#   - Simple DQN: Uses Neon DL library (Intel Nirvana) (tambetm/simple_dqn)
#   - AgentNet: Allows you to develop custom agents using Theano (yandexdataschool/AgentNet)
#     Space invaders here:

import gym
from gym import envs

print("Printing all environments\n{}".format(envs.registry.all()))

# env = gym.make('CartPole-v0')
# env = gym.make('MoutainCar-v0')
# env = gym.make('MsPacman-v0')
env = gym.make('Pong-v0')

print("Action space: {}".format(env.action_space))
print("Observation space: {}".format(env.observation_space))

initial_observation = env.reset()
i = 0
total_reward = 0
while True:
    i += 1
    env.render()
    random_action = env.action_space.sample()
    observation, reward, done, info = env.step(random_action)
    total_reward += reward
    if i % 10 == 0:
        print("We are now at: {}, reward: {}, total_reward: {}".format(i, reward, total_reward))
    if done:
        print("--> done detected")
        break
print("Total reward: {}".format(total_reward))




