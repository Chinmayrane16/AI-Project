import gym
import gym_snake
import numpy as np
np.set_printoptions(threshold=np.inf)

# Construct Environment
env = gym.make('snake-v0',grid_size=[15,15])
observation = env.reset() # Constructs an instance of the game
print(observation.shape)

for _ in range(100):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    # print(info)
    if done:
        break

env.close()
