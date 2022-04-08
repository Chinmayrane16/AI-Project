import gym
import gym_snake
import numpy as np
from Astar import SimpleSnakeGrid

np.set_printoptions(threshold=np.inf)

# Construct Environment
env = gym.make('snake-v0',grid_size=[8,8], unit_size=1,unit_gap=0,snake_size=2)
observation = env.reset() # Constructs an instance of the game
snake_grid = SimpleSnakeGrid(observation)

# Astar
while True:
    snake_grid.update_observation(observation)
    snake_grid.print_grid()
    actions = snake_grid.Astar_actions()

    if not actions:
        break
    for action in actions:
        env.render()
        observation, _, done, _ = env.step(action)  # take A* action

    if done:
        env.render()
        break

# try random actions
# for _ in range(100):
#     env.render()
#     action = env.action_space.sample()
#     print(action)
#     observation, reward, done, info = env.step(action) # take a random action

#     # print(info)
#     if done:
#         break

# # Astar search
# curr_pos = get_snake_head(observation)

env.close()
