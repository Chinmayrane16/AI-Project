import gym
import gym_snake
import numpy as np
from simple_snake_grid import SimpleSnakeGrid

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


env.close()
