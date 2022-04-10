import gym
import gym_snake
import numpy as np
from simple_snake_grid import SimpleSnakeGrid
import csv


np.set_printoptions(threshold=np.inf)

num_of_games = 100

# Construct Environment BFS
env = gym.make('snake-v0',grid_size=[8,8], unit_size=1,unit_gap=0,snake_size=2)


scores_bfs = []
steps_bfs = []
runs = [i for i in range(num_of_games)]

# perform bfs runs
for game in range(num_of_games):
    # resetting score
    score = 0
    steps = 0
    observation = env.reset() # Constructs an instance of the game
    snake_grid = SimpleSnakeGrid(observation)
    
    # start the game
    while True:
        snake_grid.update_observation(observation)
        snake_grid.print_grid()
        actions = snake_grid.bfs_actions()
        steps += len(actions)

        if not actions:
            break
        for action in actions:
            observation, reward, done, _ = env.step(action)  # take a random action
            if reward == 1:
                score += 1

        if done:
            break
    
    scores_bfs.append(score)
    steps_bfs.append(steps)


# open the file in the write mode
f = open('bfs_csv_eval.csv', 'w')

writer = csv.writer(f)
row = ["algo", "iteration number", "food count", "step count"]

writer.writerow(row)
for i in range(num_of_games):
    row = ["bfs", str(i+1),str(scores_bfs[i]),str(steps_bfs[i])]

    writer.writerow(row)

f.close()
env.close()


# a star evaluations

scores_a_star = []
steps_a_star= []
runs = [i for i in range(num_of_games)]

# perform runs
for game in range(num_of_games):
    # resetting score
    score = 0
    steps = 0
    observation = env.reset() # Constructs an instance of the game
    snake_grid = SimpleSnakeGrid(observation)
    
    # start the game
    while True:
        snake_grid.update_observation(observation)
        snake_grid.print_grid()
        actions = snake_grid.Astar_actions()
        steps += len(actions)

        print("**",actions)

        if not actions:
            break
        for action in actions:
            observation, reward, done, _ = env.step(action)  # take a random action
            if reward == 1:
                score += 1

        if done:
            break
    
    scores_a_star.append(score)
    steps_a_star.append(steps)


# open the file in the write mode
f = open('a_star_csv_eval.csv', 'w')

writer = csv.writer(f)
row = ["algo", "iteration number", "food count", "step count"]

writer.writerow(row)
for i in range(num_of_games):
    row = ["a_star", str(i+1),str(scores_a_star[i]),str(steps_a_star[i])]

    writer.writerow(row)

f.close()
env.close()

# random evaluations

scores_random = []
steps_random= []
runs = [i for i in range(num_of_games)]

# perform runs
for game in range(num_of_games):
    # resetting score
    score = 0
    steps = 0
    observation = env.reset() # Constructs an instance of the game
    
    
    # start the game
    while True:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action) # take a random action
        if reward == 1:
            score += reward
        steps += 1

        if done:
            break
    
    scores_random.append(score)
    steps_random.append(steps)


# open the file in the write mode
f = open('random_csv_eval.csv', 'w')

writer = csv.writer(f)
row = ["algo", "iteration number", "food count", "step count"]

writer.writerow(row)
for i in range(num_of_games):
    row = ["random", str(i+1),str(scores_random[i]),str(steps_random[i])]

    writer.writerow(row)

f.close()
env.close()

# perform hamilton run
scores_ham = []
steps_ham = []
hamilton_cache = {}

for game in range(num_of_games):
    # resetting score
    score = 0
    steps = 0
    observation = env.reset() # Constructs an instance of the game
    snake_grid = SimpleSnakeGrid(observation)
    snake_grid.update_observation(observation)
    
    # cache path for 100 runs
    snake_head_key = str(snake_grid.snake_head[0]) + '-' + str(snake_grid.snake_head[1])
    actions = None
    if snake_head_key not in hamilton_cache:
        actions = snake_grid.hamilton_actions()
        hamilton_cache[snake_head_key] = actions
    else:
        print("Retrieved from cache ->", snake_head_key)
        actions =  hamilton_cache[snake_head_key]

    
    # start the game
    while True:
        snake_grid.update_observation(observation)
        snake_grid.print_grid()

        steps += len(actions)

        if not actions:
            break
        for action in actions:
            observation, reward, done, _ = env.step(action)  # take a random action
            if reward == 1:
                score += 1

        if done:
            break
    
    scores_ham.append(score)
    steps_ham.append(steps)


# open the file in the write mode
f = open('hamilton_csv_eval.csv', 'w')

writer = csv.writer(f)
row = ["algo", "iteration number", "food count", "step count"]

writer.writerow(row)
for i in range(num_of_games):
    row = ["hamilton", str(i+1),str(scores_ham[i]),str(steps_ham[i])]

    writer.writerow(row)

f.close()

env.close()