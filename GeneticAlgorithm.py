import random
import gym
import gym_snake
import numpy as np
from simple_snake_grid import SimpleSnakeGrid

np.set_printoptions(threshold=np.inf)

env = gym.make('snake-v0',grid_size=[15,15], unit_size=1,unit_gap=0,snake_size=2)

def moveDistanceToFood(food_x, food_y, snake_x, snake_y, move_before):
    up = snake_y - food_y
    down = food_y - snake_y
    left = snake_x - food_x
    right = food_x - snake_x
    possible_moves = [up, down, left, right]

    move_before += 1 if move_before % 2 == 0 else -1
    if move_before <= 3:
        possible_moves[move_before] == 0.0
    moves = [i for i, ni in enumerate(possible_moves) if ni > 0.0]

    return random.randint(0, 4) if len(moves) == 0 else random.choice(moves)

def distanceMoves(action, snake_grid, move_before):
    food_x, food_y = snake_grid.snake_food
    snake_x, snake_y = snake_grid.snake_head
    
    if action == 4:
        return moveDistanceToFood(food_x, food_y, snake_x, snake_y, move_before)
    
    return action

def play_game(actions):
    print(type(actions))
    observation = env.reset()
    snake_grid = SimpleSnakeGrid(observation)

    total_reward = 0
    snake_len = 1

    while True:
        snake_grid.update_observation(observation)
        # snake_grid.print_grid()
        if not actions:
            break

        move_before = actions[0]
        for action in actions:
            action = distanceMoves(action, snake_grid, move_before)
            # env.render()
            observation, reward, done, _ = env.step(action)
            
            # Snake Length
            if reward > 0:
                snake_len = snake_len + reward
            
            # Calculate Reward
            curr_reward = reward * 1000 if reward < 0 else reward * 500
            if curr_reward == 0:
                curr_reward = curr_reward - 1
            
            total_reward = total_reward + curr_reward

            # Save prev action
            move_before = action

            if done:
                break
            

        if done:
            env.render()
            break

    env.close()
    return total_reward + snake_len * 5000, snake_len

GENERATIONS = 100

population = []
population_size = 50
step_size = 1000
# 0 Up
# 1 Down
# 2 Left
# 3 Right
# 4 Distance to food
# 5 Distance to an obstacle
# moves_set = [0, 1, 2, 3, 4, 5]
moves_set = [0, 1, 2, 3, 4]

def cross_over(parent1, parent2):
    new_child = []
    for gen1, gen2 in zip(parent1, parent2):

        luck = random.random()

        if luck < 0.45:
            new_child.append(gen1)
        elif luck < 0.90:
            new_child.append(gen2)
        else:
            new_child.append(random.randrange(0, 6))

    return new_child

# Play the game and call fitness function
def cal_fitness(population):
    fitness_val = []
    snake_lens = []
    for i in range(len(population)):
        fitness, snake_len = play_game(population[i])
        fitness_val.append(fitness)
        snake_lens.append(snake_len)

    return fitness_val, snake_lens

# First population is created.
for i in range(population_size):
    genes = random.choices(moves_set, k=step_size)
    population.append(genes)

for i in range(GENERATIONS):

    wanted, snake_lens = cal_fitness(population)

    # List population according to fitness values.
    Z = [population for (wanted, population) in sorted(zip(wanted, population), key=lambda pair: pair[0])]
    new_generation = []

    # Elitism
    take_ratio = int((20*population_size)/100)
    new_generation.extend(Z[:take_ratio])

    # Parents
    take_ratio = int((80*population_size)/100)
    for _ in range(take_ratio):
        parent1 = random.choice(Z[:5])
        parent2 = random.choice(Z[:5])
        new_generation.append(cross_over(parent1, parent2))
    population = new_generation
    
    if i == 29:
        f = open("pop_29.txt", "w")
        f.write(''.join(str(e) for e in population))
        f.close()
    if i == 99:
        f = open("pop_99.txt", "w")
        f.write(''.join(str(e) for e in population))
        f.close()

    print("------- {} Generation Ended -------".format(i+1))
    print("Max len in this generation", max(snake_lens))
    print("All lens in this generation", snake_lens)
    print("Fitness for population", wanted)