import random
import gym
import gym_snake
import numpy as np
from simple_snake_grid import SimpleSnakeGrid
from random import choice, randint

np.set_printoptions(threshold=np.inf)

# Create gym snake environment
env = gym.make('snake-v0',grid_size=[8,8], unit_size=1,unit_gap=0,snake_size=2)

# Calculate distance to food to determine the next direction
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

# Calculate moves depending on directions
def distanceMoves(action, snake_grid, move_before):
    food_x, food_y = snake_grid.snake_food
    snake_x, snake_y = snake_grid.snake_head
    
    if action == 4:
        return moveDistanceToFood(food_x, food_y, snake_x, snake_y, move_before)
    
    return action

# Run the game for given set of population
def play_game(actions):
    actions = actions.tolist()
    observation = env.reset()
    snake_grid = SimpleSnakeGrid(observation)

    total_reward = 0
    snake_len = 1
    steps_count = 0

    while True:    
        snake_grid.update_observation(observation)
        if not actions:
            break

        move_before = actions[0]
        for action in actions:
            action = distanceMoves(action, snake_grid, move_before)
            env.render()
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

            # Increment steps count
            steps_count = steps_count + 1

            if done:
                break
            
        if done:
            env.render()
            break

    env.close()
    return total_reward + snake_len * 5000, snake_len, steps_count

# calculating the fitness value by playing a game with the given weights in chromosome
def cal_pop_fitness(pop, eval=False):
    fitness = []
    snake_lens = []
    for i in range(pop.shape[0]):
        fit, snake_len, steps_count = play_game(pop[i])
        fitness.append(fit)
        snake_lens.append(snake_len)

        if eval:
            eval_file = open("genetic_csv_eval.csv", "a")
            eval_file.write("genetic," + str(i+1) + "," + str(snake_len-1) + "," + str(steps_count) + "\n" )
            eval_file.close()

    return np.array(fitness), snake_lens, steps_count

# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999
    return parents

# creating children for next generation 
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    
    for k in range(offspring_size[0]): 
  
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if random.uniform(0, 1) < 0.5:
                        offspring[k, j] = parents[parent1_idx, j]
                    else:
                        offspring[k, j] = parents[parent2_idx, j]
                break
    return offspring


# mutating the offsprings generated from crossover to maintain variation in the population
def mutation(offspring_crossover):
    
    for idx in range(offspring_crossover.shape[0]):
        for _ in range(25):
            i = randint(0,offspring_crossover.shape[1]-1)

        random_value = np.random.choice(np.arange(-1,1,step=0.001),size=(1),replace=False)
        offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value

    return offspring_crossover

# Defining the Genetic algorithm variables
GENERATIONS = 100

step_size = 1000
# # 0 Up
# # 1 Down
# # 2 Left
# # 3 Right
# # 4 Distance to food
moves_set = [0, 1, 2, 3, 4]

n_x = 7
n_h = 9
n_h2 = 15
n_y = 3

# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
sol_per_pop = 100
num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y

# Defining the population size.
pop_size = (sol_per_pop,num_weights)

# Creating the initial population.
new_population = np.random.choice(random.choices(moves_set, k=step_size),size=pop_size,replace=True)

num_parents_mating = 12

for generation in range(GENERATIONS):
    print('##############        GENERATION ' + str(generation)+ '  ###############' )
    # Measuring the fitness of each chromosome in the population.
    fitness, snake_lens, steps_count = cal_pop_fitness(new_population)
    print('#######  fittest chromosome in gneneration ' + str(generation) +' is having fitness value:  ', np.max(fitness))
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, num_parents_mating)
    
    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation