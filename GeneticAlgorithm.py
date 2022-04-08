import random
import gym
import gym_snake
import numpy as np
from simple_snake_grid import SimpleSnakeGrid

np.set_printoptions(threshold=np.inf)

env = gym.make('snake-v0',grid_size=[15,15], unit_size=1,unit_gap=0,snake_size=2)

def play_game(actions):
    observation = env.reset()
    snake_grid = SimpleSnakeGrid(observation)

    total_reward = 0
    while True:
        snake_grid.update_observation(observation)
        # snake_grid.print_grid()
        if not actions:
            break
        for action in actions:
            env.render()
            observation, reward, done, _ = env.step(action)
            curr_reward = reward * 10 if reward < 0 else reward * 50
            
            if curr_reward == 0:
                curr_reward = curr_reward - 1
            
            total_reward = total_reward + curr_reward
            if done:
                break

        if done:
            env.render()
            break

    env.close()
    return total_reward

GENERATIONS = 30

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
moves_set = [0, 1, 2, 3]

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
    for i in range(len(population)):
        fitness = play_game(population[i])
        fitness_val.append(fitness)

    return fitness_val

# First population is created.
for i in range(population_size):
    genes = random.choices(moves_set, k=step_size)
    population.append(genes)

for i in range(GENERATIONS):

    wanted = cal_fitness(population)

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

    print("------- {} Generation Ended -------".format(i+1))
    print(wanted)