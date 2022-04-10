from snake_genetic import *

# Get the fittest population for generation 100
pop = np.load("fit_100.npy")

# Run snake game with fittest population
cal_pop_fitness(pop)