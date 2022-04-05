from snake_qlearning import SnakeQlearning
import numpy as np

np.set_printoptions(threshold=np.inf)
snakeQlearning = SnakeQlearning(epsilon=0.05, lr=0.5, gamma=1, numTrainEpisodes=10000, numTestEpisodes=5)
snakeQlearning.train()
snakeQlearning.test()