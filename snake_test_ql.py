from snake_qlearning import SnakeQlearning
import numpy as np

np.set_printoptions(threshold=np.inf)
snakeQlearning = SnakeQlearning(epsilon=0.1, lr=0.5, gamma=0.9, numTrainEpisodes=20000, numTestEpisodes=100)
# snakeQlearning.train()
snakeQlearning.test()