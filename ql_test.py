from snake_qlearning import SnakeQlearning
import numpy as np

np.set_printoptions(threshold=np.inf)
snakeQlearning = SnakeQlearning(epsilon=0.3, lr=0.5, gamma=1, numEpisodes=1)
snakeQlearning.train()
