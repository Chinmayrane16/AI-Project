# New file for Qlearning implementation
'''

class SnakeAI
1) Init
2) LegalActions
    Cases - return valid actions
3) Directions
    UP - [0, -1]
    DOWN - [0, 1]
    LEFT - [-1, 0]
    RIGHT - [1, 0]
3) ComputeActionFromQvalues
4) ComputeValueFromQvalues
5)

Doubts
1) Logic in ComputeActionFromQValues
2) Changed self.QTable((deque, action))
'''
import gym
import gym_snake
import numpy as np
from collections import deque, Counter
from simple_snake_grid import SimpleSnakeGrid
import random


class SnakeQlearning:

    def __init__(self, epsilon, lr, gamma, numEpisodes) -> None:
        # UP = 0
        # RIGHT = 1
        # DOWN = 2
        # LEFT = 3
        # self.observation = observation
        self.DIRS = [([0, 1], 1), ([1, 0], 2), ([0, -1], 3), ([-1, 0], 0)]
        # self.snakeGridObj = SimpleSnakeGrid(observation)
        self.epsilon = epsilon
        self.lr = lr
        self.discount = gamma
        self.QTable = Counter()
        self.numEpisodes = numEpisodes
        self.rewards = []
        self.accumRewards = 0

    def print_grid(self, snake_grid):
        for row in snake_grid:
            print(row)

    def get_legal_actions(self, snake_grid, head):
        # Using paramaters instead of self, to get legal actions from any state, not just current state
        # snake_grid = self.snakeGridObj.snake_grid
        # head = self.snakeGridObj.snake_head
        # self.print_grid(snake_grid)

        rows = len(snake_grid)
        cols = len(snake_grid[0])
        # Return legal actions
        legal_actions = []
        for dir in self.DIRS:
            newR = head[0] + dir[0][0]
            newC = head[1] + dir[0][1]
            if newR >= 0 and newR < rows and newC >= 0 and newC < cols and (snake_grid[newR][newC] in [0, 1]):
                legal_actions.append(dir[1])

        print("Legal action", legal_actions)
        return legal_actions

    def getQValue(self, snake_state, action):
        # print(snake_state)
        return(self.QTable[(tuple(snake_state), action)])

    def computeActionFromQValues(self, snake_grid, snake_head):
        legal_actions_list = self.get_legal_actions(
            snake_grid, snake_head)

        if len(legal_actions_list) > 0:
            temp = None
            best_actions = []
            for action in legal_actions_list:
                snake_dq = self.snakeGridObj.virtual_snake_dq
                # FEATURES
                curr_state = self.get_state(snake_dq)
                curr_val = self.getQValue(curr_state, action)

                if temp is None or curr_val > temp:
                    temp = curr_val
                    best_actions.clear()
                    best_actions.append(action)
                elif curr_val == temp:
                    best_actions.append(action)
            return random.choice(best_actions)
        else:
            return None

    def get_state(self, snake_dq):
        # self.snake_grid, self.snake_head, _, self.snake_food = self.snakeGridObj.get_updated_grid(
        #     self.observation)
        snake_dq = [tuple(x) for x in snake_dq]
        return (tuple(snake_dq), tuple(self.snakeGridObj.snake_food))

    def get_action(self):
        # function returns either random action or q table action depending on epsilon
        legal_actions_list = self.get_legal_actions(
            self.snake_grid, self.snake_head)
        if(len(legal_actions_list) > 0):
            r = random.random()
            flip_coin = (r < self.epsilon)

            if(flip_coin):
                action = random.choice(legal_actions_list)
            else:
                action = self.computeActionFromQValues(
                    self.snake_grid, self.snake_head)

            return action
        else:
            return None

    def computeValueFromQValues(self, state, snake_grid, snake_head):
        actions_list = self.get_legal_actions(snake_grid, snake_head)

        if len(actions_list) == 0:
            return 0.0

        actionQValues = [self.getQValue(state, action)
                         for action in actions_list]
        maxQValue = max(actionQValues)

        return maxQValue

    def bellmanUpdate(self, state, action, nextState, reward, nextStateSnakeGrid, nextStateSnakeHead):
        sample = reward + \
            (self.discount * self.computeValueFromQValues(nextState, nextStateSnakeGrid, nextStateSnakeHead)
             )   # nextState - [DQ, food]
        self.QTable[(state, action)] = ((1 - self.lr) *
                                        self.QTable[(state, action)]) + (self.lr * sample)

    def get_reward(self, action):
        if action == None:
            return -100

        newHead = self.get_head_from_action(action)

        # Or it hits a boundary
        snake_grid = self.snake_grid
        rows = len(snake_grid)
        cols = len(snake_grid[0])
        newR = newHead[0]
        newC = newHead[1]
        print("New Head", newHead, rows, cols)
        if newR < 0 and newR >= rows and newC < 0 and newC >= cols and (snake_grid[newR][newC] not in [0, 1]):
            return -100

        if newHead == self.snakeGridObj.snake_food:
            return 100

        return -5

    def get_head_from_action(self, action):
        head = self.snake_head
        mapDirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        newHeadX = head[0] + mapDirs[action][0]
        newHeadY = head[1] + mapDirs[action][1]

        # New head
        newHead = [newHeadX, newHeadY]

        return newHead

    def create_grid_from_v_snake(self, virtual_snake_dq, food_loc):
        rows, cols = (
            self.observation.shape[0], self.observation.shape[1])
        arr = [[0 for i in range(cols)] for j in range(rows)]

        for i in range(len(virtual_snake_dq)):
            position_obj = virtual_snake_dq[i]
            arr[position_obj[0]][position_obj[1]] = self.snakeGridObj.BODY_CODE

        # finally add position to this virtual grid
        position_obj = virtual_snake_dq[-1]
        arr[position_obj[0]][position_obj[1]] = self.snakeGridObj.HEAD_CODE

        # Food location
        position_obj = food_loc
        arr[position_obj[0]][position_obj[1]] = self.snakeGridObj.FOOD_CODE

        return arr

    def update_dq(self, action, reward):
        # Update deque
        snake_dq = self.snakeGridObj.virtual_snake_dq
        if reward != 100:
            snake_dq.popleft()
        newHead = self.get_head_from_action(action)
        snake_dq.append(newHead)
        self.snake_grid = self.create_grid_from_v_snake(
            snake_dq, self.snakeGridObj.snake_food)
        self.snake_head = snake_dq[-1]

    def train(self):
        for i in range(self.numEpisodes):
            # Construct Environment
            env = gym.make(
                'snake-v0', grid_size=[6, 6], unit_size=1, unit_gap=0, snake_size=2)
            self.observation = env.reset()  # Constructs an instance of the game
            self.snakeGridObj = SimpleSnakeGrid(self.observation)
            self.snakeGridObj.update_observation(self.observation)
            self.snakeGridObj.print_grid()

            self.snake_grid = self.snakeGridObj.snake_grid.copy()
            self.snake_head = self.snakeGridObj.snake_head.copy()

            # Print status
            if (i+1) % 100 == 0:
                print("Episodes {} \nAverage rewards {} \nAverage Rewards over last 100 episodes".format(
                    i+1, np.mean(self.rewards)), np.mean(self.rewards[-100:]))

            reward = 0
            while True:
                curr_state = self.get_state(self.snakeGridObj.virtual_snake_dq)
                # print("Curr State", curr_state)
                curr_action = self.get_action()
                # print("Curr action", curr_action)
                # self.print_grid(self.snake_grid)

                # Action = None handled in get_reward
                # Reward => death (-100) food (100) otherwise (-5)
                # Dequeue is update here
                reward = self.get_reward(curr_action)
                # print("Reward", reward)
                if reward == -100:
                    print("Current state", curr_state)
                    print("Current action", curr_action)
                    self.print_grid(self.snake_grid)
                    break
                self.update_dq(curr_action, reward)
                snake_dq = self.snakeGridObj.virtual_snake_dq
                # print("Next State Deque", snake_dq)

                nextState = self.get_state(snake_dq)
                # print("Next State", nextState)

                # Bellman equation
                self.bellmanUpdate(curr_state, curr_action,
                                   nextState, reward, self.snake_grid, self.snake_head)

                env.render()
                self.observation, _, done, _ = env.step(curr_action)

                self.rewards.append(reward)
                self.accumRewards += reward

                print()

                if reward == 100:
                    self.snakeGridObj.update_observation(self.observation)

            # Scoring -> food eaten / snake length, avg rewards, no of steps taken
            # ToDO tomorrow - legalActions replace by all possible actions
            # Change features / state
            # Write test logic

            env.close()
