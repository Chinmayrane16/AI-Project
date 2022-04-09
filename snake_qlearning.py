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
# from black import get_features_used
import gym
import gym_snake
import numpy as np
from collections import deque, Counter
from simple_snake_grid import SimpleSnakeGrid
import random
import pickle
import csv
import time
from math import sqrt


class SnakeQlearning:

    def __init__(self, epsilon, lr, gamma, numTrainEpisodes, numTestEpisodes) -> None:
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
        self.numTrainEpisodes = numTrainEpisodes
        self.numTestEpisodes = numTestEpisodes
        self.rewards = []
        self.accumRewards = 0
        self.snake_lengths = []
        self.max_snake_length = 0
        self.dying_reward = -1000

    def print_grid(self, snake_grid):
        for row in snake_grid:
            print(row)

    def get_legal_actions(self, snake_grid, head):
        # Using paramaters instead of self, to get legal actions from any state, not just current state
        # snake_grid = self.snakeGridObj.snake_grid
        # head = self.snakeGridObj.snake_head
        # self.print_grid(snake_grid)

        legal_actions = [0, 1, 2, 3]
        mapDirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        snake_dq = self.snakeGridObj.virtual_snake_dq
        head = snake_dq[-1]
        neck = snake_dq[-2]

        dir = [neck[i] - head[i] for i in range(2)]
        not_legal_action = mapDirs.index(dir)

        legal_actions.remove(not_legal_action)

        # print("Legal action", legal_actions)
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

    def get_features(self, snake_dq):
        feature_set = [0] * 10    # [UP, DOWN, LEFT, RIGHT]
        snake_grid = self.snake_grid
        up = left = -1
        down = right = 1

        head = snake_dq[-1]
        food_loc = self.snakeGridObj.snake_food

        # Food relative to head
        dir = [food_loc[i] - head[i] for i in range(2)]
        feature_set[0:2] = [0, 1] if dir[0] > 0 else [1, 0] if dir[0] < 0 else [0, 0]
        feature_set[2:4] = [0, 1] if dir[1] > 0 else [1, 0] if dir[1] < 0 else [0, 0]

        # Obstacle relative to head
        rows = len(snake_grid)
        cols = len(snake_grid[0])

        legal_actions = self.get_legal_actions(snake_grid, head)

        # if 0 in legal_actions:
        newU = head[0] + up
        if newU < 0 or snake_grid[newU][head[1]] == 2:
            feature_set[4] = 1

        # if 2 in legal_actions:
        newD = head[0] + down
        if newD >= rows or snake_grid[newD][head[1]] == 2:
            feature_set[5] = 1

        # if 1 in legal_actions:
        newR = head[1] + right
        if newR >= cols or snake_grid[head[0]][newR] == 2:
            feature_set[6] = 1

        # if 3 in legal_actions:
        newL = head[1] + left
        if newL < 0 or snake_grid[head[0]][newL] == 2:
            feature_set[7] = 1

        # Snake direction
        neck = snake_dq[-2]
        dir = [head[i] - neck[i] for i in range(2)]
        # UP = 0 RIGHT = 1 DOWN = 2 LEFT = 3
        mapDirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        feature_set[8] = mapDirs.index(dir)

        # Manhattan distance to food
        feature_set[9] = self.manhattan(head, food_loc)
        # print("Feature set", tuple(feature_set))
        return tuple(feature_set)


    def manhattan(self, a, b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))
        

    def get_state(self, snake_dq):
        # snake_dq = [tuple(x) for x in snake_dq]
        # return (tuple(snake_dq), tuple(self.snakeGridObj.snake_food))
        return self.get_features(snake_dq)

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
            return self.dying_reward

        newHead = self.get_head_from_action(action)

        # Or it hits a boundary
        snake_grid = self.snake_grid
        rows = len(snake_grid)
        cols = len(snake_grid[0])
        newR = newHead[0]
        newC = newHead[1]
        # print("New Head", newHead, rows, cols)
        if newR < 0 or newR >= rows or newC < 0 or newC >= cols or (snake_grid[newR][newC] not in [0, 1]):
            return self.dying_reward

        if newHead == self.snakeGridObj.snake_food:
            return 100

        # Negative - Get to food asap
        # living reward
        return 0

    def get_head_from_action(self, action):
        head = self.snake_head
        mapDirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        newHeadX = head[0] + mapDirs[action][0]
        newHeadY = head[1] + mapDirs[action][1]

        # New head
        newHead = [newHeadX, newHeadY]

        return newHead

    def create_grid_from_v_snake(self, virtual_snake_dq, food_loc):
        rows, cols = (self.observation.shape[0], self.observation.shape[1])
        arr = [[0 for i in range(cols)] for j in range(rows)]

        # Food location
        position_obj = food_loc
        arr[position_obj[0]][position_obj[1]] = self.snakeGridObj.FOOD_CODE

        for i in range(len(virtual_snake_dq)):
            position_obj = virtual_snake_dq[i]
            arr[position_obj[0]][position_obj[1]] = self.snakeGridObj.BODY_CODE

        # finally add position to this virtual grid
        position_obj = virtual_snake_dq[-1]
        arr[position_obj[0]][position_obj[1]] = self.snakeGridObj.HEAD_CODE

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

        self.max_snake_length = max(self.max_snake_length, len(snake_dq))

    def train(self):
        for i in range(self.numTrainEpisodes):
            # Construct Environment
            env = gym.make(
                'snake-v0', grid_size=[8, 8], unit_size=1, unit_gap=0, snake_size=2)
            self.observation = env.reset()  # Constructs an instance of the game
            self.snakeGridObj = SimpleSnakeGrid(self.observation)
            self.snakeGridObj.update_observation(self.observation)
            # self.snakeGridObj.print_grid()

            self.snake_grid = self.snakeGridObj.snake_grid.copy()
            self.snake_head = self.snakeGridObj.snake_head.copy()

            # print("Episode", i+1)

            # Print status
            if (i+1) % 100 == 0:
                print("\nEpisodes: {} \nAverage rewards: {} \nAverage Rewards over last 100 episodes: {}\nMax Snake length till now: {} \nAvg Snake length till now: {} \nAvg Snake length 100 episodes: {}".format(
                    i+1, np.mean(self.rewards), np.mean(self.rewards[-100:]), self.max_snake_length, np.mean(self.snake_lengths), np.mean(self.snake_lengths[-100:])))

            reward = 0
            while True:
                # self.print_grid(self.snake_grid)
                curr_state = self.get_state(self.snakeGridObj.virtual_snake_dq)
                # print("Curr State", curr_state)
                curr_action = self.get_action()
                actions_name = ["Up", "Right", "Down", "Left"]
                # print("Curr action", actions_name[curr_action])
                # print(self.get_features(self.snakeGridObj.virtual_snake_dq))
                # self.print_grid(self.snake_grid)

                # Action = None handled in get_reward
                # Reward => death (-100) food (100) otherwise (-5)
                # Dequeue is update here
                reward = self.get_reward(curr_action)
                # print("Reward", reward)
                if reward == self.dying_reward:
                    # print("Current state", curr_state)
                    # print("Current action", curr_action)
                    # self.print_grid(self.snake_grid)
                    self.snake_lengths.append(len(self.snakeGridObj.virtual_snake_dq))
                    # print()
                    break
                self.update_dq(curr_action, reward)
                snake_dq = self.snakeGridObj.virtual_snake_dq
                # print("Next State Deque", snake_dq)

                nextState = self.get_state(snake_dq)
                # print("Next State", nextState)

                # Bellman equation
                self.bellmanUpdate(curr_state, curr_action,
                                   nextState, reward, self.snake_grid, self.snake_head)

                # env.render()
                self.observation, _, done, _ = env.step(curr_action)

                self.rewards.append(reward)
                self.accumRewards += reward

                # print()

                if reward == 100:
                    self.snakeGridObj.update_observation(self.observation)

            # Scoring -> food eaten / snake length, avg rewards, no of steps taken
            # ToDO tomorrow - legalActions replace by 3 legal actions - done
            # Change features / state - done
            # Update reward strategy
            # Write test logic

        env.close()

        # Store trained qtable
        # print(self.QTable)

        with open('QTable.pickle', 'wb') as outputfile:
            pickle.dump(self.QTable, outputfile)


    def test(self):
        with open('QTable.pickle', 'rb') as inputfile:
            self.QTable=pickle.load(inputfile)
        # print(self.QTable)
        # open the file in the write mode
        f = open('qlearning_eval.csv', 'w',newline='')
        # create the csv writer
        writer = csv.writer(f)

        writer.writerow(['algo','iteration number','food count','step count','time'])

        rows=[]
        self.epsilon = 0
        for i in range(self.numTestEpisodes):
            row=['qlearning']
            row.append(i+1)
            start = time.time()
            # Construct Environment
            env = gym.make(
                'snake-v0', grid_size=[8, 8], unit_size=1, unit_gap=0, snake_size=2)
            self.observation = env.reset()  # Constructs an instance of the game
            self.snakeGridObj = SimpleSnakeGrid(self.observation)
            self.snakeGridObj.update_observation(self.observation)
            # self.snakeGridObj.print_grid()

            self.snake_grid = self.snakeGridObj.snake_grid.copy()
            self.snake_head = self.snakeGridObj.snake_head.copy()
            steps_counter=0
            reward = 0
            while True:
                curr_action = self.get_action()
                steps_counter+=1
                reward = self.get_reward(curr_action)

                # env.render()
                self.observation, _, done, _ = env.step(curr_action)


                if reward == self.dying_reward:
                    foodcount = len(self.snakeGridObj.virtual_snake_dq) - 2
                    row.append(foodcount)
                    row.append(steps_counter)
                    row.append(time.time()-start)
                    rows.append(row)
                    env.close()
                    break

                self.update_dq(curr_action, reward)

                if reward == 100:
                    self.snakeGridObj.update_observation(self.observation)

        env.close()

        writer.writerows(rows)
        # close the file
        f.close()