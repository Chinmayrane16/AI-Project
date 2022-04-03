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

'''

import numpy as np
from collections import deque, Counter
from simple_snake_grid import SimpleSnakeGrid
from random import random

class SnakeQlearning:

    def __init__(self, observation,epsilon) -> None:
        # UP = 0
        # RIGHT = 1
        # DOWN = 2
        # LEFT = 3
        self.DIRS = [([0,1], 1), ([1,0],2), ([0,-1],3),([-1,0], 0)]
        self.snakeGridObj = SimpleSnakeGrid(observation)
        # self.actionFn = 
        self.epsilon = epsilon
        self.QTable = Counter()
        

    def get_legal_actions(self,snake_grid,head):
        # Using paramaters instead of self, to get legal actions from any state, not just current state
        # snake_grid = self.snakeGridObj.snake_grid
        # head = self.snakeGridObj.snake_head
        rows = len(snake_grid)
        cols = len(snake_grid[0])
        # Return legal actions
        legal_actions=[]
        for dir in self.DIRS:
                newR = head[0] + dir[0][0]
                newC = head[1] + dir[0][1]
                if newR >= 0 and newR < rows and newC >= 0 and newC < cols and (snake_grid[newR][newC] in [0,1]):
                    legal_actions.append(dir[1])

        return legal_actions
    


    def getQValue(self,snake_grid,action):        
        return(self.QTable[(snake_grid,action)])

    def computeActionFromQValues(self,snake_grid):
        legal_actions_list = self.get_legal_actions(self.snakeGridObj.snake_grid,self.snakeGridObj.snake_head)

        if len(legal_actions_list)>0:
            temp=None
            best_actions=[]
            for action in legal_actions_list:
                curr_val = self.getQValue(snake_grid,action)
                if temp is None or curr_val>temp:
                    temp=curr_val
                    best_actions.clear()
                    best_actions.append(action)
                elif curr_val==temp:
                    best_actions.append(action)
            return random.choice(best_actions)
        else:
            return None

    def get_action(self):
        # function returns either random action or q table action depending on epsilon
        legal_actions_list = self.get_legal_actions(self.snakeGridObj.snake_grid,self.snakeGridObj.snake_head)
        if(len(legal_actions_list)>0):
            r = random.random()
            flip_coin = (r<self.epsilon)

            if(flip_coin):
                action = random.choice(legal_actions_list)
            else:
                action = self.computeActionFromQValues(self.snakeGridObj.snake_grid)
            
            return action
        else:
            return None


    