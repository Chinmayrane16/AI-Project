import numpy as np

class SimpleSnakeGrid:
    def __init__(self, observation):
        self.observation = observation
        self.create_simple_representation()
    
    def create_simple_representation(self):

        HEAD_COLOR =np.array([255,10,0]) # red
        BODY_COLOR =np.array([1,0,0]) # black
        FOOD_COLOR =np.array([0,0,255]) # blue
        EMPTY_GRID_COLOR=np.array([0,255,0]) # green

        colorcode_dict = {
            str(HEAD_COLOR): 3,
            str(BODY_COLOR): 2,
            str(FOOD_COLOR): 1,
            str(EMPTY_GRID_COLOR): 0
        }
        
        rows, cols = (self.observation.shape[0], self.observation.shape[1])
        arr = [[0 for i in range(cols)] for j in range(rows)]

        cnt_r, cnt_c = 0,0
        for r in self.observation:
            cnt_c = 0
            for c in r:
                # store the head location
                if str(c) == str(HEAD_COLOR):
                    self.start = (cnt_r,cnt_c)

                arr[cnt_r][cnt_c] = colorcode_dict[str(c)]
                cnt_c += 1
            cnt_r += 1
        
        # store the snake grid 
        self.snake_grid = arr
        
    def print_grid(self):
        for row in self.snake_grid:
            print(row)
    
    def bfs_actions(self):
        queue = [(self.start, [])]
        rows = len(self.snake_grid)
        cols = len(self.snake_grid[0])
        visited = [[0 for i in range(cols)] for j in range(rows)]
        # UP = 0
        # RIGHT = 1
        # DOWN = 2
        # LEFT = 3
        DIRS = [([0,1], 1), ([1,0],2), ([0,-1],3),([-1,0], 0)]
        
        visited[self.start[0]][self.start[1]] = 1

        print('start loc', self.start)
        
        while len(queue) != 0:
            head, actions = queue.pop(0)
            
            if self.snake_grid[head[0]][head[1]] == 1:
                return actions
            
            for dir in DIRS:

                newR = head[0] + dir[0][0]
                newC = head[1] + dir[0][1]

                if newR >= 0 and newR < rows and newC >= 0 and newC < cols and visited[newR][newC] == 0 and (self.snake_grid[newR][newC] in [0,1]):
                    visited[newR][newC] = 1
                    # define new action
                    new_action = [action for action in actions]
                    new_action.append(dir[1])
                    queue.append(((newR, newC), new_action))
        
        # return empty array if no path found
        print("No path Found")
        return []
