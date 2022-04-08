import numpy as np
from collections import deque
import heapq as hq

class PriorityQueue:
    
    def  __init__(self):
        self.Priorityheap = []

    def isEmpty(self):
        return len(self.Priorityheap) == 0

    def push(self,total_cost,data):
        item = (total_cost,data)
        hq.heappush(self.Priorityheap,item)

    def pop(self):
        cost,data = hq.heappop(self.Priorityheap)
        return data

class SimpleSnakeGrid:
    HEAD_COLOR = np.array([255, 10, 0])  # red
    BODY_COLOR = np.array([1, 0, 0])  # black
    FOOD_COLOR = np.array([0, 0, 255])  # blue
    EMPTY_GRID_COLOR = np.array([0, 255, 0])  # green

    FOOD_CODE = 1
    HEAD_CODE = 3
    BODY_CODE = 2
    EMPTY_GRID_CODE = 0

    colorcode_dict = {
        str(HEAD_COLOR): HEAD_CODE,
        str(BODY_COLOR): BODY_CODE,
        str(FOOD_COLOR): FOOD_CODE,
        str(EMPTY_GRID_COLOR): EMPTY_GRID_CODE
    }

    def __init__(self, observation):
        self.initial_observation = observation
        self.init_virtual_snake_and_grid()

    def init_virtual_snake_and_grid(self):
        self.snake_vir_grid, snake_vir_head, snake_vir_body, _ = self.get_updated_grid(
            self.initial_observation)
        self.virtual_snake_dq = deque([snake_vir_body, snake_vir_head])

    # each update step gets observation which will be reflected here
    def update_observation(self, observation):
        self.observation = observation
        self.snake_grid, self.snake_head, _, self.snake_food = self.get_updated_grid(
            self.observation)

    def perform_virtual_moves(self, actions, virtual_snake_dq):
        # keeping the dir at the coressponding index  UP = 0 RIGHT = 1 DOWN = 2 LEFT = 3
        dir_arr = [[-1, 0], [0, 1], [1, 0], [0, -1]]

        # perform moves
        for action in actions:
            head = virtual_snake_dq[-1].copy()
            update = dir_arr[action]
            # get new head
            head[0] += update[0]
            head[1] += update[1]

            # if we have reached food..which should ideally be the last action
            if self.snake_grid[head[0]][head[1]] != self.FOOD_CODE:
                # remove tail
                virtual_snake_dq.popleft()
            else:
                print("Food reached increasing body")

            # add head
            virtual_snake_dq.append(head)
        print(virtual_snake_dq)

    def get_updated_grid(self, observation):
        rows, cols = (observation.shape[0], observation.shape[1])
        arr = [[0 for i in range(cols)] for j in range(rows)]

        cnt_r, cnt_c = 0, 0
        snake_head = None
        snake_body = None
        food_loc = None
        for r in observation:
            cnt_c = 0
            for c in r:
                # store the head location
                if str(c) == str(self.HEAD_COLOR):
                    snake_head = [cnt_r, cnt_c]

                elif str(c) == str(self.BODY_COLOR):
                    snake_body = [cnt_r, cnt_c]

                elif str(c) == str(self.FOOD_COLOR):
                    food_loc = [cnt_r, cnt_c]

                arr[cnt_r][cnt_c] = self.colorcode_dict[str(c)]
                cnt_c += 1
            cnt_r += 1

        # update the snake grid
        return arr, snake_head, snake_body, food_loc

    def print_grid(self):
        for row in self.snake_grid:
            print(row)

    def create_grid_from_v_snake(self, virtual_snake_dq):
        rows, cols = (
            self.initial_observation.shape[0], self.initial_observation.shape[1])
        arr = [[0 for i in range(cols)] for j in range(rows)]

        for i in range(len(virtual_snake_dq)):
            position_obj = virtual_snake_dq[i]
            arr[position_obj[0]][position_obj[1]] = self.BODY_CODE

        # finally add position to this virtual grid
        position_obj = virtual_snake_dq[-1]
        arr[position_obj[0]][position_obj[1]] = self.HEAD_CODE

        # make the empty cell around tail like a food temporarily to make the bfs work
        position_obj = virtual_snake_dq[0]
        arr[position_obj[0]][position_obj[1]] = self.FOOD_CODE

        return arr

    def bfs_actions(self):
        actions = self.bfs(self.snake_head, self.snake_grid, self.snake_food)
        return actions

    def bfs(self, start, snake_grid, dest):
        queue = [(start, [])]
        rows = len(snake_grid)
        cols = len(snake_grid[0])
        visited = [[0 for i in range(cols)] for j in range(rows)]
        # UP = 0
        # RIGHT = 1
        # DOWN = 2
        # LEFT = 3
        DIRS = [([0, 1], 1), ([1, 0], 2), ([0, -1], 3), ([-1, 0], 0)]

        visited[start[0]][start[1]] = 1

        print('start loc', start)

        while len(queue) != 0:
            head, actions = queue.pop(0)

            if head[0] == dest[0] and head[1] == dest[1]:
                return actions

            for dir in DIRS:

                newR = head[0] + dir[0][0]
                newC = head[1] + dir[0][1]

                if newR >= 0 and newR < rows and newC >= 0 and newC < cols and visited[newR][newC] == 0 and (snake_grid[newR][newC] in [0, 1]):
                    visited[newR][newC] = 1
                    # define new action
                    new_action = [action for action in actions]
                    new_action.append(dir[1])
                    queue.append(((newR, newC), new_action))

        # return empty array if no path found
        print("No path Found")
        return []

    def all_vertices_visited(self,visited):
        for row in visited:
            for elem in row:
                if elem == 0:
                    return False
        return True
    

    def hamilton_util(self,snake_grid, posR, posC, visited, snake_head,cum_actions):
        DIRS = [([0,1], 1), ([1,0],2), ([0,-1],3),([-1,0], 0)]
        visited[posR][posC] = 1

        cycle_potential = False
        if self.all_vertices_visited(visited):
            cycle_potential = True

        for dir in DIRS:
            newR = posR + dir[0][0]
            newC = posC + dir[0][1]

            if cycle_potential and [newR,newC] == snake_head:
                print("Found a hamilton cycle")
                new_actions = [action for action in cum_actions]
                new_actions.append(dir[1])
                return new_actions

            if newR < 0 or newR == len(snake_grid) or newC < 0 or newC == len(snake_grid) or visited[newR][newC] == 1:
                continue
            
            new_actions = [action for action in cum_actions]
            new_actions.append(dir[1])
            
            action_list = self.hamilton_util(snake_grid, newR, newC, visited,snake_head, new_actions)
            if action_list:
                return action_list

        visited[posR][posC] = 0
        return []

    def hamilton_cycle(self,snake_head, snake_grid):
        rows = len(snake_grid)
        cols = len(snake_grid[0])
        visited = [[0 for i in range(cols)] for j in range(rows)]

        return self.hamilton_util(snake_grid, snake_head[0], snake_head[1],visited,snake_head,[])

    def hamilton_actions(self):
        return self.hamilton_cycle(self.snake_head, self.snake_grid)

    def Astar_actions(self):
        virtual_snake_curr = self.virtual_snake_dq.copy()
        look_ahead_grid = self.create_grid_from_v_snake(virtual_snake_curr)

        follow_tail = self.Astar(
            virtual_snake_curr[-1], look_ahead_grid, virtual_snake_curr[0])
        print("follow tail actions ** ", follow_tail)

        actions = self.Astar(self.snake_head, self.snake_grid, self.snake_food)

        # update our virtual snake to be at the food position if the path is available
        if actions:
            virtual_snake = self.virtual_snake_dq.copy()
            self.perform_virtual_moves(actions, virtual_snake)
            look_ahead_grid = self.create_grid_from_v_snake(virtual_snake)

            # check for path to tail from here
            if self.Astar(virtual_snake[-1], look_ahead_grid, virtual_snake[0]):
                print("path to tail is availble from food location")
                self.follow_tail_ctr = 0
                self.virtual_snake_dq = virtual_snake
                return actions

        print("no path is available from food source. Following tail!")
        if self.follow_tail_ctr >= 50:
            print('followed tail for many times..!!..exiting')
            return []
        
        self.follow_tail_ctr += 1
        self.perform_virtual_moves(follow_tail, virtual_snake_curr)
        self.virtual_snake_dq = virtual_snake_curr
        return follow_tail


    def Astar(self, start, snake_grid, dest):
        pqueue = PriorityQueue()
        pqueue.push(0 + self.heuristic(start,dest),((start),[],0))
        rows = len(snake_grid)
        cols = len(snake_grid[0])
        visited = [[0 for i in range(cols)] for j in range(rows)]
        # UP = 0
        # RIGHT = 1
        # DOWN = 2
        # LEFT = 3
        DIRS = [([0, 1], 1), ([1, 0], 2), ([0, -1], 3), ([-1, 0], 0)]

        visited[start[0]][start[1]] = 1

        print('start loc', start)

        while (pqueue.isEmpty()==False):
            head,actions,path_cost = pqueue.pop()
            if head[0] == dest[0] and head[1] == dest[1]:
                return actions

            for dir in DIRS:

                newR = head[0] + dir[0][0]
                newC = head[1] + dir[0][1]

                if newR >= 0 and newR < rows and newC >= 0 and newC < cols and visited[newR][newC] == 0 and (snake_grid[newR][newC] in [0, 1]):
                    visited[newR][newC] = 1
                    # define new action
                    new_actions = [action for action in actions]
                    new_actions.append(dir[1])
                    #new cost
                    new_path_cost = path_cost + self.manhattan_dist(head,(newR,newC))
                    new_total_cost = new_path_cost + self.heuristic((newR,newC),dest) 
                    pqueue.push(new_total_cost,((newR, newC), new_actions,new_path_cost))

        # return empty array if no path found
        print("No path Found")
        return []


    def manhattan_dist(self,state,dest):
        return abs(state[0] - dest[0]) + abs( state[1] - dest[1])


    def heuristic(self,state,dest):
        return self.manhattan_dist(state,dest)
