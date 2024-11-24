
import numpy as np

class Planning:

    def __init__(self, map_data, start=(0, 0), goal=(0, 0)):
        self.map_data = map_data
        self.costs = np.ones_like(map_data) * np.inf
        self.heuristics = np.zeros_like(map_data)
        self.closed_list = np.zeros_like(map_data)
        self.previous_x = np.zeros_like(map_data)-1
        self.previous_y = np.zeros_like(map_data)-1

        self.start = start
        self.goal = goal
        
    def plan(self):
        parent = self.start
        self.costs[self.start[0], self.start[1]] = 0

        history = []
        while parent[0] != self.goal[0] or parent[1] != self.goal[1]:
            
            # generate mask to assign infinite costs for cells already visited
            closed_mask = self.closed_list.copy()
            closed_mask[closed_mask == 1] = np.inf

            # find the candidates for expansion (open list/frontier)
            open_list = self.costs + self.heuristics + closed_mask

            # check if a non-infinite entry exists in open list (list is not empty)
            if open_list.min == np.inf:
                raise ValueError("No valid path found")
            
            # find the cell with the minimum cost in the open list
            parent_y, parent_x = np.unravel_index(open_list == open_list.min(), open_list.shape)
            parent = (parent_y[0], parent_x[0])

            # put parent in closed list
            self.closed_list[parent[0], parent[1]] = 1
            history.append(parent)

            # get neighbors of parent
            neighbors = self.get_neighbors(parent)

            for neighbor in neighbors:
                child = (neighbor[0], neighbor[1])

                # calculate the cost of reaching the cell
                cost_val = self.costs[parent[0], parent[1]] + self.get_edge_cost(parent, child)

                # Exercise 2: estimate the remaining costs from the cell to the goal
                heuristic_val = self.get_heuristic(child)

                # update cost of cell
                if cost_val < self.costs[child[0], child[1]]:
                    self.costs[child[0], child[1]] = cost_val
                    self.heuristics[child[0], child[1]] = heuristic_val
                    self.previous_x[child[0], child[1]] = parent[0]
                    self.previous_y[child[0], child[1]] = parent[1]
        
        return history