
import numpy as np

class Planning:

    def __init__(self, map_data, start=np.array([0, 0]), goal=np.array([0, 0]), threshold=0.5):
        self.map_data = map_data
        self.costs = np.ones_like(map_data) * np.inf
        self.heuristics = np.zeros_like(map_data)
        self.closed_list = np.zeros_like(map_data)
        self.previous = np.zeros((*map_data.shape, 2), dtype=int) - 1

        self.start = start
        self.goal = goal
        self.threshold = threshold
        
    def plan(self):
        parent = self.start
        self.costs[tuple(self.start)] = 0

        history = []
        while np.any(parent != self.goal):
            
            # generate mask to assign infinite costs for cells already visited
            closed_mask = self.closed_list.copy()
            closed_mask[closed_mask == 1] = np.inf

            # find the candidates for expansion (open list/frontier)
            open_list = self.costs + self.heuristics + closed_mask

            # check if a non-infinite entry exists in open list (list is not empty)
            if open_list.min == np.inf:
                raise ValueError("No valid path found")
            
            # find the cell with the minimum cost in the open list
            parent = np.asarray(np.unravel_index(np.argmin(open_list), open_list.shape))

            # put parent in closed list
            self.closed_list[tuple(parent)] = 1
            history.append(parent)

            # get neighbors of parent
            neighbors = self.get_neighbors(parent)

            for child in neighbors:

                # calculate the cost of reaching the cell
                cost_val = self.costs[tuple(parent)] + self.get_edge_cost(parent, child)

                # Exercise 2: estimate the remaining costs from the cell to the goal
                heuristic_val = self.get_heuristic(child)

                # update cost of cell
                child_idx = tuple(child)
                if cost_val < self.costs[child_idx]:
                    self.costs[child_idx] = cost_val
                    self.heuristics[child_idx] = heuristic_val
                    self.previous[child_idx] = parent
        
        # Save final path, cost, distance, and number of nodes visited
        parent = self.goal
        path_cells = []
        distance2 = 0
        while self.previous[tuple(parent)][0] >= 0:
            path_cells.append(parent)
            child = self.previous[tuple(parent)]
            distance2 += np.linalg.norm(parent - child)
            parent = child
        nodes_visited = np.sum(self.closed_list)
        path_cost = self.costs[tuple(self.goal)]
        
        return {
            "history": history,
            "path_cells": path_cells,
            "path_cost": path_cost,
            "distance2": distance2,
            "nodes_visited": nodes_visited,
        }
    
    def get_neighbors(self, cell):
        size_y, size_x = self.map_data.shape
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (0 <= cell[0] + i < size_y) and (0 <= cell[1] + j < size_x):
                    neighbors.append([cell[0] + i, cell[1] + j])
        return np.array(neighbors)
    
    def cell_is_occupied(self, cell):
        return self.map_data[tuple(cell)] > self.threshold

    def get_edge_cost(self, parent, child):
        if self.cell_is_occupied(child) or self.cell_is_occupied(parent):
            return np.inf
        return np.linalg.norm(parent - child)
    
    def get_heuristic(self, cell):
        raise NotImplementedError