
import numpy as np

from .base import Planning

class AStar(Planning):

    def __init__(self, map_data, start=(0, 0), goal=(0, 0), threshold=0.5, factor=1):
        super().__init__(map_data, start=start, goal=goal, threshold=threshold)
        self.factor = factor

    def get_heuristic(self, child):
        cost_val = self.map_data[child[0], child[1]] * 10 + np.linalg.norm(np.array(child) - np.array(self.goal))
        return cost_val * self.factor
    
