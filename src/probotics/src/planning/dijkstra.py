
from .base import Planning

class Dijkstra(Planning):

    def __init__(self, map_data, start=(0, 0), goal=(0, 0), threshold=0.5):
        super().__init__(map_data, start=start, goal=goal, threshold=threshold)

    def get_heuristic(self, cell):
        return 0
    
