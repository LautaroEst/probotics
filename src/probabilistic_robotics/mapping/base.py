

import numpy as np


class Map2D:

    def __init__(self, map_array, map_resolution):
        self.map_array = map_array
        self.map_resolution = map_resolution

    @classmethod
    def from_image(cls, map_path, map_resolution):
        from PIL import Image
        map_data  = 1 - np.array(Image.open(map_path)) / 255
        return cls(map_data, map_resolution)