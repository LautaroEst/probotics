import numpy as np

class Lidar:

    def __init__(self, sensor_offset, num_scans, start_angle, end_angle, max_range, seed=None):
        self.sensor_offset = sensor_offset
        self.num_scans = num_scans
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.scan_angles = np.linspace(start_angle, end_angle, num_scans)
        self.max_range = max_range
        self._rs = np.random.RandomState(seed)

    def measure(self, pose):
        pass
 