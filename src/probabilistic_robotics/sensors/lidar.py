
import numpy as np

class Lidar:

    def __init__(self, sensor_offset, num_scans, start_angle, end_angle, max_range, seed=None):
        self.sensor_offset = np.asarray(sensor_offset)
        self.num_scans = num_scans
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.scan_angles = np.linspace(start_angle, end_angle, int(num_scans))
        self.max_range = max_range
        self._rs = np.random.RandomState(seed)
        self.ranges = None

    def measure(self, pose, map_data, resolution):
        lidar_pose = pose + np.hstack((self.sensor_offset, (0,)))
        inter_points = self.ray_intersection(map_data, resolution, lidar_pose)

    def rays_intersection(self, map_data, resolution, lidar_pose):
        scan_angles = self.scan_angles + lidar_pose[2]

        x, y, theta = lidar_pose
        x_idx, y_idx = int(x / resolution), int(y / resolution)
        ranges = np.full_like(scan_angles.angles, self.max_range)

        for i, angle in enumerate(scan_angles):
            ray_angle = theta + angle
            sin_theta = np.sin(ray_angle)
            cos_theta = np.cos(ray_angle)

            # Step through the grid cells along the ray
            for r in np.linspace(0, self.max_range, int(self.max_range / resolution)):
                xi = int((x + r * cos_theta) / resolution)
                yi = int((y + r * sin_theta) / resolution)

                # Check if ray is out of bounds
                if xi < 0 or yi < 0 or xi >= map_data.shape[1] or yi >= map_data.shape[0]:
                    break

                # Check if the cell is occupied
                if map_data[yi, xi] == 1:
                    ranges[i] = r
                    break

        return ranges


    