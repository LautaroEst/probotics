
import numpy as np

class Lidar:

    def __init__(self, sensor_offset, num_scans, start_angle, end_angle, min_range, max_range, occupation_threshold=0.5, seed=None):

        self.current_pose = None
        self.sensor_offset = np.asarray(sensor_offset)
        self.num_scans = num_scans
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.scan_angles = np.linspace(start_angle, end_angle, int(num_scans))
        self.min_range = min_range
        self.max_range = max_range
        self._rs = np.random.RandomState(seed)
        self.ranges = np.ones(num_scans) * np.nan
        self.threshold = occupation_threshold

    def update_lidar_pose(self, robot_pose):
        # Robot pose
        x, y, theta = robot_pose

        # Matriz de cambio de base de la terna global a la terna local.
        T = np.array([
            [np.cos(theta), -np.sin(theta), x], 
            [np.sin(theta), np.cos(theta), y], 
            [0, 0, 1]
        ])
        
        # Lidar pose
        lidar_pose = np.hstack((self.sensor_offset, (1,))) @ T.T
        lidar_pose[2] = theta
        self.current_pose = lidar_pose

    def measure(self, robot_pose, map2d):
        
        # Update lidar pose
        self.update_lidar_pose(robot_pose)

        # Ranges
        self.ranges = self.rays_intersection(map2d.map_array, map2d.map_resolution)
        return self.ranges

    def rays_intersection(self, map_data, resolution):

        x, y, theta = self.current_pose
        scan_angles = self.scan_angles + theta
        ranges = np.zeros_like(scan_angles)

        xi = int(x / resolution)
        yi = int(y / resolution)
        if xi < 0 or yi < 0 or xi >= map_data.shape[1] or yi >= map_data.shape[0] or map_data[map_data.shape[0]-yi, xi] >= self.threshold:
            return ranges + np.nan

        for i, ray_angle in enumerate(scan_angles):
            sin_theta = np.sin(ray_angle)
            cos_theta = np.cos(ray_angle)

            # Step through the grid cells along the ray
            for r in np.linspace(0, self.max_range, int(self.max_range / resolution)):
                xi = int((x + r * cos_theta) / resolution)
                yi = int((y + r * sin_theta) / resolution)

                # Check if ray is out of bounds
                if xi < 0 or yi < 0 or xi >= map_data.shape[1] or yi >= map_data.shape[0]:
                    ranges[i] = np.nan
                    break

                # Check if the cell is occupied
                if map_data[map_data.shape[0]-yi, xi] >= self.threshold:
                    ranges[i] = r
                    break

        return ranges


    