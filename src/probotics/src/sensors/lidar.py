
import numpy as np
from scipy.ndimage import gaussian_filter

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
    
    def compute_prob_of_measure(self, robot_pose, ranges, angles, map2d, resolution):
        x, y, theta = robot_pose
        prob = 1
        for i, (ray_angle, r) in enumerate(zip(angles, ranges)):
            if np.isnan(r):
                continue
            sin_theta = np.sin(theta + ray_angle)
            cos_theta = np.cos(theta + ray_angle)

            xi = int((x + r * cos_theta) / resolution)
            yi = int((y + r * sin_theta) / resolution)

            # Check if ray is out of bounds
            if xi < 0 or yi < 0 or xi >= map2d.shape[1] or yi >= map2d.shape[0]:
                return 1e-300
            
            # compute probability of not accupied
            xi = np.clip(xi, 0, map2d.shape[1]-1)
            yi = np.clip(yi, 1, map2d.shape[0])
            prob *= (1 - map2d[map2d.shape[0]-yi, xi])
        
        return prob

    
    # def compute_prob_of_measure(self, robot_pose, ranges, angles, map2d, resolution):
        
    #     x, y, theta = robot_pose
    #     max_range = np.max(ranges)

    #     linspace = np.linspace(0, max_range, int(max_range / resolution)).reshape(-1, 1)
    #     xi = np.clip(np.floor((x + linspace * np.sin(theta + angles)) / resolution), 0, map2d.shape[1]-1).squeeze().astype(int)
    #     yi = np.clip(np.floor((y + linspace * np.cos(theta + angles)) / resolution), 1, map2d.shape[0]).squeeze().astype(int)
    #     prob = np.prod(1 - map2d[map2d.shape[0]-yi, xi])

    #     # prob = 1
    #     # for i, ray_angle in enumerate(angles):
    #     #     sin_theta = np.sin(theta + ray_angle)
    #     #     cos_theta = np.cos(theta + ray_angle)

    #     #     # Step through the grid cells along the ray
    #     #     for r in np.linspace(0, max_range, int(max_range / resolution)):
    #     #         xi = int((x + r * cos_theta) / resolution)
    #     #         yi = int((y + r * sin_theta) / resolution)

    #     #         # Check if ray is out of bounds
    #     #         if xi < 0 or yi < 0 or xi >= map2d.shape[1] or yi >= map2d.shape[0]:
    #     #             return 1e-300
                
    #     #         # compute probability of not accupied
    #     #         xi = np.clip(xi, 0, map2d.shape[1]-1)
    #     #         yi = np.clip(yi, 1, map2d.shape[0])
    #     #         prob *= (1 - map2d[map2d.shape[0]-yi, xi])
        
    #     return prob
                
                
                

    
    # def compute_prob_of_measure(self, robot_pose, ranges, angles, map2d, resolution):
    #     # min max normalization
    #     map_prob = map2d - np.min(map2d)
    #     map_prob = map_prob / np.max(map_prob)
    #     # 2d Gaussian filter
    #     # map_prob = gaussian_filter(map_prob, sigma=0.2)
    #     # prob = map_prob[map_prob.shape[0]-int(robot_pose[1]/resolution), int(robot_pose[0]/resolution)]
    #     prob = 1
    #     for r, a in zip(ranges, angles):
    #         if np.isnan(r):
    #             continue
    #         x = robot_pose[0] + r * np.cos(robot_pose[2] + a)
    #         y = robot_pose[1] + r * np.sin(robot_pose[2] + a)
    #         if x < 0 or y < 0 or x >= map_prob.shape[1] * resolution or y >= map_prob.shape[0] * resolution:
    #             return 1e-300
    #         # nearest index
    #         x_idx = np.clip(int(x/resolution), 0, map_prob.shape[1]-1)
    #         y_idx = np.clip(int(y/resolution), 1, map_prob.shape[0])
    #         # prob *= map_prob[map_prob.shape[0]-int(y/resolution)-1, int(x/resolution)-1]
    #         prob *= map_prob[map_prob.shape[0]-y_idx, x_idx]
    #     return prob


    