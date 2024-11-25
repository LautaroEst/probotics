
import numpy as np

from .base import Robot

class GaussianNoisyOdometryRobot(Robot):

    def __init__(self, initial_pose, noise_params, radius, seed=None):
        self.noise_params = noise_params
        super().__init__(initial_pose, radius, seed)

    def apply_movement(self, r1, t, r2):

        # Unpack the current pose
        x, y, theta = self.current_pose
        
        # noise sigma for r1 
        r1_noisy = r1 + self._rs.randn() * self.noise_params[0]

        # noise sigma for translation
        t_noisy = t + self._rs.randn() * self.noise_params[2]

        # noise sigma for r2
        r2_noisy = r2 + self._rs.randn() * self.noise_params[1]

        # Estimate of the new position of the robot
        x = x  + t_noisy * np.cos(theta + r1_noisy)
        y = y  + t_noisy * np.sin(theta + r1_noisy)
        theta = theta + r1_noisy + r2_noisy
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Update the current pose
        self.current_pose = np.array([x, y, theta])
        