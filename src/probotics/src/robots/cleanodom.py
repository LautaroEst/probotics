
import numpy as np

from .base import Robot

class CleanOdometryRobot(Robot):

    def __init__(self, initial_pose, radius=1, seed=None):
        super().__init__(initial_pose, radius, seed)

    def apply_movement(self, r1, t, r2):

        # Unpack the current pose
        x, y, theta = self.current_pose

        # Compute the noise free motion.    
        x = x + t * np.cos(r1 + theta)
        y = y + t * np.sin(r1 + theta)
        theta = theta + r1 + r2

        # Update the current pose
        self.current_pose = np.array([x, y, theta])
        