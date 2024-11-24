import numpy as np
import matplotlib.pyplot as plt

class Robot:

    def __init__(self, initial_pose, seed=None):
        self._rs = np.random.RandomState(seed)
        self._current_pose = initial_pose
        self._pose_history = []

    @property
    def current_pose(self):
        return self._current_pose

    @current_pose.setter
    def current_pose(self, pose):
        self.pose_history.append(self._current_pose)
        self._current_pose = pose
    
    @property
    def pose_history(self):
        return self._pose_history

    def body2world(self, measurements):
        x, y, theta = self.current_pose

        # Matriz de cambio de base de la terna global a la terna local.
        T = np.array([
            [np.cos(theta), -np.sin(theta), x], 
            [np.sin(theta), np.cos(theta), y], 
            [0, 0, 1]
        ])
        
        measurements_1 = np.hstack((measurements[:,:2], np.ones((measurements.shape[0],1)))) # Concateno columna de 1's
        measurements_global = measurements_1 @ T.T
        measurements_global[:,2] = measurements[:,2] + theta
        return measurements_global
    
    def apply_movement(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, ax, radius=1, **kwargs):
        x, y, theta = self.current_pose
        circle = plt.Circle((x,y), radius, fill=False, color=kwargs.get('color', None))
        ax.add_artist(circle)
        ax.plot([x, x + radius * np.cos(theta)], [y, y + radius * np.sin(theta)], linewidth=2, **kwargs)

