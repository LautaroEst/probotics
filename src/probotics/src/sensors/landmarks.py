
import numpy as np
import pandas as pd

from ..utils import evaluate_lognormal
   

class LandmarkIdentificator:

    def __init__(self, landmarks, sensor_noise):
        self.landmarks = landmarks
        self.sensor_noise = sensor_noise

    def measurement_prob_range(self, current_pose, indices, ranges):
        # Calcula la probabilidad medir current_pose dadas las mediciones (indices, ranges)
        
        x, y, _ = current_pose

        logprob = 0
        for idx, rng in zip(indices, ranges):
            x_landmark = self.landmarks.loc[idx,'x']
            y_landmark = self.landmarks.loc[idx,'y']
            mu = np.sqrt((x - x_landmark)**2 + (y - y_landmark)**2)
            logprob += evaluate_lognormal(rng, mu, self.sensor_noise)
        prob = np.exp(logprob)
        prob += 1.e-300 # avoid round-off to zero
        return prob
    
    # def measurement_model(self, current_pose, landmark_id):

    #     x, y, theta = current_pose
    #     x_landmark, y_landmark = self.landmarks.loc[landmark_id, 'mu']
        
    #     # Use the current state of the particle to predict the measurment      
    #     expected_range = np.sqrt((x - x_landmark)**2 + (y - y_landmark)**2)
    #     expected_bearing = np.arctan2(y_landmark - y, x_landmark - x) - theta
    #     expected_bearing = (expected_bearing + np.pi) % (2 * np.pi) - np.pi
    #     h = np.array([expected_range, expected_bearing])
        
    #     # Compute the Jacobian H of the measurement function h wrt the landmark location
    #     H = np.array([
    #         [ (x_landmark - x) / expected_range, (y_landmark - y) / expected_range ],
    #         [ (y - y_landmark) / expected_range**2, (x_landmark - x) / expected_range**2 ]
    #     ])
        
    #     return h, H    
    
    @classmethod
    def from_file(cls, filename, sensor_noise):
        world_data = pd.read_csv(filename, sep=' ', header=None, names=["id", "x", "y"]).set_index("id")
        return cls(world_data, sensor_noise)
