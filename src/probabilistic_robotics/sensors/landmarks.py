
import numpy as np
import pandas as pd

from ..utils import evaluate_lognormal
   

class LandmarkIdentificator:

    def __init__(self, landmarks, sigma_r):
        self.landmarks = landmarks
        self.sigma_r = sigma_r

    def measurement_prob_range(self, current_pose, indices, ranges):
        """ computes the probability of a measurement 
        """
        # p = lambda x, mu: 1 / (np.sqrt(2*np.pi) * sigma_r) * np.exp(-(x - mu)**2 / 2 / sigma_r**2)
        
        x, y, _ = current_pose

        N = len(indices)
        logprob = 0
        for i in range(N):
            x_landmark = self.landmarks.loc[indices[i],'x']
            y_landmark = self.landmarks.loc[indices[i],'y']
            mu = np.sqrt((x - x_landmark)**2 + (y - y_landmark)**2)
            logprob += evaluate_lognormal(ranges[i], mu, self.sigma_r)
        return logprob
    
    @classmethod
    def from_file(cls, filename, sigma_r):
        world_data = pd.read_csv(filename, sep=' ', header=None, names=["id", "x", "y"]).set_index("id")
        return cls(world_data, sigma_r)
