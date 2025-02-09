from math import *
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

from ..robots.cleanodom import CleanOdometryRobot
from ..sensors.landmarks import DeterministicLandmarkIdentificator

class KalmanFilter:

    def __init__(self, mu0, sigma0, world_data, odometry_noise, measurement_noise, robot_radius=1):
        self.robot = CleanOdometryRobot(mu0, radius=robot_radius, seed=0)
        self._read_world_data(world_data, measurement_noise)
        self.world_data = world_data
        self.odometry_noise = odometry_noise
        self.measurement_noise = measurement_noise
        self.mu = mu0
        self.sigma = sigma0

    def _read_world_data(self, world_data, measurement_noise):
        if isinstance(world_data, str):
            if Path(world_data).exists():
                self.sensor = DeterministicLandmarkIdentificator.from_file(world_data, measurement_noise)
            else:
                raise FileNotFoundError(f"File {world_data} not found")
        elif isinstance(world_data, pd.DataFrame):
            self.sensor = DeterministicLandmarkIdentificator(world_data, measurement_noise)
        else:
            raise ValueError("world_data must be a path to a file or a pandas DataFrame")

    def update(self, odometry: dict, sensor: dict):

        # Paso de predicción
        mu_hat, sigma_hat = self.prediction_step(odometry)

        # Paso de corrección
        mu, sigma = self.correction_step(mu_hat, sigma_hat, sensor)
        self.robot.current_pose = mu

        self.mu = mu
        self.sigma = sigma
        
    def prediction_step(self, odometry):
        """
            Argumentos:
            -----------
            odometry: dict
                Diccionario con las lecturas del sensor de odometría. Las claves son 'r1', 't' y 'r2'.
            noise: np.ndarray
                Covarianza del ruido del modelo de movimiento.

            Devuelve:
            ---------
            mu_hat: np.ndarray
                Media predicha del estado.
            sigma_hat: np.ndarray
                Covarianza predicha del estado.
        """

        # Read in the state from the mu vector
        mu_x, mu_y, mu_theta = self.mu
        
        # Read in the odometry i.e r1, t , r2
        dr1, dt, dr2 = odometry['r1'], odometry['t'], odometry['r2']
        
        # Compute the noise free motion.    
        self.robot.apply_movement(dr1, dt, dr2)
        mu_hat = self.robot.current_pose
        
        # Computing the Jacobian of G with respect to the state
        Gt = np.array([
            [1, 0, -dt * np.sin(dr1 + mu_theta)],
            [0, 1, dt * np.cos(dr1 + mu_theta)],
            [0, 0, 1]
        ])
        
        # Predict the covariance
        sigma_hat = Gt @ self.sigma @ Gt.T + self.odometry_noise
        
        return mu_hat, sigma_hat
    

    def correction_step(self, mu_hat, sigma_hat, sensor):
        """
            Argumentos:
            -----------
            mu_hat: np.ndarray
                Media predicha del estado.
            sigma_hat: np.ndarray
                Covarianza predicha del estado.
            sensor: dict
                Diccionario con las lecturas del sensor. Las claves son 'id' y 'range'.

            Devuelve:
            ---------
            mu: np.ndarray
                Media corregida del estado.
            sigma: np.ndarray
                Covarianza corregida del estado.
        """
        
        # Landmarks
        world_data = self.sensor.landmarks

        # Read in the state from the uncorrected mu vector
        mu_x, mu_y, mu_theta = mu_hat
        
        # Read in the ids and ranges from measurements using dictionary indexing
        arg = np.argsort(sensor['id'])
        ids = np.asarray(sensor['id'])[arg]   
        ranges = np.asarray(sensor['range'])[arg]   
        
        # Initialize the Jacobian h
        N = np.size(ids)
        H = np.zeros((N,3))
        
        # Vectorize measurements 
        zt = ranges.reshape(N,1)
        h_mu_hat = np.sqrt(
            (mu_x - world_data.loc[ids,'x'].to_numpy()) ** 2 + (mu_y - world_data.loc[ids,'y'].to_numpy())**2
        )
        
        # Compute the columns of H
        denominator = np.sqrt(
            (mu_x - world_data.loc[ids,'x'])**2 + (mu_y - world_data.loc[ids,'y'])**2
        )
        H[:,0] = (mu_x - world_data.loc[ids,'x']) / denominator
        H[:,1] = (mu_y - world_data.loc[ids,'y']) / denominator
        H[:,2] = 0
            
        # Noise covariance for the measurements
        R = np.eye(N) * self.measurement_noise

        # Gain of Kalman
        K = sigma_hat @ H.T @ np.linalg.inv((H @ sigma_hat @ H.T) + R)    
        
        # Kalman correction for mean and covariance
        mu = (mu_hat.reshape(3,1) + K @ (zt - h_mu_hat.reshape(N,1))).squeeze()
        sigma = (np.eye(3) - (K @ H)) @ sigma_hat
        
        return mu, sigma




    
