
import numpy as np
from scipy.special import logit, expit


class GridMapping1D:

    def __init__(self, prior, grid_size, grid_interval, backgroud_limit):
        self.belief = np.ones(grid_size + 1) * prior
        self.grid_size = grid_size
        self.grid_interval = grid_interval
        self.backgroud_limit = backgroud_limit

    def position_to_cell(self, position):
        return np.digitize(position, self.grid) - 1
    
    @property
    def grid(self):
        return np.arange(0, self.grid_size + 1) * self.grid_interval

    def inverse_sensor_model(self, belief, robot_position, z):
        """
            Argumentos:
            -----------
            belief : np.array
                Probabilidad de ocupación actual del robot.
            robot_position : float
                Posición actual del robot.
            z : float
                Medida del sensor.

            Devuelve:
            ---------
            np.array
                Log-odds de ocupación actualizada.
        """
        belief = np.copy(belief)
        ids = np.arange(self.grid_size + 1)
        cell_robot = self.position_to_cell(robot_position)
        cell_measurement = self.position_to_cell(robot_position + z)
        if z > 0:
            cell_measurement_limit = self.position_to_cell(robot_position + z + self.backgroud_limit)
            belief[(cell_robot <= ids) & (ids < cell_measurement)] = .3
            belief[(cell_measurement < ids) & (ids < cell_measurement_limit)] = .6
        elif z < 0:
            cell_measurement_limit = self.position_to_cell(robot_position + z - self.backgroud_limit)
            belief[(cell_measurement_limit < ids) & (ids <= cell_robot)] = .3
            belief[(cell_measurement_limit < ids) & (ids < cell_measurement)] = .6
        else:
            cell_measurement_limit_inf = self.position_to_cell(robot_position - self.backgroud_limit)
            cell_measurement_limit_sup = self.position_to_cell(robot_position + self.backgroud_limit)
            belief[(cell_measurement_limit_inf < ids) & (ids < cell_measurement_limit_sup)] = .6
        
        logodds = logit(belief)
        return logodds

    def fit(self, measurements):
        belief = self.belief
        logodds_0 = logit(self.belief)
        logodds = logit(self.belief)

        history = [belief]
        for robot_position, z in measurements:
            logodds += self.inverse_sensor_model(belief, robot_position, z) - logodds_0
            belief = expit(logodds)
            history.append(belief)
        self.belief = belief
        return history
