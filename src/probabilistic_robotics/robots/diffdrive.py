import numpy as np
from .base import Robot

class DiffDriveRobot(Robot):

    def __init__(self, initial_pose, wheels_radius, wheels_distance, alpha, seed=None):
        self.wheels_radius = wheels_radius
        self.wheel_distance = wheels_distance
        self.alpha = alpha
        super().__init__(initial_pose, seed)

    def inverse_kinematics(self, v, w):
        raise NotImplementedError
    
    def diffdrive(self, x, y, theta, vL, vR, dt):
        w = (vR - vL) / self.wheel_distance # Velocidad angular 
        if w == 0:
            v = vR
            x_n = x + v * np.cos(theta) * dt
            y_n = y + v * np.sin(theta) * dt
            theta_n = theta
        else:
            R = self.wheel_distance / 2 * (vL + vR) / (vR - vL) # distancia del ICC al robot
            icc_x = x - R * np.sin(theta) # ICC_x
            icc_y = y + R * np.cos(theta) # ICC_y
            x_n = np.cos(w * dt) * (x - icc_x) - np.sin(w * dt) * (y - icc_y) + icc_x
            y_n = np.sin(w * dt) * (x - icc_x) + np.cos(w * dt) * (y - icc_y) + icc_y
            theta_n = theta + w * dt
        return x_n, y_n, theta_n


    def apply_movement(self, v, w, dt):

        # Obtenemos la pose actual
        x, y, theta = self.current_pose

        # Calculamos las velocidades lineales de las ruedas
        vL, vR = self.inverse_kinematics(v, w)
        
        # Calculamos la pose después de aplicar una velocidad v y w durante un intervalo de tiempo dt
        x_n, y_n, theta_n = self.diffdrive(x, y, theta, vL, vR, dt)
        
        # Actualizamos la pose actual
        self.current_pose = np.array([x_n, y_n, theta_n])


class NoisyDiffDriveRobot(DiffDriveRobot):

    def inverse_kinematics(self, v, w):
        # Velocidades de las ruedas
        wL = (v - w * self.wheel_distance / 2) / self.wheel_radius
        wR = (v + w * self.wheel_distance / 2) / self.wheel_radius

        # Velocidades con ruido
        v = self.wheels_radius * (wR + wL) / 2 + self._rs.randn() * self.alpha * (wL**2 + wR**2)
        w = self.wheels_radius * (wR - wL) / self.wheel_distance + self._rs.randn() * self.alpha * (wL**2 + wR**2)
        vL = v - w * self.wheel_distance / 2
        vR = v + w * self.wheel_distance / 2
        
        return vL, vR
    
class CleanDiffDriveRobot(DiffDriveRobot):

    def inverse_kinematics(self, v, w):

        # Velocidades de las ruedas sin ruido
        vL = (v - w * self.wheel_distance / 2)
        vR = (v + w * self.wheel_distance / 2)

        return vL, vR