
import numpy as np

def get_global_pose(measurements, reference_pose):
    """
    Recibe una matriz en donde cada fila es una medición de una pose
    con respecto a la terna reference_pose (x, y, theta) y
    devuelve las poses con respecto a la terna global.

    Argumentos:
    -----------
        pose: (Nx3 np.array) Matriz de Nx3 en donde cada fila representa 
            la pose de un objeto con respecto a la terna de referencia.
        reference_pose: tuple (float, float, float) terna de referencia.

    Devuelve:
    ---------
        measurements_global (Nx3 np.array): Matriz de Nx3 en donde 
            cada fila representa la pose del objeto con respecto a la 
            terna global.
    """

    x, y, theta = reference_pose

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



class Robot:

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    @property
    def transformation_matrix(self):
        return np.array([
            [np.cos(self.theta), -np.sin(self.theta), self.x], 
            [np.sin(self.theta), np.cos(self.theta), self.y], 
            [0, 0, 1]
        ])
    
    def local2global(self, local_measurement):
        x, y = local_measurement
        global_measurement = self.transformation_matrix @ np.array([[x, y, 1]]).T
        return global_measurement[:2].flatten()

        