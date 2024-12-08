
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..src.slam.fastslam import FastSLAM
from ..src.utils import read_sensor_data, plot_ellipse
from ..src.sensors.landmarks import LandmarkIdentificator


def plot_landmarks_with_ellipsoids(ax, robot_pose, landmarks):
    for i, landmark in landmarks[landmarks["observed"]].iterrows():
        plot_ellipse(landmark['mu'], landmark['sigma'], ax, color='b')
        x = np.asarray([robot_pose[0], landmark['mu'][0]])
        y = np.asarray([robot_pose[1], landmark['mu'][1]])
        ax.plot(x, y, 'b--')


def main(N, odometry_noise_params, measurement_noise, sensor_data, world_data, plots_dir, seed=None):

    # Leemos los landmarks verdaderos
    landmarks = LandmarkIdentificator.from_file(world_data, 0.).landmarks

    # Inicializamos el filtro de partículas
    fs = FastSLAM(N, len(landmarks), odometry_noise_params, measurement_noise, use_neff=True, seed=seed)

    # Leemos los datos del sensor
    sensor_data = read_sensor_data(sensor_data)

    # Actualización del filtro de partículas y gráfico
    for t in tqdm(range(len(sensor_data)),total=len(sensor_data)):

        # Actualizamos el filtro de partículas
        particle_poses, best_particle = fs.update(sensor_data[t]['odom'], sensor_data[t]['sensor'])

        # Graficamos
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("FastSLAM. Timestamp: " + str(t))
        plot_landmarks_with_ellipsoids(ax, best_particle['robot'].current_pose, best_particle['sensor'].landmarks)
        ax.plot(particle_poses[:,0], particle_poses[:,1], 'r.')
        ax.plot(landmarks['x'], landmarks['y'], 'ko', markersize=5)
        best_particle['robot'].plot(ax, radius=0.5, color='k')
        plt.savefig(f"{plots_dir}/plot_{t:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def setup(command_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, help='Number of particles', default=100)
    parser.add_argument("--odometry_noise_params", type=str, help="Noise parameters", default="0.1,0.1,0.05,0.05")
    parser.add_argument("--sensor_noise", type=float, help="Sensor noise", default=0.1)
    parser.add_argument("--sensor_data", help="File containing sensor data")
    parser.add_argument("--world_data", help="File containing world data")
    parser.add_argument("--plots_dir", help="Directory to save plots", default="plots")
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=1234)
    args = parser.parse_args(command_args)
    
    main(
        args.N, 
        list(map(float, args.odometry_noise_params.split(','))), 
        args.sensor_noise, 
        args.sensor_data, 
        args.world_data, 
        args.plots_dir, 
        args.seed
    )


if __name__ == "__main__":
    import sys
    setup(sys.argv[1:])
    
    