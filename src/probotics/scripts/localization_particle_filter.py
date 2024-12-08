
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..src.localization import ParticleFilter
from ..src.utils import read_sensor_data


def  main(sensor_data, world_data, N, odometry_noise_params, sensor_noise, seed, plots_dir):

    # Inicializamos el filtro de partículas
    sensor_noise = float(sensor_noise)
    pf = ParticleFilter(N, world_data, odometry_noise_params, sensor_noise, seed)

    # Leemos los datos del sensor
    sensor_data = read_sensor_data(sensor_data)

    # Actualización del filtro de partículas y gráfico
    for t in tqdm(range(len(sensor_data)),total=len(sensor_data)):

        # Actualizamos el filtro de partículas
        pf.update(sensor_data[t]['odom'], sensor_data[t]['sensor'])

        # Obtenemos las poses de las partículas y la posición promedio
        poses = pf.get_particles_poses()
        mean_robot = pf.get_mean_robot()
        
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Localización con filtro de partículas y landmarks. Timestamp: " + str(t))
        ax.plot(pf.sensor.landmarks['x'], pf.sensor.landmarks['y'], 'ko', markersize=5)
        ax.plot(poses[:,0], poses[:,1], 'r.')
        mean_robot.plot(ax, radius=0.5, color='k')
        plt.savefig(f"{plots_dir}/plot_{t:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def setup(command_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor_data", type=str, help="File containing sensor data")
    parser.add_argument("--world_data", type=str, help="File containing world data")
    parser.add_argument('--N', type=int, help='Number of particles', default=100)
    parser.add_argument("--odometry_noise_params", type=str, help="Noise parameters", default="0.1,0.1,0.05,0.05")
    parser.add_argument("--sensor_noise", type=float, help="Sensor noise", default=0.1)
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=1234)
    parser.add_argument("--plots_dir", type=str, help="Directory to save plots", default="plots")
    args = parser.parse_args(command_args)
    
    main(
        args.sensor_data, 
        args.world_data, 
        args.N, 
        args.odometry_noise_params, 
        args.sensor_noise, 
        args.seed, 
        args.plots_dir
    )


if __name__ == "__main__":
    import sys
    setup(sys.argv[1:])
    