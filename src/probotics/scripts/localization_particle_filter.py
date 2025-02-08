
import argparse
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import ffmpeg

from ..src.localization import ParticleFilter
from ..src.utils import read_sensor_data
from ..src.config import BASE_OUTPUT_DIR


def  main(sensor_data, world_data, N, odometry_noise_params, sensor_noise, seed, output_dir):

    # Creamos el directorio de salida
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Inicializamos el filtro de partículas
    sensor_noise = float(sensor_noise)
    pf = ParticleFilter(N, world_data, odometry_noise_params, sensor_noise, radius=0.5, seed=seed)

    # Leemos los datos del sensor
    sensor_data = read_sensor_data(sensor_data)

    # Actualización del filtro de partículas y gráfico
    print("Actualizando filtro de partículas y generando gráficos...")
    for t in tqdm(range(len(sensor_data)),total=len(sensor_data)):

        # Obtenemos las poses de las partículas y la posición promedio
        poses = pf.get_particles_poses()
        # best = pf.get_best_particle()
        best = pf.get_mean_robot()
        
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Localización con filtro de partículas y landmarks. Timestamp: " + str(t))
        ax.plot(pf.sensor.landmarks['x'], pf.sensor.landmarks['y'], 'ko', markersize=5)
        ax.plot(poses[:,0], poses[:,1], 'r.')
        best.plot(ax, color='k')
        plt.savefig(f"{plots_dir}/plot_{t:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Actualizamos el filtro de partículas
        pf.update(sensor_data[t]['odom'], sensor_data[t]['sensor'])

    # Último gráfico
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Localización con filtro de partículas y landmarks. Timestamp: " + str(len(sensor_data)))
    ax.plot(pf.sensor.landmarks['x'], pf.sensor.landmarks['y'], 'ko', markersize=5)
    ax.plot(poses[:,0], poses[:,1], 'r.')
    best.plot(ax, color='k')
    plt.savefig(f"{plots_dir}/plot_{t:03d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Generamos el video
    print("Generando video...")
    stream = ffmpeg.input(f"{plots_dir}/plot_%03d.png", framerate=10)
    stream = ffmpeg.output(stream, f"{output_dir}/pf.mp4")
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)

    

def setup(command_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor_data", type=str, help="File containing sensor data")
    parser.add_argument("--world_data", type=str, help="File containing world data")
    parser.add_argument('--N', type=int, help='Number of particles', default=100)
    parser.add_argument("--odometry_noise_params", type=float, nargs=4, help="Odometry noise parameters", default=[0.1, 0.1, 0.05, 0.05])
    parser.add_argument("--sensor_noise", type=float, help="Sensor noise", default=0.1)
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=1234)
    args = parser.parse_args(command_args)
    
    main(
        args.sensor_data, 
        args.world_data, 
        args.N, 
        args.odometry_noise_params, 
        args.sensor_noise, 
        args.seed, 
        os.path.join(BASE_OUTPUT_DIR, "particle_filter")
    )


if __name__ == "__main__":
    import sys
    setup(sys.argv[1:])
    