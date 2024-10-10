
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..localization import ParticleFilter
from ..utils import read_sensor_data, read_world_data, plot_robot

def get_mean_position(particles):
    x = 0.0
    y = 0.0
    orientation = 0.0
    
    for p in particles:
        x += p[0]
        y += p[1]
        orientation += p[2]
        
    x /= len(particles)
    y /= len(particles)
    orientation /= len(particles)
    
    return x, y, orientation


def  main(sensor_data, world_data, N, noise_params, seed, plots_dir):

    noise_params = list(map(float, noise_params.split(',')))
    sensor_data = read_sensor_data(sensor_data)
    world_data = read_world_data(world_data)

    pf = ParticleFilter(N, random_state=seed)
    print("Fitting particle filter...")
    history = pf.fit(sensor_data, world_data, noise_params=noise_params)

    # Plot
    print("Creating plots...")
    for t, particles in tqdm(history):

        # Obtenemos la posición de las partículas
        pos = np.array([[p[0], p[1]] for p in particles])

        # Obtenemos la posición promedio
        mean_pos = get_mean_position(particles)
        
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 18)
        ax.set_ylim(-1, 18)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Filtro de partículas para ubicar al robot. Timestamp: " + str(t))
        ax.plot(world_data['x'], world_data['y'], 'ko', markersize=5)
        ax.plot(pos[:,0], pos[:,1], 'r.')
        plot_robot(mean_pos, ax, radius=0.5, color='k')
        plt.savefig(f"{plots_dir}/plot_{t:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sensor_data", type=str, help="File containing sensor data")
    parser.add_argument("--world_data", type=str, help="File containing world data")
    parser.add_argument('--N', type=int, help='Number of particles', default=100)
    parser.add_argument("--noise_params", type=str, help="Noise parameters", default="0.1,0.1,0.05,0.05")
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=1234)
    parser.add_argument("--plots_dir", type=str, help="Directory to save plots", default="plots")

    args = parser.parse_args()
    main(args.sensor_data, args.world_data, args.N, args.noise_params, args.seed, args.plots_dir)