
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..planning.dijkstra import Dijkstra
from ..planning.a_star import AStar

def read_map(map_path):
    map_data = pd.read_csv(map_path, header=None, index_col=None, sep=' ', skiprows=5)
    map_data = map_data.values[:, 1:]
    return map_data


def main(map_path, start, goal, method, factor, threshold, plots_dir):
    map_data = read_map(map_path)

    # Initialize planning
    if method == "dijkstra":
        planning = Dijkstra(map_data, start=start, goal=goal, threshold=threshold)
    elif method == "a_star":
        planning = AStar(map_data, start=start, goal=goal, threshold=threshold, factor=factor)

    # Plan
    print("Planning...")
    summary = planning.plan()

    # Plot history
    print("Plotting...")
    for i, parent in enumerate(tqdm(summary["history"])):
        fig, ax = plt.subplots()
        ax.imshow(map_data, cmap='gray_r')
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, map_data.shape[1]+1, 1)-0.5)
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0, map_data.shape[0]+1, 1)-0.5)
        ax.set_yticklabels([])
        ax.grid(which='both')
        ax.plot(start[1], start[0], 'r.', markersize=5)
        ax.plot(goal[1], goal[0], 'g.', markersize=5)
        if np.all(parent != start):
            ax.plot(parent[1], parent[0], 'y.', markersize=5)
        plt.savefig(f"{plots_dir}/plot_{i:04d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Plot final path
    fig, ax = plt.subplots()
    ax.imshow(map_data, cmap='gray_r')
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, map_data.shape[1]+1, 1)-0.5)
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0, map_data.shape[0]+1, 1)-0.5)
    ax.set_yticklabels([])
    ax.grid(which='both')
    for pc in summary["path_cells"]:
        ax.plot(pc[1], pc[0], 'b.', markersize=5)
    ax.plot(start[1], start[0], 'r.', markersize=5)
    ax.plot(goal[1], goal[0], 'g.', markersize=5)
    plt.savefig(f"{plots_dir}/plot_{i+1:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, help="Path to the map file")
    parser.add_argument("--start", type=str, help="Start position")
    parser.add_argument("--goal", type=str, help="Goal position")
    parser.add_argument("--method", type=str, default="dijkstra", help="Planning method")
    parser.add_argument("--factor", type=int, default=1, help="Factor to scale the heuristic")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the map")
    parser.add_argument("--plots_dir", type=str, help="Path to the directory where the plots will be saved")
    
    args = parser.parse_args()
    start = np.array(list(map(int, args.start.split(','))))
    goal = np.array(list(map(int, args.goal.split(','))))
    main(args.map_path, start, goal, args.method, args.factor, args.threshold, args.plots_dir)