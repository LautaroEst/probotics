
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..planning.dijkstra import Planning

def read_map(map_path):
    map_data = pd.read_csv(map_path, header=None, index_col=None, sep=' ', skiprows=5)
    map_data = map_data.values[:, 1:]
    return map_data


def main(map_path, start, goal, plots_dir):
    map_data = read_map(map_path)

    planning = Planning(map_data, start=start, goal=goal)

    fig, ax = plt.subplots()
    ax.imshow(map_data, cmap='gray')
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, map_data.shape[1], 1))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0, map_data.shape[0], 1))
    ax.set_yticklabels([])
    ax.grid(which='both')
    ax.plot(start[1], start[0], 'ro', markersize=10)
    ax.plot(goal[1], goal[0], 'go', markersize=10)

    history = planning.plan()

    # visualization: from the goal to the start
    # draw the path as blue dots
    parent = goal
    distance2 = 0
    while planning.previous_x[parent[0], parent[1]] >= 0:
        ax.plot(parent[1], parent[0], 'bo', markersize=5)

        child = (planning.previous_y[parent[0], parent[1]], planning.previous_x[parent[0], parent[1]])
        distance2 += np.linalg.norm(np.array(parent) - np.array(child))
        parent = child



    plt.savefig(f"{plots_dir}/map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, help="Path to the map file")
    parser.add_argument("--start", type=str, help="Start position")
    parser.add_argument("--goal", type=str, help="Goal position")
    parser.add_argument("--plots_dir", type=str, help="Path to the directory where the plots will be saved")
    
    args = parser.parse_args()
    main(args.map_path, tuple(map(int, args.start.split(',')), tuple(map(int, args.goal.split(','))), args.plots_dir))