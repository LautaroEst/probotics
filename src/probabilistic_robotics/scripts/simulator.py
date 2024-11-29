import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import importlib
from scipy.io import loadmat

from ..robots import *
from ..sensors import *
from ..tasks import *
from ..mapping import *

# ROBOT SPECTS
RADIUS = 0.35 / 2 # Radio del robot [m]
WHEELS_DISTANCE = 0.235 # Distancia entre las ruedas [m]
WHEELS_RADIUS = 0.072 / 2 # Radio de las ruedas [m]
ALPHA = 0.0015 # Factor de ruido para la simulación

# LIDIAR_SPECS
SENSOR_OFFSET = (.07, 0) # Posición del sensor en el robot
MAX_NUM_SCANS = 513 # Número de escaneos por medición
START_ANGLE = -np.pi / 2 # Ángulo inicial de medición
END_ANGLE = np.pi / 2 # Ángulo final de medición
MAX_RANGE = 5 # Distancia máxima de medición
MIN_RANGE = 0.06 # Distancia mínima de medición
OCCUPATION_THRESHOLD = 0.45 # Umbral de ocupación

# SIMULATION SPECS
# MAX_DURATION = 3 * 60  # 3 minutes
MAX_DURATION = 10  # 10 seconds
SAMPLE_TIME = 0.1  # 100ms
WAITING_TIME = 5  # 5 seconds

def load_yaml(path):
    with open(path, "r") as f:
        file = yaml.safe_load(f)
    if file is None:
        return {}
    return file


def read_tasks(tasks_cfgs):
    tasks = []
    module = importlib.import_module('..tasks', package=__package__)
    for cfg in tasks_cfgs:
        cfg = load_yaml(cfg + ".yaml")
        task_name = cfg.pop('task')
        task_cls = getattr(module, task_name)
        task = task_cls(**cfg)
        tasks.append(task)
    return tasks

        

def save_state_for_visualization(history, state):
    history.append({
        "pose": state["robot"].current_pose,
        "path": [],
        "ranges": state["sensor"].ranges,
    })
    

def visualize_history(history, map_data, map_resolution, radius, sensor_offset, start_angle, end_angle, plots_dir):
    
    for i, state in enumerate(history):
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        
        # Plot map
        extent = np.array([0, map_data.shape[1], 0, map_data.shape[0]]) * map_resolution
        ax.imshow(map_data, cmap='gray_r', extent=extent)
        ax.grid(which='both')
        
        # Plot robot
        robot = Robot(state["pose"], radius)
        robot.plot(ax, color="k")
        
        # Plot path
        for dot in state["path"]:
            ax.plot(dot[0], dot[1], "r.", markersize=4)

        # Plot lidar
        x, y, theta = state['pose']
        x += sensor_offset[0]
        y += sensor_offset[1]
        angles = np.linspace(start_angle, end_angle, len(state['ranges']))
        for ang, r in zip(angles, state['ranges']):
            ax.plot([x, x + r * np.cos(theta+ang)], [y, y + r * np.sin(theta+ang)], "b--")

        # Save plots
        plt.savefig(f"{plots_dir}/plot_{i:04d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def read_map(map_path, map_resolution):
    map2d = Map2D.from_image(map_path, map_resolution)
    return map2d

def main(
    initial_pose,
    map_data_path,
    map_resolution,
    scale_factor,
    tasks_cfgs,
    plots_dir,
    use_roomba=False,
    seed=None
):
    
    # Lectura del mapa
    map2d = read_map(map_data_path, map_resolution)

    # Inicializar el robot
    robot = NoisyDiffDriveRobot(
        initial_pose = initial_pose, 
        radius = RADIUS,
        wheels_radius = WHEELS_RADIUS,
        wheels_distance = WHEELS_DISTANCE,
        alpha = ALPHA,
        seed = seed,
    )

    # Inicializar el lidar
    lidar = Lidar(
        sensor_offset = SENSOR_OFFSET,
        num_scans = MAX_NUM_SCANS / scale_factor,
        start_angle = START_ANGLE,
        end_angle = END_ANGLE,
        max_range = MAX_RANGE,
        threshold = OCCUPATION_THRESHOLD,
        seed = seed,
    )

    # Tareas:
    TASKS = read_tasks(tasks_cfgs)    

    # Estado actual
    state = {
        "robot": robot,
        "sensor": lidar,
        "task_status": "not_started",
        "current_task_id": 0,
        "cycle_start_time": None,
    }
    history = []
    state['start_time'] = time.time()
    while time.time() - state['start_time'] < MAX_DURATION:
        if int(time.time() - state['start_time']) % 1 == 0:
            print(f"Pose: {state['robot'].current_pose}")
            print(f"Simulation time: {int(time.time() - state['start_time'])}s", end="\r")

        # Tomar el tiempo
        state["cycle_start_time"] = time.time()

        # Chequeamos si terminaron todas las tareas
        if state["current_task_id"] == len(TASKS):
            print("Finished all tasks")
            break

        # Chequear el estado de la tarea y ejecutar
        current_task = TASKS[state["current_task_id"]]
        if state["task_status"] == "not_started":
            current_task.start(state)
            if time.time() - state["cycle_start_time"] < SAMPLE_TIME:
                time.sleep(SAMPLE_TIME - (time.time() - state['cycle_start_time']))
            continue
        elif state["task_status"] == "running":
            outputs = current_task.run_cycle(state)
        elif state["task_status"] == "finished":
            current_task.finish(state)
        else:
            raise ValueError("Not a valid task status")

        if use_roomba:
            pass
        else:
            # Aplicar acción diferencial con ruido y guardar la nueva pose
            state['robot'].apply_movement(outputs["linear_velocity"], outputs["angular_velocity"], SAMPLE_TIME)
            state['sensor'].measure(state['robot'].current_pose, map2d)
            save_state_for_visualization(history, state)

        # Sincronizar
        if time.time() - state["cycle_start_time"] < SAMPLE_TIME:
            time.sleep(SAMPLE_TIME - (time.time() - state['cycle_start_time']))
        else:
            raise RuntimeError("Sample time too short")

    if not use_roomba:
        visualize_history(history, map2d.map_array, map2d.map_resolution, RADIUS, SENSOR_OFFSET, START_ANGLE, END_ANGLE, plots_dir)
        



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_pose", type=str, default="0,0,0", help="Initial pose")
    parser.add_argument("--map", type=str, default="data/map.mat", help="Path to map")
    parser.add_argument("--map_resolution", type=float, default=0.05, help="Resolution of the map")
    parser.add_argument("--scans_scale_factor", type=int, default=1, help="Scale factor of the Lidiar scans")
    parser.add_argument("--tasks", type=str, help="List of configs for each task")
    parser.add_argument("--plots_dir", type=str, help="Directory where plots are saved")
    parser.add_argument("--use_roomba", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Seed for simulation")
    args = parser.parse_args()

    initial_pose = np.array(list(map(float,args.initial_pose.split(","))))
    tasks_cfgs = args.tasks.split(",")
    main(initial_pose, args.map, args.map_resolution, args.scans_scale_factor, tasks_cfgs, args.plots_dir, args.use_roomba, args.seed)