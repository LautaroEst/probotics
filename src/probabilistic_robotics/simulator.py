
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
import importlib

from .robots import *
from .sensors import *
from .tasks import *
from .mapping import *

class Simulator:

    def __init__(self, map_cfg, robot_cfg, sensor_cfg, tasks_cfgs, scans_scale_factor, use_roomba=False, max_duration=10, sample_time=0.1, xlims=None, ylims=None, seed=None):
        self.fig, self.ax = plt.subplots(1,1, figsize=(5,5))
        plt.ion()
        self.fig.show()

        # Lectura del mapa
        self.map_cfg = self._load_yaml(map_cfg)
        self.map2d = Map2D.from_image(**self.map_cfg)

        # Inicializar el robot
        self.robot_config = self._load_yaml(robot_cfg)
        self.robot = NoisyDiffDriveRobot(seed=seed, **self.robot_config)

        # Inicializar el lidar
        self.sensor_cfg = self._load_yaml(sensor_cfg)
        self.sensor_cfg["num_scans"] = self.sensor_cfg["num_scans"] // scans_scale_factor
        self.lidar = Lidar(seed=seed, **self.sensor_cfg)

        # Tareas:
        self.tasks = self._read_tasks(tasks_cfgs)    

        # Parámetros de simulación
        self.use_roomba = use_roomba
        self.max_duration = max_duration
        self.sample_time = sample_time
        self.xlims = xlims
        self.ylims = ylims
        self.seed = seed

    def run(self):

        # Estado actual
        state = {
            "robot": self.robot,
            "sensor": self.lidar,
            "task_status": "not_started",
            "current_task_id": 0,
            "cycle_start_time": None,
        }
        state['start_time'] = time.time()
        while time.time() - state['start_time'] < self.max_duration:

            # Tomar el tiempo
            state["cycle_start_time"] = time.time()

            # Chequeamos si terminaron todas las tareas
            if state["current_task_id"] == len(self.tasks):
                print("Finished all tasks")
                break

            # Chequear el estado de la tarea y ejecutar
            current_task = self.tasks[state["current_task_id"]]
            if state["task_status"] == "not_started":
                current_task.start(state)
                if time.time() - state["cycle_start_time"] < self.sample_time:
                    time.sleep(self.sample_time - (time.time() - state['cycle_start_time']))
                continue
            elif state["task_status"] == "running":
                outputs = current_task.run_cycle(state)
            elif state["task_status"] == "finished":
                current_task.finish(state)
            else:
                raise ValueError("Not a valid task status")

            if self.use_roomba:
                pass
            else:
                # Aplicar acción diferencial con ruido y guardar la nueva pose
                state['robot'].apply_movement(outputs["linear_velocity"], outputs["angular_velocity"], self.sample_time)
                state['sensor'].measure(state['robot'].current_pose, self.map2d)

            # Plotting
            self._plot_realtime(
                pose=state["robot"].current_pose,
                path=[],
                ranges=state["sensor"].ranges,
            )
            if time.time() - state["cycle_start_time"] < self.sample_time:
                time.sleep(self.sample_time - (time.time() - state['cycle_start_time']))
            else:
                raise RuntimeError("Sample time too short")

    def _load_yaml(self, path):
        with open(path, "r") as f:
            file = yaml.safe_load(f)
        if file is None:
            return {}
        return file

    def _read_tasks(self, tasks_cfgs):
        tasks = []
        module = importlib.import_module('.tasks', package=__package__)
        for cfg in tasks_cfgs:
            cfg = self._load_yaml(cfg + ".yaml")
            task_name = cfg.pop('task')
            task_cls = getattr(module, task_name)
            task = task_cls(**cfg)
            tasks.append(task)
        return tasks

    def _plot_realtime(self, pose, path, ranges):
        
        # Clear axis
        self.ax.cla()

        # Plot map
        extent = np.array([0, self.map2d.map_array.shape[1], 0, self.map2d.map_array.shape[0]]) * self.map2d.map_resolution
        self.ax.imshow(self.map2d.map_array, cmap='gray_r', extent=extent)
        self.ax.grid(which='both')
        
        # Plot robot
        robot = Robot(pose, self.robot.wheels_radius)
        robot.plot(self.ax, color="k")
        
        # Plot path
        for dot in path:
            self.ax.plot(dot[0], dot[1], "r.", markersize=4)

        # Plot lidar
        x, y, theta = pose
        x += self.lidar.sensor_offset[0]
        y += self.lidar.sensor_offset[1]
        angles = np.linspace(self.lidar.start_angle, self.lidar.end_angle, len(ranges))
        for ang, r in zip(angles, ranges):
            self.ax.plot([x, x + r * np.cos(theta+ang)], [y, y + r * np.sin(theta+ang)], "b--")

        # Update plot
        if self.xlims is not None:
            self.ax.set_xlim(*self.xlims)
        if self.ylims is not None:
            self.ax.set_ylim(*self.ylims)
        self.fig.canvas.draw()
