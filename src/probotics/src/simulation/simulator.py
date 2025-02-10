
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt

from ..robots import *
from ..sensors import *
from ..tasks import *
from ..mapping import *
from ..utils import read_tasks

# import mpld3

class Simulator:

    def __init__(
        self, 
        seed=None,
        max_duration=10, 
        sample_time=0.1,
        scans_scale_factor=1, 
        sensor_offset=(0., 0.),
        num_scans=200,
        start_angle=-np.pi/2,
        end_angle=np.pi/2,
        max_range=10,
        min_range=0.0,
        occupation_threshold=0.5,
        initial_pose=(0, 0, 0),
        radius=0.5,
        wheels_distance=0.2,
        wheels_radius=0.05,
        alpha=0.1,
        map_path="",
        map_resolution=0.05,
        tasks=[],
    ):

        # Inicializamos los gráficos
        self._init_plot()

        # Lectura del mapa
        self.map2d = Map2D.from_image(map_path, map_resolution)

        # Inicializar el robot
        initial_pose = np.asarray(initial_pose)
        self.robot = NoisyDiffDriveRobot(initial_pose, radius, wheels_radius, wheels_distance, alpha, seed)

        # Inicializar el lidar
        num_scans = num_scans // scans_scale_factor
        self.lidar = Lidar(sensor_offset, num_scans, start_angle, end_angle, min_range, max_range, occupation_threshold, seed)

        # Tareas:
        self.tasks = read_tasks(tasks)    

        # Parámetros de simulación
        self.max_duration = max_duration
        self.sample_time = sample_time
        self.seed = seed

    def run(self, xlim=None, ylim=None):
        try:
            self._run(xlim=xlim, ylim=ylim)
        except KeyboardInterrupt:
            robot_pose = self.robot.current_pose
            lidar_pose = self.lidar.current_pose
            path = []
            ranges = self.lidar.ranges
            self._plot_realtime(robot_pose, lidar_pose, self.last_outputs["particles_poses"], self.last_outputs["target_angle"], path, ranges, xlim, ylim)

    def _run(self, xlim=None, ylim=None):

        # Estado actual
        state = {
            "robot": self.robot,
            "sensor": self.lidar,
            "map": self.map2d,
            "task_status": "not_started",
            "current_task_id": 0,
            "cycle_start_time": None,
            "sample_time": self.sample_time,
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
                # Sincronizar con el tiempo de muestreo
                n = 1
                while (time.time() - state["cycle_start_time"]) / (self.sample_time * n) > 1:
                    n += 1
                time.sleep(self.sample_time * n - (time.time() - state['cycle_start_time']))
                continue
            elif state["task_status"] == "running":
                outputs = current_task.run_cycle(state)
            elif state["task_status"] == "finished":
                current_task.finish(state)
            else:
                raise ValueError("Not a valid task status")
            self.last_outputs = outputs

            # Aplicar acción diferencial con ruido y guardar la nueva pose
            state['robot'].apply_movement(outputs["linear_velocity"], outputs["angular_velocity"], self.sample_time)
            state['sensor'].measure(state['robot'].current_pose, self.map2d)

            # Graficar
            self._plot_realtime(
                robot_pose=state["robot"].current_pose,
                lidar_pose=state["sensor"].current_pose,
                particles_poses=outputs["particles_poses"],
                target_angle=outputs["target_angle"],
                path=[],
                ranges=state["sensor"].ranges,
                xlim=xlim,
                ylim=ylim,
            )

            # Sincronizar con el tiempo de muestreo
            n = 1
            while (time.time() - state["cycle_start_time"]) / (self.sample_time * n) > 1:
                n += 1
            time.sleep(self.sample_time * n - (time.time() - state['cycle_start_time']))

    def _init_plot(self):
        self.fig, self.ax = plt.subplots(1,1, figsize=(5,5))
        plt.ion()
        self.fig.show()

    def _plot_realtime(self, robot_pose, lidar_pose, particles_poses, target_angle, path, ranges, xlim=None, ylim=None):
        
        # Clear axis
        self.ax.cla()

        # Plot map
        extent = np.array([0, self.map2d.map_array.shape[1], 0, self.map2d.map_array.shape[0]]) * self.map2d.map_resolution
        self.ax.imshow(self.map2d.map_array, cmap='gray_r', extent=extent)
        self.ax.grid(which='both')
        
        # Plot robot
        robot = Robot(robot_pose, radius=self.robot.radius)
        robot.plot(self.ax, color="k")
        
        # Plot path
        for dot in path:
            self.ax.plot(dot[0], dot[1], "r.", markersize=4)
        
        # Plot particles
        self.ax.plot(particles_poses[:, 0], particles_poses[:, 1], "g.", markersize=4)

        # Plot target angle
        self.ax.text(0.5, 0.95, f"Target angle: {(target_angle / np.pi * 180):.2f}", transform=self.ax.transAxes, ha='center', va='center')

        # Plot lidar
        x, y, theta = lidar_pose
        angles = np.linspace(self.lidar.start_angle, self.lidar.end_angle, len(ranges))
        for ang, r in zip(angles, ranges):
            self.ax.plot([x, x + r * np.cos(theta+ang)], [y, y + r * np.sin(theta+ang)], "b--")
        
        # Set limits
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

        # Update plot
        self.fig.canvas.draw()
