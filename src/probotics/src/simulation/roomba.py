
from copy import deepcopy
import os
import time
from matplotlib import pyplot as plt
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String

import numpy as np
from scipy.spatial.transform import Rotation

from ..utils import read_tasks
from ..robots import Robot
from ..sensors import Lidar


class Roomba:

    def __init__(
        self, 
        ros_ip, 
        ros_master_uri, 
        tasks, 
        lidar_offset, 
        radius,
        sensor_offset=(0., 0.),
        wheels_distance=0.2,
        wheels_radius=0.05,
        max_duration=60, 
        sample_time=0.1
    ):

        # Parámetros del robot y del lidar
        self.lidar_offset = lidar_offset
        self.radius = radius
        self.sensor_offset = sensor_offset
        self.wheels_distance = wheels_distance
        self.wheels_radius = wheels_radius
        
        # Inicializamos los gráficos
        self._init_plot()

        # Configuración de ROS
        os.environ['ROS_IP'] = ros_ip
        os.environ['ROS_MASTER_URI'] = ros_master_uri

        # Duración de la simulación y tiempo de muestreo
        self.max_duration = max_duration
        self.sample_time = sample_time

        # Read tasks
        self.tasks = read_tasks(tasks)

        # Init ROS
        rospy.init_node("NodeHost")
        self.cmdPub = rospy.Publisher('/auto_cmd_vel', Twist, queue_size=2)
        self.lidarSub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.odomSub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.last_scan = {
            "ranges": np.zeros(360),
            "angle_min": 0,
            "angle_max": 0,
            "range_min": 0,
            "range_max": 0,
        }
        self.last_odom = {
            "pose": np.zeros(3),
        }

    def scan_callback(self, scan_data):
        ranges = np.asarray(scan_data.ranges)
        ranges[ranges == 0] = np.nan
        self.last_scan = {
            "ranges": ranges,
            "angle_min": scan_data.angle_min,
            "angle_max": scan_data.angle_max,
            "range_min": scan_data.range_min,
            "range_max": scan_data.range_max,
        }

    def odom_callback(self, odom_data):
        quat_odom = np.array([
            odom_data.pose.pose.orientation.x, 
            odom_data.pose.pose.orientation.y, 
            odom_data.pose.pose.orientation.z,
            odom_data.pose.pose.orientation.w,
        ])
        eul = Rotation.from_quat(quat_odom, scalar_first=False).as_euler('zxy')
        pose = np.array([
            odom_data.pose.pose.position.x,
            odom_data.pose.pose.position.y,
            eul[0],
        ])
        self.last_odom = {
            "pose": pose,
        }

    def _publish_message(self, outputs):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = outputs["linear_velocity"], 0, 0
        msg.angular.x, msg.angular.y, msg.angular.z = 0, 0, outputs["angular_velocity"]
        self.cmdPub.publish(msg)

    def run(self, xlim=None, ylim=None):
        try:
            self._run(xlim=xlim, ylim=ylim)
        except (rospy.ROSInterruptException, KeyboardInterrupt):
            pass

    @property
    def robot(self):
        robot = Robot(self.last_odom["pose"], radius=self.radius)
        robot.wheels_distance = self.wheels_distance
        robot.wheels_radius = self.wheels_radius
        return robot
    
    @property
    def lidar(self):
        last_scan = deepcopy(self.last_scan)
        lidar = Lidar(self.lidar_offset, len(last_scan["ranges"]), last_scan["angle_min"], last_scan["angle_max"], last_scan["range_min"], last_scan["range_max"])
        lidar.update_lidar_pose(self.robot.current_pose)
        return lidar
    
    def _update_state(self, state):
        state["robot"] = Robot(self.last_odom["pose"], radius=self.radius)
        state["pose_history"].append(self.robot.current_pose)
        last_scan = deepcopy(self.last_scan)
        lidar = Lidar(self.lidar_offset, len(last_scan["ranges"]), last_scan["angle_min"], last_scan["angle_max"], last_scan["range_min"], last_scan["range_max"])
        lidar.update_lidar_pose(self.robot.current_pose)
        lidar.ranges = last_scan["ranges"]
        state["sensor"] = lidar


    def _run(self, xlim=None, ylim=None):

        # Estado actual
        state = {
            "robot": self.robot,
            "pose_history": [],
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

            # Publicar mensaje y graficar
            self._publish_message(outputs)
            self._update_state(state)
            self._plot_realtime(state["robot"], state["pose_history"], state["sensor"], xlim=xlim, ylim=ylim)

            # Sincronizar con el tiempo de muestreo
            n = 1
            while (time.time() - state["cycle_start_time"]) / (self.sample_time * n) > 1:
                n += 1
            time.sleep(self.sample_time * n - (time.time() - state['cycle_start_time']))

    def _init_plot(self):
        self.fig, self.ax = plt.subplots(1,1, figsize=(5,5))
        plt.ion()
        self.fig.show()
        self.pose_history = []

    def _plot_realtime(self, robot, pose_history, lidar, xlim=None, ylim=None):
        
        # Clear axis
        self.ax.cla()

        # Plot robot
        robot.plot(self.ax, color="k")
        
        # Plot lidar
        x, y, theta = lidar.current_pose
        angles = np.linspace(lidar.start_angle, lidar.end_angle, len(lidar.ranges))
        for ang, r in zip(angles, lidar.ranges):
            self.ax.plot([x, x + r * np.cos(theta+ang)], [y, y + r * np.sin(theta+ang)], "b--")

        # Update plot
        self.ax.grid(which='both')
    
        # Set limits
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

        self.fig.canvas.draw()

