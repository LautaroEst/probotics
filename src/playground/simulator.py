import time
import numpy as np
from .robot import Robot, Lidar

# MAX_DURATION = 3 * 60  # 3 minutes
MAX_DURATION = 10  # 10 seconds
SAMPLE_TIME = 0.1  # 100ms

POINT_A = (3, 1)
POINT_B = (1.1, 2.85)


def main(
    seed=None
):
    # Inicializar el robot
    robot = Robot(
        radius = 0.35 / 2, # Radio del robot [m]
        wheels_radius = 0.072 / 2, # Radio de las ruedas [m]
        wheels_distance = 0.235, # Distancia entre las ruedas [m]
        alpha = 0.0015, # Factor de ruido
        seed = seed,
    )

    # Inicializar el lidar
    scale_factor = 10, # Decimar las lecturas del lidar acelera el algoritmo
    lidar = Lidar(
        sensor_offset = (.09, 0), # Posición del sensor en el robot
        num_scans = 720 / scale_factor, # Número de escaneos por medición
        start_angle = -np.pi, # Ángulo inicial de medición
        end_angle = np.pi, # Ángulo final de medición
        max_range = 8, # Distancia máxima de medición
        seed = seed,
    )

    # Estado actual
    state = {
        "robot": robot,
        "lidar": lidar,
        "t": time.time(),
        "task_status": "irfe",
        "remaining_waiting_time": WAITING_TIME,
    }
    history = {
        "pose": [],
        "ranges": [],
    }
    while t < MAX_DURATION:

        # Chequear el estado de la tarea
        if task_status == "finished":
            print("Task completed")
            break
        elif task_status == "going_to_A":
            v, w = plan_going_to(POINT_A, pose_history, ranges_history)
            new_task_status = check_arrival(POINT_A, robot.current_pose)
        elif task_status == "going_to_B":
            v, w = plan_going_to(POINT_B, pose_history, ranges_history)
            new_task_status = check_arrival(POINT_B, robot.current_pose)
        elif task_status == "waiting_on_A":
            if remaining_time <= 0:
                new_task_status = "going_to_B"
                remaining_time = WAITING_TIME
            else:
                v, w = 0., 0.
                remaining_time -= SAMPLE_TIME
        elif task_status == "waiting_on_B":
            if remaining_time <= 0:
                new_task_status = "finished"
                remaining_time = WAITING_TIME
            else:
                v, w = 0., 0.
                remaining_time -= SAMPLE_TIME
        else:
            raise ValueError(f"Invalid task status: {task_status}")
        
        

        # Incrementar el tiempo
        t += SAMPLE_TIME

        # Actualizar el valor de la tarea
        task_status = new_task_status

        



if __name__ == "__main__":
    from fire import Fire
    Fire(setup)