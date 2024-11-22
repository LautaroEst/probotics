import time
import numpy as np
from .robot import Robot, Lidar

# MAX_DURATION = 3 * 60  # 3 minutes
MAX_DURATION = 10  # 10 seconds
SAMPLE_TIME = 0.1  # 100ms
WAITING_TIME = 5  # 5 seconds

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
        "task_status": "irfe",
    }
    task_data = {}
    history = {
        "pose": [],
        "ranges": [],
    }
    start_time = time.time()
    while start_time - time.time() < MAX_DURATION:

        # Tomar el timpo
        state['current_time'] = time.time()

        # Chequear el estado de la tarea
        if state['task_status'] == "finished":
            if task_data['status'] == "success":
                print("Task completed successfully")
            else:
                print("Task failed")
            break
        elif state['task_status'] == "irfe":
            ## TODO: Implementar la lógica de la tarea IRFE
            pass
        elif state['task_status'] == "going_to_A":
            if has_arrived(POINT_A, robot.current_pose):
                state['task_status'] = "waiting_on_A"
                task_data = {
                    "current_time": time.time(),
                    "remaining_time": WAITING_TIME
                }
                v, w = 0., 0.
            else:
                v, w = plan_going_to(POINT_A, robot.current_pose, robot.world)
        elif state['task_status'] == "going_to_B":
            if has_arrived(POINT_B, robot.current_pose):
                state['task_status'] = "finished"
                task_data = {
                    "status": "success"
                }
                v, w = 0., 0.
            else:
                v, w = plan_going_to(POINT_B, robot.current_pose, robot.world)
        elif state['task_status'] == "waiting_on_A":
            if task_data["remaining_time"] <= 0:
                state['task_status'] = "going_to_B"
                task_data = {}
            else:
                v, w = 0., 0.
                task_data["remaining_time"] -= time.time() - task_data["current_time"]
                task_data["current_time"] = time.time()
        else:
            raise ValueError(f"Invalid task status: {state['task_status']}")
        
        # Aplicar acción diferencial con ruido y guardar la nueva pose
        robot.apply_diffdrive(v, w, SAMPLE_TIME)
        history['pose'].append(robot.current_pose)

        # Tomar una nueva medición de lidiar con ruido
        ranges = lidar.measure(robot.current_pose)
        history['range'].append(ranges)

        # Sincronizar
        if time.time() - state['current_time'] < SAMPLE_TIME:
            time.sleep(SAMPLE_TIME - (time.time() - state['current_time']))

        



if __name__ == "__main__":
    from fire import Fire
    Fire(main)