

class BaseTask:
    
    def __init__(self):
        pass

    def apply(self):
        # Aplicar acción diferencial con ruido y guardar la nueva pose
        robot.apply_diffdrive(v, w, SAMPLE_TIME)
        pose_history.append(robot.current_pose)

        # Tomar una nueva medición de lidiar con ruido
        ranges = lidar.measure(robot.current_pose)
        ranges_history.append(ranges)


class GoToPoint:

    def __init__(self):
        pass

    def run_cycle(self):
        v, w = plan_going_to(POINT_A, pose_history, ranges_history)
        new_task_status = check_arrival(POINT_A, robot.current_pose)
