
from .scripts.localization_particle_filter import main as particle_filter
from .scripts.localization_kf import main as kalman
from .scripts.fastslam import main as fastslam
from .scripts.planning import main as planning
from .scripts.simulator import main as simulator

def main(action, *args):
    if action == "particle_filter":
        particle_filter(*args)
    elif action == "kalman":
        kalman(*args)
    elif action == "fastslam":
        fastslam(*args)
    elif action == "planning":
        planning(*args)
    elif action == "simulator":
        simulator(*args)
    else:
        raise ValueError("Not a valid action")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(*args)

