
from .scripts.localization_particle_filter import setup as particle_filter
from .scripts.localization_kf import setup as kalman
from .scripts.fastslam import setup as fastslam
from .scripts.planning import setup as planning
from .scripts.simulator import setup as simulator

def setup(action, *args):
    if action == "particle_filter":
        particle_filter(args)
    elif action == "kalman":
        kalman(args)
    elif action == "fastslam":
        fastslam(args)
    elif action == "planning":
        planning(args)
    elif action == "simulator":
        simulator(args)
    else:
        raise ValueError("Not a valid action")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    setup(*args)

