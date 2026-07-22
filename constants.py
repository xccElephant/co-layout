from pathlib import Path

PATH_OF_PROJECT = Path(__file__).parent.absolute()
PATH_OF_OUTPUT = PATH_OF_PROJECT / "output"
# Every generation run gets its own directory here, named after its session_id:
#   output/sessions/<session_id>/
#     memory.json, debug_info.md, user_info.md   (agent workflow state/logs)
#     optimization/                               (Gurobi logs/.lp models/result.json -- raw solver artifacts)
#     visualization/                               (Coarse/Fine 2D png+svg, 3D layout JSON, render, .blend -- everything meant to be looked at)
# This keeps every artifact produced by one run together instead of scattering
# it across type-based top-level folders (figures/, optimization_logs/, ...).
PATH_OF_SESSIONS = PATH_OF_OUTPUT / "sessions"

PATH_OF_OUTPUT.mkdir(parents=True, exist_ok=True)
PATH_OF_SESSIONS.mkdir(parents=True, exist_ok=True)


def get_session_dir(session_id: str) -> Path:
    """output/sessions/<session_id>/ -- root directory for one generation run."""
    path = PATH_OF_SESSIONS / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_optimization_dir(session_id: str) -> Path:
    """output/sessions/<session_id>/optimization/ -- raw Gurobi logs/.lp models/result.json."""
    path = get_session_dir(session_id) / "optimization"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_visualization_dir(session_id: str) -> Path:
    """output/sessions/<session_id>/visualization/ -- everything meant to be looked at:
    Coarse/Fine 2D floorplan png+svg, the 3D layout JSON, render image, and .blend."""
    path = get_session_dir(session_id) / "visualization"
    path.mkdir(parents=True, exist_ok=True)
    return path

# Direction
EAST = 0
WEST = 1
SOUTH = 2
NORTH = 3

# Door Size (m)
DOOR_SIZE = 1

# Window Size (m)
WINDOW_SIZE = 1

# Coarse-Grid cell size (m)
C = 2
# Fine-Grid cell size (m)
F = 1
N = C//F

# Optimization
CPU_THREADS = 10

COARSE_MODEL_WEIGHTS = {
    "area": 0.8,
    "perimeter": 0.1,
    "rectangularity": 1.0,
    "shape": 2.0,
    "privacy": 0.3,
}

FINE_MODEL_WEIGHTS = {
    "rectangularity": 1.0,
    "perimeter": 0.1,
    "area": 0.8,
    "distance": 0.6,
    "balance": 1.0,
    "reference": 0.3,
    "shape": 2.0,
    "privacy": 0.3,
}
