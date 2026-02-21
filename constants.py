from pathlib import Path

PATH_OF_PROJECT = Path(__file__).parent.absolute()
PATH_OF_OUTPUT = PATH_OF_PROJECT / "output"
PATH_OF_MEMORY = PATH_OF_OUTPUT / "memory"
PATH_OF_FIGURES = PATH_OF_OUTPUT / "figures"
PATH_OF_OPTIMIZATION_RESULTS = PATH_OF_OUTPUT / "optimization_results"
PATH_OF_OPTIMIZATION_MODELS = PATH_OF_OUTPUT / "optimization_models"
PATH_OF_OPTIMIZATION_LOGS = PATH_OF_OUTPUT / "optimization_logs"

# Create Path
for path in [PATH_OF_MEMORY, PATH_OF_OUTPUT, PATH_OF_FIGURES, PATH_OF_OPTIMIZATION_RESULTS, PATH_OF_OPTIMIZATION_MODELS, PATH_OF_OPTIMIZATION_LOGS]:
    path.mkdir(parents=True, exist_ok=True)

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
