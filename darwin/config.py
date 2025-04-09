from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
SCORES_DIR = DATA_DIR / "scores"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---- My Variables -----
FEATURE_NUM = 20
RANDOM_STATE = 42
CRITERION = "entropy"
MAX_DEPTH = 5
CLASS_WEIGHT = "balanced"
TEST_SIZE = 0.2
METRICS = ["accuracy", "precision", "recall", "f1"]
RANDOM_SEEDS = [
    454,
    167,
    332,
    322,
    222,
    464,
    955,
    35,
    691,
    292,
    142,
    42,
    665,
    384,
    462,
    700,
    707,
    841,
    447,
    208,
]
# ---------------------

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
