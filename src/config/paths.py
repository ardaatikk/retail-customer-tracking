from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CITIC_MTMC_DIR = RAW_DATA_DIR / "citic_mtmc"

REFERENCE_FRAMES_DIR = (
    PROJECT_ROOT
    / "src"
    / "config"
    / "reference_frames"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
VIDEOS_DIR = ARTIFACTS_DIR / "videos"
CITIC_MTMC_DATASET_DIR = CITIC_MTMC_DIR / "dataset"
SESSIONS_DIR = ARTIFACTS_DIR / "sessions"
