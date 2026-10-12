"""Manual offline camera replay; provide your own local recording."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.replay_pipeline import main


if __name__ == '__main__':
    main()
