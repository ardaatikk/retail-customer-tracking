"""Human-readable recording IDs shared by video and track outputs."""
from datetime import datetime
import re
from zoneinfo import ZoneInfo


def make_session_id(camera_id: str, kind: str, label: str = None) -> str:
    for value in [camera_id, kind] + ([label] if label is not None else []):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
            raise ValueError("Session labels must use letters, digits, hyphens or underscores (1-64 characters)")
    timestamp = datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%Y-%m-%d_%H-%M-%S")
    parts = [camera_id, kind] + ([label] if label else []) + [timestamp]
    return "_".join(parts)
