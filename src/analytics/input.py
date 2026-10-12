"""Select one camera/session without merging unrelated local person IDs."""
from pathlib import Path
import pandas as pd


def load_selected_tracks(path: Path, camera_id: str = None, capture_id: str = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Track data not found: {path}")
    tracks = pd.read_parquet(path)
    required = {"camera_id", "capture_id", "customer_id", "timestamp", "zone"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"Missing track columns: {', '.join(sorted(missing))}")
    if camera_id is not None:
        tracks = tracks[tracks.camera_id == camera_id]
    if capture_id is not None:
        tracks = tracks[tracks.capture_id == capture_id]
    if tracks.empty:
        raise ValueError("No observations match the selected camera/session")
    scopes = tracks[["camera_id", "capture_id"]].drop_duplicates()
    if len(scopes) != 1 or scopes.isna().any().any():
        raise ValueError("Select exactly one camera/session with --camera-id and --capture-id")
    tracks = tracks.copy()
    # Raw exports retain observation-time IDs; use final IDs to retain confirmed histories.
    if "final_customer_id" in tracks.columns:
        tracks["customer_id"] = tracks["final_customer_id"]
    tracks = tracks[tracks.customer_id.notna()].copy()
    if tracks.empty:
        raise ValueError("No confirmed tracks in the selected camera/session")
    tracks["customer_id"] = tracks.customer_id.astype(int)
    tracks["timestamp"] = pd.to_numeric(tracks.timestamp, errors="raise")
    return tracks
