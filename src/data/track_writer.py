from pathlib import Path

import pandas as pd

from src.tracking.customer_state import CustomerState
from src.tracking.tracker import Track


class TrackWriter:
    def __init__(
        self,
        output_path: Path,
        *,
        confirmed_output_path: Path | None = None,
        confirmed_frames_only: bool = False,
    ) -> None:
        self.output_path = output_path
        self.confirmed_output_path = confirmed_output_path
        self.confirmed_frames_only = confirmed_frames_only
        if confirmed_output_path is not None and output_path.resolve() == confirmed_output_path.resolve():
            raise ValueError("Raw and confirmed paths must differ")
        self.confirmed_record_count = 0
        self.records: list[dict] = []

    def add(
        self,
        *,
        camera_id: str,
        capture_id: str,
        frame_id: int,
        timestamp: float,
        track: Track,
        state: CustomerState,
        zone: str | None,
    ) -> None:
        self.records.append(
            {
                "camera_id": camera_id,
                "capture_id": capture_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "local_track_id": (
                    track.local_track_id
                ),
                "customer_id": (
                    state.customer_id
                ),
                "x1": track.x1,
                "y1": track.y1,
                "x2": track.x2,
                "y2": track.y2,
                "ground_x": track.ground_x,
                "ground_y": track.ground_y,
                "zone": zone,
                "confidence": track.confidence,
                **({"confirmed_at_frame": state.confirmed} if self.confirmed_output_path is not None else {}),
            }
        )

    def save(
        self,
        customer_states: dict,
    ) -> int:
        if self.confirmed_output_path is not None:
            return self._save_split(customer_states)
        if not self.records:
            return 0

        customer_id_map = {
            local_track_id: state.customer_id
            for local_track_id, state
            in customer_states.items()
            if state.confirmed
            and state.customer_id is not None
        }

        dataframe = pd.DataFrame(
            self.records
        )

        dataframe["customer_id"] = (
            dataframe["local_track_id"]
            .map(customer_id_map)
            .astype("Int64")
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_parquet(
            self.output_path,
            index=False,
        )

        return len(dataframe)

    def _save_split(self, customer_states: dict) -> int:
        """Raw snapshots plus full histories of tracks confirmed by session end."""
        columns = ["camera_id", "capture_id", "frame_id", "timestamp", "local_track_id",
                   "customer_id", "x1", "y1", "x2", "y2", "ground_x", "ground_y",
                   "zone", "confidence", "confirmed_at_frame"]
        dataframe = pd.DataFrame(self.records, columns=columns)
        confirmed = {
            track_id: state for track_id, state in customer_states.items()
            if state.confirmed and state.customer_id is not None
        }
        dataframe["customer_id"] = dataframe["customer_id"].astype("Int64")
        dataframe["confirmed_at_frame"] = dataframe["confirmed_at_frame"].astype(bool)
        dataframe["final_customer_id"] = dataframe["local_track_id"].map(
            {track_id: state.customer_id for track_id, state in confirmed.items()}
        ).astype("Int64")
        dataframe["confirmed_at"] = dataframe["local_track_id"].map(
            {track_id: state.confirmed_at for track_id, state in confirmed.items()}
        ).astype("float64")
        dataframe["first_seen"] = dataframe["local_track_id"].map(
            {track_id: state.first_seen for track_id, state in customer_states.items()}
        ).astype("float64")
        mask = dataframe["final_customer_id"].notna()
        if self.confirmed_frames_only:
            mask &= dataframe["confirmed_at_frame"]
        confirmed_data = dataframe[mask].copy()
        # Full histories remain in the debug table; strict exports mirror green-box frames.
        confirmed_data["customer_id"] = confirmed_data["final_customer_id"]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.confirmed_output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(self.output_path, index=False)
        confirmed_data.to_parquet(self.confirmed_output_path, index=False)
        self.confirmed_record_count = len(confirmed_data)
        return len(dataframe)
