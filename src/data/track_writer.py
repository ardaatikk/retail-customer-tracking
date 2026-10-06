from pathlib import Path

import pandas as pd

from src.tracking.customer_state import CustomerState
from src.tracking.tracker import Track


class TrackWriter:
    def __init__(
        self,
        output_path: Path,
    ) -> None:
        self.output_path = output_path
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
            }
        )

    def save(
        self,
        customer_states: dict,
    ) -> int:
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