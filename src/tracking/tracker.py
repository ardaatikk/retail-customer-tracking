from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Track:
    local_track_id: int

    x1: float
    y1: float
    x2: float
    y2: float

    confidence: float
    class_id: int

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def ground_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def ground_y(self) -> float:
        return self.y2


class PersonTracker:
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.40,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config

        self.model = YOLO(self.model_name)

    def track(
        self,
        frame: np.ndarray,
    ) -> list[Track]:
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            tracker=self.tracker_config,
            verbose=False,
        )

        tracks: list[Track] = []

        if not results:
            return tracks

        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return tracks

        for box in boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )

            local_track_id = int(
                box.id[0]
                .detach()
                .cpu()
                .item()
            )

            tracks.append(
                Track(
                    local_track_id=local_track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                )
            )

        return tracks
    
    def reset(self) -> None:
        self.model = YOLO(self.model_name)