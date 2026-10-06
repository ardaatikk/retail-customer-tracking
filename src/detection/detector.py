from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class PersonDetector:
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []

        if not results:
            return detections

        boxes = results[0].boxes

        if boxes is None:
            return detections

        for box in boxes:
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

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                )
            )

        return detections