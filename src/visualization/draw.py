import cv2
import numpy as np

from src.detection.detector import Detection
from src.tracking.tracker import Track



def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    output = frame.copy()

    for detection in detections:
        x1 = int(detection.x1)
        y1 = int(detection.y1)
        x2 = int(detection.x2)
        y2 = int(detection.y2)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"person "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output

def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    customer_states: dict,
    timestamp: float,
    zone_manager,
    show_zone: bool = True,
) -> np.ndarray:
    output = frame.copy()

    for track in tracks:
        state = customer_states.get(
            track.local_track_id
        )

        if state is None:
            continue

        # Candidate tracks are intentionally hidden.
        if not state.confirmed:
            continue

        x1 = int(track.x1)
        y1 = int(track.y1)
        x2 = int(track.x2)
        y2 = int(track.y2)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        elapsed_seconds = int(
            state.elapsed_time(timestamp)
        )

        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60

        label = f"Person #{state.customer_id} "
        if show_zone:
            zone = zone_manager.get_zone(track.ground_x, track.ground_y)
            label += f"Zone: {zone if zone is not None else 'Outside'} | "
        label += f"Store Time {minutes:02d}:{seconds:02d}"

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.circle(
            output,
            (
                int(track.ground_x),
                int(track.ground_y),
            ),
            5,
            (0, 255, 0),
            -1,
        )

    return output

def draw_zones(
    frame: np.ndarray,
    zones: dict[str, np.ndarray],
) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()

    for zone_name, polygon in zones.items():
        cv2.fillPoly(
            overlay,
            [polygon],
            (255, 255, 255),
        )

    cv2.addWeighted(
        overlay,
        0.10,
        output,
        0.90,
        0,
        output,
    )

    for zone_name, polygon in zones.items():
        cv2.polylines(
            output,
            [polygon],
            True,
            (255, 255, 255),
            2,
        )

        x, y = polygon[0]

        cv2.putText(
            output,
            zone_name,
            (int(x), max(int(y) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output