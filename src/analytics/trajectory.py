import cv2
import numpy as np
import pandas as pd

from src.config.paths import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
)
from src.zones.zone_manager import ZoneManager


CAMERA_ID = "Showroom"
CAPTURE_ID = "capture1"

TRACKS_PATH = (
    PROCESSED_DATA_DIR
    / "tracks.parquet"
)

OUTPUT_PATH = (
    FIGURES_DIR
    / f"{CAPTURE_ID}_{CAMERA_ID}_trajectory.png"
)


def draw_zone_background(
    image: np.ndarray,
    zone_manager: ZoneManager,
) -> np.ndarray:
    overlay = image.copy()

    for zone_name, polygon in zone_manager.zones.items():
        cv2.fillPoly(
            overlay,
            [polygon],
            (255, 255, 255),
        )

        cv2.polylines(
            image,
            [polygon],
            True,
            (180, 180, 180),
            2,
        )

        x, y = polygon[0]

        cv2.putText(
            image,
            zone_name,
            (int(x), max(int(y) - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(
        overlay,
        0.08,
        image,
        0.92,
        0,
        image,
    )

    return image


def draw_customer_trajectory(
    image: np.ndarray,
    customer_data: pd.DataFrame,
    customer_id: int,
) -> np.ndarray:
    customer_data = customer_data.sort_values(
        "timestamp"
    )

    points = customer_data[
        ["ground_x", "ground_y"]
    ].to_numpy(dtype=np.int32)

    if len(points) == 0:
        return image

    if len(points) >= 2:
        cv2.polylines(
            image,
            [points],
            False,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    start_point = tuple(
        points[0]
    )

    end_point = tuple(
        points[-1]
    )

    cv2.circle(
        image,
        start_point,
        8,
        (255, 255, 255),
        -1,
    )

    cv2.circle(
        image,
        end_point,
        8,
        (100, 100, 100),
        -1,
    )

    cv2.putText(
        image,
        f"Person #{customer_id} START",
        (
            start_point[0] + 10,
            start_point[1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"Person #{customer_id} END",
        (
            end_point[0] + 10,
            end_point[1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        2,
        cv2.LINE_AA,
    )

    return image


def generate_trajectory() -> None:
    if not TRACKS_PATH.exists():
        raise FileNotFoundError(
            f"Track data not found: {TRACKS_PATH}"
        )

    tracks = pd.read_parquet(
        TRACKS_PATH
    )

    tracks = tracks[
        (tracks["camera_id"] == CAMERA_ID)
        & (tracks["capture_id"] == CAPTURE_ID)
        & (tracks["customer_id"].notna())
    ].copy()

    if tracks.empty:
        raise RuntimeError(
            "No confirmed customer tracks found."
        )

    zone_manager = ZoneManager(
        CAMERA_ID
    )

    image = np.zeros(
        (
            zone_manager.image_height,
            zone_manager.image_width,
            3,
        ),
        dtype=np.uint8,
    )

    image = draw_zone_background(
        image,
        zone_manager,
    )

    customer_ids = sorted(
        tracks["customer_id"]
        .astype(int)
        .unique()
    )

    for customer_id in customer_ids:
        customer_data = tracks[
            tracks["customer_id"]
            == customer_id
        ]

        image = draw_customer_trajectory(
            image=image,
            customer_data=customer_data,
            customer_id=customer_id,
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Could not save trajectory image: "
            f"{OUTPUT_PATH}"
        )

    print("Trajectory analytics")
    print("--------------------")
    print(
        f"Customers: {len(customer_ids)}"
    )
    print(
        f"Track points: {len(tracks)}"
    )
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    generate_trajectory()