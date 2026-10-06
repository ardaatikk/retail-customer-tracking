import cv2
import numpy as np
import pandas as pd

from src.config.paths import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    REFERENCE_FRAMES_DIR,
)
from src.zones.zone_manager import ZoneManager


CAMERA_ID = "Showroom"
CAPTURE_ID = "capture1"

TRACKS_PATH = (
    PROCESSED_DATA_DIR
    / "tracks.parquet"
)

REFERENCE_FRAME_PATH = (
    REFERENCE_FRAMES_DIR
    / f"{CAMERA_ID}.jpg"
)

OUTPUT_PATH = (
    FIGURES_DIR
    / f"{CAPTURE_ID}_{CAMERA_ID}_trajectory.png"
)


TRAJECTORY_COLOR = (255, 255, 0)
END_COLOR = (0, 0, 255)
ZONE_COLOR = (255, 255, 255)


def load_background(
    zone_manager: ZoneManager,
) -> np.ndarray:
    image = cv2.imread(
        str(REFERENCE_FRAME_PATH)
    )

    if image is None:
        raise FileNotFoundError(
            f"Reference frame not found: "
            f"{REFERENCE_FRAME_PATH}"
        )

    expected_size = (
        zone_manager.image_width,
        zone_manager.image_height,
    )

    actual_size = (
        image.shape[1],
        image.shape[0],
    )

    if actual_size != expected_size:
        raise ValueError(
            "Reference frame dimensions do not "
            "match zone configuration."
        )

    dark_overlay = np.zeros_like(
        image
    )

    image = cv2.addWeighted(
        image,
        0.55,
        dark_overlay,
        0.45,
        0,
    )

    return image


def draw_zones(
    image: np.ndarray,
    zone_manager: ZoneManager,
) -> np.ndarray:
    for zone_name, polygon in zone_manager.zones.items():
        cv2.polylines(
            image,
            [polygon],
            True,
            ZONE_COLOR,
            2,
            cv2.LINE_AA,
        )

        x, y = polygon[0]

        cv2.putText(
            image,
            zone_name,
            (
                int(x),
                max(int(y) - 10, 25),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ZONE_COLOR,
            2,
            cv2.LINE_AA,
        )

    return image


def smooth_points(
    points: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    if len(points) < kernel_size:
        return points

    smoothed_x = (
        pd.Series(points[:, 0])
        .rolling(
            window=kernel_size,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    smoothed_y = (
        pd.Series(points[:, 1])
        .rolling(
            window=kernel_size,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    return np.column_stack(
        (smoothed_x, smoothed_y)
    ).astype(np.int32)


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
    ].to_numpy(
        dtype=np.int32
    )

    if len(points) == 0:
        return image

    points = smooth_points(
        points
    )

    start_point = tuple(
        points[0]
    )

    end_point = tuple(
        points[-1]
    )

    if len(points) >= 2:
        cv2.polylines(
            image,
            [points],
            False,
            TRAJECTORY_COLOR,
            3,
            cv2.LINE_AA,
        )

    cv2.circle(
        image,
        start_point,
        7,
        TRAJECTORY_COLOR,
        -1,
    )

    cv2.circle(
        image,
        end_point,
        7,
        END_COLOR,
        -1,
    )
    return image


def generate_trajectory() -> None:
    if not TRACKS_PATH.exists():
        raise FileNotFoundError(
            f"Track data not found: "
            f"{TRACKS_PATH}"
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

    confirmed_track_points = len(
        tracks
    )

    zone_manager = ZoneManager(
        CAMERA_ID
    )

    frame_bottom = (
        zone_manager.image_height - 1
    )

    tracks = tracks[
        tracks["ground_y"] < frame_bottom
    ].copy()

    if tracks.empty:
        raise RuntimeError(
            "No valid confirmed customer "
            "track points found."
        )

    spatial_points = len(
        tracks
    )

    excluded_points = (
        confirmed_track_points
        - spatial_points
    )

    image = load_background(
        zone_manager
    )

    image = draw_zones(
        image=image,
        zone_manager=zone_manager,
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
        f"Customers: "
        f"{len(customer_ids)}"
    )

    print(
        f"Confirmed track points: "
        f"{confirmed_track_points}"
    )

    print(
        f"Spatial points used: "
        f"{spatial_points}"
    )

    print(
        f"Excluded truncated points: "
        f"{excluded_points}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    generate_trajectory()