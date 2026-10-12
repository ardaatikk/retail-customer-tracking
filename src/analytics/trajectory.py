from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from src.config.paths import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    REFERENCE_FRAMES_DIR,
)
from src.zones.zone_manager import ZoneManager, ZONE_CONFIG_DIR


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
    reference_frame_path: Path = None,
) -> np.ndarray:
    reference_frame_path = reference_frame_path or REFERENCE_FRAME_PATH
    image = cv2.imread(str(reference_frame_path))

    if image is None:
        raise FileNotFoundError(
            f"Reference frame not found: "
            f"{reference_frame_path}"
        )

    expected_size = (
        (zone_manager.image_width, zone_manager.image_height)
        if zone_manager is not None else (image.shape[1], image.shape[0])
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


def generate_trajectory(
    tracks: pd.DataFrame = None,
    camera_id: str = None,
    reference_frame_path: Path = None,
    output_path: Path = None,
) -> None:
    if tracks is None:
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

    else:
        tracks = tracks.copy()
    camera_id = camera_id or CAMERA_ID
    output_path = Path(output_path) if output_path is not None else OUTPUT_PATH
    confirmed_track_points = len(
        tracks
    )

    zone_manager = (
        ZoneManager(camera_id) if (ZONE_CONFIG_DIR / f"{camera_id}.json").exists() else None
    )
    background = load_background(zone_manager, reference_frame_path)
    frame_bottom = background.shape[0] - 1

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

    image = background
    if zone_manager is not None:
        image = draw_zones(image=image, zone_manager=zone_manager)

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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Could not save trajectory image: "
            f"{output_path}"
        )

    print("Trajectory analytics")
    print("--------------------")

    print(
        f"Confirmed tracks: "
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
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    generate_trajectory()