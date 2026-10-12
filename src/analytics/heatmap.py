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
    / f"{CAPTURE_ID}_{CAMERA_ID}_heatmap.png"
)


HEATMAP_SIGMA = 25
HEATMAP_OPACITY = 0.55

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
        0.60,
        dark_overlay,
        0.40,
        0,
    )

    return image


def create_heatmap(
    tracks: pd.DataFrame,
    width: int,
    height: int,
) -> np.ndarray:
    heatmap = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    for row in tracks.itertuples():
        x = int(round(row.ground_x))
        y = int(round(row.ground_y))

        if (
            0 <= x < width
            and 0 <= y < height
        ):
            heatmap[y, x] += 1.0

    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=HEATMAP_SIGMA,
        sigmaY=HEATMAP_SIGMA,
    )

    maximum = heatmap.max()

    if maximum > 0:
        heatmap = (
            heatmap
            / maximum
            * 255
        )

    return heatmap.astype(
        np.uint8
    )


def overlay_heatmap(
    background: np.ndarray,
    raw_heatmap: np.ndarray,
) -> np.ndarray:
    colored_heatmap = cv2.applyColorMap(
        raw_heatmap,
        cv2.COLORMAP_JET,
    )

    output = background.copy()

    active_mask = (
        raw_heatmap > 10
    )

    blended = cv2.addWeighted(
        background,
        1.0 - HEATMAP_OPACITY,
        colored_heatmap,
        HEATMAP_OPACITY,
        0,
    )

    output[
        active_mask
    ] = blended[
        active_mask
    ]

    return output


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


def generate_heatmap(
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

    raw_heatmap = create_heatmap(
        tracks=tracks,
        width=background.shape[1],
        height=background.shape[0],
    )

    output = overlay_heatmap(
        background=background,
        raw_heatmap=raw_heatmap,
    )

    if zone_manager is not None:
        output = draw_zones(image=output, zone_manager=zone_manager)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(
        str(output_path),
        output,
    )

    if not success:
        raise RuntimeError(
            f"Could not save heatmap: "
            f"{output_path}"
        )

    customer_count = (
        tracks["customer_id"]
        .nunique()
    )

    print("Customer heatmap")
    print("----------------")

    print(
        f"Confirmed tracks: "
        f"{customer_count}"
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
    generate_heatmap()