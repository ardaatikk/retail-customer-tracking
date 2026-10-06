import json
from pathlib import Path

import cv2
import numpy as np


ZONE_CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "zones"
)


class ZoneManager:
    def __init__(
        self,
        camera_id: str,
    ) -> None:
        self.camera_id = camera_id

        config_path = (
            ZONE_CONFIG_DIR
            / f"{camera_id}.json"
        )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Zone configuration not found: "
                f"{config_path}"
            )

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        self.image_width = config["image_width"]
        self.image_height = config["image_height"]

        self.zones: dict[str, np.ndarray] = {}

        for zone_name, points in config["zones"].items():
            self.zones[zone_name] = np.array(
                points,
                dtype=np.int32,
            )

    def get_zone(
        self,
        x: float,
        y: float,
        tolerance_pixels: float = 20.0,
    ) -> str | None:
        point = (
            float(x),
            float(y),
        )

        nearest_zone: str | None = None
        nearest_distance = float("inf")

        for zone_name, polygon in self.zones.items():
            distance = cv2.pointPolygonTest(
                polygon,
                point,
                True,
            )

            # Point is inside or directly on the polygon.
            if distance >= 0:
                return zone_name

            outside_distance = abs(distance)

            if outside_distance < nearest_distance:
                nearest_distance = outside_distance
                nearest_zone = zone_name

        if nearest_distance <= tolerance_pixels:
            return nearest_zone

        return None