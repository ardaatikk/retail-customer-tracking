import json
from pathlib import Path

import cv2
import numpy as np

from src.config.paths import FIGURES_DIR


CAMERA_ID = "Showroom"

IMAGE_PATH = (
    FIGURES_DIR
    / "showroom_zone_reference.jpg"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "zones"
    / f"{CAMERA_ID}.json"
)


class ZoneConfigurator:
    def __init__(
        self,
        image_path: Path,
    ) -> None:
        self.image = cv2.imread(
            str(image_path)
        )

        if self.image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        self.points: list[
            tuple[int, int]
        ] = []

        self.zones: dict[
            str,
            list[list[int]]
        ] = {}

        self.current_zone_name: str | None = None

        self.window_name = (
            f"Zone Configurator - {CAMERA_ID}"
        )

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        if (
            event
            == cv2.EVENT_LBUTTONDOWN
        ):
            self.points.append((x, y))

            print(
                f"Point added: ({x}, {y})"
            )

    def draw(self) -> np.ndarray:
        canvas = self.image.copy()

        # Draw saved zones.
        for zone_name, points in self.zones.items():
            polygon = np.array(
                points,
                dtype=np.int32,
            )

            if len(polygon) >= 3:
                overlay = canvas.copy()

                cv2.fillPoly(
                    overlay,
                    [polygon],
                    (0, 255, 0),
                )

                cv2.addWeighted(
                    overlay,
                    0.20,
                    canvas,
                    0.80,
                    0,
                    canvas,
                )

                cv2.polylines(
                    canvas,
                    [polygon],
                    True,
                    (0, 255, 0),
                    2,
                )

                x, y = polygon[0]

                cv2.putText(
                    canvas,
                    zone_name,
                    (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Draw current polygon.
        for point in self.points:
            cv2.circle(
                canvas,
                point,
                5,
                (0, 0, 255),
                -1,
            )

        if len(self.points) >= 2:
            polygon = np.array(
                self.points,
                dtype=np.int32,
            )

            cv2.polylines(
                canvas,
                [polygon],
                False,
                (0, 0, 255),
                2,
            )

        return canvas

    def save_zone(
        self,
        zone_name: str,
    ) -> None:
        if len(self.points) < 3:
            print(
                "A zone needs at least 3 points."
            )
            return

        self.zones[zone_name] = [
            [x, y]
            for x, y in self.points
        ]

        self.points.clear()

        print(
            f"Zone saved: {zone_name}"
        )

    def undo_point(self) -> None:
        if self.points:
            removed = self.points.pop()

            print(
                f"Removed point: {removed}"
            )

    def save_config(self) -> None:
        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        config = {
            "camera_id": CAMERA_ID,
            "image_width": self.image.shape[1],
            "image_height": self.image.shape[0],
            "zones": self.zones,
        }

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                config,
                file,
                indent=2,
            )

        print()
        print(
            f"Configuration saved: {OUTPUT_PATH}"
        )

    def run(self) -> None:
        cv2.namedWindow(
            self.window_name
        )

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback,
        )

        print()
        print("Zone Configurator")
        print("-----------------")
        print("Left click : Add polygon point")
        print("U          : Undo last point")
        print("N          : Finish current zone")
        print("S          : Save configuration")
        print("Q          : Quit")
        print()

        while True:
            canvas = self.draw()

            cv2.imshow(
                self.window_name,
                canvas,
            )

            key = cv2.waitKey(30) & 0xFF

            if key == ord("u"):
                self.undo_point()

            elif key == ord("n"):
                if len(self.points) < 3:
                    print(
                        "Add at least 3 points first."
                    )
                    continue

                zone_name = input(
                    "Zone name: "
                ).strip()

                if not zone_name:
                    print(
                        "Zone name cannot be empty."
                    )
                    continue

                self.save_zone(
                    zone_name
                )

            elif key == ord("s"):
                self.save_config()

            elif key == ord("q"):
                break

        cv2.destroyAllWindows()


def main() -> None:
    configurator = ZoneConfigurator(
        IMAGE_PATH
    )

    configurator.run()


if __name__ == "__main__":
    main()
    