import time

import cv2

from src.config.paths import (
    CITIC_MTMC_DATASET_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    VIDEOS_DIR,
)
from src.data.track_writer import TrackWriter
from src.tracking.customer_state import (
    CustomerStateManager,
)
from src.tracking.tracker import PersonTracker
from src.visualization.draw import (
    draw_tracks,
    draw_zones,
)
from src.zones.zone_manager import ZoneManager


CAMERA_ID = "Showroom"
CAPTURE_ID = "capture1"
START_SECOND = 0


def run_tracking_pipeline() -> None:
    video_path = (
        CITIC_MTMC_DATASET_DIR
        / CAPTURE_ID
        / f"{CAMERA_ID}.mp4"
    )

    output_path = (
        VIDEOS_DIR
        / f"{CAPTURE_ID}_{CAMERA_ID}_tracked.mp4"
    )

    track_output_path = (
        PROCESSED_DATA_DIR
        / "tracks.parquet"
    )

    VIDEOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    start_frame = int(
        START_SECOND * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    tracker = PersonTracker()

    zone_manager = ZoneManager(
        CAMERA_ID
    )

    track_writer = TrackWriter(
        track_output_path
    )

    customer_manager = CustomerStateManager(
        min_observation_seconds=0.5,
        min_displacement_pixels=10.0,
        min_movement_frames=5,
        movement_step_pixels=2.0,
    )

    paused = False
    last_frame = None
    frame_id = 0

    print(f"Camera: {CAMERA_ID}")
    print(f"Capture: {CAPTURE_ID}")
    print(f"FPS: {fps:.2f}")
    print(
        f"Duration: "
        f"{total_frames / fps:.1f}s"
    )

    print()
    print("Controls:")
    print("  Space : Pause / Resume")
    print("  Q     : Quit")
    print()

    try:
        while True:
            if not paused:
                frame_start_time = (
                    time.perf_counter()
                )

                success, frame = cap.read()

                if not success:
                    break

                frame_id = (
                    int(
                        cap.get(
                            cv2.CAP_PROP_POS_FRAMES
                        )
                    )
                    - 1
                )

                timestamp = (
                    frame_id / fps
                )

                tracks = tracker.track(
                    frame
                )

                for track in tracks:
                    state = (
                        customer_manager.update(
                            track,
                            timestamp,
                        )
                    )

                    zone = (
                        zone_manager.get_zone(
                            track.ground_x,
                            track.ground_y,
                        )
                    )

                    state.update_zone(
                        zone,
                        timestamp,
                    )

                    track_writer.add(
                        camera_id=CAMERA_ID,
                        capture_id=CAPTURE_ID,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        track=track,
                        state=state,
                        zone=zone,
                    )

                annotated_frame = draw_zones(
                    frame=frame,
                    zones=zone_manager.zones,
                )

                annotated_frame = draw_tracks(
                    frame=annotated_frame,
                    tracks=tracks,
                    customer_states=(
                        customer_manager.states
                    ),
                    timestamp=timestamp,
                    zone_manager=zone_manager,
                )

                cv2.putText(
                    annotated_frame,
                    f"{timestamp:.1f}s",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(
                    annotated_frame
                )

                last_frame = (
                    annotated_frame
                )

            if last_frame is not None:
                cv2.imshow(
                    f"Tracking - {CAMERA_ID}",
                    last_frame,
                )

            if paused:
                wait_ms = 100

            else:
                processing_time = (
                    time.perf_counter()
                    - frame_start_time
                )

                target_frame_time = (
                    1.0 / fps
                )

                remaining_time = max(
                    target_frame_time
                    - processing_time,
                    0.001,
                )

                wait_ms = max(
                    1,
                    int(
                        remaining_time
                        * 1000
                    ),
                )

            key = (
                cv2.waitKey(
                    wait_ms
                )
                & 0xFF
            )

            if key == ord("q"):
                break

            if key == ord(" "):
                paused = not paused

                print(
                    "Paused"
                    if paused
                    else "Resumed"
                )

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    track_record_count = (
        track_writer.save(
            customer_manager.states
        )
    )

    confirmed_states = [
        state
        for state
        in customer_manager.states.values()
        if state.confirmed
    ]

    print()
    print("Processing complete")
    print("-------------------")

    print()
    print("Customer analytics")
    print("------------------")

    for state in confirmed_states:
        dwell_times = (
            state.get_zone_dwell_times(
                state.last_seen
            )
        )

        print()
        print(
            f"Person #{state.customer_id}"
        )

        print(
            f"  Store time: "
            f"{state.observed_time:.2f}s"
        )

        print(
            "  Path: "
            + " -> ".join(
                state.zone_history or []
            )
        )

        print(
            "  Zone dwell times:"
        )

        for (
            zone_name,
            duration,
        ) in dwell_times.items():
            print(
                f"    {zone_name}: "
                f"{duration:.2f}s"
            )

    print()
    print("Summary")
    print("-------")

    print(
        f"Confirmed customers: "
        f"{len(confirmed_states)}"
    )

    print(
        f"Track records: "
        f"{track_record_count}"
    )

    print()
    print("Outputs")
    print("-------")

    print(
        "Track data: "
        f"{track_output_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        "Video: "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    run_tracking_pipeline()