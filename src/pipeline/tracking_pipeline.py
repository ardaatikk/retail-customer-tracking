import argparse
from pathlib import Path
from datetime import datetime, timezone
import re
from uuid import uuid4
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
from src.video.source import VideoInput, VideoSource
from src.video.recorder import StreamRecorder


CAMERA_ID = "Showroom"
CAPTURE_ID = "capture1"
START_SECOND = 0
RECOVERY_TRACKER_CONFIG = Path(__file__).resolve().parents[1] / "config" / "trackers" / "bytetrack_recovery.yaml"


def run_tracking_pipeline(
    source: VideoInput = None,
    *,
    camera_id: str = None,
    record_stream: bool = False,
    model_name: str = "yolo11n.pt",
    tracker_profile: str = "auto",
) -> None:
    if tracker_profile not in ("auto", "baseline", "recovery"):
        raise ValueError("tracker_profile must be auto, baseline or recovery")
    if record_stream and not str(source).lower().startswith(("rtsp://", "rtsps://")):
        raise ValueError("--record-stream requires an RTSP source")
    if camera_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", camera_id):
        raise ValueError("camera_id must contain letters, digits, underscores or hyphens")
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

    confirmed_track_output_path = None
    video = VideoSource(video_path if source is None else source)
    writer = None
    recorder = None
    recorder_warning = False
    fps = video.info.fps
    total_frames = 0
    start_frame = int(START_SECOND * fps)
    live_start_time = time.monotonic()

    try:
        recording_camera_id = camera_id or ("live1" if video.is_live else CAMERA_ID)
        recording_capture_id = CAPTURE_ID
        if video.is_live:
            recording_capture_id = (
                datetime.now(timezone.utc).strftime("live_%Y%m%dT%H%M%SZ")
                + "_" + uuid4().hex[:8]
            )
            output_path = VIDEOS_DIR / recording_camera_id / f"{recording_capture_id}_tracked.mp4"
            track_output_path = (
                PROCESSED_DATA_DIR / recording_camera_id / recording_capture_id / "tracks_raw.parquet"
            )
            confirmed_track_output_path = track_output_path.with_name("tracks_confirmed.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video.is_live:
            total_frames = int(video.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (video.info.width, video.info.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")

        active_profile = tracker_profile
        if active_profile == "auto":
            active_profile = "recovery" if video.is_live else "baseline"
        if active_profile == "recovery":
            tracker = PersonTracker(
                model_name=model_name,
                confidence_threshold=0.1,
                tracker_config=str(RECOVERY_TRACKER_CONFIG),
            )
        else:
            tracker = PersonTracker(model_name=model_name)
        if record_stream:
            raw_output_path = output_path.with_name(f"{recording_capture_id}_raw.mkv")
            recorder = StreamRecorder(str(source), raw_output_path)
            print(f"Raw recording requested: {raw_output_path}")

        # Live camera zones need their own calibration; never reuse Showroom polygons.
        zone_manager = None if video.is_live else ZoneManager(recording_camera_id)

        track_writer = TrackWriter(
            track_output_path,
            **({"confirmed_output_path": confirmed_track_output_path}
               if confirmed_track_output_path is not None else {}),
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

        print(f"Camera: {recording_camera_id}")
        print(f"Capture: {recording_capture_id}")
        print(f"FPS: {fps:.2f}")
        print(f"Model: {model_name}")
        print(f"Tracker profile: {active_profile}")
        if video.is_live:
            print("Live source")
        else:
            print(f"Duration: {total_frames / fps:.1f}s")

        print()
        print("Controls:")
        print("  Space : Pause / Resume")
        print("  Q     : Quit")
        print()

        while True:
            if recorder is not None and not recorder.running and not recorder_warning:
                print("WARNING: Raw recording stopped; detection continues. Check camera/FFmpeg.")
                recorder_warning = True
            if not paused:
                frame_start_time = (
                    time.perf_counter()
                )

                success, frame = video.read(timeout=0.1) if video.is_live else video.read()

                if not success:
                    if not video.is_live:
                        break
                    # Keep GUI events responsive while waiting for reconnect.
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord(" "):
                        paused = True
                    continue

                if video.is_live:
                    frame_id = video._last_read_id - 1
                    timestamp = time.monotonic() - live_start_time
                    if writer is None:
                        fps = video.info.fps
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {output_path}")
                    elif frame.shape[:2] != (height, width):
                        raise RuntimeError("Live resolution changed; restart recording")
                else:
                    frame_id = int(video.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    timestamp = frame_id / fps

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
                        zone_manager.get_zone(track.ground_x, track.ground_y)
                        if zone_manager is not None else None
                    )

                    state.update_zone(
                        zone,
                        timestamp,
                    )

                    track_writer.add(
                        camera_id=recording_camera_id,
                        capture_id=recording_capture_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        track=track,
                        state=state,
                        zone=zone,
                    )

                annotated_frame = frame
                if not video.is_live:
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
                    show_zone=not video.is_live,
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
                    f"Tracking - {recording_camera_id}",
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
        try:
            video.release()
        finally:
            try:
                if recorder is not None:
                    recorder.stop()
            finally:
                if writer is not None:
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
        f"Confirmed tracks: "
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

    if confirmed_track_output_path is not None:
        print(f"Confirmed track data: {confirmed_track_output_path.relative_to(PROJECT_ROOT)}")
        print(f"Confirmed track records: {track_writer.confirmed_record_count}")

    if writer is not None:
        print("Video: " f"{output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track a video file or live RTSP stream")
    parser.add_argument("--source", help="Video file path or RTSP URL")
    parser.add_argument("--camera-id", help="Camera name (live default: live1)")
    parser.add_argument("--record-stream", action="store_true",
                        help="Also record original RTSP video using FFmpeg stream copy")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights or local path")
    parser.add_argument("--tracker-profile", choices=["auto", "baseline", "recovery"],
                        default="auto", help="auto: recovery for live sources, baseline for files")
    args = parser.parse_args()
    run_tracking_pipeline(source=args.source, camera_id=args.camera_id,
                          record_stream=args.record_stream, model_name=args.model, tracker_profile=args.tracker_profile)