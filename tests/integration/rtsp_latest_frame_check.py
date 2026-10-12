import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.video.source import VideoSource


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test latest-frame behavior with a live RTSP stream."
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="RTSP stream URL",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of frames to process",
    )

    parser.add_argument(
        "--processing-delay",
        type=float,
        default=0.1,
        help="Simulated processing time in seconds",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    source = VideoSource(args.source)

    print("\nRTSP Latest-Frame Integration Test")
    print("----------------------------------")
    print(f"Camera FPS: {source.info.fps:.2f}")
    print(
        f"Resolution: "
        f"{source.info.width}x{source.info.height}"
    )
    print(f"Processing delay: {args.processing_delay:.3f}s\n")

    previous_frame_id = 0
    total_skipped = 0

    try:
        for iteration in range(args.iterations):
            success, frame = source.read()

            if not success:
                print("Failed to read frame.")
                break

            current_frame_id = source._last_read_id

            skipped = max(
                0,
                current_frame_id - previous_frame_id - 1,
            )

            total_skipped += skipped

            print(
                f"Iteration {iteration + 1:02d} | "
                f"Frame ID: {current_frame_id:04d} | "
                f"Skipped: {skipped:02d} | "
                f"Shape: {frame.shape}"
            )

            previous_frame_id = current_frame_id

            # Simulate YOLO inference time.
            time.sleep(args.processing_delay)

    finally:
        source.release()

    print("\nTest Summary")
    print("------------")
    print(f"Iterations: {args.iterations}")
    print(f"Total skipped frames: {total_skipped}")


if __name__ == "__main__":
    main()