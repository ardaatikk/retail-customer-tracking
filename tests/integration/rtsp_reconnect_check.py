import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.video.source import VideoSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    source = VideoSource(
        args.source,
        read_timeout=1.0,
        reconnect_interval=1.0,
    )

    try:
        while True:
            ok, frame = source.read()

            if not ok:
                print(f"Kare bekleniyor | Bağlantı: {source.connected}")
                continue

            print(
                f"Frame ID: {source._last_read_id} "
                f"| Boyut: {frame.shape}"
            )
    except KeyboardInterrupt:
        print("\nTest durduruldu.")
    finally:
        source.release()


if __name__ == "__main__":
    main()