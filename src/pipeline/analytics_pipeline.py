import argparse
from pathlib import Path

from src.analytics.heatmap import generate_heatmap
from src.analytics.summary import generate_summary, load_tracks
from src.analytics.trajectory import generate_trajectory


def run_analytics_pipeline(
    tracks_path: Path = None,
    camera_id: str = None,
    capture_id: str = None,
    output_dir: Path = None,
    reference_frame: Path = None,
) -> None:
    tracks = load_tracks(tracks_path, camera_id, capture_id)
    selected_camera = str(tracks.camera_id.iloc[0])
    selected_capture = str(tracks.capture_id.iloc[0])
    custom_input = tracks_path is not None
    if output_dir is not None:
        output_dir = Path(output_dir)
    elif custom_input:
        output_dir = Path(tracks_path).resolve().parent / "analytics"
    print("Retail analytics pipeline")
    print(f"Camera: {selected_camera} | Capture: {selected_capture}")
    print(f"Confirmed observations: {len(tracks)}")
    spatial_enabled = reference_frame is not None or (
        not custom_input and selected_camera == "Showroom"
    )
    if spatial_enabled:
        missing = {"ground_x", "ground_y"} - set(tracks.columns)
        if missing:
            raise ValueError(f"Missing spatial columns: {', '.join(sorted(missing))}")
        for name, generate in [("trajectory", generate_trajectory), ("heatmap", generate_heatmap)]:
            print(f"Generating {name}...")
            destination = output_dir / f"{name}.png" if output_dir is not None else None
            generate(tracks=tracks, camera_id=selected_camera,
                     reference_frame_path=reference_frame, output_path=destination)
    else:
        print("Spatial plots skipped: supply --reference-frame from the same camera at original resolution")
    generate_summary(tracks=tracks, output_dir=output_dir)
    print("Analytics pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze one confirmed camera/session history")
    parser.add_argument("--tracks", type=Path, help="Parquet input; default: original Showroom tracks.parquet")
    parser.add_argument("--camera-id", help="Select a camera; inferred for single-session custom input")
    parser.add_argument("--capture-id", help="Select a capture/session; inferred for single-session custom input")
    parser.add_argument("--output-dir", type=Path, help="Custom outputs; custom input default: adjacent analytics directory")
    parser.add_argument("--reference-frame", type=Path, help="Same-camera image at the original recorded resolution")
    args = parser.parse_args()
    run_analytics_pipeline(tracks_path=args.tracks, camera_id=args.camera_id,
                           capture_id=args.capture_id, output_dir=args.output_dir,
                           reference_frame=args.reference_frame)
