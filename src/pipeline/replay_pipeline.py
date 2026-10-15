"""Replay a local camera recording without a camera connection or embedded media."""
import argparse
import json
import math
from pathlib import Path
import re
from statistics import median
import time

import cv2
import pandas as pd

from src.config.paths import SESSIONS_DIR
from src.data.track_writer import TrackWriter
from src.analytics.summary import export_session_summary
from src.tracking.customer_state import CustomerStateManager
from src.tracking.tracker import PersonTracker
from src.video.source import VideoSource
from src.video.session import make_session_id
from src.visualization.draw import draw_tracks

RECOVERY_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'trackers' / 'bytetrack_recovery.yaml'


def validate_exports(raw_path: Path, confirmed_path: Path) -> tuple[int, int, int]:
    raw = pd.read_parquet(raw_path)
    confirmed = pd.read_parquet(confirmed_path)
    keys = ['camera_id', 'capture_id', 'frame_id', 'local_track_id']
    selected = raw[raw.final_customer_id.notna() & raw.confirmed_at_frame]
    checks = [
        list(selected[keys].itertuples(index=False, name=None))
        == list(confirmed[keys].itertuples(index=False, name=None)),
        confirmed.customer_id.notna().all(),
        raw.loc[~raw.confirmed_at_frame, 'customer_id'].isna().all(),
        (raw.loc[raw.confirmed_at_frame, 'timestamp']
         >= raw.loc[raw.confirmed_at_frame, 'confirmed_at']).all(),
    ]
    if not all(checks):
        raise RuntimeError('Raw/confirmed export consistency check failed')
    return len(raw), len(confirmed), int(confirmed.confirmed_at_frame.sum())


def run_replay(
    source: Path,
    output_dir: Path = None,
    *,
    camera_id: str = 'camera1',
    model_name: str = 'yolo11n.pt',
    tracker_profile: str = 'recovery',
    stride: int = 3,
    save_video: bool = False,
    max_frames: int = None,
    session_name: str = None,
) -> dict:
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Local recording not found: {source}')
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError('stride and max_frames must be positive')
    if tracker_profile not in ('baseline', 'recovery'):
        raise ValueError('tracker_profile must be baseline or recovery')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', camera_id):
        raise ValueError('Invalid camera_id')
    capture_id = make_session_id(camera_id, 'replay', session_name)
    output = Path(output_dir) if output_dir is not None else SESSIONS_DIR / camera_id / capture_id
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    video = VideoSource(source)
    writer = None
    try:
        output.mkdir(parents=True)
        fps = video.info.fps
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Recording FPS must be finite and positive')
        sampled_fps = fps / stride
        tracker = PersonTracker(
            model_name=model_name,
            confidence_threshold=.1 if tracker_profile == 'recovery' else .4,
            tracker_config=str(RECOVERY_CONFIG) if tracker_profile == 'recovery' else 'bytetrack.yaml',
        )
        manager = CustomerStateManager(min_observation_seconds=.5, min_displacement_pixels=10,
                                       min_movement_frames=5, movement_step_pixels=2)
        raw_path = output / f'{capture_id}_all_tracks_debug.parquet'
        confirmed_path = output / f'{capture_id}_confirmed_tracks.parquet'
        exporter = TrackWriter(raw_path, confirmed_output_path=confirmed_path, confirmed_frames_only=True)
        index = processed = empty = 0
        first_pts = None
        last_time = 0.0
        durations = []
        stop_reason = 'end_of_readable_recording'
        print(f'Replay: {tracker_profile}, sampled FPS {sampled_fps:.2f}', flush=True)
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame_id = index
            index += 1
            if frame_id % stride:
                continue
            pts = video.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if first_pts is None:
                first_pts = pts if math.isfinite(pts) else 0.0
            timestamp = pts - first_pts
            if not math.isfinite(timestamp) or timestamp < 0 or (processed and timestamp <= last_time):
                timestamp = max(frame_id / fps, last_time + 1 / sampled_fps if processed else 0.0)
            last_time = timestamp
            started = time.perf_counter()
            tracks = tracker.track(frame)
            durations.append((time.perf_counter() - started) * 1000)
            processed += 1
            empty += not bool(tracks)
            for track in tracks:
                state = manager.update(track, timestamp)
                exporter.add(camera_id=camera_id, capture_id=capture_id, frame_id=frame_id,
                             timestamp=timestamp, track=track, state=state, zone=None)
            if save_video:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output / f'{capture_id}_annotated.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), sampled_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError('Could not create annotated replay video')
                if frame.shape[:2] != (height, width):
                    raise RuntimeError('Recording resolution changed')
                annotated = draw_tracks(frame, tracks, manager.states, timestamp, None, show_zone=False)
                cv2.putText(annotated, f'Replay {timestamp:.2f}s | {tracker_profile}', (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                writer.write(annotated)
            if processed % 80 == 0:
                print(f'Processed {processed} frames; media time {timestamp:.1f}s', flush=True)
            if max_frames is not None and processed >= max_frames:
                stop_reason = 'max_frames'
                break
        exporter.save(manager.states)
        raw_rows, confirmed_rows, post_confirmation_rows = validate_exports(
            raw_path, confirmed_path)
        export_session_summary(confirmed_path)
        summary = {
            'source_name': source.name, 'camera_id': camera_id, 'capture_id': capture_id,
            'model': Path(model_name).name, 'tracker_profile': tracker_profile,
            'source_frames_read': index, 'frames_processed': processed, 'frame_stride': stride,
            'sample_fps': sampled_fps, 'last_media_timestamp': last_time, 'stop_reason': stop_reason,
            'raw_rows': raw_rows, 'confirmed_rows': confirmed_rows,
            'confirmed_track_history_rows': int(pd.read_parquet(raw_path).final_customer_id.notna().sum()),
            'post_confirmation_rows': post_confirmation_rows, 'sampled_frames_without_tracks': empty,
            'track_ids': len(manager.states),
            'confirmed_track_ids': sum(state.confirmed for state in manager.states.values()),
            'median_tracker_ms': median(durations[1:] or durations) if durations else None,
            'export_checks_passed': True, 'video_saved': writer is not None,
            'track_summaries': [
                {'local_track_id': tid, 'customer_id': state.customer_id,
                 'confirmed_at': state.confirmed_at, 'first_seen': state.first_seen,
                 'last_seen': state.last_seen, 'observed_span_seconds': state.observed_time}
                for tid, state in manager.states.items()
            ],
            'limitations': ['No live network, reconnect or latency test',
                            'Constant frame sampling; live frame skipping may differ',
                            'Track IDs are not unique-person identities',
                            'Observation spans include gaps'],
        }
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(f'Raw observations: {raw_rows}; confirmed rows: {confirmed_rows}')
        print(f'Confirmed tracks: {summary["confirmed_track_ids"]}')
        print(f'Outputs: {output}')
        return summary
    finally:
        try:
            video.release()
        finally:
            if writer is not None:
                writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Local video; no camera URL')
    parser.add_argument('--output-dir', type=Path, help='New local output directory')
    parser.add_argument('--camera-id', default='camera1')
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--tracker-profile', choices=['baseline', 'recovery'], default='recovery')
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--max-frames', type=int, help='Optional limit on processed frames')
    parser.add_argument('--save-video', action='store_true', help='Also save a local annotated video')
    parser.add_argument('--session-name', help='Optional readable recording label')
    args = parser.parse_args()
    run_replay(args.source, args.output_dir, camera_id=args.camera_id, model_name=args.model,
               tracker_profile=args.tracker_profile, stride=args.stride,
               max_frames=args.max_frames, save_video=args.save_video, session_name=args.session_name)


if __name__ == '__main__':
    main()
