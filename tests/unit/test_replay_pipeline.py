from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest

from src.pipeline import replay_pipeline as replay
from src.tracking.tracker import Track


class SyntheticVideo:
    def __init__(self, count=8, timestamps=True):
        self.info = SimpleNamespace(fps=8.)
        self.cap = self
        self.count = count
        self.index = 0
        self.released = False
        self.timestamps = timestamps

    def read(self):
        if self.index >= self.count:
            return False, None
        self.index += 1
        return True, np.zeros((48, 80, 3), dtype=np.uint8)

    def get(self, prop):
        return (self.index - 1) / 8 * 1000 if self.timestamps else 0

    def release(self):
        self.released = True


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / 'synthetic.input'
    source.write_bytes(b'fake source; decoder is mocked')
    video = SyntheticVideo()
    monkeypatch.setattr(replay, 'VideoSource', lambda path: video)
    tracker = Mock()
    counter = iter(range(100))
    def predict(frame):
        offset = next(counter)*3
        return [Track(1, 5+offset, 5, 25+offset, 30, .8, 0)]
    tracker.track.side_effect = predict
    factory = Mock(return_value=tracker)
    monkeypatch.setattr(replay, 'PersonTracker', factory)
    return source, video, factory


def test_default_writes_tables_without_camera_media(prepared, tmp_path, monkeypatch):
    source, video, factory = prepared
    monkeypatch.setattr(replay.cv2, 'VideoWriter', Mock(side_effect=AssertionError('Video must be opt-in')))
    output = tmp_path/'run'
    result = replay.run_replay(source, output, stride=1)
    assert video.released
    assert result['confirmed_track_ids'] == 1
    assert result['raw_rows'] == result['confirmed_track_history_rows'] == 8
    assert result['post_confirmation_rows'] < result['raw_rows']
    assert result['source_name'] == source.name
    assert not result['video_saved']
    assert sorted(p.name for p in output.iterdir()) == sorted(['summary.json', result['capture_id']+'_confirmed_tracks.parquet', result['capture_id']+'_all_tracks_debug.parquet'])
    assert result['confirmed_rows'] == result['post_confirmation_rows']
    assert pd.read_parquet(output/(result['capture_id']+'_confirmed_tracks.parquet')).confirmed_at_frame.all()
    assert factory.call_args.kwargs['confidence_threshold'] == .1


def test_video_is_native_size_and_sampled_fps(prepared, tmp_path, monkeypatch):
    source, video, factory = prepared
    writer = Mock()
    writer.isOpened.return_value = True
    create = Mock(return_value=writer)
    monkeypatch.setattr(replay.cv2,'VideoWriter',create)
    result = replay.run_replay(source,tmp_path/'run',stride=2,save_video=True,tracker_profile='baseline')
    assert create.call_args.args[-2:] == (4., (80,48))
    assert writer.write.call_count == 4
    assert writer.write.call_args.args[0].shape == (48,80,3)
    writer.release.assert_called_once()
    assert result['video_saved'] and result['frames_processed'] == 4
    assert factory.call_args.kwargs['tracker_config'] == 'bytetrack.yaml'


def test_empty_recording_writes_empty_tables(prepared,tmp_path):
    source, video, _ = prepared
    video.count = 0
    result = replay.run_replay(source,tmp_path/'run')
    assert result['frames_processed'] == 0
    assert result['median_tracker_ms'] is None
    assert pd.read_parquet(tmp_path/'run'/(result['capture_id']+'_confirmed_tracks.parquet')).empty
    assert video.released


def test_max_frames_and_timestamp_fallback(prepared,tmp_path):
    source, video, _ = prepared
    video.timestamps = False
    result = replay.run_replay(source,tmp_path/'run',stride=2,max_frames=2)
    assert result['frames_processed'] == 2 and result['stop_reason'] == 'max_frames'
    assert result['last_media_timestamp'] == .25
    assert video.released


def test_existing_output_is_not_overwritten(prepared,tmp_path):
    source, video, _ = prepared
    with pytest.raises(FileExistsError):
        replay.run_replay(source,tmp_path)
    assert video.index == 0


def test_failure_releases_capture(prepared,tmp_path,monkeypatch):
    source, video, _ = prepared
    monkeypatch.setattr(replay,'PersonTracker',Mock(side_effect=RuntimeError('model failed')))
    with pytest.raises(RuntimeError,match='model failed'):
        replay.run_replay(source,tmp_path/'run')
    assert video.released


@pytest.mark.parametrize('kwargs',[{'stride':0},{'max_frames':0},{'camera_id':'../bad'},{'tracker_profile':'other'}])
def test_invalid_options(prepared,tmp_path,kwargs):
    source, _, _ = prepared
    with pytest.raises(ValueError):
        replay.run_replay(source,tmp_path/'run',**kwargs)
