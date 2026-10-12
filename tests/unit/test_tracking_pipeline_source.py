from types import SimpleNamespace
import numpy as np
import pytest
from src.pipeline import tracking_pipeline as pipeline


class FakeVideo:
    def __init__(self, is_live, reads):
        self.is_live = is_live
        self.reads = iter(reads)
        self.info = SimpleNamespace(fps=25, width=4, height=3)
        self.cap = self
        self._last_read_id = 0
        self.released = False
        self.seek = None

    def read(self, **kwargs):
        result = next(self.reads)
        if result[0]:
            self._last_read_id += 1
        return result

    def get(self, prop):
        if prop == pipeline.cv2.CAP_PROP_FRAME_COUNT:
            return 10
        return self._last_read_id

    def set(self, prop, value):
        self.seek = (prop, value)

    def release(self):
        self.released = True


@pytest.fixture
def setup(monkeypatch, tmp_path):
    calls = SimpleNamespace(frames=[], records=[], writers=[], paths=[], video_paths=[], confirmed_paths=[], zones=[], saved=False, destroyed=False)
    monkeypatch.setattr(pipeline, 'VIDEOS_DIR', tmp_path)
    monkeypatch.setattr(pipeline, 'PROCESSED_DATA_DIR', tmp_path)
    monkeypatch.setattr(pipeline, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'PersonTracker', lambda **kw: SimpleNamespace(
        track=lambda frame: calls.frames.append(frame) or [SimpleNamespace(ground_x=0, ground_y=0)]))
    monkeypatch.setattr(pipeline, 'ZoneManager', lambda camera: setup_zone(calls, camera))
    monkeypatch.setattr(pipeline, 'CustomerStateManager', lambda **kw: SimpleNamespace(
        states={}, update=lambda *a: SimpleNamespace(update_zone=lambda *a: None)))
    def setup_zone(calls, camera):
        calls.zones.append(camera)
        return SimpleNamespace(zones=[], get_zone=lambda *a: 'zone')
    def track_writer(path, **kwargs):
        calls.paths.append(path)
        calls.confirmed_paths.append(kwargs.get("confirmed_output_path"))
        return SimpleNamespace(
        add=lambda **kw: calls.records.append(kw), save=lambda states: 0, confirmed_record_count=0)
    monkeypatch.setattr(pipeline, 'TrackWriter', track_writer)
    monkeypatch.setattr(pipeline, 'draw_zones', lambda **kw: kw['frame'])
    monkeypatch.setattr(pipeline, 'draw_tracks', lambda **kw: kw['frame'])
    monkeypatch.setattr(pipeline.cv2, 'putText', lambda *a: None)
    monkeypatch.setattr(pipeline.cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(pipeline.cv2, 'destroyAllWindows', lambda: setattr(calls, 'destroyed', True))
    def writer(*args):
        calls.video_paths.append(args[0])
        w = SimpleNamespace(released=False, frames=[], isOpened=lambda: True)
        w.write = w.frames.append
        w.release = lambda: setattr(w, 'released', True)
        calls.writers.append(w)
        return w
    monkeypatch.setattr(pipeline.cv2, 'VideoWriter', writer)
    return calls


def test_live_timeout_recovers_without_processing_stale_frame(monkeypatch, setup):
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    video = FakeVideo(True, [(False, None), (True, frame), (False, None)])
    monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
    keys = iter([-1, -1, ord('q')])
    monkeypatch.setattr(pipeline.cv2, 'waitKey', lambda delay: next(keys))
    clock = iter([100.0, 105.0])
    monkeypatch.setattr(pipeline.time, 'monotonic', lambda: next(clock))
    pipeline.run_tracking_pipeline('rtsp://fake/live')
    assert len(setup.frames) == 1
    assert setup.records[0]['timestamp'] == 5.0
    assert setup.records[0]['camera_id'] == 'faydam1'
    assert setup.records[0]['capture_id'].startswith('live_')
    assert setup.records[0]['zone'] is None
    assert not setup.zones
    assert setup.paths[0].parent.parent.name == 'faydam1'
    assert setup.paths[0].name == 'tracks_raw.parquet'
    assert setup.confirmed_paths[0] == setup.paths[0].with_name('tracks_confirmed.parquet')
    assert '/faydam1/' in setup.video_paths[0]
    assert video.seek is None
    assert video.released and setup.writers[0].released and setup.destroyed


def test_file_eof_and_seek_are_preserved(monkeypatch, setup):
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    video = FakeVideo(False, [(True, frame), (True, frame), (False, None)])
    monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
    monkeypatch.setattr(pipeline.cv2, 'waitKey', lambda delay: -1)
    monkeypatch.setattr(pipeline, 'START_SECOND', 2)
    pipeline.run_tracking_pipeline()
    assert video.seek == (pipeline.cv2.CAP_PROP_POS_FRAMES, 50)
    assert [r['timestamp'] for r in setup.records] == [0, 1 / 25]
    assert len(setup.frames) == 2
    assert setup.records[0]['camera_id'] == 'Showroom'
    assert setup.records[0]['capture_id'] == 'capture1'
    assert setup.zones == ['Showroom']
    assert setup.paths[0].name == 'tracks.parquet'
    assert setup.confirmed_paths[0] is None
    assert video.released and setup.writers[0].released


def test_quit_during_initial_connection_does_not_create_writer(monkeypatch, setup):
    video = FakeVideo(True, [(False, None)])
    monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
    monkeypatch.setattr(pipeline.cv2, 'waitKey', lambda delay: ord('q'))
    pipeline.run_tracking_pipeline('rtsp://fake/live')
    assert not setup.frames and not setup.writers
    assert video.released and setup.destroyed


def test_model_initialization_failure_releases_resources(monkeypatch, setup):
    video = FakeVideo(False, [])
    monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
    def fail(**kwargs):
        raise RuntimeError('model unavailable')
    monkeypatch.setattr(pipeline, 'PersonTracker', fail)
    with pytest.raises(RuntimeError, match='model unavailable'):
        pipeline.run_tracking_pipeline()
    assert video.released and setup.writers[0].released and setup.destroyed


def test_custom_live_camera_and_distinct_sessions(monkeypatch, setup):
    for _ in range(2):
        frame = np.zeros((3, 4, 3), dtype=np.uint8)
        video = FakeVideo(True, [(True, frame)])
        monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
        monkeypatch.setattr(pipeline.cv2, 'waitKey', lambda delay: ord('q'))
        pipeline.run_tracking_pipeline('rtsp://fake/live', camera_id='faydam2')
    assert [r['camera_id'] for r in setup.records] == ['faydam2', 'faydam2']
    assert setup.records[0]['capture_id'] != setup.records[1]['capture_id']
    assert setup.paths[0] != setup.paths[1]
    assert setup.video_paths[0] != setup.video_paths[1]


@pytest.mark.parametrize('camera_id', ['../oops', '', '/absolute', 'a/b'])
def test_invalid_camera_name(camera_id):
    with pytest.raises(ValueError):
        pipeline.run_tracking_pipeline('rtsp://fake/live', camera_id=camera_id)


def test_independent_recorder_stops_on_quit(monkeypatch, setup):
    from unittest.mock import Mock
    video = FakeVideo(True, [(False, None)])
    monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
    recorder = Mock(running=True)
    factory = Mock(return_value=recorder)
    monkeypatch.setattr(pipeline, 'StreamRecorder', factory)
    monkeypatch.setattr(pipeline.cv2, 'waitKey', lambda delay: ord('q'))
    pipeline.run_tracking_pipeline('rtsp://fake/live', record_stream=True)
    assert factory.call_args.args[1].name.endswith('_raw.mkv')
    recorder.stop.assert_called_once()
    assert video.released


def test_recorder_failure_does_not_stop_detection(monkeypatch, setup, capsys):
    from unittest.mock import Mock
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    video = FakeVideo(True, [(True, frame)])
    monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
    recorder = Mock(running=False)
    monkeypatch.setattr(pipeline, 'StreamRecorder', lambda *a: recorder)
    monkeypatch.setattr(pipeline.cv2, 'waitKey', lambda delay: ord('q'))
    pipeline.run_tracking_pipeline('rtsp://fake/live', record_stream=True)
    assert len(setup.frames) == 1
    assert 'Raw recording stopped' in capsys.readouterr().out
    recorder.stop.assert_called_once()


def test_selected_model_is_loaded_once(monkeypatch, setup):
    from unittest.mock import Mock
    video = FakeVideo(True, [(False, None)])
    monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
    tracker = Mock()
    factory = Mock(return_value=tracker)
    monkeypatch.setattr(pipeline, 'PersonTracker', factory)
    monkeypatch.setattr(pipeline.cv2, 'waitKey', lambda delay: ord('q'))
    pipeline.run_tracking_pipeline('rtsp://fake/live', model_name='yolo11s.pt')
    factory.assert_called_once_with(model_name='yolo11s.pt', confidence_threshold=0.1,
                                    tracker_config=str(pipeline.RECOVERY_TRACKER_CONFIG))
    tracker.track.assert_not_called()
    assert video.released


@pytest.mark.parametrize('is_live,profile,expected', [
    (True, 'auto', 'recovery'), (False, 'auto', 'baseline'),
    (True, 'baseline', 'baseline'), (False, 'recovery', 'recovery'),
])
def test_tracker_profile_selection(monkeypatch, setup, is_live, profile, expected):
    from unittest.mock import Mock
    video = FakeVideo(is_live, [(False, None)])
    monkeypatch.setattr(pipeline, 'VideoSource', lambda source: video)
    factory = Mock()
    monkeypatch.setattr(pipeline, 'PersonTracker', factory)
    monkeypatch.setattr(pipeline.cv2, 'waitKey', lambda delay: ord('q'))
    pipeline.run_tracking_pipeline('rtsp://fake/live' if is_live else None,
                                   tracker_profile=profile)
    if expected == 'recovery':
        factory.assert_called_once_with(model_name='yolo11n.pt', confidence_threshold=0.1,
                                        tracker_config=str(pipeline.RECOVERY_TRACKER_CONFIG))
    else:
        factory.assert_called_once_with(model_name='yolo11n.pt')
    assert video.released


def test_invalid_tracker_profile():
    with pytest.raises(ValueError, match='tracker_profile'):
        pipeline.run_tracking_pipeline(tracker_profile='unknown')
