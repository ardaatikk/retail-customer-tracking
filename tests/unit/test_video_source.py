from pathlib import Path
from queue import Queue, Empty
from threading import Event, Thread
import time
import cv2
import pytest
from src.video.source import VideoSource


class FakeCapture:
    def __init__(self, opened=True, fps=24, width=640):
        self.opened, self.fps, self.width = opened, fps, width
        self.frames = Queue()
        self.released, self.read_entered = Event(), Event()
        self.read_count = 0

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.width,
                cv2.CAP_PROP_FRAME_HEIGHT: 480}.get(prop, 0)

    def read(self):
        self.read_count += 1
        self.read_entered.set()
        try:
            item = self.frames.get(timeout=0.1)
        except Empty:
            return False, None
        if isinstance(item, Exception):
            raise item
        return item

    def release(self):
        self.released.set()


def install(monkeypatch, *captures):
    pending, calls = list(captures), []
    def factory(*args):
        calls.append(args)
        return pending.pop(0) if pending else FakeCapture(opened=False)
    monkeypatch.setattr(cv2, 'VideoCapture', factory)
    return calls


def wait_for(predicate):
    deadline = time.monotonic() + 1
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def live(**kwargs):
    return VideoSource('rtsp://fake/live', read_timeout=0.02,
                       reconnect_interval=0.01, **kwargs)


def test_latest_frame_and_no_duplicates(monkeypatch):
    cap = FakeCapture()
    for frame in ('one', 'two', 'three'):
        cap.frames.put((True, frame))
    install(monkeypatch, cap)
    source = live()
    try:
        wait_for(lambda: source._frame_id == 3)
        assert source.read() == (True, 'three')
        assert source.read(timeout=0) == (False, None)
        cap.frames.put((True, 'four'))
        assert source.read(timeout=0.2) == (True, 'four')
    finally:
        source.release()
    assert cap.released.is_set()
    assert not source._thread.is_alive()


def test_timeout_allows_later_reads(monkeypatch):
    cap = FakeCapture()
    install(monkeypatch, cap)
    source = live()
    try:
        start = time.monotonic()
        assert source.read() == (False, None)
        assert 0.01 <= time.monotonic() - start < 0.3
        cap.frames.put((True, 'later'))
        assert source.read(timeout=0.2) == (True, 'later')
    finally:
        source.release()


@pytest.mark.parametrize('failure', [(False, None), cv2.error('decode failed')])
def test_reconnect_discards_stale_frame(monkeypatch, failure):
    first, second = FakeCapture(), FakeCapture(fps=30, width=1280)
    first.frames.put((True, 'stale'))
    first.frames.put(failure)
    failed = FakeCapture(opened=False)
    install(monkeypatch, first, failed, second)
    source = live()
    try:
        wait_for(first.released.is_set)
        assert source.read(timeout=0) == (False, None)
        assert not source.connected
        wait_for(second.read_entered.is_set)
        second.frames.put((True, 'fresh'))
        assert source.read(timeout=0.2) == (True, 'fresh')
        assert source.connected
        assert source._last_read_id == 2
        assert source.info.fps == 30
        assert source.info.width == 1280
        assert failed.released.is_set()
    finally:
        source.release()


def test_initial_failure_and_backend_timeouts(monkeypatch):
    failed, recovered = FakeCapture(opened=False), FakeCapture()
    recovered.frames.put((True, 'recovered'))
    calls = install(monkeypatch, failed, recovered)
    source = live(open_timeout=0.4, capture_read_timeout=0.3)
    try:
        assert source.read(timeout=0.2) == (True, 'recovered')
        assert failed.released.is_set()
        assert calls[0] == ('rtsp://fake/live', cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 400, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 300])
    finally:
        source.release()


def test_shutdown_wakes_reader_and_interrupts_retry_wait(monkeypatch):
    calls = install(monkeypatch, FakeCapture(opened=False))
    source = VideoSource('rtsp://fake/live', reconnect_interval=10)
    result = []
    reader = Thread(target=lambda: result.append(source.read(timeout=10)))
    reader.start()
    source.release()
    reader.join(timeout=0.5)
    assert not reader.is_alive()
    assert result == [(False, None)]
    assert not source._thread.is_alive()
    assert len(calls) == 1
    source.release()
    assert source.read() == (False, None)


def test_shutdown_during_native_read(monkeypatch):
    cap = FakeCapture()
    install(monkeypatch, cap)
    source = live()
    assert cap.read_entered.wait(1)
    source.release()
    assert cap.released.is_set()
    assert not source._thread.is_alive()
    assert not source.connected


def test_file_sequence_eof_and_fallback(monkeypatch):
    cap = FakeCapture(fps=0)
    for item in [(True, 'one'), (True, 'two'), (False, None), (False, None)]:
        cap.frames.put(item)
    calls = install(monkeypatch, cap)
    source = VideoSource(Path('movie.mp4'))
    assert source._thread is None
    assert source.info.fps == 25
    assert calls == [('movie.mp4',)]
    assert cap.read_count == 0
    assert [source.read() for _ in range(4)] == [
        (True, 'one'), (True, 'two'), (False, None), (False, None)]
    assert len(calls) == 1
    source.release()
    assert cap.released.is_set()


def test_unopened_file_cleanup(monkeypatch):
    cap = FakeCapture(opened=False)
    install(monkeypatch, cap)
    with pytest.raises(RuntimeError):
        VideoSource('missing.mp4')
    assert cap.released.is_set()


def test_non_rtsp_live_failure_stays_terminal(monkeypatch):
    cap = FakeCapture()
    cap.frames.put((False, None))
    calls = install(monkeypatch, cap)
    source = VideoSource(0)
    try:
        assert source.read() == (False, None)
        assert len(calls) == 1
    finally:
        source.release()


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
@pytest.mark.parametrize('option', ['read_timeout', 'reconnect_interval', 'open_timeout', 'capture_read_timeout'])
def test_invalid_config(monkeypatch, option, value):
    calls = install(monkeypatch)
    with pytest.raises(ValueError):
        VideoSource('rtsp://fake/live', **{option: value})
    assert not calls


def test_invalid_read_timeout(monkeypatch):
    install(monkeypatch, FakeCapture(opened=False))
    source = live()
    try:
        for value in [-1, float('inf'), float('nan')]:
            with pytest.raises(ValueError):
                source.read(timeout=value)
    finally:
        source.release()
