from unittest.mock import Mock
import signal
import subprocess
import pytest
from src.video import recorder as module


@pytest.fixture
def process(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/fake/ffmpeg')
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    return process, popen


def test_stream_copy_and_graceful_stop(process, tmp_path):
    child, popen = process
    recorder = module.StreamRecorder('rtsp://fake/live', tmp_path/'raw.mkv')
    command = popen.call_args.args[0]
    assert command[command.index('-c:v')+1] == 'copy'
    assert '-r' not in command and '-vf' not in command
    assert '-rw_timeout' not in command
    assert command[command.index('-timeout')+1] == '5000000'
    assert command[command.index('-i')+1] == 'rtsp://fake/live'
    assert recorder.running
    recorder.stop()
    child.send_signal.assert_called_once_with(signal.SIGINT)
    child.wait.assert_called_once_with(timeout=7)


def test_already_exited_is_safe(process, tmp_path):
    child, _ = process
    recorder = module.StreamRecorder('rtsp://fake/live', tmp_path/'raw.mkv')
    child.poll.return_value = 1
    assert not recorder.running
    recorder.stop()
    child.send_signal.assert_not_called()


def test_stuck_child_is_killed(process, tmp_path):
    child, _ = process
    recorder = module.StreamRecorder('rtsp://fake/live', tmp_path/'raw.mkv')
    child.wait.side_effect = [subprocess.TimeoutExpired('ffmpeg',7),
                             subprocess.TimeoutExpired('ffmpeg',2), 0]
    recorder.stop()
    child.terminate.assert_called_once()
    child.kill.assert_called_once()


def test_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(module.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='FFmpeg'):
        module.StreamRecorder('rtsp://fake/live', tmp_path/'raw.mkv')


def test_file_source_rejected(tmp_path):
    with pytest.raises(ValueError):
        module.StreamRecorder('movie.mp4', tmp_path/'raw.mkv')
