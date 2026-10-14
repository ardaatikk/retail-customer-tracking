from src.pipeline import replay_pipeline as replay
from tests.unit.test_replay_pipeline import prepared


def test_replay_default_groups_outputs_under_camera_session(prepared, tmp_path, monkeypatch):
    source, video, _ = prepared
    monkeypatch.setattr(replay, "SESSIONS_DIR", tmp_path / "sessions")
    result = replay.run_replay(source, camera_id="live1", stride=1)
    directory = tmp_path / "sessions" / "live1" / result["capture_id"]
    assert (directory / "summary.json").exists()
    assert (directory / (result["capture_id"] + "_confirmed_tracks.parquet")).exists()
    assert (directory / (result["capture_id"] + "_all_tracks_debug.parquet")).exists()
    assert video.released
