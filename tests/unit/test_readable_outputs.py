import re
import pandas as pd
import pytest
from src.video.session import make_session_id
from src.data.track_writer import TrackWriter
from src.tracking.customer_state import CustomerState
from src.tracking.tracker import Track
from src.analytics.summary import calculate_customer_summary


def test_readable_ids_are_labeled_without_random_suffix():
    first = make_session_id("faydam1", "live", "doorway-walk")
    assert re.fullmatch(r"faydam1_live_doorway-walk_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", first)


@pytest.mark.parametrize("label", ["../bad", "", "a/b"])
def test_invalid_recording_labels(label):
    with pytest.raises(ValueError):
        make_session_id("camera1", "replay", label)


def test_strict_primary_excludes_unconfirmed_people_and_frames(tmp_path):
    writer = TrackWriter(tmp_path/"debug.parquet", confirmed_output_path=tmp_path/"confirmed.parquet", confirmed_frames_only=True)
    track = Track(3, 10, 10, 20, 30, .8, 0)
    state = CustomerState(3, 0, 2, 15, 30)
    def add(t):
        writer.add(camera_id="faydam1", capture_id="run", frame_id=int(t), timestamp=t, track=track, state=state, zone=None)
    add(0)
    state.confirmed = True
    state.customer_id = 1
    state.confirmed_at = 1
    add(1)
    add(2)
    rejected = CustomerState(12, 0, 0, 15, 30)
    writer.add(camera_id="faydam1", capture_id="run", frame_id=0, timestamp=0, track=Track(12,10,10,20,30,.4,0), state=rejected, zone=None)
    writer.save({3:state,12:rejected})
    raw = pd.read_parquet(tmp_path/"debug.parquet")
    accepted = pd.read_parquet(tmp_path/"confirmed.parquet")
    assert len(raw) == 4 and len(accepted) == 2
    assert accepted.local_track_id.tolist() == [3,3]
    assert accepted.confirmed_at_frame.all()
    assert accepted.first_seen.tolist() == [0,0]
    assert calculate_customer_summary(accepted).iloc[0].store_time_seconds == 2
