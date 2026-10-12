import pandas as pd
from src.data.track_writer import TrackWriter
from src.tracking.customer_state import CustomerState, CustomerStateManager
from src.tracking.tracker import Track


def make_track(tid=1):
    return Track(tid, 10, 20, 30, 40, .8, 0)


def add(writer, state, second):
    writer.add(camera_id='faydam1', capture_id='session1', frame_id=int(second*10),
               timestamp=second, track=make_track(state.local_track_id), state=state, zone=None)


def test_split_preserves_raw_state_and_confirmed_history(tmp_path):
    raw, accepted = tmp_path/'raw.parquet', tmp_path/'confirmed.parquet'
    writer = TrackWriter(raw, confirmed_output_path=accepted)
    first = CustomerState(1, 0, 2, 20, 40)
    rejected = CustomerState(2, 0, 2, 20, 40)
    add(writer, first, 0)
    add(writer, rejected, 0)
    first.confirmed, first.customer_id, first.confirmed_at = True, 7, 1.0
    add(writer, first, 1)
    add(writer, rejected, 1)
    assert writer.save({1:first, 2:rejected}) == 4
    df, confirmed = pd.read_parquet(raw), pd.read_parquet(accepted)
    assert writer.confirmed_record_count == 2
    assert pd.isna(df.loc[0, 'customer_id'])
    assert df.loc[2,'customer_id'] == 7
    assert df.confirmed_at_frame.tolist() == [False, False, True, False]
    assert confirmed.local_track_id.tolist() == [1, 1]
    assert confirmed.customer_id.tolist() == [7, 7]
    assert confirmed.confirmed_at_frame.tolist() == [False, True]
    assert confirmed.confirmed_at.tolist() == [1, 1]
    assert pd.isna(df.loc[1,'confirmed_at'])
    assert pd.isna(df.loc[1,'final_customer_id'])
    assert confirmed.frame_id.tolist() == df[df.final_customer_id.notna()].frame_id.tolist()


def test_unconfirmed_only_produces_empty_confirmed_table(tmp_path):
    raw, accepted = tmp_path/'raw.parquet', tmp_path/'confirmed.parquet'
    writer = TrackWriter(raw, confirmed_output_path=accepted)
    state = CustomerState(1, 0, 0, 20, 40)
    add(writer, state, 0)
    writer.save({1:state})
    assert len(pd.read_parquet(raw)) == 1
    assert pd.read_parquet(accepted).empty
    assert list(pd.read_parquet(raw).columns) == list(pd.read_parquet(accepted).columns)


def test_empty_session_produces_two_empty_tables(tmp_path):
    raw, accepted = tmp_path/'raw.parquet', tmp_path/'confirmed.parquet'
    writer = TrackWriter(raw, confirmed_output_path=accepted)
    assert writer.save({}) == 0
    assert pd.read_parquet(raw).empty and pd.read_parquet(accepted).empty


def test_confirmation_time_is_not_replaced_by_later_observations():
    manager = CustomerStateManager(min_observation_seconds=.5,
                                   min_displacement_pixels=0, min_movement_frames=0)
    state = manager.update(make_track(), 0)
    assert state.confirmed_at is None
    manager.update(make_track(), 1)
    assert state.confirmed and state.confirmed_at == 1
    manager.update(make_track(), 2)
    assert state.confirmed_at == 1
