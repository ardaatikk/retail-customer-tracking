import pytest

from src.tracking.customer_state import (
    CustomerState,
    CustomerStateManager,
)
from src.tracking.tracker import Track


def make_track(
    track_id: int = 1,
    ground_x: float = 100.0,
    ground_y: float = 200.0,
    confidence: float = 0.90,
) -> Track:
    width = 40.0
    height = 100.0

    return Track(
        local_track_id=track_id,
        x1=ground_x - width / 2,
        y1=ground_y - height,
        x2=ground_x + width / 2,
        y2=ground_y,
        confidence=confidence,
        class_id=0,
    )


def test_observed_time() -> None:
    state = CustomerState(
        local_track_id=1,
        first_seen=10.0,
        last_seen=15.5,
        start_ground_x=100.0,
        start_ground_y=200.0,
    )

    assert state.observed_time == pytest.approx(
        5.5
    )


def test_elapsed_time_never_negative() -> None:
    state = CustomerState(
        local_track_id=1,
        first_seen=10.0,
        last_seen=10.0,
        start_ground_x=100.0,
        start_ground_y=200.0,
    )

    assert state.elapsed_time(8.0) == 0.0
    assert state.elapsed_time(12.5) == pytest.approx(
        2.5
    )


def test_average_confidence() -> None:
    state = CustomerState(
        local_track_id=1,
        first_seen=0.0,
        last_seen=1.0,
        start_ground_x=100.0,
        start_ground_y=200.0,
        confidence_sum=1.6,
        confidence_count=2,
    )

    assert state.average_confidence == pytest.approx(
        0.8
    )


def test_zone_history_and_dwell_time() -> None:
    state = CustomerState(
        local_track_id=1,
        first_seen=0.0,
        last_seen=10.0,
        start_ground_x=100.0,
        start_ground_y=200.0,
    )

    state.update_zone(
        "Entrance",
        0.0,
    )

    state.update_zone(
        "Main Area",
        3.0,
    )

    state.update_zone(
        "Demo Area",
        8.0,
    )

    dwell_times = state.get_zone_dwell_times(
        10.0
    )

    assert state.zone_history == [
        "Entrance",
        "Main Area",
        "Demo Area",
    ]

    assert dwell_times[
        "Entrance"
    ] == pytest.approx(
        3.0
    )

    assert dwell_times[
        "Main Area"
    ] == pytest.approx(
        5.0
    )

    assert dwell_times[
        "Demo Area"
    ] == pytest.approx(
        2.0
    )


def test_none_zone_does_not_break_current_zone() -> None:
    state = CustomerState(
        local_track_id=1,
        first_seen=0.0,
        last_seen=5.0,
        start_ground_x=100.0,
        start_ground_y=200.0,
    )

    state.update_zone(
        "Entrance",
        0.0,
    )

    state.update_zone(
        None,
        2.0,
    )

    assert state.current_zone == "Entrance"
    assert state.zone_history == [
        "Entrance"
    ]

    dwell_times = state.get_zone_dwell_times(
        5.0
    )

    assert dwell_times[
        "Entrance"
    ] == pytest.approx(
        5.0
    )


def test_customer_is_not_confirmed_too_early() -> None:
    manager = CustomerStateManager(
        min_observation_seconds=0.5,
        min_displacement_pixels=10.0,
        min_movement_frames=2,
        movement_step_pixels=2.0,
        min_average_confidence=0.50,
    )

    state = manager.update(
        make_track(
            ground_x=100.0,
        ),
        timestamp=0.0,
    )

    state = manager.update(
        make_track(
            ground_x=105.0,
        ),
        timestamp=0.2,
    )

    assert state.confirmed is False
    assert state.customer_id is None


def test_customer_is_confirmed_when_requirements_are_met() -> None:
    manager = CustomerStateManager(
        min_observation_seconds=0.5,
        min_displacement_pixels=10.0,
        min_movement_frames=2,
        movement_step_pixels=2.0,
        min_average_confidence=0.50,
    )

    manager.update(
        make_track(
            ground_x=100.0,
        ),
        timestamp=0.0,
    )

    manager.update(
        make_track(
            ground_x=106.0,
        ),
        timestamp=0.3,
    )

    state = manager.update(
        make_track(
            ground_x=112.0,
        ),
        timestamp=0.6,
    )

    assert state.confirmed is True
    assert state.customer_id == 1
    assert state.max_displacement == pytest.approx(
        12.0
    )
    assert state.movement_frames == 2


def test_low_confidence_track_is_not_confirmed() -> None:
    manager = CustomerStateManager(
        min_observation_seconds=0.5,
        min_displacement_pixels=10.0,
        min_movement_frames=2,
        movement_step_pixels=2.0,
        min_average_confidence=0.50,
    )

    manager.update(
        make_track(
            ground_x=100.0,
            confidence=0.30,
        ),
        timestamp=0.0,
    )

    manager.update(
        make_track(
            ground_x=106.0,
            confidence=0.30,
        ),
        timestamp=0.3,
    )

    state = manager.update(
        make_track(
            ground_x=112.0,
            confidence=0.30,
        ),
        timestamp=0.6,
    )

    assert state.confirmed is False
    assert state.customer_id is None


def test_confirmed_customers_get_sequential_ids() -> None:
    manager = CustomerStateManager(
        min_observation_seconds=0.5,
        min_displacement_pixels=10.0,
        min_movement_frames=2,
        movement_step_pixels=2.0,
        min_average_confidence=0.50,
    )

    for timestamp, x in [
        (0.0, 100.0),
        (0.3, 106.0),
        (0.6, 112.0),
    ]:
        first_state = manager.update(
            make_track(
                track_id=10,
                ground_x=x,
            ),
            timestamp=timestamp,
        )

    for timestamp, x in [
        (1.0, 200.0),
        (1.3, 206.0),
        (1.6, 212.0),
    ]:
        second_state = manager.update(
            make_track(
                track_id=20,
                ground_x=x,
            ),
            timestamp=timestamp,
        )

    assert first_state.confirmed is True
    assert second_state.confirmed is True

    assert first_state.customer_id == 1
    assert second_state.customer_id == 2