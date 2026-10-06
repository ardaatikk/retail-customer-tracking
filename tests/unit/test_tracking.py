from src.tracking.tracker import Track


def test_track_center_coordinates() -> None:
    track = Track(
        local_track_id=1,
        x1=100.0,
        y1=200.0,
        x2=300.0,
        y2=600.0,
        confidence=0.90,
        class_id=0,
    )

    assert track.center_x == 200.0
    assert track.center_y == 400.0


def test_track_ground_coordinates() -> None:
    track = Track(
        local_track_id=1,
        x1=100.0,
        y1=200.0,
        x2=300.0,
        y2=600.0,
        confidence=0.90,
        class_id=0,
    )

    assert track.ground_x == 200.0
    assert track.ground_y == 600.0