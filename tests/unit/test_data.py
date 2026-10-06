from pathlib import Path

import pandas as pd
import pytest

import src.data.validate as validate_module
from src.data.track_writer import TrackWriter
from src.tracking.customer_state import CustomerState
from src.tracking.tracker import Track


def make_track(
    track_id: int = 1,
    confidence: float = 0.90,
) -> Track:
    return Track(
        local_track_id=track_id,
        x1=100.0,
        y1=200.0,
        x2=300.0,
        y2=600.0,
        confidence=confidence,
        class_id=0,
    )


def make_state(
    track_id: int = 1,
    confirmed: bool = False,
    customer_id: int | None = None,
) -> CustomerState:
    return CustomerState(
        local_track_id=track_id,
        first_seen=0.0,
        last_seen=1.0,
        start_ground_x=200.0,
        start_ground_y=600.0,
        confirmed=confirmed,
        customer_id=customer_id,
    )


def test_track_writer_adds_record(
    tmp_path: Path,
) -> None:
    writer = TrackWriter(
        tmp_path / "tracks.parquet"
    )

    track = make_track()
    state = make_state()

    writer.add(
        camera_id="Showroom",
        capture_id="capture1",
        frame_id=10,
        timestamp=0.4,
        track=track,
        state=state,
        zone="Entrance",
    )

    assert len(writer.records) == 1

    record = writer.records[0]

    assert record["camera_id"] == "Showroom"
    assert record["capture_id"] == "capture1"
    assert record["frame_id"] == 10
    assert record["local_track_id"] == 1
    assert record["ground_x"] == pytest.approx(
        200.0
    )
    assert record["ground_y"] == pytest.approx(
        600.0
    )
    assert record["zone"] == "Entrance"


def test_track_writer_returns_zero_when_empty(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "tracks.parquet"
    )

    writer = TrackWriter(
        output_path
    )

    saved_count = writer.save({})

    assert saved_count == 0
    assert output_path.exists() is False


def test_track_writer_saves_parquet(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path
        / "nested"
        / "tracks.parquet"
    )

    writer = TrackWriter(
        output_path
    )

    track = make_track()
    state = make_state(
        confirmed=True,
        customer_id=1,
    )

    writer.add(
        camera_id="Showroom",
        capture_id="capture1",
        frame_id=10,
        timestamp=0.4,
        track=track,
        state=state,
        zone="Entrance",
    )

    saved_count = writer.save(
        {1: state}
    )

    assert saved_count == 1
    assert output_path.exists()

    dataframe = pd.read_parquet(
        output_path
    )

    assert len(dataframe) == 1
    assert dataframe.iloc[0][
        "customer_id"
    ] == 1


def test_customer_id_is_backfilled_after_confirmation(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "tracks.parquet"
    )

    writer = TrackWriter(
        output_path
    )

    track = make_track()

    state = make_state(
        confirmed=False,
        customer_id=None,
    )

    writer.add(
        camera_id="Showroom",
        capture_id="capture1",
        frame_id=1,
        timestamp=0.0,
        track=track,
        state=state,
        zone="Entrance",
    )

    writer.add(
        camera_id="Showroom",
        capture_id="capture1",
        frame_id=2,
        timestamp=0.04,
        track=track,
        state=state,
        zone="Entrance",
    )

    state.confirmed = True
    state.customer_id = 7

    writer.save(
        {1: state}
    )

    dataframe = pd.read_parquet(
        output_path
    )

    assert dataframe[
        "customer_id"
    ].tolist() == [
        7,
        7,
    ]


def test_unconfirmed_track_keeps_null_customer_id(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "tracks.parquet"
    )

    writer = TrackWriter(
        output_path
    )

    track = make_track(
        track_id=5
    )

    state = make_state(
        track_id=5,
        confirmed=False,
        customer_id=None,
    )

    writer.add(
        camera_id="Showroom",
        capture_id="capture1",
        frame_id=1,
        timestamp=0.0,
        track=track,
        state=state,
        zone="Demo Area",
    )

    writer.save(
        {5: state}
    )

    dataframe = pd.read_parquet(
        output_path
    )

    assert pd.isna(
        dataframe.iloc[0][
            "customer_id"
        ]
    )


def create_fake_dataset(
    root: Path,
) -> None:
    for capture_name in (
        validate_module.EXPECTED_CAPTURES
    ):
        capture_dir = (
            root / capture_name
        )

        capture_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for camera_name in (
            validate_module.EXPECTED_CAMERAS
        ):
            video_path = (
                capture_dir
                / f"{camera_name}.mp4"
            )

            video_path.write_bytes(
                b"fake-video"
            )

        gt_path = (
            root
            / f"{capture_name}_gt.mot.txt"
        )

        gt_path.write_text(
            "fake-ground-truth",
            encoding="utf-8",
        )

    map_config_dir = (
        root / "map_config"
    )

    map_config_dir.mkdir()

    for camera_name in (
        validate_module.EXPECTED_CAMERAS
    ):
        config_path = (
            map_config_dir
            / f"Node{camera_name}.json"
        )

        config_path.write_text(
            "{}",
            encoding="utf-8",
        )


def test_validate_dataset_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = (
        tmp_path / "dataset"
    )

    dataset_dir.mkdir()

    monkeypatch.setattr(
        validate_module,
        "CITIC_MTMC_DATASET_DIR",
        dataset_dir,
    )

    validate_module.validate_dataset_directory()


def test_missing_dataset_directory_raises_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = (
        tmp_path / "missing"
    )

    monkeypatch.setattr(
        validate_module,
        "CITIC_MTMC_DATASET_DIR",
        dataset_dir,
    )

    with pytest.raises(
        FileNotFoundError
    ):
        validate_module.validate_dataset_directory()


def test_validate_complete_dataset_structure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = (
        tmp_path / "dataset"
    )

    create_fake_dataset(
        dataset_dir
    )

    monkeypatch.setattr(
        validate_module,
        "CITIC_MTMC_DATASET_DIR",
        dataset_dir,
    )

    validate_module.validate_dataset_directory()
    validate_module.validate_captures()

    video_count = (
        validate_module.validate_videos()
    )

    gt_count = (
        validate_module.validate_ground_truth()
    )

    validate_module.validate_map_config()

    assert video_count == 15
    assert gt_count == 3


def test_missing_video_raises_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = (
        tmp_path / "dataset"
    )

    create_fake_dataset(
        dataset_dir
    )

    missing_video = (
        dataset_dir
        / "capture1"
        / "Showroom.mp4"
    )

    missing_video.unlink()

    monkeypatch.setattr(
        validate_module,
        "CITIC_MTMC_DATASET_DIR",
        dataset_dir,
    )

    with pytest.raises(
        FileNotFoundError
    ):
        validate_module.validate_videos()


def test_empty_video_raises_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = (
        tmp_path / "dataset"
    )

    create_fake_dataset(
        dataset_dir
    )

    empty_video = (
        dataset_dir
        / "capture1"
        / "Showroom.mp4"
    )

    empty_video.write_bytes(b"")

    monkeypatch.setattr(
        validate_module,
        "CITIC_MTMC_DATASET_DIR",
        dataset_dir,
    )

    with pytest.raises(
        ValueError
    ):
        validate_module.validate_videos()