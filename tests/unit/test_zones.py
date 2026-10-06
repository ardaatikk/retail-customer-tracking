import json

import pytest

import src.zones.zone_manager as zone_manager_module
from src.zones.zone_manager import ZoneManager


@pytest.fixture
def zone_manager(
    tmp_path,
    monkeypatch,
) -> ZoneManager:
    config = {
        "camera_id": "TestCamera",
        "image_width": 1280,
        "image_height": 720,
        "zones": {
            "Entrance": [
                [100, 100],
                [300, 100],
                [300, 300],
                [100, 300],
            ],
            "Main Area": [
                [300, 100],
                [600, 100],
                [600, 300],
                [300, 300],
            ],
        },
    }

    config_path = (
        tmp_path
        / "TestCamera.json"
    )

    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        zone_manager_module,
        "ZONE_CONFIG_DIR",
        tmp_path,
    )

    return ZoneManager(
        "TestCamera"
    )


def test_loads_zone_configuration(
    zone_manager: ZoneManager,
) -> None:
    assert zone_manager.camera_id == "TestCamera"
    assert zone_manager.image_width == 1280
    assert zone_manager.image_height == 720

    assert set(
        zone_manager.zones.keys()
    ) == {
        "Entrance",
        "Main Area",
    }


def test_point_inside_zone(
    zone_manager: ZoneManager,
) -> None:
    zone = zone_manager.get_zone(
        200.0,
        200.0,
    )

    assert zone == "Entrance"


def test_point_inside_second_zone(
    zone_manager: ZoneManager,
) -> None:
    zone = zone_manager.get_zone(
        450.0,
        200.0,
    )

    assert zone == "Main Area"


def test_point_on_zone_boundary(
    zone_manager: ZoneManager,
) -> None:
    zone = zone_manager.get_zone(
        100.0,
        200.0,
    )

    assert zone == "Entrance"


def test_point_within_tolerance(
    zone_manager: ZoneManager,
) -> None:
    zone = zone_manager.get_zone(
        90.0,
        200.0,
        tolerance_pixels=20.0,
    )

    assert zone == "Entrance"


def test_point_outside_tolerance(
    zone_manager: ZoneManager,
) -> None:
    zone = zone_manager.get_zone(
        70.0,
        200.0,
        tolerance_pixels=20.0,
    )

    assert zone is None


def test_custom_tolerance_can_disable_nearby_match(
    zone_manager: ZoneManager,
) -> None:
    zone = zone_manager.get_zone(
        90.0,
        200.0,
        tolerance_pixels=5.0,
    )

    assert zone is None


def test_nearest_zone_is_selected(
    zone_manager: ZoneManager,
) -> None:
    zone = zone_manager.get_zone(
        620.0,
        200.0,
        tolerance_pixels=25.0,
    )

    assert zone == "Main Area"


def test_missing_configuration_raises_error(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        zone_manager_module,
        "ZONE_CONFIG_DIR",
        tmp_path,
    )

    with pytest.raises(
        FileNotFoundError
    ):
        ZoneManager(
            "MissingCamera"
        )