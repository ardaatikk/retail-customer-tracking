import pandas as pd
import pytest

from src.analytics.summary import (
    calculate_customer_summary,
    calculate_zone_summary,
)


@pytest.fixture
def sample_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": 1,
                "timestamp": 0.0,
                "zone": "Entrance",
            },
            {
                "customer_id": 1,
                "timestamp": 2.0,
                "zone": "Entrance",
            },
            {
                "customer_id": 1,
                "timestamp": 4.0,
                "zone": "Main Area",
            },
            {
                "customer_id": 1,
                "timestamp": 7.0,
                "zone": "Main Area",
            },
            {
                "customer_id": 1,
                "timestamp": 10.0,
                "zone": "Demo Area",
            },
            {
                "customer_id": 2,
                "timestamp": 20.0,
                "zone": "Entrance",
            },
            {
                "customer_id": 2,
                "timestamp": 23.0,
                "zone": "Main Area",
            },
            {
                "customer_id": 2,
                "timestamp": 26.0,
                "zone": "Main Area",
            },
            {
                "customer_id": 2,
                "timestamp": 30.0,
                "zone": "Demo Area",
            },
        ]
    )


def test_customer_summary_count(
    sample_tracks: pd.DataFrame,
) -> None:
    summary = calculate_customer_summary(
        sample_tracks
    )

    assert len(summary) == 2


def test_customer_store_time(
    sample_tracks: pd.DataFrame,
) -> None:
    summary = calculate_customer_summary(
        sample_tracks
    )

    customer_1 = summary[
        summary["customer_id"] == 1
    ].iloc[0]

    customer_2 = summary[
        summary["customer_id"] == 2
    ].iloc[0]

    assert (
        customer_1["store_time_seconds"]
        == pytest.approx(10.0)
    )

    assert (
        customer_2["store_time_seconds"]
        == pytest.approx(10.0)
    )


def test_customer_zone_path(
    sample_tracks: pd.DataFrame,
) -> None:
    summary = calculate_customer_summary(
        sample_tracks
    )

    customer_1 = summary[
        summary["customer_id"] == 1
    ].iloc[0]

    assert customer_1["zone_path"] == (
        "Entrance -> Main Area -> Demo Area"
    )


def test_consecutive_duplicate_zones_are_removed(
    sample_tracks: pd.DataFrame,
) -> None:
    summary = calculate_customer_summary(
        sample_tracks
    )

    customer_1 = summary[
        summary["customer_id"] == 1
    ].iloc[0]

    assert customer_1["zone_path"] == (
        "Entrance -> Main Area -> Demo Area"
    )

    assert customer_1["zones_visited"] == 3


def test_zone_customer_count(
    sample_tracks: pd.DataFrame,
) -> None:
    summary = calculate_zone_summary(
        sample_tracks
    )

    entrance = summary[
        summary["zone"] == "Entrance"
    ].iloc[0]

    main_area = summary[
        summary["zone"] == "Main Area"
    ].iloc[0]

    demo_area = summary[
        summary["zone"] == "Demo Area"
    ].iloc[0]

    assert entrance["customer_count"] == 2
    assert main_area["customer_count"] == 2
    assert demo_area["customer_count"] == 2


def test_zone_dwell_times(
    sample_tracks: pd.DataFrame,
) -> None:
    summary = calculate_zone_summary(
        sample_tracks
    )

    entrance = summary[
        summary["zone"] == "Entrance"
    ].iloc[0]

    main_area = summary[
        summary["zone"] == "Main Area"
    ].iloc[0]

    assert (
        entrance["total_dwell_seconds"]
        == pytest.approx(7.0)
    )

    assert (
        main_area["total_dwell_seconds"]
        == pytest.approx(13.0)
    )


def test_average_zone_dwell_time(
    sample_tracks: pd.DataFrame,
) -> None:
    summary = calculate_zone_summary(
        sample_tracks
    )

    entrance = summary[
        summary["zone"] == "Entrance"
    ].iloc[0]

    main_area = summary[
        summary["zone"] == "Main Area"
    ].iloc[0]

    assert (
        entrance["average_dwell_seconds"]
        == pytest.approx(3.5)
    )

    assert (
        main_area["average_dwell_seconds"]
        == pytest.approx(6.5)
    )


def test_zone_dwell_matches_total_observed_time(
    sample_tracks: pd.DataFrame,
) -> None:
    customer_summary = (
        calculate_customer_summary(
            sample_tracks
        )
    )

    zone_summary = (
        calculate_zone_summary(
            sample_tracks
        )
    )

    total_store_time = (
        customer_summary[
            "store_time_seconds"
        ].sum()
    )

    total_zone_dwell = (
        zone_summary[
            "total_dwell_seconds"
        ].sum()
    )

    assert total_zone_dwell == pytest.approx(
        total_store_time
    )