from pathlib import Path
import pandas as pd
from src.analytics.input import load_selected_tracks

from src.config.paths import (
    PROCESSED_DATA_DIR,
)


CAMERA_ID = "Showroom"
CAPTURE_ID = "capture1"

TRACKS_PATH = (
    PROCESSED_DATA_DIR
    / "tracks.parquet"
)

CUSTOMER_SUMMARY_PATH = (
    PROCESSED_DATA_DIR
    / "customer_summary.parquet"
)

ZONE_SUMMARY_PATH = (
    PROCESSED_DATA_DIR
    / "zone_summary.parquet"
)


def load_tracks(
    tracks_path: Path = None,
    camera_id: str = None,
    capture_id: str = None,
) -> pd.DataFrame:
    if tracks_path is None:
        return load_selected_tracks(TRACKS_PATH, camera_id or CAMERA_ID, capture_id or CAPTURE_ID)
    return load_selected_tracks(tracks_path, camera_id, capture_id)


def _scoped(tracks: pd.DataFrame) -> pd.DataFrame:
    tracks = tracks.copy()
    if "camera_id" not in tracks:
        tracks["camera_id"] = CAMERA_ID
    if "capture_id" not in tracks:
        tracks["capture_id"] = CAPTURE_ID
    return tracks


def calculate_customer_summary(
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    tracks = _scoped(tracks)
    records = []

    for (camera_id, capture_id, customer_id), customer_data in tracks.groupby(
        ["camera_id", "capture_id", "customer_id"]
    ):
        customer_data = customer_data.sort_values(
            "timestamp"
        )

        first_seen = float(
            customer_data["timestamp"].min()
        )

        if "first_seen" in customer_data and customer_data["first_seen"].notna().any():
            first_seen = min(first_seen, float(customer_data["first_seen"].min()))
        last_seen = float(
            customer_data["timestamp"].max()
        )

        store_time = (
            last_seen
            - first_seen
        )

        zones = (
            customer_data["zone"]
            .dropna()
            .astype(str)
            .tolist()
        )

        zone_history = []

        for zone in zones:
            if (
                not zone_history
                or zone_history[-1] != zone
            ):
                zone_history.append(
                    zone
                )

        records.append(
            {
                "camera_id": camera_id,
                "capture_id": capture_id,
                "customer_id": int(
                    customer_id
                ),
                "first_seen": first_seen,
                "last_seen": last_seen,
                "store_time_seconds": store_time,
                "zones_visited": len(
                    set(zones)
                ),
                "zone_path": " -> ".join(
                    zone_history
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def calculate_zone_summary(
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    tracks = _scoped(tracks)
    tracks = tracks.sort_values(
        [
            "camera_id",
            "capture_id",
            "customer_id",
            "timestamp",
        ]
    ).copy()

    tracks["next_timestamp"] = (
        tracks
        .groupby(["camera_id", "capture_id", "customer_id"])["timestamp"]
        .shift(-1)
    )

    tracks["dwell_seconds"] = (
        tracks["next_timestamp"]
        - tracks["timestamp"]
    )

    tracks["dwell_seconds"] = (
        tracks["dwell_seconds"]
        .clip(lower=0)
        .fillna(0)
    )

    zone_tracks = tracks[
        tracks["zone"].notna()
    ].copy()

    if zone_tracks.empty:
        return pd.DataFrame(
            columns=[
                "camera_id",
                "capture_id",
                "zone",
                "customer_count",
                "total_dwell_seconds",
                "average_dwell_seconds",
            ]
        )

    zone_summary = (
        zone_tracks
        .groupby(["camera_id", "capture_id", "zone"])
        .agg(
            customer_count=(
                "customer_id",
                "nunique",
            ),
            total_dwell_seconds=(
                "dwell_seconds",
                "sum",
            ),
        )
        .reset_index()
    )

    customer_zone_dwell = (
        zone_tracks
        .groupby(
            [
                "camera_id",
                "capture_id",
                "customer_id",
                "zone",
            ]
        )["dwell_seconds"]
        .sum()
        .reset_index()
    )

    average_dwell = (
        customer_zone_dwell
        .groupby(["camera_id", "capture_id", "zone"])[
            "dwell_seconds"
        ]
        .mean()
        .rename(
            "average_dwell_seconds"
        )
        .reset_index()
    )

    zone_summary = zone_summary.merge(
        average_dwell,
        on=["camera_id", "capture_id", "zone"],
        how="left",
    )

    return zone_summary


def generate_summary(
    tracks: pd.DataFrame = None,
    output_dir: Path = None,
) -> None:
    tracks = load_tracks() if tracks is None else tracks
    customer_summary = calculate_customer_summary(tracks)
    zone_summary = calculate_zone_summary(tracks)
    customer_path = CUSTOMER_SUMMARY_PATH if output_dir is None else Path(output_dir) / "customer_summary.parquet"
    zone_path = ZONE_SUMMARY_PATH if output_dir is None else Path(output_dir) / "zone_summary.parquet"
    customer_path.parent.mkdir(parents=True, exist_ok=True)
    customer_summary.to_parquet(customer_path, index=False)
    print("Confirmed track summary")
    print(f"Confirmed tracks: {len(customer_summary)}")
    for row in customer_summary.itertuples():
        print(f"  Person #{row.customer_id}: {row.store_time_seconds:.2f}s | {row.zone_path}")
    print(f"Track summary: {customer_path}")
    if zone_summary.empty:
        print("Zone summary skipped: selected tracks have no zone assignments")
    else:
        zone_path.parent.mkdir(parents=True, exist_ok=True)
        zone_summary.to_parquet(zone_path, index=False)
        print(f"Zone summary: {zone_path}")
        for row in zone_summary.itertuples():
            print(f"  {row.zone}: {row.customer_count} tracks | total {row.total_dwell_seconds:.2f}s")


if __name__ == "__main__":
    generate_summary()