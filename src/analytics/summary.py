import pandas as pd

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


def load_tracks() -> pd.DataFrame:
    if not TRACKS_PATH.exists():
        raise FileNotFoundError(
            f"Track data not found: "
            f"{TRACKS_PATH}"
        )

    tracks = pd.read_parquet(
        TRACKS_PATH
    )

    tracks = tracks[
        (tracks["camera_id"] == CAMERA_ID)
        & (tracks["capture_id"] == CAPTURE_ID)
        & (tracks["customer_id"].notna())
    ].copy()

    if tracks.empty:
        raise RuntimeError(
            "No confirmed customer tracks found."
        )

    tracks["customer_id"] = (
        tracks["customer_id"]
        .astype(int)
    )

    return tracks


def calculate_customer_summary(
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for customer_id, customer_data in tracks.groupby(
        "customer_id"
    ):
        customer_data = customer_data.sort_values(
            "timestamp"
        )

        first_seen = float(
            customer_data["timestamp"].min()
        )

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
                "camera_id": CAMERA_ID,
                "capture_id": CAPTURE_ID,
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
    tracks = tracks.sort_values(
        [
            "customer_id",
            "timestamp",
        ]
    ).copy()

    tracks["next_timestamp"] = (
        tracks
        .groupby("customer_id")["timestamp"]
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
        .groupby("zone")
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
                "customer_id",
                "zone",
            ]
        )["dwell_seconds"]
        .sum()
        .reset_index()
    )

    average_dwell = (
        customer_zone_dwell
        .groupby("zone")[
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
        on="zone",
        how="left",
    )

    zone_summary.insert(
        0,
        "capture_id",
        CAPTURE_ID,
    )

    zone_summary.insert(
        0,
        "camera_id",
        CAMERA_ID,
    )

    return zone_summary


def generate_summary() -> None:
    tracks = load_tracks()

    customer_summary = (
        calculate_customer_summary(
            tracks
        )
    )

    zone_summary = (
        calculate_zone_summary(
            tracks
        )
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    customer_summary.to_parquet(
        CUSTOMER_SUMMARY_PATH,
        index=False,
    )

    zone_summary.to_parquet(
        ZONE_SUMMARY_PATH,
        index=False,
    )

    print("Customer & zone summary")
    print("-----------------------")

    print(
        f"Customers: "
        f"{len(customer_summary)}"
    )

    print()

    print("Customer summary:")

    for row in customer_summary.itertuples():
        print(
            f"  Person #{row.customer_id}: "
            f"{row.store_time_seconds:.2f}s | "
            f"{row.zone_path}"
        )

    print()

    print("Zone summary:")

    for row in zone_summary.itertuples():
        print(
            f"  {row.zone}: "
            f"{row.customer_count} customers | "
            f"total {row.total_dwell_seconds:.2f}s | "
            f"avg {row.average_dwell_seconds:.2f}s"
        )

    print()

    print(
        f"Customer data: "
        f"{CUSTOMER_SUMMARY_PATH}"
    )

    print(
        f"Zone data: "
        f"{ZONE_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    generate_summary()