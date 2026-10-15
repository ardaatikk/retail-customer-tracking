import pandas as pd
import pytest
from src.analytics.summary import export_session_summary


@pytest.mark.parametrize("zone", [None, "Entrance"])
def test_summary_has_one_row_per_customer_and_optional_zones(tmp_path, zone):
    path = tmp_path / "confirmed.parquet"
    pd.DataFrame([
        dict(camera_id="live1", capture_id="run", customer_id=customer,
             timestamp=t, first_seen=0., zone=zone)
        for customer in [1, 2] for t in [1., 2., 3.]
    ]).to_parquet(path, index=False)
    export_session_summary(path)
    summary = pd.read_parquet(tmp_path / "analytics/customer_summary.parquet")
    assert summary.customer_id.tolist() == [1, 2]
    assert summary.store_time_seconds.tolist() == [3., 3.]
    assert (tmp_path / "analytics/zone_summary.parquet").exists() == (zone is not None)
    assert len(pd.read_parquet(path)) == 6


def test_empty_session_has_readable_empty_customer_summary(tmp_path):
    path = tmp_path / "confirmed.parquet"
    pd.DataFrame(columns=["camera_id", "capture_id", "customer_id", "timestamp", "zone"]).to_parquet(path, index=False)
    export_session_summary(path)
    result = pd.read_parquet(tmp_path / "analytics/customer_summary.parquet")
    assert result.empty
    assert "customer_id" in result.columns
    assert "store_time_seconds" in result.columns
