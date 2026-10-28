import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import build, quality_checks  # noqa: E402
from models import forecast  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixture_transactions.csv")


def test_silver_rules_and_reconciliation():
    con = build(FIXTURE)
    assert quality_checks(con)
    log = dict(con.execute("SELECT reason, row_count FROM silver_dq_log").fetchall())
    assert log == {"kept": 5, "exact duplicates removed": 1, "cancellation": 2, "order line cancelled later": 1, "no customer id": 1,
                   "non-product code (postage, fees, adjustments)": 1, "non-positive price": 1}
    assert con.execute("SELECT count(*) FROM silver_sales").fetchone()[0] == 4


def test_lapse_label_uses_only_future_window():
    con = build(FIXTURE)
    # Customer 1 bought on 1 Oct 2011, inside the 90 days after the 1 Sep snapshot: not lapsed. Customer 2 last bought in March.
    labels = dict(con.execute("SELECT customer_id, lapsed FROM gold_lapse_labels").fetchall())
    assert labels == {1: 0, 2: 1}
    f = con.execute("SELECT frequency, monetary FROM gold_lapse_features WHERE customer_id = 1").fetchone()
    assert f == (2, 34.0)  # Oct purchase excluded from features (no leakage); duplicate line counted once


def test_forecast_beats_naive_on_seasonal_series():
    weeks = pd.date_range("2009-12-07", periods=104, freq="W-MON")
    rng = np.random.default_rng(0)
    season = 100 + 60 * np.sin(2 * np.pi * np.arange(104) / 52)
    wide = pd.DataFrame({f"p{i}": season * (1 + i / 10) + rng.normal(0, 5, 104) for i in range(5)}, index=weeks)
    fc, actual, wape, _, _ = forecast(wide)
    assert set(fc) >= {"naive (last week)", "gradient boosting (global model)"}
    assert all(np.isfinite(v) for v in wape.values())
    assert wape["seasonal naive (same week last year)"] < wape["naive (last week)"]
