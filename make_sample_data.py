"""
Generates sample data for the example backtest.

Run once before launching the Streamlit UI:
    python make_sample_data.py

Creates:
    data/prices.parquet
    data/resolutions.parquet
    data/factors.parquet
"""
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

random.seed(42)
Path("data").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERIES   = "DEMO"
N_DAYS   = 30
START    = datetime(2025, 1, 1, tzinfo=timezone.utc)
TICKERS_PER_DAY = 5          # buckets available each day
SNAPSHOTS_PER_TICKER = 8     # price snapshots per ticker per day

# ---------------------------------------------------------------------------
# Generate tickers + price snapshots
# ---------------------------------------------------------------------------
price_rows = []
resolution_rows = []

for day_offset in range(N_DAYS):
    day = START + timedelta(days=day_offset)
    date_str = day.strftime("%y%b%d").upper()

    # Each day has 5 buckets (e.g. B60, B65, B70, B75, B80)
    buckets = [60 + i * 5 for i in range(TICKERS_PER_DAY)]

    for bucket in buckets:
        ticker = f"{SERIES}-{date_str}-B{bucket}"

        # All tickers resolve YES and are priced in the 35-55c range,
        # drifting up toward 88-96c by end of day.
        # This gives a clean upward equity curve to demo the framework.
        base_ask  = random.uniform(0.35, 0.55)
        final_ask = random.uniform(0.88, 0.96)

        for snap_i in range(SNAPSHOTS_PER_TICKER):
            ts = day + timedelta(hours=9 + snap_i)
            t = snap_i / (SNAPSHOTS_PER_TICKER - 1)
            target = base_ask + t * (final_ask - base_ask)
            ask = max(0.02, min(0.98, target + random.gauss(0, 0.015)))
            bid = max(0.01, ask - random.uniform(0.02, 0.05))
            price_rows.append({
                "ts":          ts,
                "ticker":      ticker,
                "bid_close":   round(bid, 4),
                "ask_close":   round(ask, 4),
                "price_close": round((bid + ask) / 2, 4),
                "volume":      random.randint(10, 300),
            })

        close_time = day + timedelta(hours=20)
        resolution_rows.append({
            "ticker":     ticker,
            "close_time": close_time,
            "result":     "yes",
        })

# ---------------------------------------------------------------------------
# Generate factor data — one value per day (e.g. a model confidence score)
# ---------------------------------------------------------------------------
factor_rows = []
for day_offset in range(N_DAYS):
    day = START + timedelta(days=day_offset)
    factor_rows.append({
        "date":  day.date(),
        "value": round(random.uniform(0.3, 1.0), 4),  # confidence score 0–1
    })

# ---------------------------------------------------------------------------
# Write parquets
# ---------------------------------------------------------------------------
prices_df = pd.DataFrame(price_rows)
prices_df["ts"] = pd.to_datetime(prices_df["ts"], utc=True)
prices_df.to_parquet("data/prices.parquet", index=False)
print(f"Wrote data/prices.parquet  ({len(prices_df)} rows, {N_DAYS} days, {TICKERS_PER_DAY} tickers/day)")

res_df = pd.DataFrame(resolution_rows)
res_df["close_time"] = pd.to_datetime(res_df["close_time"], utc=True)
res_df.to_parquet("data/resolutions.parquet", index=False)
print(f"Wrote data/resolutions.parquet  ({len(res_df)} rows)")

factors_df = pd.DataFrame(factor_rows)
factors_df.to_parquet("data/factors.parquet", index=False)
print(f"Wrote data/factors.parquet  ({len(factors_df)} rows)")

print("\nDone. Run:  streamlit run backtest/app.py")
