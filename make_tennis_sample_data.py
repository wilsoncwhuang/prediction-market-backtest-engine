"""
Generates sample data for the tennis win-probability example backtest.

ILLUSTRATIVE ONLY — this is synthetic data to demonstrate the strategy and the
Factor feed; it is not real market history and implies nothing about profit.

Run once before launching the Streamlit UI:
    python make_tennis_sample_data.py

Creates:
    data/tennis_prices.parquet
    data/tennis_resolutions.parquet
    data/tennis_factors.parquet

Model: one binary "will FAVOURITE win?" match market per day. The YES ask
drifts from an early number toward the outcome. The daily Factor is a
live-tennis win-probability signal for the YES side (0..1); on a subset of days
the signal is deliberately higher than the market-implied probability so the
TennisWinProbStrategy finds an entry edge.
"""
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

random.seed(7)
Path("data").mkdir(exist_ok=True)

SERIES  = "TENNIS"
N_DAYS  = 30
START   = datetime(2025, 3, 1, tzinfo=timezone.utc)
SNAPSHOTS_PER_MATCH = 8

# A rotating cast so each day's ticker names a plausible matchup (labels only).
MATCHUPS = [
    ("ALCARAZ", "SINNER"), ("DJOKOVIC", "MEDVEDEV"), ("ZVEREV", "RUNE"),
    ("FRITZ", "RUBLEV"), ("HURKACZ", "DIMITROV"), ("TSITSIPAS", "DE_MINAUR"),
]

price_rows, resolution_rows, factor_rows = [], [], []

for day_offset in range(N_DAYS):
    day = START + timedelta(days=day_offset)
    date_str = day.strftime("%y%b%d").upper()
    fav, dog = MATCHUPS[day_offset % len(MATCHUPS)]
    ticker = f"{SERIES}-{date_str}-{fav}_{dog}"

    # True (hidden) win probability for the favourite (YES side).
    p_true = random.uniform(0.45, 0.85)

    # Market opens near p_true (with noise) and converges toward the result.
    open_ask  = max(0.03, min(0.97, p_true + random.gauss(0, 0.06)))
    result_yes = random.random() < p_true
    final_ask = random.uniform(0.90, 0.97) if result_yes else random.uniform(0.03, 0.10)

    for snap_i in range(SNAPSHOTS_PER_MATCH):
        ts = day + timedelta(hours=13 + snap_i)          # match window
        t  = snap_i / (SNAPSHOTS_PER_MATCH - 1)
        ask = max(0.02, min(0.98, open_ask + t * (final_ask - open_ask)
                            + random.gauss(0, 0.01)))
        bid = max(0.01, ask - random.uniform(0.02, 0.05))
        price_rows.append({
            "ts":          ts,
            "ticker":      ticker,
            "bid_close":   round(bid, 4),
            "ask_close":   round(ask, 4),
            "price_close": round((bid + ask) / 2, 4),
            "volume":      random.randint(20, 400),
        })

    resolution_rows.append({
        "ticker":     ticker,
        "close_time": day + timedelta(hours=22),
        "result":     "yes" if result_yes else "no",
    })

    # Live-tennis win-prob signal for the YES side. On ~half the days we nudge it
    # a few cents above the opening implied prob so the strategy sees an edge.
    signal = p_true
    if random.random() < 0.5:
        signal = min(0.98, open_ask + random.uniform(0.06, 0.12))
    factor_rows.append({"date": day.date(), "value": round(signal, 4)})

prices_df = pd.DataFrame(price_rows)
prices_df["ts"] = pd.to_datetime(prices_df["ts"], utc=True)
prices_df.to_parquet("data/tennis_prices.parquet", index=False)
print(f"Wrote data/tennis_prices.parquet  ({len(prices_df)} rows, {N_DAYS} matches)")

res_df = pd.DataFrame(resolution_rows)
res_df["close_time"] = pd.to_datetime(res_df["close_time"], utc=True)
res_df.to_parquet("data/tennis_resolutions.parquet", index=False)
print(f"Wrote data/tennis_resolutions.parquet  ({len(res_df)} rows)")

factors_df = pd.DataFrame(factor_rows)
factors_df.to_parquet("data/tennis_factors.parquet", index=False)
print(f"Wrote data/tennis_factors.parquet  ({len(factors_df)} rows)")

print("\nDone. Run:  streamlit run backtest/app.py  (pick the 'tennis_winprob' config)")
