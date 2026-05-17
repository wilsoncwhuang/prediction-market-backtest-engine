from __future__ import annotations

import numpy as np
import pandas as pd
import pytz

from core.portfolio import EquityPoint, Portfolio

_ET = pytz.timezone("America/New_York")

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def equity_to_df(equity_curve: list[EquityPoint], scale: float = 100) -> pd.DataFrame:
    """Convert equity curve to a USD-denominated DataFrame indexed by timestamp.

    scale=100  for Kalshi (internal cents → dollars)
    scale=1    for OHLCV  (already in dollar units)
    """
    if not equity_curve:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "timestamp":      p.timestamp,
        "equity":         p.equity / scale,
        "cash":           p.cash / scale,
        "position_value": p.position_value / scale,
        "realized_pnl":   p.realized_pnl / scale,
        "unrealized_pnl": p.unrealized_pnl / scale,
    } for p in equity_curve])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(_ET)
    return df.set_index("timestamp")


def trade_log_to_df(trade_log: list[dict], scale: float = 100) -> pd.DataFrame:
    """Convert portfolio trade_log to a DataFrame with USD columns."""
    if not trade_log:
        return pd.DataFrame()
    df = pd.DataFrame(trade_log)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(_ET)
    df["fill_price_usd"]   = df["fill_price"]   / scale
    df["fees_usd"]         = df["fees"]         / scale
    df["realized_pnl_usd"] = df["realized_pnl"] / scale
    return df


def trade_log_to_round_trips(trade_log: list[dict], scale: float = 100) -> pd.DataFrame:
    """
    Pair BUY fills with their corresponding exit (SELL, settlement, or expiry).
    Each row represents one complete round trip: entry → exit.

    Columns: entry_time, exit_time, ticker, side, qty,
             entry_price_c, exit_price_c, pnl_usd, cum_pnl_usd, exit_type

    scale=100 for Kalshi, scale=1 for OHLCV.
    """
    if not trade_log:
        return pd.DataFrame()

    _FIXED = {"timestamp", "ticker", "side", "action", "quantity",
              "fill_price", "fees", "realized_pnl"}

    def _meta(row: dict) -> dict:
        return {k: v for k, v in row.items() if k not in _FIXED}

    entries: dict[tuple[str, str], list[dict]] = {}   # (ticker, side) → FIFO buy queue
    trips = []

    for row in sorted(trade_log, key=lambda r: r["timestamp"]):
        key    = (row["ticker"], row["side"])
        action = row["action"]

        if action == "buy":
            entries.setdefault(key, []).append(row)

        elif action == "sell":
            queue = entries.get(key, [])
            if queue:
                entry = queue.pop(0)
                trips.append({
                    "entry_time":    entry["timestamp"],
                    "exit_time":     row["timestamp"],
                    "ticker":        entry["ticker"],
                    "side":          entry["side"],
                    "qty":           entry["quantity"],
                    "entry_price_c": entry["fill_price"],
                    "exit_price_c":  row["fill_price"],
                    "fees_usd":      entry["fees"] / scale,
                    "pnl_usd":       row["realized_pnl"] / scale,
                    "exit_type":     "sell",
                    **_meta(entry),
                })
            else:
                entries.setdefault(key, []).append(row)

        elif action in ("settlement", "expiry"):
            queue = entries.pop(key, [])
            for entry in queue:
                trips.append({
                    "entry_time":    entry["timestamp"],
                    "exit_time":     row["timestamp"],
                    "ticker":        entry["ticker"],
                    "side":          entry["side"],
                    "qty":           entry["quantity"],
                    "entry_price_c": entry["fill_price"],
                    "exit_price_c":  row["fill_price"],
                    "fees_usd":      entry["fees"] / scale,
                    "pnl_usd":       row["realized_pnl"] / scale,
                    "exit_type":     action,
                    **_meta(entry),
                })

    # Unmatched entries (still open)
    for queue in entries.values():
        for entry in queue:
            trips.append({
                "entry_time":    entry["timestamp"],
                "exit_time":     None,
                "ticker":        entry["ticker"],
                "side":          entry["side"],
                "qty":           entry["quantity"],
                "entry_price_c": entry["fill_price"],
                "exit_price_c":  None,
                "fees_usd":      entry["fees"] / scale,
                "pnl_usd":       None,
                "exit_type":     "open",
            })

    if not trips:
        return pd.DataFrame()

    df = pd.DataFrame(trips)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert(_ET)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True).dt.tz_convert(_ET)
    df["cum_pnl_usd"] = df["pnl_usd"].cumsum()
    return df


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Peak-to-trough drawdown as a percentage (always <= 0)."""
    peak = equity.cummax()
    return (equity - peak) / peak * 100


def _daily_returns(equity_df: pd.DataFrame) -> pd.Series:
    daily = equity_df["equity"].resample("D").last().dropna()
    return daily.pct_change().dropna()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(portfolio: Portfolio, scale: float = 100) -> dict:
    """
    Compute a full set of performance metrics from a completed Portfolio.

    All monetary values returned in USD; percentages as floats (e.g. -5.2 for -5.2%).
    NaN is returned for metrics that cannot be computed (e.g. no trades, zero variance).

    scale=100 for Kalshi (cents → dollars), scale=1 for OHLCV.
    """
    eq_df = equity_to_df(portfolio.equity_curve, scale=scale)
    if eq_df.empty:
        return {}

    equity        = eq_df["equity"]
    initial_cash  = portfolio.initial_cash / scale
    final_equity  = equity.iloc[-1]

    # ---- Return ----
    total_return = (final_equity - initial_cash) / initial_cash * 100

    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    # Only annualize when there's enough data and ending equity is positive;
    # blown-up runs (final ≤ 0) and short periods produce nonsense via compounding.
    if n_days >= 365 and final_equity > 0:
        ann_return = ((final_equity / initial_cash) ** (365 / n_days) - 1) * 100
    else:
        ann_return = float("nan")

    # ---- Risk-adjusted ----
    daily_rets = _daily_returns(eq_df)

    def _sharpe(rets: pd.Series) -> float:
        return rets.mean() / rets.std() * np.sqrt(TRADING_DAYS) \
            if len(rets) > 1 and rets.std() > 0 else float("nan")

    sharpe = _sharpe(daily_rets)

    downside = daily_rets[daily_rets < 0]
    sortino = daily_rets.mean() / downside.std() * np.sqrt(TRADING_DAYS) \
        if len(downside) > 1 and downside.std() > 0 else float("nan")

    # ---- Drawdown ----
    dd        = compute_drawdown(equity)
    max_dd    = dd.min()

    # Duration: longest consecutive period below high-water mark (in periods)
    peak      = equity.cummax()
    underwater = (equity < peak).astype(int)
    # Find longest run of 1s
    groups = underwater.ne(underwater.shift()).cumsum()
    max_dd_periods = int(
        underwater[underwater == 1].groupby(groups[underwater == 1]).count().max()
    ) if underwater.any() else 0

    calmar = (ann_return / abs(max_dd)) \
        if (not np.isnan(ann_return) and max_dd < 0) else float("nan")

    # ---- Trade stats ----
    trade_df = trade_log_to_df(portfolio.trade_log, scale=scale)
    exits = trade_df[trade_df["action"].isin(["sell", "settlement", "expiry"])] if not trade_df.empty else pd.DataFrame()

    if not exits.empty:
        pnl       = exits["realized_pnl_usd"]
        wins      = pnl[pnl > 0]
        losses    = pnl[pnl < 0]
        win_rate  = len(wins) / len(exits) * 100
        profit_factor = wins.sum() / abs(losses.sum()) \
            if not losses.empty and losses.sum() != 0 else float("nan")
        avg_win   = wins.mean()  if not wins.empty   else float("nan")
        avg_loss  = losses.mean() if not losses.empty else float("nan")
        best      = pnl.max()
        worst     = pnl.min()
        n_trades  = len(exits)
    else:
        win_rate = profit_factor = avg_win = avg_loss = best = worst = float("nan")
        n_trades = 0

    total_volume = int(trade_df["quantity"].sum()) if not trade_df.empty else 0

    return {
        "total_return":   total_return,
        "ann_return":     ann_return,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "max_drawdown":   max_dd,
        "max_dd_periods": max_dd_periods,
        "calmar":         calmar,
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "best_trade":     best,
        "worst_trade":    worst,
        "total_trades":   n_trades,
        "total_volume":   total_volume,
    }
