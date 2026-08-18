"""
Prediction Market Backtest Report
Run with:  streamlit run backtest/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, ".")

from backtest.broker import BrokerConfig, PredictionSimulatedBroker
from backtest.engine import BacktestEngine
from core.event_bus import EventBus
from backtest.feeds.generic_ohlcv_feed import GenericOHLCVFeed
from backtest.feeds.factor_feed import FactorFeed
from backtest.feeds.prediction_market_price_feed import PredictionMarketPriceFeed
from core.constants import Side
from backtest.feeds.prediction_market_resolution_feed import PredictionMarketResolutionFeed
from core.portfolio import Portfolio
from backtest.reporting.metrics import (
    compute_drawdown,
    compute_metrics,
    equity_to_df,
    trade_log_to_df,
    trade_log_to_round_trips,
)
from strategies.example.example_strategy import ExampleStrategy
from strategies.tennis_winprob.strategy import TennisWinProbStrategy

# ---------------------------------------------------------------------------
# Strategy registry — add your strategies here
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY = {
    "ExampleStrategy": ExampleStrategy,
    "TennisWinProbStrategy": TennisWinProbStrategy,
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Backtest Report",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS — terminal / trading-desk aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

/* ---- root — override Streamlit primary color ---- */
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
:root {
    --primary-color: #4a6fa8 !important;
    --primary-color-light: #1e2d40 !important;
}

/* ---- hide streamlit chrome ---- */
#MainMenu, footer, [data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { display: none; }

/* ---- main background ---- */
.stApp { background: #13131e !important; }
.main .block-container { padding-top: 1.5rem; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: #0d0d17 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }
[data-testid="stSidebar"] h1 { color: #b0b8c8 !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.12em; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #888899 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* ---- dividers ---- */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

/* ---- popover buttons (filter bar) ---- */
[data-testid="stPopover"] button,
[data-testid="stPopover"] [data-testid="stBaseButton-secondary"] {
    background: #1d1d2b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
    color: #c8c8d8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stPopover"] button:hover,
[data-testid="stPopover"] button:focus,
[data-testid="stPopover"] button:active,
[data-testid="stPopover"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stPopover"] [data-testid="stBaseButton-secondary"]:focus,
[data-testid="stPopover"] [data-testid="stBaseButton-secondary"]:active {
    background: #1e2d40 !important;
    border-color: #4a6fa8 !important;
    color: #a8c4e0 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ---- sidebar buttons (Run Backtest, expander, etc.) ---- */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #1e2a3a !important;
    border: 1px solid #2d3f55 !important;
    color: #a8bdd4 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: background 0.15s ease, border-color 0.15s ease;
}
[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    background: #253448 !important;
    border-color: #388bfd !important;
    color: #e8eef8 !important;
}

/* ---- main area buttons (Clear filters, export, etc.) ---- */
[data-testid="stMainBlockContainer"] button,
[data-testid="stMainBlockContainer"] [data-testid="stBaseButton-secondary"] {
    background: #1e2a3a !important;
    border: 1px solid #2d3f55 !important;
    color: #a8bdd4 !important;
    border-radius: 6px !important;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stMainBlockContainer"] button:hover,
[data-testid="stMainBlockContainer"] [data-testid="stBaseButton-secondary"]:hover {
    background: #253448 !important;
    border-color: #4a6fa8 !important;
    color: #a8c4e0 !important;
    box-shadow: none !important;
}

/* ---- checkboxes — override Streamlit primary color ---- */
[data-baseweb="checkbox"] [role="checkbox"] {
    background-color: transparent !important;
    border-color: #4a6fa8 !important;
}
[data-baseweb="checkbox"] [role="checkbox"][aria-checked="true"],
[data-baseweb="checkbox"] [role="checkbox"][aria-checked="mixed"] {
    background-color: #1e2d40 !important;
    border-color: #4a6fa8 !important;
}
[data-baseweb="checkbox"] [role="checkbox"] svg { fill: #a8c4e0 !important; }
[data-baseweb="checkbox"] [role="checkbox"]:focus { outline-color: #4a6fa8 !important; box-shadow: 0 0 0 3px rgba(74,111,168,0.3) !important; }

/* ---- radio buttons — override Streamlit primary color ---- */
[data-baseweb="radio"] [role="radio"] {
    border-color: #4a6fa8 !important;
    background-color: transparent !important;
}
[data-baseweb="radio"] [role="radio"][aria-checked="true"] {
    border-color: #4a6fa8 !important;
    background-color: #1e2d40 !important;
}
[data-baseweb="radio"] [role="radio"][aria-checked="true"]::before {
    background-color: #4a6fa8 !important;
}
[data-baseweb="radio"] [role="radio"]:focus { box-shadow: 0 0 0 3px rgba(74,111,168,0.3) !important; }

/* ---- multiselect option checkmarks ---- */
[role="option"] svg, [role="option"] [data-baseweb="icon"] { color: #4a6fa8 !important; fill: #4a6fa8 !important; }

/* ---- dataframe ---- */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}

/* ---- multiselect tags (chips) — global, Streamlit portals these outside popover ---- */
[data-baseweb="tag"] {
    background-color: #1e2d40 !important;
    border-color: #4a6fa8 !important;
}
[data-baseweb="tag"] span {
    color: #a8c4e0 !important;
}
[data-baseweb="tag"] button,
[data-baseweb="tag"] [role="presentation"] {
    color: #6a90b8 !important;
}
[data-baseweb="tag"]:hover {
    background-color: #253448 !important;
}

/* ---- multiselect inside popovers: dropdown list ---- */
[data-testid="stPopover"] [data-baseweb="menu"],
[data-testid="stPopover"] [data-baseweb="popover"] {
    background: #1a2030 !important;
    border: 1px solid #2d3f55 !important;
}
[data-testid="stPopover"] [role="option"]:hover,
[data-testid="stPopover"] [role="option"][aria-selected="true"] {
    background: #1e2d40 !important;
    color: #a8c4e0 !important;
}
[data-testid="stPopover"] [role="option"] {
    color: #c8c8d8 !important;
}
/* selected tags / chips */
[data-testid="stPopover"] [data-baseweb="tag"] {
    background: #1e2d40 !important;
    border-color: #4a6fa8 !important;
    color: #a8c4e0 !important;
}
/* checkmark / tick color */
[data-testid="stPopover"] svg { fill: #a8c4e0 !important; }
/* date input inside popover */
[data-testid="stPopover"] [data-baseweb="calendar"] {
    background: #1a2030 !important;
}
[data-testid="stPopover"] [data-baseweb="calendar"] button:hover {
    background: #1e2d40 !important;
    color: #a8c4e0 !important;
}
[data-testid="stPopover"] [data-baseweb="calendar"] [aria-selected="true"] {
    background: #2d4a6a !important;
}

/* ---- inputs ---- */
[data-baseweb="input"] input, [data-baseweb="select"] {
    background: #1d1d2b !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #e8e8f0 !important;
}

/* ---- caption ---- */
[data-testid="stCaptionContainer"] { color: #888899 !important; }

/* ---- general text ---- */
p, label, .stMarkdown { color: #c8c8d8; }

/* ---- custom dark tables ---- */
.bt-table-wrap {
    overflow-x: auto; overflow-y: auto; max-height: 500px;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
}
.bt-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.78rem; color: #8b949e;
}
.bt-table thead th {
    position: sticky; top: 0; z-index: 1;
    background: #0d0d17;
    color: #555566;
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 8px 12px; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    white-space: nowrap;
}
.bt-table tbody td {
    padding: 6px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    white-space: nowrap; font-family: 'JetBrains Mono', monospace;
}
.bt-table tbody tr:hover td { background: rgba(255,255,255,0.025); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def _load_strategy_configs() -> dict[str, dict]:
    """Scan strategies/*/backtest.yaml and return {folder_name: config_dict}."""
    strategies_dir = Path("strategies")
    if not strategies_dir.exists():
        return {}
    return {
        f.parent.name: yaml.safe_load(f.read_text(encoding="utf-8"))
        for f in sorted(strategies_dir.glob("*/backtest.yaml"))
    }


@st.cache_data
def _load_price_df(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from all datetime columns (required for Excel export)."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    return df

def _fmt_pct(v: float) -> str:
    return f"{v:+.2f}%" if not np.isnan(v) else "—"

def _fmt_ratio(v: float) -> str:
    return f"{v:.2f}" if not np.isnan(v) else "—"

def _fmt_usd(v: float) -> str:
    return f"${v:+.2f}" if not np.isnan(v) else "—"


# ---- Visual helpers --------------------------------------------------------

BLUE   = "#388bfd"
GREEN  = "#4ade80"
RED    = "#f87171"
GRAY   = "rgba(200,200,216,0.25)"
CHART_MARGIN = dict(t=10, b=40, l=60, r=20)

_GRID  = "rgba(48,54,61,0.8)"
_LINE  = "rgba(48,54,61,0.9)"
_PAPER = "rgba(0,0,0,0)"

def _chart_layout(**kwargs) -> dict:
    """Base plotly layout for all charts — dark, transparent, subtle grid."""
    base = dict(
        template="plotly_dark",
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#888899"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", tickcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", tickcolor="rgba(255,255,255,0.08)", zeroline=False),
        margin=CHART_MARGIN,
    )
    base.update(kwargs)
    return base


def _section(title: str, subtitle: str = "") -> None:
    """Render a styled section header."""
    sub_html = f'<span style="font-size:0.9rem;font-weight:400;opacity:0.45;margin-left:10px">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
    <div style="
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #e8e8f0;
        border-bottom: 2px solid rgba(232,232,240,0.15);
        padding-bottom: 8px;
        margin: 28px 0 16px 0;
    ">{title}{sub_html}</div>
    """, unsafe_allow_html=True)


def _render_table(df: pd.DataFrame, pnl_cols: tuple = ()) -> None:
    """Render a DataFrame as a fully-styled dark HTML table."""
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col, val in zip(df.columns, row):
            if col in pnl_cols:
                try:
                    fval = float(val)
                    color = GREEN if fval > 0 else (RED if fval < 0 else GRAY)
                    cells += f'<td style="color:{color}">{val:+.3f}</td>'
                except (TypeError, ValueError):
                    cells += f"<td>{val}</td>"
            else:
                cells += f"<td>{val if val is not None else '—'}</td>"
        rows_html += f"<tr>{cells}</tr>"
    st.markdown(
        f'<div class="bt-table-wrap"><table class="bt-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


def _kpi(label: str, value: str, accent: str = BLUE) -> str:
    """Return HTML for a single KPI card."""
    return f"""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-left: 3px solid {accent};
        border-radius: 8px;
        padding: 14px 18px;
        min-height: 76px;
    ">
        <div style="
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            opacity: 0.55;
            margin-bottom: 8px;
        ">{label}</div>
        <div style="
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 1.4rem;
            font-weight: 600;
            line-height: 1.2;
        ">{value}</div>
    </div>"""


def _accent_for(v: float, positive_good: bool = True) -> str:
    if np.isnan(v):
        return GRAY
    if positive_good:
        return GREEN if v > 0 else (RED if v < 0 else GRAY)
    return RED if v > 0 else (GREEN if v < 0 else GRAY)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
strategy_configs = _load_strategy_configs()

with st.sidebar:
    st.title("⬡  Backtest Config")

    st.subheader("Strategy")
    if not strategy_configs:
        st.error("No config files found in strategies/*/backtest.yaml")
        st.stop()

    display_names = {
        stem: cfg.get("strategy", stem)
        for stem, cfg in strategy_configs.items()
    }
    selected_stem = st.selectbox(
        "Strategy",
        list(display_names.keys()),
        format_func=lambda s: display_names[s],
    )
    selected_cfg = strategy_configs[selected_stem]

    with st.expander("Config details"):
        st.code(yaml.dump(selected_cfg, default_flow_style=False, allow_unicode=True), language="yaml")

    st.divider()

    st.subheader("Portfolio")
    _default_cash = 10_000.0 if selected_cfg.get("instrument") == "ohlcv" else 100.0
    initial_cash = st.number_input(
        "Initial cash ($)", value=_default_cash, min_value=1.0, step=100.0, format="%.2f"
    )

    st.subheader("Broker")
    _is_pm = selected_cfg.get("instrument", "prediction_market") != "ohlcv"
    if _is_pm:
        st.caption("Fee: taker formula  `rate × C × P × (1−P)`  (default rate = 7%)")
        fee = 0.0
    else:
        fee = st.number_input("Fee per contract (¢)", value=0.0, min_value=0.0, step=0.5)
    slippage      = st.number_input("Slippage (¢)",         value=0.0, min_value=0.0, step=0.5)
    fill_at_limit = st.checkbox("Fill at limit price", value=True)

    st.divider()
    st.subheader("Date Range")
    col_s, col_e = st.columns(2)
    with col_s:
        bt_start = st.date_input("Start", value=None, key="bt_start")
    with col_e:
        bt_end   = st.date_input("End",   value=None, key="bt_end")

    st.divider()

    st.subheader("Benchmark")
    enable_bench = st.checkbox("Enable benchmark", value=False, key="enable_bench")
    if enable_bench:
        bench_stem = st.selectbox(
            "Benchmark strategy",
            list(display_names.keys()),
            format_func=lambda s: display_names[s],
            index=list(display_names.keys()).index(selected_stem),
            key="bench_stem",
        )
        bench_cfg = strategy_configs[bench_stem]
        with st.expander("Benchmark config"):
            st.code(yaml.dump(bench_cfg, default_flow_style=False, allow_unicode=True), language="yaml")

    st.divider()
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Helpers — build engine from a config dict
# ---------------------------------------------------------------------------
def _build_feeds(cfg: dict):
    instrument   = cfg.get("instrument", "prediction_market")
    price_cfgs   = cfg.get("price", [])
    if instrument == "ohlcv":
        return [
            GenericOHLCVFeed(fc["path"], fc["ticker"])
            for fc in price_cfgs
        ]
    no_side = cfg.get("trade_no_side", False)
    return (
        [PredictionMarketPriceFeed(fc["path"], side=Side.YES) for fc in price_cfgs]
        + ([PredictionMarketPriceFeed(fc["path"], side=Side.NO) for fc in price_cfgs] if no_side else [])
        + [PredictionMarketResolutionFeed(fc["path"]) for fc in cfg.get("resolutions", [])]
        + [FactorFeed(fc["series"], fc["pred_path"], value_col=fc.get("value_col", "value"))
           for fc in cfg.get("factors", [])]
    )


def _build_engine(cfg: dict, cash: float, fee: float, slippage: float, fill_at_limit: bool,
                  start=None, end=None):
    instrument = cfg.get("instrument", "prediction_market")
    is_ohlcv   = instrument == "ohlcv"

    excluded = {"strategy", "instrument", "price", "resolutions", "factors"}
    strategy_cls    = cfg["strategy"]
    strategy_params = {k: v for k, v in cfg.items() if k not in excluded}
    strategy = STRATEGY_REGISTRY[strategy_cls](**strategy_params)

    broker_cfg = BrokerConfig(
        fee_per_contract=fee,
        slippage_cents=slippage,
        fill_at_limit=fill_at_limit,
        price_floor=0.0            if is_ohlcv else 1.0,
        price_ceiling=float("inf") if is_ohlcv else 99.0,
        use_platform_fee_formula=not is_ohlcv,
    )
    portfolio_cash = cash if is_ohlcv else cash * 100
    portfolio = Portfolio(initial_cash=portfolio_cash)
    strategy.set_equity_callback(lambda: portfolio.equity)

    engine = BacktestEngine(
        bus=EventBus(),
        broker=PredictionSimulatedBroker(config=broker_cfg),
        portfolio=portfolio,
        strategy=strategy,
        feeds=_build_feeds(cfg),
    )
    return engine, strategy_params, start, end


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------
if run_btn:
    strategy_class_name = selected_cfg.get("strategy")
    if strategy_class_name not in STRATEGY_REGISTRY:
        st.error(f"Unknown strategy: '{strategy_class_name}'. Add it to STRATEGY_REGISTRY in app.py.")
        st.stop()

    price_configs = selected_cfg.get("price", [])
    if not price_configs:
        st.error("No price feeds defined in config. Add a 'price' list with at least one path.")
        st.stop()

    # Clear previous results so old data doesn't show during computation
    for _k in ["portfolio", "bench_portfolio", "bench_name", "events_processed", "run_config"]:
        st.session_state.pop(_k, None)

    with st.spinner("Running backtest…"):
        try:
            from datetime import datetime
            def _to_dt(d):
                return datetime(d.year, d.month, d.day) if d else None

            run_start = _to_dt(bt_start)
            run_end   = _to_dt(bt_end)

            engine, strategy_params, *_ = _build_engine(
                selected_cfg, initial_cash, fee, slippage, fill_at_limit,
                start=run_start, end=run_end,
            )
            engine.run(start=run_start, end=run_end)
            st.session_state["portfolio"]        = engine.portfolio
            st.session_state["events_processed"] = engine.bus.processed
            st.session_state["run_config"] = {
                "strategy_name":   display_names[selected_stem],
                "strategy_params": strategy_params,
                "feeds":           price_configs,
                "initial_cash":    initial_cash,
                "instrument":      selected_cfg.get("instrument", "prediction_market"),
            }

            # Run benchmark if enabled
            if enable_bench:
                bench_engine, *_ = _build_engine(
                    bench_cfg, initial_cash, fee, slippage, fill_at_limit,
                    start=run_start, end=run_end,
                )
                bench_engine.run(start=run_start, end=run_end)
                st.session_state["bench_portfolio"] = bench_engine.portfolio
                st.session_state["bench_name"]      = display_names[bench_stem]
            else:
                st.session_state.pop("bench_portfolio", None)
                st.session_state.pop("bench_name", None)

        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

if "portfolio" not in st.session_state:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 80px 0;
        color: #2a4a6a;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
    ">
        Configure parameters in the sidebar and click <strong style="color:#388bfd">▶ Run Backtest</strong>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
pf       = st.session_state["portfolio"]
cfg      = st.session_state["run_config"]
scale    = 1 if cfg.get("instrument") == "ohlcv" else 100
p_unit   = "$" if cfg.get("instrument") == "ohlcv" else "¢"

metrics  = compute_metrics(pf, scale=scale)
eq_df    = equity_to_df(pf.equity_curve, scale=scale)
trade_df = trade_log_to_df(pf.trade_log, scale=scale)
trips_df = trade_log_to_round_trips(pf.trade_log, scale=scale)

bench_pf      = st.session_state.get("bench_portfolio")
bench_name    = st.session_state.get("bench_name", "Benchmark")
bench_eq_df   = equity_to_df(bench_pf.equity_curve, scale=scale) if bench_pf else None
bench_metrics = compute_metrics(bench_pf, scale=scale) if bench_pf else None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
params_str = "  ·  ".join(f"{k}={v}" for k, v in cfg["strategy_params"].items())
feeds_str  = "  ·  ".join(f["path"] for f in cfg["feeds"])
st.markdown(f"""
<div style="border-bottom: 1px solid rgba(56,139,253,0.15); padding-bottom: 16px; margin-bottom: 4px;">
    <div style="
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6edf3;
        letter-spacing: -0.01em;
    ">{cfg['strategy_name']}</div>
    <div style="
        margin-top: 6px;
        font-size: 0.72rem;
        color: #8b949e;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
    ">{params_str}  ·  {feeds_str}  ·  ${cfg['initial_cash']:,.2f} initial  ·  {st.session_state['events_processed']:,} events</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI cards — row 1: returns & risk-adjusted
# ---------------------------------------------------------------------------
_section("Performance")

tr   = metrics.get("total_return",  float("nan"))
ar   = metrics.get("ann_return",    float("nan"))
sh   = metrics.get("sharpe",        float("nan"))
so   = metrics.get("sortino",       float("nan"))

r1 = st.columns(4)
for col, (label, val, accent) in zip(r1, [
    ("Total Return",  _fmt_pct(tr),   _accent_for(tr)),
    ("Ann. Return",   _fmt_pct(ar),   _accent_for(ar)),
    ("Sharpe Ratio",  _fmt_ratio(sh), GREEN if (not np.isnan(sh) and sh > 1) else (BLUE if (not np.isnan(sh) and sh > 0) else RED)),
    ("Sortino Ratio", _fmt_ratio(so), GREEN if (not np.isnan(so) and so > 1) else (BLUE if (not np.isnan(so) and so > 0) else RED)),

]):
    col.markdown(_kpi(label, val, accent), unsafe_allow_html=True)

st.write("")

# row 2: drawdown & trade stats
md  = metrics.get("max_drawdown",   float("nan"))
cal = metrics.get("calmar",         float("nan"))
wr  = metrics.get("win_rate",       float("nan"))
pf_ = metrics.get("profit_factor",  float("nan"))

r2 = st.columns(4)
for col, (label, val, accent) in zip(r2, [
    ("Max Drawdown",  _fmt_pct(md),    RED if (not np.isnan(md) and md < 0) else GRAY),
    ("Calmar Ratio",  _fmt_ratio(cal), _accent_for(cal)),
    ("Win Rate",      f"{wr:.2f}%" if not np.isnan(wr) else "—", GREEN if (not np.isnan(wr) and wr >= 50) else RED),
    ("Profit Factor", _fmt_ratio(pf_), GREEN if (not np.isnan(pf_) and pf_ > 1) else RED),
]):
    col.markdown(_kpi(label, val, accent), unsafe_allow_html=True)

st.write("")

# row 3: per-trade stats
r3 = st.columns(4)
for col, (label, val, accent) in zip(r3, [
    ("Avg Win",     _fmt_usd(metrics.get("avg_win",     float("nan"))), GREEN),
    ("Avg Loss",    _fmt_usd(metrics.get("avg_loss",    float("nan"))), RED),
    ("Best Trade",  _fmt_usd(metrics.get("best_trade",  float("nan"))), GREEN),
    ("Worst Trade", _fmt_usd(metrics.get("worst_trade", float("nan"))), RED),
]):
    col.markdown(_kpi(label, val, accent), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# vs Benchmark metrics
# ---------------------------------------------------------------------------
if bench_metrics is not None:
    _section("vs Benchmark", subtitle=bench_name)

    main_tr  = metrics.get("total_return", float("nan"))
    bench_tr = bench_metrics.get("total_return", float("nan"))
    excess   = main_tr - bench_tr if not (np.isnan(main_tr) or np.isnan(bench_tr)) else float("nan")

    main_sh  = metrics.get("sharpe", float("nan"))
    bench_sh = bench_metrics.get("sharpe", float("nan"))
    sharpe_diff = main_sh - bench_sh if not (np.isnan(main_sh) or np.isnan(bench_sh)) else float("nan")

    # Daily returns for correlation & tracking error
    main_daily  = eq_df["equity"].resample("D").last().pct_change().dropna()
    bench_daily = bench_eq_df["equity"].resample("D").last().pct_change().dropna()
    aligned     = pd.concat([main_daily, bench_daily], axis=1, join="inner").dropna()
    if len(aligned) > 1:
        correlation    = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        tracking_error = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).std() * np.sqrt(252) * 100)
    else:
        correlation = tracking_error = float("nan")

    rb = st.columns(4)
    for col, (label, val, accent) in zip(rb, [
        ("Excess Return",   _fmt_pct(excess),                   _accent_for(excess)),
        ("Sharpe Δ",        _fmt_ratio(sharpe_diff),             _accent_for(sharpe_diff)),
        ("Correlation",     _fmt_ratio(correlation),             BLUE),
        ("Tracking Error",  f"{tracking_error:.2f}%" if not np.isnan(tracking_error) else "—", GRAY),
    ]):
        col.markdown(_kpi(label, val, accent), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------
_section("Equity Curve")

fig_eq = go.Figure()
fig_eq.add_hline(
    y=cfg["initial_cash"],
    line=dict(color=GRAY, dash="dash", width=1),
    annotation_text="initial",
    annotation_font=dict(size=10, color="#4a6b8a"),
    annotation_position="top right",
)
fig_eq.add_trace(go.Scatter(
    x=eq_df.index, y=eq_df["equity"],
    name=cfg["strategy_name"],
    line=dict(color="#4ade80", width=2),
    fill="tozeroy",
    fillcolor="rgba(74,222,128,0.07)",
    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
))
if bench_eq_df is not None:
    fig_eq.add_trace(go.Scatter(
        x=bench_eq_df.index, y=bench_eq_df["equity"],
        name=bench_name,
        line=dict(color="#64748b", width=1.5, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra>Benchmark</extra>",
    ))
fig_eq.update_layout(**_chart_layout(
    xaxis_title="Date",
    yaxis=dict(title="USD", tickprefix="$", gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False),
    hovermode="x unified",
    height=340,
    showlegend=bench_eq_df is not None,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
))
st.plotly_chart(fig_eq, use_container_width=True)

# ---------------------------------------------------------------------------
# Drawdown  |  PnL components
# ---------------------------------------------------------------------------
col_dd, col_pnl = st.columns(2)

with col_dd:
    _section("Drawdown")
    dd = compute_drawdown(eq_df["equity"])
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd,
        line=dict(color=RED, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,61,87,0.1)",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
    ))
    fig_dd.update_layout(**_chart_layout(
        xaxis_title="Date",
        yaxis=dict(ticksuffix="%", gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False),
        height=280,
        showlegend=False,
    ))
    st.plotly_chart(fig_dd, use_container_width=True)

with col_pnl:
    _section("PnL Components")
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df["realized_pnl"],
        name="Realized",
        line=dict(color="#60a5fa", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>Realized</extra>",
    ))
    fig_pnl.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df["unrealized_pnl"],
        name="Unrealized",
        line=dict(color="#93c5fd", width=2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>Unrealized</extra>",
    ))
    fig_pnl.update_layout(**_chart_layout(
        xaxis_title="Date",
        yaxis=dict(title="USD", tickprefix="$", gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False),
        hovermode="x unified",
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
    ))
    st.plotly_chart(fig_pnl, use_container_width=True)

# ---------------------------------------------------------------------------
# Returns distribution  |  Per-ticker PnL
# ---------------------------------------------------------------------------
col_hist, col_ticker = st.columns(2)

with col_hist:
    _section("Daily Returns Distribution")
    daily_rets = eq_df["equity"].resample("D").last().dropna().pct_change().dropna() * 100
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=daily_rets, nbinsx=30,
        marker_color="#4a6fa8",
        marker_line=dict(width=0),
        opacity=0.55,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(x=0, line=dict(color=GRAY, dash="dash", width=1))
    fig_hist.update_layout(**_chart_layout(
        xaxis=dict(title="Daily Return (%)", ticksuffix="%", gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False),
        yaxis=dict(title="Frequency", gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False),
        height=280,
        showlegend=False,
    ))
    st.plotly_chart(fig_hist, use_container_width=True)

with col_ticker:
    _section("Per-Ticker Realized PnL")
    ticker_rows = [
        {"label": f"{ticker} ({side.value})", "pnl": pos.realized_pnl / scale}
        for (ticker, side, _sid), pos in sorted(pf._positions.items())
        if pos.realized_pnl != 0
    ]
    if ticker_rows:
        tkr_df = pd.DataFrame(ticker_rows).sort_values("pnl", ascending=True)
        fig_tkr = go.Figure(go.Bar(
            x=tkr_df["pnl"],
            y=tkr_df["label"],
            orientation="h",
            marker_color=[GREEN if v >= 0 else RED for v in tkr_df["pnl"]],
            marker_line=dict(width=0),
            hovertemplate="%{y}<br>$%{x:+.2f}<extra></extra>",
        ))
        fig_tkr.update_layout(**_chart_layout(
            xaxis=dict(title="Realized PnL (USD)", tickprefix="$", gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False),
            yaxis=dict(gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False),
            height=280,
            showlegend=False,
        ))
        st.plotly_chart(fig_tkr, use_container_width=True)
    else:
        st.info("No realized PnL yet.")

# ---------------------------------------------------------------------------
# Price chart with trade events
# ---------------------------------------------------------------------------
_section("Price Chart")

traded_tickers = sorted(trips_df["ticker"].unique().tolist()) if not trips_df.empty else []
all_tickers    = sorted(pf._positions.keys(), key=lambda k: k[0])
chart_tickers  = traded_tickers if traded_tickers else [k[0] for k in all_tickers]

if chart_tickers:
    col_ticker_sel, col_chart_type = st.columns([3, 1])
    with col_ticker_sel:
        chart_ticker = st.selectbox("Ticker", chart_tickers, key="chart_ticker")
    with col_chart_type:
        chart_type = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True)

    instrument = cfg.get("instrument", "prediction_market")
    # For prediction markets: match feed by series prefix of the selected ticker
    if instrument == "prediction_market":
        chart_series = chart_ticker.split("-")[0]
        feed_cfg = next(
            (fc for fc in cfg.get("feeds", [])
             if Path(fc["path"]).stem.startswith(chart_series)),
            cfg["feeds"][0] if cfg.get("feeds") else None,
        )
    else:
        feed_map = {fc.get("ticker", fc.get("path")): fc for fc in cfg.get("feeds", [])}
        feed_cfg = feed_map.get(chart_ticker) or (cfg["feeds"][0] if cfg.get("feeds") else None)

    if feed_cfg:
        price_path = feed_cfg["path"]
        p = Path(price_path)
        raw_df = _load_price_df(price_path)

        if instrument == "ohlcv":
            raw_df["ts"] = pd.to_datetime(raw_df["timestamp"], utc=True)
            ticker_df    = raw_df.sort_values("ts")
            has_ohlc     = {"open", "high", "low", "close"}.issubset(ticker_df.columns)
            price_mul    = 1.0
            y_title      = f"Price ($)"
            y_range      = None
        else:
            raw_df["ts"] = pd.to_datetime(raw_df["ts"], utc=True)
            ticker_df    = raw_df[raw_df["ticker"] == chart_ticker].sort_values("ts")
            has_ohlc     = {"price_high", "price_low", "price_open", "price_close"}.issubset(ticker_df.columns)
            price_mul    = 100.0
            y_title      = "Price (¢)"
            y_range      = [0, 102]

        if not ticker_df.empty:
            fig_price = go.Figure()

            if chart_type == "Candlestick" and has_ohlc:
                if instrument == "ohlcv":
                    o, h, l, c = ticker_df["open"], ticker_df["high"], ticker_df["low"], ticker_df["close"]
                else:
                    o = ticker_df["price_open"]  * price_mul
                    h = ticker_df["price_high"]  * price_mul
                    l = ticker_df["price_low"]   * price_mul
                    c = ticker_df["price_close"] * price_mul
                fig_price.add_trace(go.Candlestick(
                    x=ticker_df["ts"], open=o, high=h, low=l, close=c,
                    name="Price",
                    increasing_line_color=GREEN,
                    decreasing_line_color=RED,
                ))
            else:
                if instrument == "ohlcv":
                    mid = ticker_df["close"]
                    fig_price.add_trace(go.Scatter(
                        x=ticker_df["ts"], y=mid,
                        name="Close", line=dict(color="#e6edf3", width=1.5),
                    ))
                else:
                    mid = (ticker_df["bid_close"] + ticker_df["ask_close"]) / 2 * price_mul
                    fig_price.add_trace(go.Scatter(
                        x=ticker_df["ts"], y=ticker_df["bid_close"] * price_mul,
                        name="Bid", line=dict(color="#60a5fa", width=1.5, dash="dot"),
                    ))
                    fig_price.add_trace(go.Scatter(
                        x=ticker_df["ts"], y=ticker_df["ask_close"] * price_mul,
                        name="Ask", line=dict(color="#fb923c", width=1.5, dash="dot"),
                        fill="tonexty", fillcolor="rgba(251,146,60,0.06)",
                    ))
                    fig_price.add_trace(go.Scatter(
                        x=ticker_df["ts"], y=mid,
                        name="Mid", line=dict(color="#e6edf3", width=1.5),
                    ))

            # Trade event overlays
            if not trips_df.empty:
                t_trips = trips_df[trips_df["ticker"] == chart_ticker]
                hover_fmt = f"%{{y:.2f}}{p_unit}"

                buys = t_trips[t_trips["exit_type"] != "open"]
                if not buys.empty:
                    fig_price.add_trace(go.Scatter(
                        x=buys["entry_time"], y=buys["entry_price_c"],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=13, color=GREEN,
                                    line=dict(color="#080d14", width=1)),
                        name="Buy",
                        hovertemplate=f"BUY %{{x|%Y-%m-%d %H:%M}}<br>{hover_fmt}<extra></extra>",
                    ))

                sells_t = t_trips[t_trips["exit_type"] == "sell"]
                if not sells_t.empty:
                    fig_price.add_trace(go.Scatter(
                        x=sells_t["exit_time"], y=sells_t["exit_price_c"],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=13, color=RED,
                                    line=dict(color="#080d14", width=1)),
                        name="Sell",
                        hovertemplate=f"SELL %{{x|%Y-%m-%d %H:%M}}<br>{hover_fmt}<extra></extra>",
                    ))

                for exit_type, label, sym in [("settlement", "Settlement", "circle"), ("expiry", "Expiry", "circle")]:
                    subset = t_trips[t_trips["exit_type"] == exit_type]
                    if not subset.empty:
                        colors = [GREEN if p >= 0 else RED for p in subset["pnl_usd"]]
                        fig_price.add_trace(go.Scatter(
                            x=subset["exit_time"], y=subset["exit_price_c"],
                            mode="markers",
                            marker=dict(symbol=sym, size=10, color=colors,
                                        line=dict(color="#080d14", width=1.5)),
                            name=label,
                            hovertemplate=f"{label.upper()} %{{x|%Y-%m-%d %H:%M}}<br>{hover_fmt}<extra></extra>",
                        ))

            yaxis_kwargs = dict(title=y_title, gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False)
            if y_range:
                yaxis_kwargs["range"] = y_range
            fig_price.update_layout(**_chart_layout(
                xaxis=dict(title="Date", gridcolor=_GRID, linecolor=_LINE, tickcolor=_LINE, zeroline=False),
                yaxis=yaxis_kwargs,
                hovermode="x unified",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
                xaxis_rangeslider_visible=False,
            ))
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info(f"No price data found for {chart_ticker}.")

# ---------------------------------------------------------------------------
# Trade log (round trips)
# ---------------------------------------------------------------------------
_section("Trade Log")

if not trips_df.empty:
    tickers_in_log = sorted(trips_df["ticker"].unique().tolist())
    exit_types_all = sorted(trips_df["exit_type"].dropna().unique().tolist())
    entry_min = trips_df["entry_time"].min().date()
    entry_max = trips_df["entry_time"].max().date()
    _exit_times = trips_df["exit_time"].dropna()
    exit_min = _exit_times.min().date() if not _exit_times.empty else entry_min
    exit_max = _exit_times.max().date() if not _exit_times.empty else entry_max

    # Jira-style per-filter popovers
    fc1, fc2, fc3, fc4, fc5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])
    with fc1:
        with st.popover("Ticker", use_container_width=True):
            sel_tickers = st.multiselect(
                "Ticker", tickers_in_log, default=tickers_in_log,
                key="tl_tickers", label_visibility="collapsed",
            )
    with fc2:
        with st.popover("Exit Type", use_container_width=True):
            sel_exit_types = st.multiselect(
                "Exit Type", exit_types_all, default=exit_types_all,
                key="tl_exit_types", label_visibility="collapsed",
            )
    with fc3:
        with st.popover("Entry Date", use_container_width=True):
            entry_range = st.date_input(
                "Entry Date", value=(entry_min, entry_max),
                min_value=entry_min, max_value=entry_max,
                key="tl_entry_range", label_visibility="collapsed",
            )
    with fc4:
        with st.popover("Exit Date", use_container_width=True):
            exit_range = st.date_input(
                "Exit Date", value=(exit_min, exit_max),
                min_value=exit_min, max_value=exit_max,
                key="tl_exit_range", label_visibility="collapsed",
            )
    with fc5:
        if st.button("Clear filters", use_container_width=True, key="tl_clear"):
            for k in ["tl_tickers", "tl_exit_types", "tl_entry_range", "tl_exit_range"]:
                st.session_state.pop(k, None)
            st.rerun()

    # Apply filters
    mask = (
        trips_df["ticker"].isin(sel_tickers or tickers_in_log) &
        trips_df["exit_type"].isin(sel_exit_types or exit_types_all)
    )
    if isinstance(entry_range, (list, tuple)) and len(entry_range) == 2:
        mask &= (
            (trips_df["entry_time"].dt.date >= entry_range[0]) &
            (trips_df["entry_time"].dt.date <= entry_range[1])
        )
    if isinstance(exit_range, (list, tuple)) and len(exit_range) == 2:
        mask &= (
            trips_df["exit_time"].isna() |
            (
                (trips_df["exit_time"].dt.date >= exit_range[0]) &
                (trips_df["exit_time"].dt.date <= exit_range[1])
            )
        )

    display_trips = trips_df[mask].reset_index(drop=True).copy()
    display_trips["cum_pnl_usd"] = display_trips["pnl_usd"].cumsum()

    active = []
    if sorted(sel_tickers) != tickers_in_log:
        active.append(f"{len(sel_tickers)} ticker(s)")
    if sorted(sel_exit_types) != exit_types_all:
        active.append(f"exit: {', '.join(sel_exit_types)}")
    filter_note = f"  ·  {', '.join(active)}" if active else ""
    st.caption(f"{len(display_trips):,} of {len(trips_df):,} round trips{filter_note}")

    _fixed_cols = ["entry_time", "exit_time", "ticker", "side", "qty",
                   "entry_price_c", "exit_price_c", "fees_usd", "pnl_usd", "cum_pnl_usd", "exit_type"]
    _meta_cols  = [c for c in display_trips.columns if c not in _fixed_cols]
    display = display_trips[_fixed_cols + _meta_cols].copy()
    fmt_price = (lambda x: f"{x:.2f}{p_unit}") if p_unit == "¢" else (lambda x: f"${x:.2f}")
    display["entry_price_c"] = display["entry_price_c"].map(lambda x: fmt_price(x) if x is not None else "—")
    display["exit_price_c"]  = display["exit_price_c"].map(lambda x: fmt_price(x) if x is not None else "—")
    display.columns = [
        "Entry Time", "Exit Time", "Ticker", "Side", "Qty",
        "Entry Price", "Exit Price", "Fee ($)", "PnL ($)", "Cum. PnL ($)", "Exit Type",
    ] + _meta_cols
    _render_table(display, pnl_cols=("PnL ($)", "Cum. PnL ($)"))
else:
    st.info("No completed trades recorded.")

# ---------------------------------------------------------------------------
# Open positions
# ---------------------------------------------------------------------------
open_pos = {k: v for k, v in pf._positions.items() if not v.is_flat}
_section("Open Positions", subtitle=str(len(open_pos)))
pos_rows = []
if open_pos:
    for (ticker, side, _sid), pos in sorted(open_pos.items()):
        mid = pf._mids.get((ticker, side), pos.avg_cost)
        pos_rows.append({
            "Ticker":                   ticker,
            "Side":                     side.value,
            "Qty":                      pos.quantity,
            f"Avg Cost ({p_unit})":     f"{pos.avg_cost:.2f}",
            f"Mid ({p_unit})":          f"{mid:.2f}",
            "Unrealized PnL ($)":       round(pos.unrealized_pnl(mid) / scale, 2),
        })
    _render_table(pd.DataFrame(pos_rows), pnl_cols=("Unrealized PnL ($)",))
else:
    st.info("No open positions.")

st.divider()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
_section("Export")
col_csv, col_xlsx, _ = st.columns([1, 1, 2])

with col_csv:
    st.download_button(
        "⬇  Equity Curve CSV",
        data=eq_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="equity_curve.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_xlsx:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _strip_tz(eq_df.reset_index()).to_excel(writer, sheet_name="Equity Curve", index=False)
        if not trade_df.empty:
            _strip_tz(trade_df).to_excel(writer, sheet_name="Trade Log", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="Metrics", index=False)
        if pos_rows:
            pd.DataFrame(pos_rows).to_excel(writer, sheet_name="Open Positions", index=False)
    st.download_button(
        "⬇  Full Report (Excel)",
        data=buf.getvalue(),
        file_name="backtest_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
