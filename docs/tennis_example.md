# Tennis win-probability example strategy

> **Illustrative demo, not a profit claim.** This example exists to show how to
> feed an external per-match signal through the engine's existing `Factor` feed
> and compare it against a binary market's implied probability. The sample data
> is synthetic. Nothing here implies the strategy is profitable.
>
> **Disclosure:** contributed by the [Live Tennis API](https://livetennisapi.com)
> team. The tennis win-probability is a *data input* to the strategy — the
> engine still owns fills, resolution and settlement, and a real market resolves
> on its own venue. This is not a market, a broker, or a settlement oracle.

## What it demonstrates

A binary "will PLAYER win this match?" market prices an implied probability in
its YES ask (cents). `TennisWinProbStrategy` compares that implied probability
against an independent live-tennis win-probability signal and buys YES when the
signal is higher than the market by at least an edge threshold.

- **Market-implied probability** comes from `PredictionMarketSnapshot.ask`
  (cents) via the existing `PredictionMarketPriceFeed` — `ask / 100`.
- **The tennis signal** arrives as a daily `Factor` value in `[0, 1]` via the
  existing `FactorFeed` (the win-probability of the YES side). In a live setup
  this value would be derived from ranking/Elo/H2H, or the paid model
  win-probability field; here it is synthetic sample data.

No new engine interfaces are added — the strategy only subclasses
`strategies/base.py` and consumes the `Factor` and `PredictionMarketSnapshot`
events the engine already emits.

## Files

| File | Purpose |
|------|---------|
| `strategies/tennis_winprob/strategy.py` | `TennisWinProbStrategy` (subclasses `Strategy`) |
| `strategies/tennis_winprob/backtest.yaml` | Config: strategy params + feed paths |
| `make_tennis_sample_data.py` | Generates the synthetic parquet sample data |

`TennisWinProbStrategy` is registered in `backtest/app.py`'s `STRATEGY_REGISTRY`.

## Run it

```bash
python make_tennis_sample_data.py      # writes data/tennis_*.parquet
streamlit run backtest/app.py          # pick the "tennis_winprob" config
```

## Signal logic

On each YES `PredictionMarketSnapshot` (at most one order per ticker per day):

```
implied_cents = ask                      # YES ask, already in cents
signal_cents  = factor_value * 100       # live-tennis win-prob of YES side
edge          = signal_cents - implied_cents
if edge >= edge_cents:  buy YES (one lot at the ask)
```

### Parameters (`backtest.yaml`)

| Param | Default | Meaning |
|-------|---------|---------|
| `edge_cents` | `5.0` | Minimum `signal% − implied%` (in cents) to enter |
| `quantity` | `10` | Contracts per order |
| `max_ask_cents` | `95.0` | Skip near-certain YES (little to gain) |
| `min_ask_cents` | `5.0` | Skip near-zero YES (illiquid tails) |

## Using real Live Tennis data

Replace the synthetic factor with a real signal keyed to each match, written to
the same `Factor` parquet schema (`date`, `value`). Sources on the Live Tennis
API (base `https://api.livetennisapi.com/api/public/v1`, header `X-API-Key`):

- **Free tier** (30 req/min, 100/day — develop-and-test, not continuous fast
  polling): `/players` (rankings/Elo), `/matches?status=live` (live score).
  A ranking/Elo gap can seed a prior win-probability. Free key:
  <https://livetennisapi.com/subscribe/free>.
- **Paid tiers**: `/h2h` (head-to-head, BASIC) and the model win-probability
  field on `/matches/{id}/score` (ULTRA).

Discovery/matching of tennis markets on Polymarket's public Gamma API is shown
(observe-only) in <https://github.com/livetennisapi/polymarket-tennis>.
