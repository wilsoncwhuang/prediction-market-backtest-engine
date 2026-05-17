from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.events import Factor


class FactorFeed:
    """
    Loads a factor parquet and emits Factor events.

    Parquet columns: date (required), <value_col> (required), model_id (optional).

    Parameters
    ----------
    series       : label for this factor (e.g. "DEMO", "KXHIGHNY")
    parquet_path : path to parquet with columns [date, <value_col>]
    model_id     : optional identifier for the model/source
    emit_hour    : UTC hour at which to timestamp each Factor event (default 0)
    value_col    : column name for the factor value (default "value")
    """

    def __init__(
        self,
        series: str,
        parquet_path: Path | str,
        model_id: str = "",
        emit_hour: int = 0,
        value_col: str = "value",
    ) -> None:
        self._series    = series
        self._model_id  = model_id
        self._emit_hour = emit_hour
        self._value_col = value_col

        df = pd.read_parquet(parquet_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        self._df = df

    def load(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Factor]:
        events: list[Factor] = []
        for _, row in self._df.iterrows():
            d = row["date"]
            ts = datetime(d.year, d.month, d.day, self._emit_hour, tzinfo=timezone.utc)

            if start is not None:
                s = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
                if ts < s:
                    continue
            if end is not None:
                e = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
                if ts > e:
                    continue

            events.append(Factor(
                timestamp=ts,
                series=self._series,
                target_date=d,
                value=float(row[self._value_col]),
                model_id=row["model_id"] if "model_id" in row.index else self._model_id,
            ))

        return events
