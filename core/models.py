from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Predictor(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def evaluate(model: Predictor, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict(X)
    residuals = y.values - preds
    return {
        "mae":  round(float(np.abs(residuals).mean()), 3),
        "rmse": round(float(np.sqrt((residuals ** 2).mean())), 3),
    }


def save_artifact(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_artifact(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
