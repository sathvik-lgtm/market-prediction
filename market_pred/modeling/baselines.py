"""Naive baselines to compare trained models against: a majority-class
classifier and a buy-and-hold trading benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_always_up_accuracy(y_true: pd.Series) -> float:
    """Accuracy of a classifier that always predicts 'up' -- equals the actual
    fraction of up days in the given period."""
    return float((y_true == 1).mean())


def mean_daily_return(next_returns: pd.Series) -> float:
    """Average next-day return over the period -- the buy-and-hold benchmark:
    what you'd earn on average per day just holding the position throughout."""
    return float(next_returns.mean())


def strategy_mean_return(y_pred, next_returns: pd.Series) -> float:
    """Average return under a simple long/flat strategy: capture the next-day
    return on days predicted 'up', earn 0 (stay in cash) on days predicted 'down'.
    No shorting, no transaction costs -- a simple comparison point, not a full
    backtest.
    """
    captured = next_returns.to_numpy() * np.asarray(y_pred)
    return float(captured.mean())
