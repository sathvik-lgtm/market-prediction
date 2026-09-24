"""Walk-forward (expanding-window) date-based validation splits.

Never a random split on time-series data -- shuffling would leak future rows
into training. Splits are by calendar date across ALL pooled tickers at once,
so a fold's train/test boundary is a single date cutoff shared by every
ticker, not a per-ticker one (avoiding any chance of one ticker's future
leaking into another's "past" fold via misaligned boundaries).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_end: str  # inclusive, ISO date string
    test_start: str  # inclusive
    test_end: str  # inclusive


def yearly_expanding_folds(dates: pd.Series, min_train_years: int = 2) -> list[Fold]:
    """One fold per calendar year after an initial `min_train_years`-year seed,
    with an ever-expanding training window (train = everything before the fold's
    test year; test = that whole year).
    """
    years = sorted(pd.to_datetime(dates).dt.year.unique())
    folds = []
    for i in range(min_train_years, len(years)):
        test_year = years[i]
        folds.append(Fold(
            train_end=f"{years[i - 1]}-12-31",
            test_start=f"{test_year}-01-01",
            test_end=f"{test_year}-12-31",
        ))
    return folds


def split_fold(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a pooled dataset (with a `date` column of ISO strings) into this
    fold's train/test rows."""
    train = df[df["date"] <= fold.train_end]
    test = df[(df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)]
    return train, test


def simple_holdout_split(df: pd.DataFrame, test_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A single chronological train/test split, for datasets too small for a
    multi-fold walk-forward to mean anything (the sentiment ablation's ~29-day
    overlap window). Splits on unique sorted dates so every ticker shares the
    same cutoff, not on row count.
    """
    unique_dates = sorted(df["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError(f"Need at least 2 distinct dates to split, got {len(unique_dates)}")
    cutoff_idx = min(max(1, int(len(unique_dates) * (1 - test_fraction))), len(unique_dates) - 1)
    train_end = unique_dates[cutoff_idx - 1]
    test_start = unique_dates[cutoff_idx]
    return df[df["date"] <= train_end], df[df["date"] >= test_start]
