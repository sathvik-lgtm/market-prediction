import pandas as pd
import pytest

from market_pred.modeling.validation import Fold, simple_holdout_split, split_fold, yearly_expanding_folds


def test_yearly_expanding_folds_seeds_then_expands():
    dates = pd.Series(["2018-01-01", "2019-06-01", "2020-01-01", "2021-01-01", "2022-01-01"])
    folds = yearly_expanding_folds(dates, min_train_years=2)

    assert [f.test_start[:4] for f in folds] == ["2020", "2021", "2022"]
    assert folds[0].train_end == "2019-12-31"
    assert folds[1].train_end == "2020-12-31"  # expanding, not a fixed-size window


def test_split_fold_boundaries_are_inclusive_and_non_overlapping():
    df = pd.DataFrame({"date": ["2019-12-31", "2020-01-01", "2020-06-01", "2020-12-31", "2021-01-01"]})
    fold = Fold(train_end="2019-12-31", test_start="2020-01-01", test_end="2020-12-31")

    train, test = split_fold(df, fold)

    assert train["date"].tolist() == ["2019-12-31"]
    assert test["date"].tolist() == ["2020-01-01", "2020-06-01", "2020-12-31"]
    assert set(train["date"]).isdisjoint(set(test["date"]))


def test_simple_holdout_split_is_chronological_not_random():
    df = pd.DataFrame({"date": [f"2024-01-{d:02d}" for d in range(1, 11)]})  # 10 unique dates

    train, test = simple_holdout_split(df, test_fraction=0.3)

    assert train["date"].max() < test["date"].min()
    assert len(train) + len(test) == len(df)


def test_simple_holdout_split_raises_on_too_few_dates():
    df = pd.DataFrame({"date": ["2024-01-01"]})
    with pytest.raises(ValueError):
        simple_holdout_split(df)
