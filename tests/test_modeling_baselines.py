import pandas as pd
import pytest

from market_pred.modeling.baselines import mean_daily_return, naive_always_up_accuracy, strategy_mean_return


def test_naive_always_up_accuracy_equals_fraction_of_up_days():
    y_true = pd.Series([1, 1, 1, 0, 0])
    assert naive_always_up_accuracy(y_true) == pytest.approx(0.6)


def test_mean_daily_return_is_plain_average():
    returns = pd.Series([0.01, -0.02, 0.03])
    assert mean_daily_return(returns) == pytest.approx((0.01 - 0.02 + 0.03) / 3)


def test_strategy_mean_return_only_captures_predicted_up_days():
    y_pred = [1, 0, 1, 0]
    next_returns = pd.Series([0.02, 0.05, -0.01, 0.03])  # 2nd/4th days would-be gains are skipped (predicted down)
    # Captured: 0.02 (pred up) + 0 (pred down) + -0.01 (pred up) + 0 (pred down)
    assert strategy_mean_return(y_pred, next_returns) == pytest.approx((0.02 + 0 - 0.01 + 0) / 4)


def test_strategy_mean_return_all_flat_is_zero():
    y_pred = [0, 0, 0]
    next_returns = pd.Series([0.05, -0.05, 0.1])
    assert strategy_mean_return(y_pred, next_returns) == 0.0
