from market_pred.sentiment.baseline import vader_score


def test_vader_score_positive_text():
    assert vader_score("The company reported excellent profits and strong growth.") > 0.3


def test_vader_score_negative_text():
    assert vader_score("The company reported terrible losses and a disastrous quarter.") < -0.3


def test_vader_score_empty_text_is_zero():
    assert vader_score("") == 0.0
    assert vader_score("   ") == 0.0
