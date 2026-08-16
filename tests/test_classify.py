from app.model import THRESHOLDS, classify


def test_below_low_threshold():
    assert classify(0.0) == "low"
    assert classify(0.19) == "low"


def test_between_threshold():
    assert classify(0.30) == "medium"
    assert classify(0.49) == "medium"


def test_above_high_threshold():
    assert classify(0.50) == "high"
    assert classify(1.0) == "high"


def test_boudaries_are_inclusive_upwards():
    """Genau auf der Schwelle gehoert der Wert zur hoeheren Zone"""
    assert classify(THRESHOLDS["low"]) == "medium"
    assert classify(THRESHOLDS["high"]) == "high"