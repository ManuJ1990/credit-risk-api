"""Known-Answer- und Regressionstests fuer die Vorhersage."""

import pytest

from app.model import model, predict, to_frame


def test_model_loaded():
    assert model is not None


def test_low_profile_lands_in_low(low_profile):
    assert predict(low_profile)["risk_class"] == "low"


def test_medium_profile_lands_in_medium(medium_profile):
    assert predict(medium_profile)["risk_class"] == "medium"


def test_high_profile_lands_in_high(high_profile):
    assert predict(high_profile)["risk_class"] == "high"


def test_score_stays_between_zero_and_one(medium_profile):
    assert 0.0 <= predict(medium_profile)["risk_score"] <= 1.0


def test_field_order_in_request_does_not_matter(low_profile):
    shuffled = dict(reversed(list(low_profile.items())))

    assert predict(shuffled) == predict(low_profile)


def test_missing_field_raises(low_profile):
    del low_profile["age"]

    with pytest.raises(KeyError):
        to_frame(low_profile)