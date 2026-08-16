"""Known-Answer- und Regressionstests fuer die Vorhersage."""

import pytest

from app.model import model, predict, to_frame

LOW_PROFILE = {
    "checking_status": 2, "duration": 12, "credit_history": 2, "purpose": 4,
    "credit_amount": 2000, "savings_account": 1, "employment": 3,
    "installment_rate": 2, "personal_status": 2, "other_debtors": 0,
    "residence_since": 3, "property": 0, "age": 35, "other_installment": 2,
    "housing": 1, "existing_credits": 1, "job": 2, "liable_people": 1,
    "telephone": 1, "foreign_worker": 1,
}

MEDIUM_PROFILE = {
    "checking_status": 0, "duration": 12, "credit_history": 2, "purpose": 3,
    "credit_amount": 1500, "savings_account": 0, "employment": 2,
    "installment_rate": 2, "personal_status": 2, "other_debtors": 0,
    "residence_since": 2, "property": 0, "age": 35, "other_installment": 2,
    "housing": 1, "existing_credits": 1, "job": 1, "liable_people": 1,
    "telephone": 0, "foreign_worker": 0,
}

HIGH_PROFILE = {
    "checking_status": 0, "duration": 48, "credit_history": 3, "purpose": 0,
    "credit_amount": 8000, "savings_account": 0, "employment": 1,
    "installment_rate": 4, "personal_status": 1, "other_debtors": 0,
    "residence_since": 1, "property": 3, "age": 22, "other_installment": 1,
    "housing": 0, "existing_credits": 2, "job": 1, "liable_people": 2,
    "telephone": 0, "foreign_worker": 0,
}


def test_model_loaded():
    assert model is not None


def test_low_profile_lands_in_low():
    assert predict(LOW_PROFILE)["risk_class"] == "low"


def test_medium_profile_lands_in_medium():
    assert predict(MEDIUM_PROFILE)["risk_class"] == "medium"


def test_high_profile_lands_in_high():
    assert predict(HIGH_PROFILE)["risk_class"] == "high"


def test_score_stays_between_zero_and_one():
    assert 0.0 <= predict(MEDIUM_PROFILE)["risk_score"] <= 1.0


def test_field_order_in_request_does_not_matter():
    """to_frame sortiert nach feature_order, die Reihenfolge der Eingabe ist egal."""
    shuffled = dict(reversed(list(LOW_PROFILE.items())))

    assert predict(shuffled) == predict(LOW_PROFILE)


def test_missing_field_raises():
    incomplete = {k: v for k, v in LOW_PROFILE.items() if k != "age"}

    with pytest.raises(KeyError):
        to_frame(incomplete)