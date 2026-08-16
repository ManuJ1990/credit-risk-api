"""Gemeinsame Testprofile."""

import pytest

LOW = {
    "checking_status": 2, "duration": 12, "credit_history": 2, "purpose": 4,
    "credit_amount": 2000, "savings_account": 1, "employment": 3,
    "installment_rate": 2, "personal_status": 2, "other_debtors": 0,
    "residence_since": 3, "property": 0, "age": 35, "other_installment": 2,
    "housing": 1, "existing_credits": 1, "job": 2, "liable_people": 1,
    "telephone": 1, "foreign_worker": 1,
}

MEDIUM = {
    "checking_status": 0, "duration": 12, "credit_history": 2, "purpose": 3,
    "credit_amount": 1500, "savings_account": 0, "employment": 2,
    "installment_rate": 2, "personal_status": 2, "other_debtors": 0,
    "residence_since": 2, "property": 0, "age": 35, "other_installment": 2,
    "housing": 1, "existing_credits": 1, "job": 1, "liable_people": 1,
    "telephone": 0, "foreign_worker": 0,
}

HIGH = {
    "checking_status": 0, "duration": 48, "credit_history": 3, "purpose": 0,
    "credit_amount": 8000, "savings_account": 0, "employment": 1,
    "installment_rate": 4, "personal_status": 1, "other_debtors": 0,
    "residence_since": 1, "property": 3, "age": 22, "other_installment": 1,
    "housing": 0, "existing_credits": 2, "job": 1, "liable_people": 2,
    "telephone": 0, "foreign_worker": 0,
}


@pytest.fixture
def low_profile():
    """Solides Profil, erwartete Zone: low."""
    return dict(LOW)


@pytest.fixture
def medium_profile():
    """Grenzwertiges Profil, erwartete Zone: medium."""
    return dict(MEDIUM)


@pytest.fixture
def high_profile():
    """Riskantes Profil, erwartete Zone: high."""
    return dict(HIGH)