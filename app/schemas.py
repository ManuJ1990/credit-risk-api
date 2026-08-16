"""Pydantic-Schema fuer die Eingabe von /predict."""

from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    """Eingabe für /predict.

    Kategoriale Felder sind Integer-Codes, die Zuordnung steht in
    models/feature_mapping.json. Die Grenzen entsprechen den Werten,
    die im Training vorkamen.
    """

    checking_status: int = Field(..., ge=0, le=3)
    duration: int = Field(..., ge=4, le=72)
    credit_history: int = Field(..., ge=0, le=4)
    purpose: int = Field(..., ge=0, le=9)
    credit_amount: int = Field(..., ge=250, le=18424)
    savings_account: int = Field(..., ge=0, le=4)
    employment: int = Field(..., ge=0, le=4)
    installment_rate: int = Field(..., ge=1, le=4)
    personal_status: int = Field(..., ge=0, le=3)
    other_debtors: int = Field(..., ge=0, le=2)
    residence_since: int = Field(..., ge=1, le=4)
    property: int = Field(..., ge=0, le=3)
    age: int = Field(..., ge=19, le=75)
    other_installment: int = Field(..., ge=0, le=2)
    housing: int = Field(..., ge=0, le=2)
    existing_credits: int = Field(..., ge=1, le=4)
    job: int = Field(..., ge=0, le=3)
    liable_people: int = Field(..., ge=1, le=2)
    telephone: int = Field(..., ge=0, le=1)
    foreign_worker: int = Field(..., ge=0, le=1)