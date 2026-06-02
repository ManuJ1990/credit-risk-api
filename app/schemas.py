from pydantic import BaseModel

class ApplicantInput(BaseModel):
    checking_status: int
    duration: int
    credit_history: int
    purpose: int
    credit_amount: int
    savings_account: int
    employment: int
    installment_rate: int
    personal_status: int
    other_debtors: int
    residence_since: int
    property: int
    age: int
    other_installment: int
    housing: int
    existing_credits: int
    job: int
    liable_people: int
    telephone: int
    foreign_worker: int