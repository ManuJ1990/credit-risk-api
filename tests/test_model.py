from app.model import model, predict

# Beispiel-Input
SAMPLE_INPUT = {
    "checking_status": 0,
    "duration": 12,
    "credit_history": 2,
    "purpose": 3,
    "credit_amount": 1500,
    "savings_account": 0,
    "employment": 2,
    "installment_rate": 2,
    "personal_status": 2,
    "other_debtors": 0,
    "residence_since": 2,
    "property": 0,
    "age": 35,
    "other_installment": 2,
    "housing": 1,
    "existing_credits": 1,
    "job": 1,
    "liable_people": 1,
    "telephone": 0,
    "foreign_worker": 0
}

def test_model_loaded():
    assert model is not None

def test_prediction_output():
    result = predict(SAMPLE_INPUT)
    assert result["risk_class"] in ["low", "high"]

def test_score_range():
    result = predict(SAMPLE_INPUT)
    assert 0 <= result["risk_score"] <= 1