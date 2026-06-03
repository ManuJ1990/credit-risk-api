from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

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

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_valid_input():
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    assert "risk_score" in response.json()
    assert "risk_class" in response.json()
    assert "top_factors" in response.json()

def test_predict_invalid_input():
    response = client.post("/predict", json={"checking_status": "falsch"})
    assert response.status_code == 422