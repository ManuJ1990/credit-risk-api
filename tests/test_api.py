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


def test_out_of_range_value_returns_422():
    response = client.post("/predict", json={**SAMPLE_INPUT, "purpose": 99})

    assert response.status_code == 422


def test_top_factors_have_the_documented_shape():
    factors = client.post("/predict", json=SAMPLE_INPUT).json()["top_factors"]

    assert len(factors) == 3

    for factor in factors:
        assert set(factor) == {"feature", "value", "impact", "direction"}
        assert factor["direction"] in ("increases_risk", "decreases_risk")


def test_schema_endpoint_exposes_the_contract():
    body = client.get("/schema").json()

    assert set(body) >= {"feature_order", "risk_thresholds", "categorical", "numeric"}
    assert body["categorical"]["housing"]["A152"] == 1


def test_predict_rejects_a_missing_api_key(monkeypatch, medium_profile):
    monkeypatch.setenv("API_KEY", "test-secret")

    response = client.post("/predict", json=medium_profile)

    assert response.status_code == 401


def test_predict_accepts_the_configured_api_key(monkeypatch, medium_profile):
    monkeypatch.setenv("API_KEY", "test-secret")

    response = client.post(
        "/predict", json=medium_profile, headers={"X-API-Key": "test-secret"}
    )

    assert response.status_code == 200