"""FastAPI-Endpunkte der Credit Risk API."""

import logging
import os

from fastapi import FastAPI, Header, HTTPException

from app.explainer import get_top_factors
from app.model import FEATURE_ORDER, MAPPING, THRESHOLDS, TRAINED_AT, predict
from app.schemas import ApplicantInput

logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk API")


@app.get("/health")
def health():
    """Status, Trainingsdatum des Modells und aktive Schwellen."""
    return {
        "status": "ok",
        "model_trained_at": TRAINED_AT,
        "n_features": len(FEATURE_ORDER),
        "risk_thresholds": THRESHOLDS,
        "auth": "enabled" if os.environ.get("API_KEY") else "disabled",
    }


@app.post("/predict")
def predict_risk(
    input_data: ApplicantInput,
    x_api_key: str | None = Header(default=None),
    ):
    """Risikozone und die drei einflussreichsten Felder fuer einen Antrag."""
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    data = input_data.model_dump()

    try:
        result = predict(data)
        result["top_factors"] = get_top_factors(data)
    except Exception as exc:
        logger.exception("Vorhersage fehlgeschlagen fuer %s", data)
        raise HTTPException(
            status_code=500, detail="Vorhersage fehlgeschlagen"
        ) from exc

    return result


@app.get("/schema")
def schema():
    """Feature-Reihenfolge, Encoding, Wertebereiche und Schwellen."""
    return MAPPING