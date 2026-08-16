"""Modell laden, Eingaben aufbereiten, Risikozone bestimmen."""

import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"
MAPPING_PATH = Path(__file__).parent.parent / "models" / "feature_mapping.json"

model = joblib.load(MODEL_PATH)
_mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
THRESHOLDS = _mapping["risk_thresholds"]
FEATURE_ORDER = _mapping["feature_order"]
TRAINED_AT = _mapping["trained_at"]


def to_frame(input_data: dict) -> pd.DataFrame:
    """Eingabe in die Spaltenreihenfolge bringen, mit der trainiert wurde."""
    return pd.DataFrame([input_data])[FEATURE_ORDER]


def classify(score: float) -> str:
    """Risikozone anhand der Schwellen aus feature_mapping.json."""
    if score < THRESHOLDS["low"]:
        return "low"
    if score < THRESHOLDS["high"]:
        return "medium"
    return "high"


def predict(input_data: dict) -> dict:
    """Score und Risikozone fuer einen Antrag."""
    risk_score = float(model.predict_proba(to_frame(input_data))[0][1])

    return {
        "risk_score": round(risk_score, 3),
        "risk_class": classify(risk_score),
    }