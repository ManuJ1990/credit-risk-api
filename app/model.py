import json
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"
MAPPING_PATH = Path(__file__).parent.parent / "models" / "feature_mapping.json"

model = joblib.load(MODEL_PATH)
THRESHOLDS = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["risk_thresholds"]


def classify(score: float) -> str:
    """Risikozone anhand der Schwellen aus feature_mapping.json."""
    if score < THRESHOLDS["low"]:
        return "low"
    if score < THRESHOLDS["high"]:
        return "medium"
    return "high"


def predict(input_data: dict) -> dict:
    features = np.array(list(input_data.values())).reshape(1, -1)
    risk_score = float(model.predict_proba(features)[0][1])

    return {
        "risk_score": round(risk_score, 3),
        "risk_class": classify(risk_score),
    }