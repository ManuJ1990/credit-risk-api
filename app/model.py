import joblib
import numpy as np
from pathlib import Path

# Pfad zum Modell
MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"

# Modell einmalig beim Start laden
model = joblib.load(MODEL_PATH)


def predict(input_data: dict) -> dict:
    # Input in numpy Array umwandeln
    features = np.array(list(input_data.values())).reshape(1, -1)
    
    # Vorhersage
    risk_class = int(model.predict(features)[0])
    risk_score = float(model.predict_proba(features)[0][1])
    
    return {
        "risk_score": round(risk_score, 3),
        "risk_class": "high" if risk_class == 1 else "low"
    }