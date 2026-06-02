import shap
import numpy as np
from app.model import model

# SHAP Explainer einmalig initialisieren
explainer = shap.TreeExplainer(model)

def get_top_factors(input_data: dict, top_n: int = 3) -> list:
    # Input in numpy Array umwandeln
    features = np.array(list(input_data.values())).reshape(1, -1)
    
    # SHAP Werte berechnen
    shap_values = explainer.shap_values(features)[0]
    
    # Feature Namen und SHAP Werte zusammenbringen
    feature_names = list(input_data.keys())
    factors = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]
    
    # Ergebnis formatieren
    return [
        {
            "feature": name,
            "impact": round(float(value), 3),
            "direction": "increases_risk" if value > 0 else "decreases_risk"
        }
        for name, value in factors
    ]