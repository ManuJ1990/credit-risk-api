"""SHAP-Erklaerungen fuer einzelne Vorhersagen."""

import shap

from app.model import FEATURE_ORDER, model, to_frame

explainer = shap.TreeExplainer(model)


def get_top_factors(input_data: dict, top_n: int = 3) -> list[dict]:
    """Die top_n Felder mit dem stärksten Einfluss auf diese Vorhersage."""
    shap_values = explainer.shap_values(to_frame(input_data))[0]

    factors = sorted(
        zip(FEATURE_ORDER, shap_values, strict=True),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_n]

    return [
        {
            "feature": name,
            "value": input_data[name],
            "impact": round(float(value), 3),
            "direction": "increases_risk" if value > 0 else "decreases_risk",
        }
        for name, value in factors
    ]