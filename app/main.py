from fastapi import FastAPI
from app.schemas import ApplicantInput
from app.model import predict
from app.explainer import get_top_factors

# FastAPI App initialisieren
app = FastAPI(title="Credit Risk API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_risk(input_data: ApplicantInput):
    # Input zu Dictionary umwandeln
    data = input_data.model_dump()
    
    # Vorhersage + Erklärung
    result = predict(data)
    result["top_factors"] = get_top_factors(data)
    
    return result