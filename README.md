# Credit Risk API

A REST API that evaluates the credit default risk of a loan applicant.  
Input: applicant financial data → Output: risk score + classification (low / high) including an explanation of which factors influenced the score.

Built with FastAPI, XGBoost, and SHAP — deployed via Docker on Railway.

🚀 **Live:** https://credit-risk-api-production-322d.up.railway.app  
📄 **API Docs:** https://credit-risk-api-production-322d.up.railway.app/docs

---

## How It Works

```
POST /predict
{ checking_status, duration, credit_amount, ... }

→ {
    "risk_score": 0.82,
    "risk_class": "high",
    "top_factors": [
      { "feature": "credit_amount", "impact": 0.34, "direction": "increases_risk" },
      { "feature": "employment",    "impact": -0.18, "direction": "decreases_risk" },
      { "feature": "age",           "impact": 0.11, "direction": "increases_risk" }
    ]
  }
```

---

## Tech Stack

| Area | Tool |
|---|---|
| API Framework | FastAPI |
| ML Model | XGBoost |
| Explainability | SHAP |
| Data Processing | pandas, scikit-learn |
| Input Validation | Pydantic |
| Testing | pytest |
| Deployment | Docker + Railway |

---

## Project Structure

```
credit-risk-api/
├── app/
│   ├── main.py           # FastAPI app + endpoints
│   ├── model.py          # Model loading + prediction
│   ├── schemas.py        # Pydantic input/output models
│   └── explainer.py      # SHAP integration
├── notebooks/
│   └── training.ipynb    # Exploratory analysis + model training
├── models/
│   └── model.pkl         # Saved model
├── data/                 # Not committed — see Getting Started
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── requirements.txt
└── Dockerfile
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
git clone https://github.com/ManuJ1990/credit-risk-api.git
cd credit-risk-api

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Dataset

This project uses the [UCI German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).  
Download it and place `german.data` at `data/raw/german.data`.

### Run locally

```bash
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`

### Run with Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### Run tests

```bash
pytest tests/ -v
```

---

*Part of my portfolio — [manuel-portfolio-alpha.vercel.app](https://manuel-portfolio-alpha.vercel.app)*
