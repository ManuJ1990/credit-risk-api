# Credit Risk API

A REST API that scores the default risk of a loan application and explains the result.

Input: 20 applicant features → Output: a risk score, one of three risk zones, and the three fields that influenced the score most.

**Live:** [API docs](https://credit-risk-api-50139538822.europe-west3.run.app/docs) ·
**Demo:** [interactive widget](https://manueljunker.at/#projects) ·
**Write-up:** [what I got wrong and how I found it](https://manueljunker.at/blog/credit-risk-api)

---

## What is worth looking at

Most of the interesting decisions in this project are not in the model.

**The decision threshold is derived, not inherited.** `model.predict()` cuts at 0.5, which is a library default. The dataset documents a cost matrix instead: letting a bad loan through costs 5, rejecting a good applicant costs 1. Minimising expected cost over out-of-fold predictions puts the boundary at roughly 0.20 and lowers cost per case from 0.679 to 0.557 on unseen data. See `experiments/find_thresholds.py`.

**Three zones instead of a binary verdict.** Below 0.20 accept, above 0.50 reject, in between send to manual review. The observed default rates in those zones are 9 %, 25 % and 57 %.

**The headline metric is cross-validated.** A single train/test split reported AUC 0.813 for this model. Five-fold cross-validation gives **0.79 ± 0.02**, and the single split was optimistic enough to reverse the outcome of an encoding comparison. See `experiments/compare_encodings.py`.

**One source of truth.** `train.py` writes `models/feature_mapping.json` with the feature order, the code-to-integer mapping, the numeric ranges and the thresholds. The API reads it, exposes it under `/schema`, and the portfolio widget consumes it instead of hardcoding values.

**Input is named, not positional.** Requests are turned into a `DataFrame` with column names, so XGBoost validates the feature names. Previously two swapped fields produced a plausible but wrong answer with no error.

---

## API

```
GET  /health    status, training timestamp, active thresholds
GET  /schema    feature order, encodings, ranges, thresholds
POST /predict   risk score, zone, top 3 factors
```

```jsonc
// POST /predict
{ "checking_status": 2, "duration": 12, "credit_amount": 2000, ... }

// →
{
  "risk_score": 0.078,
  "risk_class": "low",
  "top_factors": [
    { "feature": "property",        "value": 0, "impact": -0.379, "direction": "decreases_risk" },
    { "feature": "employment",      "value": 3, "impact": -0.368, "direction": "decreases_risk" },
    { "feature": "checking_status", "value": 2, "impact": -0.344, "direction": "decreases_risk" }
  ]
}
```

`impact` values are SHAP values in log-odds: positive raises the score, negative lowers it. They are meaningful for ranking, not as probabilities. `value` is the field value that produced the contribution, so a client can render it without looking it up again.

`/predict` requires an `X-API-Key` header when the `API_KEY` environment variable is set, so only the portfolio proxy can trigger a prediction.

---

## Stack

| Area | Tool |
|---|---|
| Model | XGBoost |
| Explainability | SHAP |
| API | FastAPI, Pydantic |
| Experiment tracking | MLflow |
| Tests | pytest |
| Linting | ruff |
| Deployment | Docker on Google Cloud Run |

---

## Project structure

```
credit-risk-api/
├── train.py                        # the only way a model is produced
├── app/
│   ├── main.py                     # endpoints, error handling, API key
│   ├── model.py                    # loading, input framing, zone classification
│   ├── schemas.py                  # input contract with value ranges
│   └── explainer.py                # SHAP top factors
├── experiments/                    # one-off investigations, not production code
│   ├── compare_encodings.py        # ordinal vs. categorical vs. one-hot
│   └── find_thresholds.py          # cost curve and threshold validation
├── models/
│   ├── model.pkl
│   └── feature_mapping.json        # contract shared by model, API and widget
├── tests/                          # 23 tests, see below
├── data/raw/german.data            # UCI German Credit, committed for reproducibility
├── notebooks/training.ipynb        # exploratory analysis
├── Dockerfile
├── requirements.txt                # runtime only
└── requirements-dev.txt            # plus MLflow, pytest, ruff
```

---

## Getting started

```bash
git clone https://github.com/ManuJ1990/credit-risk-api.git
cd credit-risk-api

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements-dev.txt
```

The dataset is committed, so training works right after cloning:

```bash
python train.py
```

This prints the holdout metrics and the cross-validated AUC, writes `models/model.pkl` and `models/feature_mapping.json`, and logs the run to MLflow. Inspect runs with `mlflow ui`.

Run the API:

```bash
uvicorn app.main:app --reload      # http://localhost:8000/docs
```

Reproduce the experiments:

```bash
python -m experiments.compare_encodings
python -m experiments.find_thresholds
```

---

## Tests

```bash
pytest -v
```

| File | Kind | Covers |
|---|---|---|
| `test_classify.py` | unit | zone boundaries, including the exact threshold values |
| `test_encoding.py` | unit | the documented code-to-integer mapping, and that encoding does not mutate its input |
| `test_model.py` | known-answer | three fixed profiles land in their expected zone; field order in a request does not change the result |
| `test_contract.py` | contract | value ranges in `schemas.py` match `feature_mapping.json` |
| `test_api.py` | integration | endpoints, 422 on out-of-range input, shape of `top_factors` |

---

## Limitations

The dataset is the UCI German Credit set from 1994. This is a technical exercise, not a usable credit model.

Two categories of `credit_history` carry only 40 and 49 cases, so their striking default rates are statistically shaky. The score is deliberately not calibrated: it is useful for ranking and zoning, but it is not a probability of default. And the thresholds depend on the model, so retraining means deriving them again.

---

## License

MIT
