"""
Trainiert das Credit-Risk-Modell und schreibt Modell + Feature-Mapping nach models/.

    python train.py

Ersetzt den Trainingsteil aus notebooks/training.ipynb.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "german.data"
MODEL_PATH = ROOT / "models" / "model.pkl"
MAPPING_PATH = ROOT / "models" / "feature_mapping.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# german.data hat keine Kopfzeile – Namen und Reihenfolge stammen aus german.doc.
COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installment", "housing", "existing_credits",
    "job", "liable_people", "telephone", "foreign_worker",
    "risk",
]

TARGET = "risk"


df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=COLUMNS)

# Encoding von Hand statt LabelEncoder: der Encoder wird in der Schleife
# überschrieben, danach ist nur noch das Mapping der letzten Spalte übrig.
# Wir brauchen aber alle für feature_mapping.json.
categorical_cols = [c for c in df.columns if df[c].dtype == "object"]

encoding = {}
for col in categorical_cols:
    # alphabetisch, identisch zu LabelEncoder – "A410" liegt dadurch vor "A42"
    codes = sorted(df[col].unique())
    encoding[col] = {code: i for i, code in enumerate(codes)}
    df[col] = df[col].map(encoding[col])

# 1/2 -> 0/1; Ausfall wird zur 1 und damit zur positiven Klasse.
df[TARGET] = df[TARGET] - 1

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    # TODO: Verhältnis aus y_train nehmen (559/241) statt aus dem
    # Gesamtdatensatz. Bleibt vorerst so, damit dieser Lauf das
    # bestehende Modell exakt reproduziert.
    scale_pos_weight=700 / 300,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

# TODO: predict() schneidet bei 0.5 – sklearn-Default, keine bewusste
# Entscheidung. Ersetzen durch Schwellen aus der 5:1-Kostenmatrix.
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["gut", "Ausfall"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

# Single source of truth für app/ und das Portfolio-Widget.
mapping = {
    "feature_order": list(X.columns),
    "categorical": encoding,
    "numeric": {
        col: {"min": int(df[col].min()), "max": int(df[col].max())}
        for col in X.columns
        if col not in encoding
    },
}
MAPPING_PATH.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

print("Modell   ->", MODEL_PATH)
print("Mapping  ->", MAPPING_PATH)