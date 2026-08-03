"""
train.py – Reproduzierbares Training des Credit-Risk-Modells.
Aufruf: python train.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import json
import joblib


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "german.data"
MODEL_PATH = ROOT / "models" / "model.pkl"
MAPPING_PATH = ROOT / "models" / "feature_mapping.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2


COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installment", "housing", "existing_credits",
    "job", "liable_people", "telephone", "foreign_worker",
    "risk",
]

df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=COLUMNS)

categorical_cols = [c for c in df.columns if df[c].dtype == "object"]

encoding = {}
for col in categorical_cols:
    codes = sorted(df[col].unique())
    encoding[col] = {code: i for i, code in enumerate(codes)}
    df[col] = df[col].map(encoding[col])

df["risk"] = df["risk"] - 1

X = df.drop(columns=["risk"])
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=700 / 300,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["gut", "Ausfall"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

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