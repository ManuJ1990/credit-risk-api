"""
Trainiert das Credit-Risk-Modell und schreibt Modell + Feature-Mapping nach models/.

    python train.py

Ersetzt den Trainingsteil aus notebooks/training.ipynb.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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


def load_data():
    df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=COLUMNS)
    # 1/2 -> 0/1; Ausfall wird zur 1 und damit zur positiven Klasse.
    df[TARGET] = df[TARGET] - 1
    return df


def encode_ordinal(df):
    """Kategorische Spalten alphabetisch durchnummerieren.

    Gibt eine encodierte Kopie und die Mappings zurück. Kein LabelEncoder,
    dessen Mappings werden für feature_mapping.json gebraucht.
    """
    df = df.copy()
    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]

    encoding = {}
    for col in categorical_cols:
        # alphabetisch: "A410" liegt dadurch vor "A42"
        codes = sorted(df[col].unique())
        encoding[col] = {code: i for i, code in enumerate(codes)}
        df[col] = df[col].map(encoding[col])

    return df, encoding


def encode_categorical(df):
    """Kategorische Spalten als pandas-Typ 'category' markieren.

    XGBoost erkennt den Typ und darf dann beliebige Kategorien-Gruppen
    trennen, statt nur entlang der alphabetischen Reihenfolge zu schneiden.
    """
    df = df.copy()
    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df


def build_model(enable_categorical=False):
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        # TODO: Verhältnis aus y_train nehmen (559/241) statt aus dem
        # Gesamtdatensatz. Bleibt vorerst so, damit dieser Lauf das
        # bestehende Modell exakt reproduziert.
        scale_pos_weight=700 / 300,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        enable_categorical=enable_categorical,
    )


def train_and_evaluate(X, y, enable_categorical=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_model(enable_categorical)
    model.fit(X_train, y_train)

    # TODO: predict() schneidet bei 0.5 – sklearn-Default, keine bewusste
    # Entscheidung. Ersetzen durch Schwellen aus der 5:1-Kostenmatrix.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    return model, metrics


def cross_val_auc(X, y, enable_categorical=False, n_splits=5):
    """Mittlere AUC über mehrere Aufteilungen, plus Streuung.

    Ein einzelner Train/Test-Split schwankt bei 1000 Zeilen zu stark,
    um kleine Unterschiede zwischen Varianten zu beurteilen.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        build_model(enable_categorical), X, y, cv=folds, scoring="roc_auc"
    )
    return scores.mean(), scores.std()


def save(model, encoding, X):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Single source of truth für app/ und das Portfolio-Widget.
    mapping = {
        "feature_order": list(X.columns),
        "categorical": encoding,
        "numeric": {
            col: {"min": int(X[col].min()), "max": int(X[col].max())}
            for col in X.columns
            if col not in encoding
        },
    }
    MAPPING_PATH.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def main():
    df = load_data()
    y = df[TARGET]

    df_ordinal, encoding = encode_ordinal(df)
    X_ordinal = df_ordinal.drop(columns=[TARGET])
    model, metrics_ordinal = train_and_evaluate(X_ordinal, y)

    X_categorical = encode_categorical(df).drop(columns=[TARGET])
    _, metrics_categorical = train_and_evaluate(
        X_categorical, y, enable_categorical=True
    )

    for name, m, X, cat in [
        ("ordinal", metrics_ordinal, X_ordinal, False),
        ("categorical", metrics_categorical, X_categorical, True),
    ]:
        mean, std = cross_val_auc(X, y, enable_categorical=cat)
        print(
            f"{name:12s} AUC {m['auc']:.3f}  P {m['precision']:.2f}  "
            f"R {m['recall']:.2f}  CV {mean:.3f} +/- {std:.3f}"
        )

    save(model, encoding, X_ordinal)
    print("Modell   ->", MODEL_PATH)
    print("Mapping  ->", MAPPING_PATH)


if __name__ == "__main__":
    main()