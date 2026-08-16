"""Einmaliger Vergleich der Encoding-Varianten.

    python -m experiments.compare_encodings

Kein Produktionscode – das Ergebnis steht im Commit und im README.
"""

import mlflow
import pandas as pd

from train import (
    TARGET,
    cross_val_auc,
    encode_ordinal,
    load_data,
    train_and_evaluate,
)

mlflow.set_experiment("encoding-comparison")


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


def encode_onehot(df):
    """Jede Kategorie bekommt eine eigene 0/1-Spalte.

    Für lineare Modelle Standard, für Bäume meist nachteilig.
    """
    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
    return pd.get_dummies(df, columns=categorical_cols, dtype=int)


df = load_data()
y = df[TARGET]

df_ordinal, _ = encode_ordinal(df)

variants = [
    ("ordinal", df_ordinal, False),
    ("categorical", encode_categorical(df), True),
    ("onehot", encode_onehot(df), False),
]

for name, df_encoded, enable_categorical in variants:
    X = df_encoded.drop(columns=[TARGET])

    with mlflow.start_run(run_name=name):
        _, metrics = train_and_evaluate(X, y, enable_categorical=enable_categorical)
        mean, std = cross_val_auc(X, y, enable_categorical=enable_categorical)

        mlflow.log_param("encoding", name)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metric("holdout_auc", metrics["auc"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("cv_auc", mean)
        mlflow.log_metric("cv_std", std)

        print(
            f"{name:12s} {X.shape[1]:3d} Merkmale  AUC {metrics['auc']:.3f}  "
            f"P {metrics['precision']:.2f}  R {metrics['recall']:.2f}  "
            f"CV {mean:.3f} +/- {std:.3f}"
        )