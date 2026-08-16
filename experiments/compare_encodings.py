"""Einmaliger Vergleich der Encoding-Varianten.

    python -m experiments.compare_encodings

Kein Produktionscode – das Ergebnis steht im Commit und im README.
"""

import pandas as pd

from train import (
    TARGET,
    cross_val_auc,
    encode_categorical,
    encode_ordinal,
    load_data,
    train_and_evaluate,
)


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
    _, metrics = train_and_evaluate(X, y, enable_categorical=enable_categorical)
    mean, std = cross_val_auc(X, y, enable_categorical=enable_categorical)
    print(
        f"{name:12s} {X.shape[1]:3d} Merkmale  AUC {metrics['auc']:.3f}  "
        f"P {metrics['precision']:.2f}  R {metrics['recall']:.2f}  "
        f"CV {mean:.3f} +/- {std:.3f}"
    )