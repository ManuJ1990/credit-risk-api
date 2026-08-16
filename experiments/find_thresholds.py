"""Herleitung der Risiko-Schwellen aus der Kostenmatrix in german.doc.

    python -m experiments.find_thresholds
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from train import (
    COST_FN,
    COST_FP,
    RANDOM_STATE,
    TARGET,
    build_model,
    encode_ordinal,
    load_data,
)


# untere Grenze aus dem Kostenminimum der Tabelle oben,
# obere Grenze als Politikentscheidung: darueber wird nicht mehr geprueft
T_LOW = 0.20
T_HIGH = 0.50


def total_cost(scores, labels, threshold):
    """Gesamtkosten der Kostenmatrix bei dieser Schwelle."""
    fn = ((scores < threshold) & (labels == 1)).sum()
    fp = ((scores >= threshold) & (labels == 0)).sum()
    return COST_FN * fn + COST_FP * fp


df = load_data()
y = df[TARGET].to_numpy()

df_encoded, _ = encode_ordinal(df)
X = df_encoded.drop(columns=[TARGET])

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_predict(build_model(y), X, y, cv=folds, method="predict_proba")[:, 1]

print("Schwelle   FN    FP   Kosten   abgelehnt")
for t in np.arange(0.05, 1.0, 0.05):
    fn = ((scores < t) & (y == 1)).sum()
    fp = ((scores >= t) & (y == 0)).sum()
    print(
        f"   {t:.2f}   {fn:4d}  {fp:4d}   {total_cost(scores, y, t):6d}   "
        f"{100 * (scores >= t).mean():5.1f}%"
    )

zones = [
    ("low", scores < T_LOW),
    ("medium", (scores >= T_LOW) & (scores < T_HIGH)),
    ("high", scores >= T_HIGH),
]

print()
print("Zone     Anteil   Ausfallrate")
for name, mask in zones:
    print(f"{name:8s} {100 * mask.mean():5.1f}%   {100 * y[mask].mean():5.1f}%")

# Gegenprobe: Schwelle auf einer Haelfte bestimmen, Kosten auf der anderen
# messen. Zeigt, ob sich der Vorteil auf ungesehene Faelle uebertraegt.
GRID = np.arange(0.05, 0.95, 0.01)

found, transferred, baseline = [], [], []

for seed in range(30):
    a, b = train_test_split(
        np.arange(len(y)), test_size=0.5, random_state=seed, stratify=y
    )
    costs = [total_cost(scores[a], y[a], t) for t in GRID]
    best_t = GRID[int(np.argmin(costs))]

    found.append(best_t)
    transferred.append(total_cost(scores[b], y[b], best_t) / len(b))
    baseline.append(total_cost(scores[b], y[b], 0.5) / len(b))

print()
print("Gegenprobe ueber 30 Aufteilungen")
print(f"  Schwelle auf Haelfte A                 {np.mean(found):.2f} +/- {np.std(found):.2f}")
print(f"  Kosten je Fall auf B, Schwelle von A   {np.mean(transferred):.3f}")
print(f"  Kosten je Fall auf B, t = 0.50         {np.mean(baseline):.3f}")