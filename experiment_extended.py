"""
Extended feature-reduction experiment for the MathQuest paper.

Run from the project root, where data/TCGA-PANCAN-HiSeq-801x20531/
contains data.csv and labels.csv.

This script compares:
- ANOVA-selected features
- Randomly selected features at the same feature counts
- A majority-class baseline

It evaluates five stratified 75:25 holdout splits (random_state 42–46).
These test splits overlap, so they are not independent trials.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA = "data/TCGA-PANCAN-HiSeq-801x20531/"
OUT = "results/extended_feature_experiment"
os.makedirs(OUT, exist_ok=True)

print("Loading dataset...", flush=True)
Xdf = pd.read_csv(DATA + "data.csv").set_index("Unnamed: 0")
ydf = pd.read_csv(DATA + "labels.csv").set_index("Unnamed: 0")

ids = Xdf.index.intersection(ydf.index)
X = Xdf.loc[ids].to_numpy(dtype=np.float32)
y = ydf.loc[ids, "Class"].to_numpy()

print(f"Samples: {len(y)} | Original input features: {X.shape[1]}", flush=True)

# Requested levels. Counts are based on the number of features that survive
# zero-variance filtering in each training split.
def requested_counts(n_features):
    levels = [
        ("fraction_0.5pct", max(1, round(n_features * 0.005))),
        ("fraction_1pct", max(1, round(n_features * 0.01))),
        ("fixed_20", 20),
        ("fixed_100", 100),
        ("fraction_10pct", max(1, round(n_features * 0.10))),
        ("fraction_25pct", max(1, round(n_features * 0.25))),
        ("fraction_50pct", max(1, round(n_features * 0.50))),
        ("fraction_75pct", max(1, round(n_features * 0.75))),
        ("all", n_features),
    ]
    # Keep requested counts within the available feature count and avoid
    # duplicate settings within a split.
    output, seen = [], set()
    for label, count in levels:
        count = min(int(count), n_features)
        if count not in seen:
            output.append((label, count))
            seen.add(count)
    return output

rows = []

for split_no in range(5):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.25,
        stratify=y,
        random_state=42 + split_no,
    )
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit preprocessing on training data only.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    variance_filter = VarianceThreshold()
    X_train = variance_filter.fit_transform(X_train_imp)
    X_test = variance_filter.transform(X_test_imp)
    n_available = X_train.shape[1]

    print(
        f"\nSplit {split_no + 1}/5: {X.shape[1]} original features; "
        f"{n_available} survive zero-variance filtering",
        flush=True,
    )

    # Calculate ANOVA scores using training data only.
    scores, _ = f_classif(X_train, y_train)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    anova_order = np.argsort(scores)[::-1]

    # Include majority-class baseline once per split.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_pred = baseline.predict(np.zeros((len(y_test), 1)))
    rows.append({
        "split": split_no + 1,
        "method": "majority_baseline",
        "setting": "0_features",
        "requested_features": 0,
        "features_used": 0,
        "accuracy": accuracy_score(y_test, baseline_pred),
        "macro_f1": f1_score(y_test, baseline_pred, average="macro", zero_division=0),
    })

    for setting, k in requested_counts(n_available):
        random_rng = np.random.default_rng(1000 + split_no * 100 + k)
        random_order = random_rng.permutation(n_available)

        for method, order in [
            ("ANOVA", anova_order),
            ("random", random_order),
        ]:
            selected = order[:k]
            Xtr_sel = X_train[:, selected]
            Xte_sel = X_test[:, selected]

            scaler = StandardScaler()
            Xtr_scaled = scaler.fit_transform(Xtr_sel)
            Xte_scaled = scaler.transform(Xte_sel)

            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            )
            model.fit(Xtr_scaled, y_train)
            pred = model.predict(Xte_scaled)

            rows.append({
                "split": split_no + 1,
                "method": method,
                "setting": setting,
                "requested_features": k,
                "features_used": len(selected),
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(
                    y_test, pred, average="macro", zero_division=0
                ),
            })

            print(
                f"Split {split_no + 1} | {method:6s} | {setting:18s} | "
                f"k={k:5d} | accuracy={accuracy_score(y_test, pred):.4f} | "
                f"macro-F1={f1_score(y_test, pred, average='macro', zero_division=0):.4f}",
                flush=True,
            )

results = pd.DataFrame(rows)
results.to_csv(f"{OUT}/trial_results.csv", index=False)

summary = (
    results.groupby(["method", "setting", "requested_features"], dropna=False)
    .agg(
        features_used_mean=("features_used", "mean"),
        features_used_min=("features_used", "min"),
        features_used_max=("features_used", "max"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        splits=("split", "count"),
    )
    .reset_index()
)
summary.to_csv(f"{OUT}/summary.csv", index=False)

print("\nSUMMARY:\n", summary.to_string(index=False), flush=True)
print(f"\nSaved trial and summary CSVs in {OUT}/", flush=True)
