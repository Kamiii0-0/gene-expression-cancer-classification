
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA = "data/TCGA-PANCAN-HiSeq-801x20531/"
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# Use the feature count from our earlier experiment.
K = 2026
MISSING_LEVELS = [0, 10, 25, 50, 75, 90]

print("Loading dataset...", flush=True)

Xdf = pd.read_csv(DATA + "data.csv").set_index("Unnamed: 0")
ydf = pd.read_csv(DATA + "labels.csv").set_index("Unnamed: 0")

ids = Xdf.index.intersection(ydf.index)
X = Xdf.loc[ids].to_numpy(dtype=np.float32)
y = ydf.loc[ids, "Class"].to_numpy()

print("Samples:", len(y), "| Features:", X.shape[1], flush=True)
print("Selected features per model:", K, flush=True)

rows = []

for trial in range(5):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.25,
        stratify=y,
        random_state=42 + trial
    )

    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    for missing_percent in MISSING_LEVELS:
        rate = missing_percent / 100.0

        X_train_masked = X_train.copy()
        X_test_masked = X_test.copy()

        # Use reproducible, independent masks for train and test.
        rng_train = np.random.default_rng(
            10000 + trial * 100 + missing_percent
        )
        rng_test = np.random.default_rng(
            20000 + trial * 100 + missing_percent
        )

        if rate > 0:
            train_mask = (
                rng_train.random(X_train_masked.shape) < rate
            )
            test_mask = (
                rng_test.random(X_test_masked.shape) < rate
            )

            X_train_masked[train_mask] = np.nan
            X_test_masked[test_mask] = np.nan

        actual_train_missing = (
            np.isnan(X_train_masked).mean() * 100
        )
        actual_test_missing = (
            np.isnan(X_test_masked).mean() * 100
        )

        # Imputation, feature filtering and selection are fitted
        # using training data only.
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(),
            SelectKBest(f_classif, k=K),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )

        model.fit(X_train_masked, y_train)
        pred = model.predict(X_test_masked)

        accuracy = accuracy_score(y_test, pred)
        macro_f1 = f1_score(
            y_test, pred, average="macro"
        )

        rows.append({
            "trial": trial + 1,
            "missing_percent": missing_percent,
            "features_selected": K,
            "train_missing_actual_pct": actual_train_missing,
            "test_missing_actual_pct": actual_test_missing,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        print(
            f"Trial {trial + 1}/5 | "
            f"Missing: {missing_percent}% | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f}",
            flush=True
        )

# Save every individual trial.
results = pd.DataFrame(rows)
results.to_csv(
    f"{RESULTS}/missing_data_trial_results.csv",
    index=False
)

# Calculate mean performance and variation across trials.
summary = results.groupby("missing_percent").agg(
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std"),
    train_missing_mean=("train_missing_actual_pct", "mean"),
    test_missing_mean=("test_missing_actual_pct", "mean")
).reset_index()

summary.to_csv(
    f"{RESULTS}/missing_data_summary.csv",
    index=False
)

print("\nMISSING-DATA EXPERIMENT SUMMARY:")
print(summary.to_string(index=False))

# Plot accuracy and macro-F1 separately.
for metric in ["accuracy", "macro_f1"]:
    plt.figure(figsize=(8, 5))

    x = summary["missing_percent"]
    mean = summary[f"{metric}_mean"]
    std = summary[f"{metric}_std"].fillna(0)

    plt.errorbar(
        x, mean, yerr=std,
        marker="o", capsize=4
    )

    plt.xlabel("Expression measurements hidden (%)")
    plt.ylabel("Mean test " + metric.replace("_", "-"))
    plt.title(
        metric.replace("_", " ").title()
        + " under simulated missing measurements"
    )
    plt.xticks(MISSING_LEVELS)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS}/missing_data_{metric}.png",
        dpi=300
    )
    plt.close()

print("\nDone! Missing-data results and graphs saved in results/.")
