
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold

DATA = "data/TCGA-PANCAN-HiSeq-801x20531/"
os.makedirs("results", exist_ok=True)

print("Loading dataset...", flush=True)
Xdf = pd.read_csv(DATA + "data.csv").set_index("Unnamed: 0")
ydf = pd.read_csv(DATA + "labels.csv").set_index("Unnamed: 0")

ids = Xdf.index.intersection(ydf.index)
X = Xdf.loc[ids].to_numpy(dtype=np.float32)
y = ydf.loc[ids, "Class"].to_numpy()

print("Samples:", len(y), "| Gene features:", X.shape[1], flush=True)

rows = []
fractions = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

for trial in range(5):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.25,
        stratify=y, random_state=42 + trial
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Count non-constant genes using training data only.
    available_genes = int(
        VarianceThreshold().fit(X_train).get_support().sum()
    )
    for fraction in fractions:
        if fraction == 0:
            model = DummyClassifier(strategy="most_frequent")
            model.fit(np.zeros((len(y_train), 1)), y_train)
            pred = model.predict(np.zeros((len(y_test), 1)))
            k = 0
        else:
            requested_k = max(
                1, int(available_genes * fraction)
            )

            # At 100%, select every feature that survives
            # constant-feature filtering in this training split.
            selector_k = (
                "all" if fraction == 1.0 else requested_k
            )

            model = make_pipeline(
                SimpleImputer(strategy="median"),
                VarianceThreshold(),
                SelectKBest(f_classif, k=selector_k),
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            # Record the actual number of selected features.
            k = int(
                model.named_steps["selectkbest"]
                .get_support()
                .sum()
            )

        rows.append({
            "trial": trial + 1,
            "retained_fraction": fraction,
            "genes_retained": k,
             "available_genes": available_genes if fraction > 0 else 0,
            "actual_retained_fraction": (
                k / available_genes if fraction > 0 else 0.0
            ),
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro")
        })
        print(
            f"Trial {trial+1}/5 | {fraction:.0%} features | "
            f"accuracy={accuracy_score(y_test, pred):.3f} | "
            f"macro-F1={f1_score(y_test, pred, average='macro'):.3f}",
            flush=True
        )

results = pd.DataFrame(rows)
results.to_csv("results/trial_results.csv", index=False)

summary = results.groupby("retained_fraction").agg(
    genes_retained_mean=("genes_retained", "mean"),
    genes_retained_min=("genes_retained", "min"),
    genes_retained_max=("genes_retained", "max"),
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std")
).reset_index()

summary.to_csv("results/summary.csv", index=False)
print("\nFINAL RESULTS:\n", summary.to_string(index=False))

for metric in ["accuracy", "macro_f1"]:
    plt.figure(figsize=(8, 5))
    x = summary["retained_fraction"] * 100
    mean = summary[f"{metric}_mean"]
    std = summary[f"{metric}_std"].fillna(0)
    plt.errorbar(x, mean, yerr=std, marker="o", capsize=4)
    plt.xlabel("Gene features retained (%)")
    plt.ylabel("Test " + metric.replace("_", "-"))
    plt.title(metric.replace("_", " ").title() + " vs. retained gene features")
    plt.xticks([0, 10, 25, 50, 75, 100])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{metric}_vs_genes.png", dpi=300)
    plt.close()

print("\nDone! Results and graphs saved in ~/cancer_project/results/")
