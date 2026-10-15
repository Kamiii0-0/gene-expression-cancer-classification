
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA = "data/TCGA-PANCAN-HiSeq-801x20531/"
RESULTS = "results"
K = 2026
N_SHUFFLES = 10

os.makedirs(RESULTS, exist_ok=True)

Xdf = pd.read_csv(DATA + "data.csv").set_index("Unnamed: 0")
ydf = pd.read_csv(DATA + "labels.csv").set_index("Unnamed: 0")

ids = Xdf.index.intersection(ydf.index)
X = Xdf.loc[ids].to_numpy(dtype=np.float32)
y = ydf.loc[ids, "Class"].to_numpy()

records = []

def build_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(),
        SelectKBest(f_classif, k=K),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )

for trial in range(5):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.25,
        stratify=y,
        random_state=42 + trial
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Baseline: genuine training labels
    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    records.append({
        "trial": trial + 1,
        "condition": "real_labels",
        "shuffle": 0,
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(
            y_test, pred, average="macro"
        )
    })

    # Control: shuffle training labels only.
    # Test labels remain unchanged.
    for shuffle in range(1, N_SHUFFLES + 1):
        rng = np.random.default_rng(
            50000 + trial * 100 + shuffle
        )
        y_train_shuffled = rng.permutation(y_train)

        model = build_model()
        model.fit(X_train, y_train_shuffled)
        pred = model.predict(X_test)

        records.append({
            "trial": trial + 1,
            "condition": "shuffled_labels",
            "shuffle": shuffle,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(
                y_test, pred, average="macro"
            )
        })

    print(
        f"Completed split {trial + 1}/5",
        flush=True
    )

results = pd.DataFrame(records)
results.to_csv(
    f"{RESULTS}/repeated_label_control_trials.csv",
    index=False
)

summary = results.groupby("condition").agg(
    runs=("accuracy", "count"),
    accuracy_mean=("accuracy", "mean"),
    accuracy_sd=("accuracy", "std"),
    accuracy_min=("accuracy", "min"),
    accuracy_max=("accuracy", "max"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_sd=("macro_f1", "std"),
    macro_f1_min=("macro_f1", "min"),
    macro_f1_max=("macro_f1", "max")
).reset_index()

summary.to_csv(
    f"{RESULTS}/repeated_label_control_summary.csv",
    index=False
)

print("\nREPEATED LABEL-CONTROL SUMMARY")
print(summary.to_string(index=False))
