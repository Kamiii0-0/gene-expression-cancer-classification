import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA = "data/TCGA-PANCAN-HiSeq-801x20531/"
OUT = "results/extended_feature_experiment"
N_RANDOM = 100
K = 20

print("Loading dataset...", flush=True)

Xdf = pd.read_csv(DATA + "data.csv").set_index("Unnamed: 0")
ydf = pd.read_csv(DATA + "labels.csv").set_index("Unnamed: 0")
ids = Xdf.index.intersection(ydf.index)
X = Xdf.loc[ids].to_numpy(dtype=np.float32)
y = ydf.loc[ids, "Class"].to_numpy()

rows = []

for split_no in range(5):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.25,
        stratify=y,
        random_state=42 + split_no,
    )

    y_train, y_test = y[train_idx], y[test_idx]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X[train_idx])
    X_test = imputer.transform(X[test_idx])

    selector = VarianceThreshold()
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    scores, _ = f_classif(X_train, y_train)
    scores = np.nan_to_num(scores, nan=-np.inf)
    anova_order = np.argsort(scores)[::-1]
    n_features = X_train.shape[1]

    if K > n_features:
        raise ValueError("Fewer than 20 available features.")

    rng = np.random.default_rng(5000 + split_no)
    random_subsets = [
        rng.choice(n_features, size=K, replace=False)
        for _ in range(N_RANDOM)
    ]

    methods = [("ANOVA", anova_order[:K])]
    methods += [
        (f"random_{i+1:03d}", subset)
        for i, subset in enumerate(random_subsets)
    ]

    for method, selected in methods:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train[:, selected])
        Xte = scaler.transform(X_test[:, selected])

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)

        rows.append({
            "split": split_no + 1,
            "method": method,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(
                y_test, pred, average="macro", zero_division=0
            ),
        })

    print(f"Completed split {split_no + 1}/5", flush=True)

results = pd.DataFrame(rows)
results.to_csv(f"{OUT}/repeated_random_trials.csv", index=False)

anova = results[results["method"] == "ANOVA"]
random_results = results[results["method"].str.startswith("random_")]

print("\nANOVA: 20 selected genes")
print(anova[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(5))

print("\nRandom selection: 100 subsets per split")
print(random_results[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(5))

print("\nRandom-subset performance by split:")
print(
    random_results.groupby("split")[["accuracy", "macro_f1"]]
    .mean()
    .round(5)
)

print(f"\nSaved raw results to {OUT}/repeated_random_trials.csv")
