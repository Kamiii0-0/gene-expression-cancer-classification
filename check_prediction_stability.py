
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA = "data/TCGA-PANCAN-HiSeq-801x20531/"
K = 2026
MISSING_LEVELS = [0, 10, 25, 50, 75, 90]

Xdf = pd.read_csv(DATA + "data.csv").set_index("Unnamed: 0")
ydf = pd.read_csv(DATA + "labels.csv").set_index("Unnamed: 0")

ids = Xdf.index.intersection(ydf.index)
X = Xdf.loc[ids].to_numpy(dtype=np.float32)
y = ydf.loc[ids, "Class"].to_numpy()

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

    predictions = {}

    for missing_percent in MISSING_LEVELS:
        rate = missing_percent / 100.0
        train_masked = X_train.copy()
        test_masked = X_test.copy()

        rng_train = np.random.default_rng(
            10000 + trial * 100 + missing_percent
        )
        rng_test = np.random.default_rng(
            20000 + trial * 100 + missing_percent
        )

        if rate > 0:
            train_mask = rng_train.random(train_masked.shape) < rate
            test_mask = rng_test.random(test_masked.shape) < rate

            train_masked[train_mask] = np.nan
            test_masked[test_mask] = np.nan

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

        model.fit(train_masked, y_train)
        pred = model.predict(test_masked)
        predictions[missing_percent] = pred

    baseline = predictions[0]

    print(f"\nTRIAL {trial + 1}")
    print("Missing | Accuracy | Changed predictions vs 0%")

    for level in MISSING_LEVELS:
        pred = predictions[level]
        changed = int(np.sum(pred != baseline))

        print(
            f"{level:>6}% | "
            f"{accuracy_score(y_test, pred):.4f}   | "
            f"{changed}/{len(y_test)}"
        )
