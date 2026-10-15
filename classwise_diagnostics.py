
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DATA = "data/TCGA-PANCAN-HiSeq-801x20531/"
RESULTS = "results"
K = 2026
CLASSES = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]

Xdf = pd.read_csv(DATA + "data.csv").set_index("Unnamed: 0")
ydf = pd.read_csv(DATA + "labels.csv").set_index("Unnamed: 0")

ids = Xdf.index.intersection(ydf.index)
X = Xdf.loc[ids].to_numpy(dtype=np.float32)
y = ydf.loc[ids, "Class"].to_numpy()

records = []
combined_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

for trial in range(5):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.25,
        stratify=y,
        random_state=42 + trial
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

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

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = classification_report(
        y_test,
        pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    combined_cm += cm

    for class_name in CLASSES:
        records.append({
            "trial": trial + 1,
            "class": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1": report[class_name]["f1-score"],
            "support": report[class_name]["support"]
        })

    print(
        f"Trial {trial + 1}: "
        f"accuracy={(pred == y_test).mean():.4f}"
    )

details = pd.DataFrame(records)
details.to_csv(
    f"{RESULTS}/classwise_trial_results.csv",
    index=False
)

summary = details.groupby("class").agg(
    precision_mean=("precision", "mean"),
    precision_sd=("precision", "std"),
    recall_mean=("recall", "mean"),
    recall_sd=("recall", "std"),
    f1_mean=("f1", "mean"),
    f1_sd=("f1", "std")
).reindex(CLASSES)

summary.to_csv(
    f"{RESULTS}/classwise_summary.csv"
)

print("\nPER-CLASS MEAN METRICS ACROSS FIVE SPLITS")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

cm_df = pd.DataFrame(
    combined_cm,
    index=[f"True_{c}" for c in CLASSES],
    columns=[f"Pred_{c}" for c in CLASSES]
)
cm_df.to_csv(f"{RESULTS}/classwise_confusion_matrix.csv")

print("\nCONFUSION MATRIX")
print("(Counts accumulated across five split runs; test samples can recur.)")
print(cm_df.to_string())
print("\nSaved class-wise results in the results/ folder.")
