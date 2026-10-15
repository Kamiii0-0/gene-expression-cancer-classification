import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA = "data/TCGA-PANCAN-HiSeq-801x20531/"

Xdf = pd.read_csv(DATA + "data.csv").set_index("Unnamed: 0")
ydf = pd.read_csv(DATA + "labels.csv").set_index("Unnamed: 0")

ids = Xdf.index.intersection(ydf.index)
X = Xdf.loc[ids].to_numpy(dtype=np.float32)
y = ydf.loc[ids, "Class"].to_numpy()

train_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=0.25,
    stratify=y, random_state=42
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        SelectKBest(f_classif, k=int(X.shape[1] * 0.10)),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )

# Normal-label reference on the same split
normal_model = make_model()
normal_model.fit(X_train, y_train)
normal_pred = normal_model.predict(X_test)

# Shuffle training labels only, then rerun the full pipeline
rng = np.random.default_rng(42)
shuffled_y_train = rng.permutation(y_train)

control_model = make_model()
control_model.fit(X_train, shuffled_y_train)
control_pred = control_model.predict(X_test)

print("Same train/test split; 10% of features")
print(f"Normal labels:   accuracy={accuracy_score(y_test, normal_pred):.4f}, "
      f"macro-F1={f1_score(y_test, normal_pred, average='macro'):.4f}")
print(f"Shuffled labels: accuracy={accuracy_score(y_test, control_pred):.4f}, "
      f"macro-F1={f1_score(y_test, control_pred, average='macro'):.4f}")
