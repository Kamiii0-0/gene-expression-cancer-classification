
from pathlib import Path
import numpy as np
import pandas as pd

R = Path("results")
checks = []

def check(name, condition, detail=""):
    status = bool(condition)
    checks.append(status)
    mark = "PASS" if status else "CHECK"
    print(f"[{mark}] {name}" + (f" | {detail}" if detail else ""))

required_files = [
    "trial_results.csv",
    "summary.csv",
    "accuracy_vs_genes.png",
    "macro_f1_vs_genes.png",
    "missing_data_trial_results.csv",
    "missing_data_summary.csv",
    "missing_data_accuracy.png",
    "missing_data_macro_f1.png",
    "classwise_trial_results.csv",
    "classwise_summary.csv",
    "classwise_confusion_matrix.csv",
    "repeated_label_control_trials.csv",
    "repeated_label_control_summary.csv",
]

print("PROJECT RESULTS AUDIT\n")

for filename in required_files:
    check(
        f"File exists: {filename}",
        (R / filename).is_file()
    )

try:
    feature_trials = pd.read_csv(R / "trial_results.csv")
    feature_summary = pd.read_csv(R / "summary.csv")
    missing_trials = pd.read_csv(R / "missing_data_trial_results.csv")
    missing_summary = pd.read_csv(R / "missing_data_summary.csv")
    classwise_trials = pd.read_csv(R / "classwise_trial_results.csv")
    classwise_summary = pd.read_csv(R / "classwise_summary.csv")
    confusion = pd.read_csv(
        R / "classwise_confusion_matrix.csv",
        index_col=0
    )
    label_trials = pd.read_csv(R / "repeated_label_control_trials.csv")

    print("\nROW COUNTS")
    check("Feature experiment has 30 rows", len(feature_trials) == 30)
    check("Feature summary has 6 rows", len(feature_summary) == 6)
    check("Missingness experiment has 30 rows", len(missing_trials) == 30)
    check("Missingness summary has 6 rows", len(missing_summary) == 6)
    check("Class-wise trials have 25 rows", len(classwise_trials) == 25)
    check("Class-wise summary has 5 classes", len(classwise_summary) == 5)
    check("Confusion matrix is 5 by 5", confusion.shape == (5, 5))
    check("Label-control experiment has 55 rows", len(label_trials) == 55)

    # Check that feature summary metrics match the raw trials.
    calculated = feature_trials.groupby("retained_fraction").agg(
        accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean")
    ).reset_index()

    merged = calculated.merge(
        feature_summary,
        on="retained_fraction",
        suffixes=("_calculated", "_saved")
    )

    summary_matches = (
        len(merged) == 6
        and np.allclose(
            merged["accuracy_mean_calculated"],
            merged["accuracy_mean_saved"],
            atol=1e-9
        )
        and np.allclose(
            merged["macro_f1_mean_calculated"],
            merged["macro_f1_mean_saved"],
            atol=1e-9
        )
    )

    check("Feature summary matches raw trials", summary_matches)

    # The 0% missingness runs should match the 10%-feature runs.
    # Both use 2,026 selected features and the same five splits.
    f10 = feature_trials[
        feature_trials["retained_fraction"] == 0.10
    ][["trial", "accuracy", "macro_f1"]].sort_values("trial")

    m0 = missing_trials[
        missing_trials["missing_percent"] == 0
    ][["trial", "accuracy", "macro_f1"]].sort_values("trial")

    baseline_matches = (
        len(f10) == 5 and len(m0) == 5
        and np.allclose(f10["accuracy"], m0["accuracy"], atol=1e-9)
        and np.allclose(f10["macro_f1"], m0["macro_f1"], atol=1e-9)
    )

    check("Feature baseline matches 0% missingness baseline",
          baseline_matches)

    # Five test folds of 201 samples each were accumulated in this matrix.
    check(
        "Confusion matrix totals 1,005 predictions",
        int(confusion.to_numpy().sum()) == 1005
    )

    print("\nAUDIT COMPLETE")
    if all(checks):
        print("All structural and consistency checks passed.")
    else:
        print("Some checks need review. See the [CHECK] lines above.")

except Exception as exc:
    print("\nAudit stopped because of an error:")
    print(type(exc).__name__ + ":", exc)
