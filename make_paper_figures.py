"""Create paper-ready figures from the saved cancer-classification results.

Run from the project root:
    python make_paper_figures.py

Inputs are read from results/. Figures are written to results/paper_figures/
with both PNG (300 dpi) and PDF (vector) formats where appropriate.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
OUT = RESULTS / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Consistent typography and legibility for paper figures.
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "savefig.dpi": 300,
})


def need_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the relevant experiment script first and launch this script "
            "from the cancer_project directory."
        )


def save_figure(fig, stem: str) -> None:
    png_path = OUT / f"{stem}.png"
    pdf_path = OUT / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def errorbar_panel(ax, x, mean, std, title, ylabel, ylim=None):
    ax.errorbar(
        x, mean, yerr=std,
        marker="o", markersize=5, linewidth=1.8,
        capsize=4, elinewidth=1.2
    )
    ax.set_title(title)
    ax.set_xlabel("Percentage")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)


# ---------------------------------------------------------------------------
# Figure 1: feature-reduction performance
# ---------------------------------------------------------------------------
feature_path = RESULTS / "summary.csv"
need_file(feature_path)
feature = pd.read_csv(feature_path).sort_values("retained_fraction")
required_feature_cols = {
    "retained_fraction", "accuracy_mean", "accuracy_std",
    "macro_f1_mean", "macro_f1_std"
}
missing_cols = required_feature_cols - set(feature.columns)
if missing_cols:
    raise ValueError(f"{feature_path} is missing columns: {sorted(missing_cols)}")

x_feature = feature["retained_fraction"].to_numpy() * 100
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
errorbar_panel(
    axes[0], x_feature,
    feature["accuracy_mean"].to_numpy(),
    feature["accuracy_std"].fillna(0).to_numpy(),
    "(a) Test accuracy", "Mean accuracy", (0, 1.02)
)
errorbar_panel(
    axes[1], x_feature,
    feature["macro_f1_mean"].to_numpy(),
    feature["macro_f1_std"].fillna(0).to_numpy(),
    "(b) Macro-F1", "Mean macro-F1", (0, 1.02)
)
for ax in axes:
    ax.set_xlabel("Nominal feature-retention level (%)")
fig.suptitle("Classification performance versus feature retention", y=1.03)
fig.text(
    0.5, -0.035,
    "Error bars show ±1 standard deviation across five overlapping test splits. "
    "The 0% point is a majority-class baseline, not a trained gene-feature model.",
    ha="center", va="top", fontsize=8
)
fig.tight_layout()
save_figure(fig, "figure1_feature_reduction")


# ---------------------------------------------------------------------------
# Figure 2 and 3: simulated missing-measurement performance, full and zoomed
# ---------------------------------------------------------------------------
missing_path = RESULTS / "missing_data_summary.csv"
need_file(missing_path)
missing = pd.read_csv(missing_path).sort_values("missing_percent")
required_missing_cols = {
    "missing_percent", "accuracy_mean", "accuracy_std",
    "macro_f1_mean", "macro_f1_std"
}
missing_cols = required_missing_cols - set(missing.columns)
if missing_cols:
    raise ValueError(f"{missing_path} is missing columns: {sorted(missing_cols)}")

x_missing = missing["missing_percent"].to_numpy()

def make_missing_figure(stem: str, zoom: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    y_limits = (0.95, 1.005) if zoom else (0, 1.02)
    errorbar_panel(
        axes[0], x_missing,
        missing["accuracy_mean"].to_numpy(),
        missing["accuracy_std"].fillna(0).to_numpy(),
        "(a) Test accuracy", "Mean accuracy", y_limits
    )
    errorbar_panel(
        axes[1], x_missing,
        missing["macro_f1_mean"].to_numpy(),
        missing["macro_f1_std"].fillna(0).to_numpy(),
        "(b) Macro-F1", "Mean macro-F1", y_limits
    )
    for ax in axes:
        ax.set_xlabel("Individual measurements masked (%)")
    suffix = "zoomed view" if zoom else "full score scale"
    fig.suptitle(f"Performance under simulated random missingness ({suffix})", y=1.03)
    note = (
        "Zoomed y-axis: 0.95–1.005; small score differences are magnified. "
        "Error bars show ±1 standard deviation across five overlapping test splits."
        if zoom else
        "Full y-axis shown to preserve context. Error bars show ±1 standard deviation "
        "across five overlapping test splits. Missingness is simulated independently "
        "at the individual-measurement level."
    )
    fig.text(0.5, -0.035, note, ha="center", va="top", fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)

make_missing_figure("figure2_missingness_full_scale", zoom=False)
make_missing_figure("figure3_missingness_zoomed", zoom=True)


# ---------------------------------------------------------------------------
# Figure 4: class-wise normalized confusion matrix with counts
# ---------------------------------------------------------------------------
cm_path = RESULTS / "classwise_confusion_matrix.csv"
need_file(cm_path)
cm_df = pd.read_csv(cm_path, index_col=0)
classes = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]
row_names = [f"True_{c}" for c in classes]
col_names = [f"Pred_{c}" for c in classes]
if not set(row_names).issubset(cm_df.index) or not set(col_names).issubset(cm_df.columns):
    raise ValueError(
        "Confusion matrix labels did not match the expected classes. "
        f"Rows found: {list(cm_df.index)}; columns found: {list(cm_df.columns)}"
    )
counts = cm_df.loc[row_names, col_names].to_numpy(dtype=int)
row_totals = counts.sum(axis=1, keepdims=True)
normalized = np.divide(
    counts, row_totals,
    out=np.zeros_like(counts, dtype=float),
    where=row_totals != 0
)

fig, ax = plt.subplots(figsize=(6.7, 5.5))
image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="equal")
ax.set_xticks(range(len(classes)), labels=classes, rotation=35, ha="right")
ax.set_yticks(range(len(classes)), labels=classes)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Row-normalized confusion matrix")
for i in range(len(classes)):
    for j in range(len(classes)):
        pct = normalized[i, j] * 100
        count = counts[i, j]
        text_color = "white" if normalized[i, j] >= 0.55 else "black"
        ax.text(
            j, i, f"{count}\n({pct:.1f}%)",
            ha="center", va="center", color=text_color, fontsize=9
        )
bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
bar.set_label("Fraction of each true class")
fig.text(
    0.5, 0.015,
    "Counts are accumulated over five overlapping test splits; a sample may recur.",
    ha="center", va="bottom", fontsize=8
)
fig.tight_layout(rect=(0, 0.045, 1, 1))
save_figure(fig, "figure4_confusion_matrix")


# ---------------------------------------------------------------------------
# Figure 5: genuine versus shuffled training-label control
# ---------------------------------------------------------------------------
label_path = RESULTS / "repeated_label_control_trials.csv"
need_file(label_path)
label_results = pd.read_csv(label_path)
required_label_cols = {"condition", "accuracy", "macro_f1"}
missing_cols = required_label_cols - set(label_results.columns)
if missing_cols:
    raise ValueError(f"{label_path} is missing columns: {sorted(missing_cols)}")

condition_order = ["real_labels", "shuffled_labels"]
condition_labels = ["Genuine training labels", "Shuffled training labels"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, metric, title in [
    (axes[0], "accuracy", "(a) Test accuracy"),
    (axes[1], "macro_f1", "(b) Macro-F1"),
]:
    data = [
        label_results.loc[label_results["condition"] == cond, metric].dropna().to_numpy()
        for cond in condition_order
    ]
    if any(len(values) == 0 for values in data):
        raise ValueError(f"No values found for both label-control conditions in {label_path}")
    ax.boxplot(data, tick_labels=condition_labels, showfliers=True)
    # Show the observed values so five genuine-label results are transparent.
    for i, values in enumerate(data, start=1):
        jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(i + jitter, values, s=15, alpha=0.7, zorder=3)
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", "-").title())
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", labelrotation=12)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Classification performance with genuine versus shuffled labels", y=1.03)
fig.text(
    0.5, -0.035,
    "Five genuine-label runs and 50 shuffled-label runs. Test sets overlap across splits; "
    "shuffled-label runs are a control, not independent external validation.",
    ha="center", va="top", fontsize=8
)
fig.tight_layout()
save_figure(fig, "figure5_label_control")


# Write concise draft captions and figure caveats next to the files.
notes = """PAPER FIGURE NOTES AND DRAFT CAPTIONS

Figure 1. Test accuracy and macro-F1 versus feature-retention level. Error bars show
one standard deviation across five repeated stratified 75:25 splits. The 0% point
uses a majority-class DummyClassifier baseline. Test sets overlap across the splits.

Figure 2. Test accuracy and macro-F1 under simulated random masking of individual
gene-expression measurements. The full 0–1 score axis is shown. The data contain no
native missing values; this is a simulation, not a test of naturally occurring missingness.

Figure 3. Zoomed view of Figure 2 with the y-axis restricted to 0.95–1.005 to make
small score differences visible. This truncated scale must be stated if used in the paper.

Figure 4. Row-normalized confusion matrix with accumulated counts and row percentages
across five test splits. Test sets overlap, so accumulated counts are prediction events,
not a count of unique independent samples.

Figure 5. Test metrics for genuine training labels versus shuffled training labels.
There are five genuine-label runs and 50 shuffled-label runs. The test sets overlap and
this control does not rule out all possible leakage or establish external generalization.

General caution: observed very high performance on this dataset should be described as
preliminary and dataset-specific. These figures do not establish clinical utility or
independent-patient generalization. Confirm data provenance and validation design before
making broader claims.
"""
(OUT / "figure_notes_and_draft_captions.txt").write_text(notes, encoding="utf-8")
print(f"Saved: {OUT / 'figure_notes_and_draft_captions.txt'}")
print("\nAll paper figures created successfully.")
