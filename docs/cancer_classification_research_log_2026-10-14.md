# Cancer Classification Research Log

**Last updated:** 14 October 2026
**Project stage:** Dataset checks and baseline feature-reduction experiment completed; missing-data experiment not yet run.

## 1. Project purpose and research direction

The project is being developed for a paper under the theme **Applications of Mathematics in Engineering/Technology**. The aim is to foreground the mathematical methods used in a machine-learning system, not present the work as only a cancer-biology study.

The broad motivating question is:

> **Can AI identify the type of cancer when most of the genetic data is missing?**

A related question we have discussed is:

> **Does AI learn a meaningful pattern in the data, or does it rely on clues or artefacts specific to the dataset?**

The work completed so far examines a related but narrower question:

> **How does reducing the number of gene-expression features affect a classifier's ability to classify samples into five cancer classes?**

These are related questions, but they are not the same experiment. The current experiment removes whole features from the model. It does not yet simulate missing individual measurements inside otherwise available features.

### Mathematical ideas in the current experiment

- **High-dimensional data representation:** each sample is represented by a vector of gene-expression measurements.
- **Statistical feature selection:** `SelectKBest` with the ANOVA F-statistic (`f_classif`) ranks features according to their relationship with the class labels.
- **Classification:** multinomial logistic regression is used to predict one of five classes.
- **Preprocessing:** median imputation, constant-feature removal, and standardization are implemented in a pipeline.
- **Evaluation:** accuracy and macro-F1 are recorded, along with means and standard deviations across repeated train/test splits.

The planned missing-data experiment will add a further mathematical question: how performance changes as increasing fractions of expression measurements are masked and imputed.

## 2. Dataset

**Dataset name:** Gene Expression Cancer RNA-Seq
**Source:** UCI Machine Learning Repository
**Dataset page:** https://archive.ics.uci.edu/dataset/401/gene%2Bexpression%2Bcancer%2Brna%2Bseq
**DOI:** https://doi.org/10.24432/C5R88H

The local dataset files are:

- `data/TCGA-PANCAN-HiSeq-801x20531/data.csv`
- `data/TCGA-PANCAN-HiSeq-801x20531/labels.csv`

Observed dimensions:

- Expression matrix: **801 samples x 20,531 features**
- Label table: **801 samples x 1 label column**
- Classes: **5**

Observed class counts:

| Class label | Number of samples |
|---|---:|
| BRCA | 300 |
| KIRC | 146 |
| LUAD | 141 |
| PRAD | 136 |
| COAD | 78 |
| **Total** | **801** |

The expression columns are named generically, for example `gene_0`, `gene_1`, ..., rather than providing directly interpretable biological gene names in this file. The row IDs inspected were also generic, for example `sample_0`, `sample_1`, and so on. These IDs are not sufficient by themselves to establish whether samples are from distinct patients. Any claim about patient-level independence needs support from dataset metadata, not just unique row IDs.

## 3. Working environment and files

The main working copy is in Ubuntu under WSL 2, with Python 3.12.3 and a virtual environment.

- Project folder: `~/cancer_project`
- Virtual environment: `~/cancer_project/.venv`
- Main experiment script: `~/cancer_project/experiment.py`
- Results folder: `~/cancer_project/results/`
- Data folder: `~/cancer_project/data/TCGA-PANCAN-HiSeq-801x20531/`

The project has also been backed up as a ZIP in Google Drive under `Cancer Classification Research/cancer_project_backup.zip`, and a Colab notebook has been used for demonstrations. The local WSL project is the main working copy. Files stored only in a Colab runtime may disappear when that runtime resets, so the Drive backup and local folder matter for reproducibility.

To return to the local project:

```bash
cd ~/cancer_project
source .venv/bin/activate
code .
```

To check that the core packages import correctly:

```bash
python -c 'import pandas, sklearn, matplotlib; print("Research environment ready!")'
```

## 4. Dataset quality checks completed

A diagnostic script loaded the expression data and labels and checked table dimensions, ID uniqueness, class counts, exact duplicate expression profiles, constant features, and missing values.

### First diagnostic output

```text
Expression matrix: (801, 20531)
Label table: (801, 1)
Unique expression sample IDs: 801
Unique label sample IDs: 801
Expression IDs duplicated: 0
Label IDs duplicated: 0

Class counts:
Class
BRCA    300
KIRC    146
LUAD    141
PRAD    136
COAD     78
Name: count, dtype: int64

Sample ID examples:
['sample_0', 'sample_1', 'sample_2', 'sample_3', 'sample_4',
 'sample_5', 'sample_6', 'sample_7', 'sample_8', 'sample_9']
```

### Second diagnostic output

```text
Exact duplicate expression profiles: 0
Constant gene features: 267
Missing expression values: 0
Labels missing: 0

Index name: None
Column examples: ['gene_0', 'gene_1', 'gene_2', 'gene_3', 'gene_4',
 'gene_5', 'gene_6', 'gene_7', 'gene_8', 'gene_9']
```

### Interpretation of these checks

- No duplicate row IDs were found in either CSV file.
- No exact duplicate full expression profiles were found.
- **267 features were constant across all 801 samples** in the diagnostic.
- There were no missing expression values and no missing class labels in the original dataset.
- These checks establish some basic data-quality facts. They **do not** prove that there is no patient overlap, no hidden dataset artefact, or no other form of leakage.
- Because the original data has no missing expression measurements, missingness will need to be simulated for the planned missing-data experiment. Simulated missingness must be clearly described as a controlled experiment, not as naturally occurring missing data in this dataset.

## 5. Baseline experiment: feature reduction

### Experimental design

The initial `experiment.py` script implemented the following procedure:

1. Load the expression matrix and align the labels to the matching sample IDs.
2. Convert expression values to NumPy `float32` format.
3. Run **5 stratified random train/test splits**, using `test_size=0.25` and random seeds 42, 43, 44, 45, and 46.
4. For each split, train and evaluate a classifier at each of these feature-retention settings: 0%, 10%, 25%, 50%, 75%, and 100%.
5. For nonzero feature settings, use a scikit-learn pipeline with median imputation, ANOVA F-statistic feature selection, standardization, and logistic regression.
6. For the 0% setting, use a `DummyClassifier(strategy="most_frequent")` as a majority-class baseline.
7. Evaluate predictions using accuracy and macro-F1.
8. Save each trial's scores, calculate means and standard deviations across the five trials, and generate plots.

Main model components:

- `SimpleImputer(strategy="median")`
- `SelectKBest(f_classif, k=k)`
- `StandardScaler()`
- `LogisticRegression(max_iter=2000, class_weight="balanced")`

The feature selector and preprocessing steps are inside the pipeline and are fitted using training data for each split. This avoids fitting those supervised feature-selection steps to the held-out test labels.

### Initial experiment results

Before adding the constant-feature filter, the saved `summary.csv` showed:

| Retained fraction | Features selected | Mean accuracy | Accuracy SD | Mean macro-F1 | Macro-F1 SD |
|---:|---:|---:|---:|---:|---:|
| 0% baseline | 0 | 0.373134 | 0.000000 | 0.108696 | 0.000000 |
| 10% | 2,053 | 0.997015 | 0.002725 | 0.997466 | 0.002313 |
| 25% | 5,132 | 0.995025 | 0.003518 | 0.993654 | 0.004473 |
| 50% | 10,265 | 0.993035 | 0.002725 | 0.991592 | 0.004383 |
| 75% | 15,398 | 0.990050 | 0.008617 | 0.987087 | 0.012703 |
| 100% | 20,531 | 0.984080 | 0.010786 | 0.978964 | 0.015168 |

These results suggested that the 10% feature setting performed better than the all-feature setting on these splits. The result was treated as preliminary and in need of validation, especially because the observed accuracy was unusually high.

## 6. Label-shuffling control

A separate diagnostic compared the model trained with the real training labels against a version in which the training labels were shuffled. It used the same train/test split and a 10% feature setting. The test labels were not shuffled.

Output:

```text
Same train/test split; 10% of features
Normal labels:   accuracy=0.9950, macro-F1=0.9958
Shuffled labels: accuracy=0.2637, macro-F1=0.2053
```

This control showed that performance dropped substantially when the training labels were shuffled. It is evidence against the simplest possibility that the classifier scores highly regardless of the training label relationships.

**It is not proof that the full pipeline is free of all leakage or that the classifier will generalize to independent patients or an external dataset.** It was a single split control, and its output should not be presented as a definitive validation study.

The run also produced warnings from `f_classif` about constant features and invalid values during calculations. Those warnings motivated adding an explicit constant-feature filter to the main experiment pipeline.

## 7. Pipeline cleanup: constant-feature filtering

The script was edited to import `VarianceThreshold` and place `VarianceThreshold()` between median imputation and `SelectKBest`:

```python
model = make_pipeline(
    SimpleImputer(strategy="median"),
    VarianceThreshold(),
    SelectKBest(f_classif, k=k),
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)
```

The feature-count line was also changed from the original-data count:

```python
k = max(1, int(X.shape[1] * fraction))
```

to:

```python
k = max(1, int((X.shape[1] - 267) * fraction))
```

This used the diagnostic count of 267 globally constant features as a simple adjustment so that the feature counts corresponded approximately to percentages of the 20,264 nonconstant features.

### Implementation caveat to revisit

The number 267 comes from the full-dataset diagnostic. `VarianceThreshold()` is fitted within each training split, and a feature could also be constant within a particular training split even if it is not constant across all 801 samples. The current `k` calculation is therefore a practical adjustment, not the most general dynamic implementation. The current run completed successfully, but the implementation should be reviewed for a final reproducible paper before drawing strong conclusions.

## 8. Cleaned-pipeline experiment results

After adding `VarianceThreshold()` and changing the feature-count calculation, the script was rerun successfully.

Exact terminal output:

```text
FINAL RESULTS:
  retained_fraction  genes_retained  accuracy_mean  accuracy_std  macro_f1_mean  macro_f1_std
              0.00               0       0.373134      0.000000       0.108696      0.000000
              0.10            2026       0.997015      0.002725       0.997466      0.002313
              0.25            5066       0.995025      0.003518       0.993654      0.004473
              0.50           10132       0.993035      0.002725       0.991592      0.004383
              0.75           15198       0.990050      0.008617       0.987087      0.012703
              1.00           20264       0.984080      0.010786       0.978964      0.015168
```

### Cleaned result table

| Retained fraction | Features selected | Mean accuracy | Accuracy SD | Mean macro-F1 | Macro-F1 SD |
|---:|---:|---:|---:|---:|---:|
| 0% baseline | 0 | 0.373134 | 0.000000 | 0.108696 | 0.000000 |
| 10% | 2,026 | 0.997015 | 0.002725 | 0.997466 | 0.002313 |
| 25% | 5,066 | 0.995025 | 0.003518 | 0.993654 | 0.004473 |
| 50% | 10,132 | 0.993035 | 0.002725 | 0.991592 | 0.004383 |
| 75% | 15,198 | 0.990050 | 0.008617 | 0.987087 | 0.012703 |
| 100% | 20,264 | 0.984080 | 0.010786 | 0.978964 | 0.015168 |

The means and standard deviations are numerically unchanged from the initial experiment at the displayed precision, even though the selected feature counts have changed. The 10% setting uses 2,026 features (approximately 10% of 20,264 nonconstant features) and has mean accuracy 0.997015, or **about 99.70%**, and mean macro-F1 0.997466, or **about 99.75%**. The all-feature setting after constant filtering has mean accuracy 0.984080, or **about 98.41%**, and mean macro-F1 0.978964, or **about 97.90%**.

The best observed setting among the tested settings was 10% feature retention. This does not establish that 10% is the optimum in general: only six settings were tested, and additional robust validation is needed.

## 9. Current interpretation and limitations

### What we can say from the work so far

- The dataset loaded with 801 samples, 20,531 initial features, and five classes.
- The diagnostic found 267 globally constant features, no exact duplicate full expression profiles, no missing expression values, and no missing labels.
- On the five observed stratified splits, selecting 2,026 features produced mean accuracy of about 99.70% and mean macro-F1 of about 99.75%.
- On those same five-split experiments, the all-feature model performed slightly worse than the 10% selected-feature model.
- Shuffling training labels in one split reduced performance substantially.

### What we cannot conclude yet

- We cannot claim clinical reliability or readiness for medical use.
- We have not established performance on an independent external dataset or confirmed patient-level independence from the IDs alone.
- The label-shuffling control does not exclude every kind of leakage or dataset-specific shortcut.
- The five train/test splits can have overlapping test samples, so the five scores are not independent experimental replicates.
- The current feature-reduction experiment does not answer whether a model can tolerate missing expression measurements.
- The current result does not show that AI identifies cancer type independently of dataset-specific clues.
- The feature fraction giving the highest score in this limited grid is not necessarily the universally optimal fraction.

## 10. Current project files and outputs

The main script currently writes:

- `results/trial_results.csv` - scores for each split and feature setting.
- `results/summary.csv` - means and standard deviations grouped by feature setting.
- `results/accuracy_vs_genes.png` - accuracy plot.
- `results/macro_f1_vs_genes.png` - macro-F1 plot.

A separate `label_control.py` script was also created for the label-shuffling control. Confirm the contents of these files locally if they are changed later; this log records the outputs reported on 14 October 2026.

## 11. Proposed next step: test missing-data robustness

The next planned experiment is to address the original question more directly by masking a controlled proportion of individual expression measurements. Candidate missingness levels discussed are **10%, 25%, 50%, 75%, and 90%**.

A careful version of this experiment should:

1. Preserve the current feature-reduction experiment as a separate baseline.
2. Define exactly how the missing entries are generated and whether masking is random across all sample-feature cells or uses another specified pattern.
3. Fit imputation only within the training pipeline, rather than using information from held-out test samples to estimate imputation values.
4. Evaluate accuracy and macro-F1 at every missingness level and compare them with the no-artificial-missingness baseline.
5. Repeat the evaluation over fixed splits and report mean and standard deviation, while acknowledging overlap between repeated test sets.
6. Include suitable controls and inspect confusion matrices or class-level metrics, not accuracy alone.
7. Record the random seeds and exact masking procedure so the experiment can be reproduced.

The missing-data experiment has **not been run yet** as of this log. Results must be added only after they are actually generated.

## 12. One-paragraph status summary

As of 14 October 2026, the project has loaded and checked the UCI Gene Expression Cancer RNA-Seq dataset (801 samples, 20,531 expression features, five cancer classes), found 267 globally constant features and no missing values or exact duplicate expression profiles, run a five-split logistic-regression feature-selection baseline, and performed a one-split shuffled-label control. After adding `VarianceThreshold()` and adjusting the selected-feature counts, the highest mean score among the tested settings was observed at 10% feature retention: 2,026 features, mean accuracy 99.70% and mean macro-F1 99.75%. These results are promising but preliminary. Patient-level independence, broader leakage risks, and external generalization have not been established. Most importantly, the original missing-data question remains open because the completed experiments concern feature reduction rather than simulated missing measurements.
