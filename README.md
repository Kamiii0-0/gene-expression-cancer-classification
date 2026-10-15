# Gene Expression Cancer Classification

## Overview

This project investigates how much gene-expression information can be
reduced while maintaining machine-learning performance in cancer-type
classification.

The study uses a public gene-expression dataset containing 801 samples,
20,531 features, and five cancer classes.

## Research question

How much gene-expression information can a machine-learning model discard
while maintaining cancer-classification performance?

## Experiments

1. **Feature reduction:** Select gene-expression features using ANOVA
   F-statistics and evaluate a logistic-regression classifier.
2. **Simulated missing data:** Artificially mask individual expression
   values to investigate performance under simulated missingness.
3. **Shuffled-label control:** Shuffle training labels to test whether
   classification performance falls when the true label relationship is
   disrupted.
4. **Class-wise evaluation:** Examine performance across the five cancer
   classes using precision, recall, F1-score, and a confusion matrix.

## Dataset

The project uses the UCI Machine Learning Repository's
[Gene Expression Cancer RNA-Seq dataset](https://archive.ics.uci.edu/dataset/401/gene%2Bexpression%2Bcancer%2Brna%2Bseq).

Citation:

Fiorini, S. (2016). *Gene Expression Cancer RNA-Seq*.
UCI Machine Learning Repository.
https://doi.org/10.24432/C5R88H

The raw dataset is not included in this repository. Obtain it from the
official source and place the required files in the expected local data
directory before running the scripts.

## Project structure

- `experiment.py` - main feature-reduction experiment
- `missing_data_experiment.py` - simulated missing-data experiment
- `repeated_label_control.py` - repeated shuffled-label control
- `label_control.py` - label-control experiment script
- `classwise_diagnostics.py` - class-wise metrics and confusion matrix
- `check_prediction_stability.py` - prediction stability checks
- `project_audit.py` - checks project outputs and consistency
- `make_paper_figures.py` - generates paper figures
- `results/` - experiment outputs, summaries, and figures
- `docs/` - research log and supporting documentation

## Preliminary results

Across five stratified 75/25 holdout splits, the feature-reduction experiment achieved a mean accuracy of approximately 99.70% and a mean macro-F1 score of approximately 99.75% when retaining about 10% of the available nonconstant features. This corresponds to approximately 2,025 retained features out of roughly 20,253 available features.

A shuffled-label control produced substantially lower mean accuracy (approximately 25.83%) than the real-label condition (approximately 99.70%). This supports the presence of a relationship between the expression features and class labels under the current experimental setup, but does not rule out every possible source of bias or leakage.

These are preliminary results from the current dataset and evaluation setup. The test splits overlap, patient-level independence has not been established, and external generalization has not been demonstrated. The results should not be interpreted as evidence of clinical diagnostic performance.

The simulated missing-data experiment uses artificial random masking and does not represent every real-world missingness mechanism.

## Environment

The project was developed with Python in Ubuntu under WSL 2.
The experiments use pandas, scikit-learn, and matplotlib.

## Status

Computational experiments and preliminary output audits are complete.
Further validation, literature review, mathematical explanation, and paper
writing remain in progress.
