# Advanced Data Science Architecture
## Problem Formulation
Supervised binary classification of the dataset's documented default-risk target.

## Dataset Design
Document exact source/version, row count, features, target, data types, missingness, duplicates, class balance, leakage risks and applicability to the micro-loan context. Do not invent schema details.

## EDA
Target distribution, missingness, distributions, outliers, correlations where appropriate, feature-target relationships and train/test distribution comparison. Do not infer causality from correlation.

## Feature Engineering
Use leakage-safe transformations. Every engineered feature must have a rationale. Address missingness, outliers, scaling where required, feature selection and potential leakage.

## Baseline
Use a simple baseline such as a dummy classifier and/or logistic regression. Compare it with AdaBoost.

## Primary Model
AdaBoost classifier, grounded conceptually in the supplied 1997 boosting paper. Explain weak learners, iterative boosting, emphasis on difficult observations and ensemble prediction. Do not claim the paper specifies the exact modern scikit-learn implementation.

## Candidate Models
AdaBoost is mandatory. Optional comparison: Random Forest, Gradient Boosting or HistGradientBoosting.

## Training Protocol
80/20 stratified train-test split. CV, tuning and threshold selection happen only inside the training partition. The test partition remains untouched until final evaluation.

## Hyperparameter Optimization
Investigate number of estimators, learning rate and base-estimator complexity. Record search space, CV strategy, selection metric and chosen configuration.

## Evaluation
Required: ROC-AUC and precision/recall at a chosen threshold. Also report confusion matrix, ROC/PR curves, class counts and threshold rationale.

## Threshold
Select threshold using training/CV, not the final test set. Explain precision/recall trade-offs.

## Explainability and Error Analysis
Use appropriate feature importance, false-positive/false-negative analysis and boundary-case analysis. Feature importance is not causality.

## Experiment Tracking
Record run ID, dataset version, split seed, pipeline version, model, hyperparameters, CV strategy, metrics, threshold and artifact version.

## Business Impact
Translate results into flagged volume, captured defaults at threshold, false-positive workload and evidence required before production use.
