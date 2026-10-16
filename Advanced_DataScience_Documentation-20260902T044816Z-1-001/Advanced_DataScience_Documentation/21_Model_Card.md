# Model Card — AdaBoost Credit Risk Scorer
## Model Details
AdaBoost binary classifier implemented with scikit-learn. Intended to flag elevated estimated default risk.

## Intended Use
Educational credit-risk modeling, analyst prioritization and ML lifecycle demonstration.

## Out of Scope
Autonomous approval/rejection, real lending decisions, regulatory claims or use outside the validated population.

## Data
Give Me Some Credit dataset as specified by the brief. Final card must record exact source/version, sample count, features, target, missingness and class balance from the actual dataset.

## Split
80/20 stratified train-test. CV and tuning occur only within the training partition.

## Evaluation
ROC-AUC, precision and recall at the selected threshold, plus confusion matrix and ROC/PR curves.

## Threshold
Record selected threshold, selection objective and training/CV procedure.

## Limitations
Dataset population may not represent micro-loan applicants; historical labels may contain bias; future distribution shift can reduce performance; score is not certainty.

## Ethical Considerations
Discuss fairness, proxy variables, data minimization, human oversight and responsible use.

## Monitoring
Input drift, missingness drift, prediction distribution, calibration and outcome-based performance when labels become available.

## Versioning
Prediction must be traceable to model, preprocessing and threshold versions.
