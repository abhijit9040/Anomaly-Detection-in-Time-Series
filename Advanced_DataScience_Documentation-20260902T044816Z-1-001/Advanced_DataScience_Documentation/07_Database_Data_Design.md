# Database / Data Design
## MVP Decision
A database is not mandatory. The core product can be stateless: request → preprocessing → model → response.

## Model Artifact Metadata
Store model name/version, dataset identifier/version, training date, feature schema, preprocessing version, hyperparameters, threshold and evaluation metrics.

## Optional Prediction History
If history is required, store prediction ID, model version, non-reversible input hash, score, tier, timestamp and operational metadata excluding raw sensitive inputs.

## Data Pipeline
Raw dataset → validated dataset → training representation → feature matrix → model artifact.

Do not create a database merely to satisfy a checklist.
