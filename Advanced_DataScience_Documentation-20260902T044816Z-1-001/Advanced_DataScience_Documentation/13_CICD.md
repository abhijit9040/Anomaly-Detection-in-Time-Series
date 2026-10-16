# CI/CD
PR → lint → unit tests → data/ML tests → API tests → frontend tests → security/dependency scan → build → artifact validation → deployment → smoke tests.

ML quality gates: schema validation, reproducibility check, artifact creation, metrics presence, threshold documentation and model-card update.

Use `main` as stable and `feature/*` branches for work. Tag application/model versions for releases.
