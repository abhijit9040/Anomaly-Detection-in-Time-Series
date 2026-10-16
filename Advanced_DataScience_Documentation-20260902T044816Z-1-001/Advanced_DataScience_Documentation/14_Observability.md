# Observability
## Logs
Request ID, endpoint, status, latency, model version and error code. Never log raw applicant records.

## Metrics
Request rate, error rate, p50/p95 latency, model-load failures, prediction count, risk-tier distribution and model-version usage.

## ML Monitoring
Future production-like extension should monitor input drift, missingness drift, prediction distribution, outcome-based performance when labels arrive, calibration and subgroup performance where appropriate.

## Health
`GET /health` should fail when the required model artifact cannot be loaded.
