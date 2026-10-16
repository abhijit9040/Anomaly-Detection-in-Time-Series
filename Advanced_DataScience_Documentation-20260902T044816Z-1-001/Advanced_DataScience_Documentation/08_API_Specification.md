# API Specification
Base URL: `/api/v1`

## GET /health
Returns service/model health.

## POST /score
Request:
```json
{"features":{"<dataset_feature_1>":0,"<dataset_feature_2>":0}}
```
Exact feature names must come from the actual dataset schema.

Response:
```json
{"riskScore":0.0,"riskTier":"LOW|MEDIUM|HIGH","modelVersion":"adaboost-v1","timestamp":"ISO-8601"}
```

## GET /model
Returns non-sensitive model metadata and evaluation metrics.

## GET /model-card
Returns model-card information or a link.

## Errors
Use structured errors with code, message and request ID. 422/400 for validation, 429 for rate limit, 503 for unavailable model, 500 for unexpected server errors.
