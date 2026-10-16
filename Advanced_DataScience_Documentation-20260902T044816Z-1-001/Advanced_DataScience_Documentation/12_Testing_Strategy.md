# Testing Strategy
## Unit
Validation, transformations, tier mapping, threshold logic, predictor service.

## Integration
FastAPI + artifact + preprocessing; React + API contract.

## API
Valid request, missing field, wrong type, malformed payload, model unavailable, unknown route and rate limit.

## UI
Form, validation, loading, success, failure and responsive behavior.

## Data Science
Schema validation, reproducible split, no preprocessing leakage, CV correctness, artifact loading and prediction consistency.

## Sample Test Cases
| ID | Scenario | Expected |
|---|---|---|
| TEST-001 | Valid score request | 200 + score + tier + model version |
| TEST-002 | Missing feature | 422 |
| TEST-003 | Wrong type | 422 |
| TEST-004 | Model unavailable | 503 |
| TEST-005 | Threshold boundary | Documented tier |
| TEST-006 | Same input/artifact | Deterministic result |
| TEST-007 | Fixed split seed | Reproducible split |
| TEST-008 | Preprocessing | No test-data fitting |
| TEST-009 | Backend unavailable | UI error state |
| TEST-010 | Model metadata | Correct version/metrics |

## Performance
Measure cold start, warm inference, p50/p95 latency and concurrent requests.
