# Architecture Decision Records
## ADR-001 — FastAPI
Use FastAPI because it keeps model serving close to the Python/scikit-learn artifact and provides typed validation.

## ADR-002 — React
Use React + Tailwind for the analyst-facing product UI.

## ADR-003 — AdaBoost
AdaBoost is the primary model because it is the explicit project objective and connects the supplied boosting reference to a modern implementation.

## ADR-004 — 80/20 Stratified Split
Required by the project brief and preserves class proportions.

## ADR-005 — CV Only on Training
Prevents test-set leakage and preserves an honest final evaluation.

## ADR-006 — CPU First
GPU is not required for the core AdaBoost workload.

## ADR-007 — No Mandatory Database
Avoid architecture complexity when the MVP can be stateless.

## ADR-008 — Human Decision Support
The model provides a risk signal, not an autonomous lending decision.
