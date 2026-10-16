# Cost Analysis
## MVP Drivers
- FastAPI compute.
- React static hosting.
- Model artifact storage.
- Optional monitoring and CI/CD.

No LLM/vector-database cost is required for the stated project.

## GPU
GPU is listed as a constraint, but AdaBoost should be evaluated CPU-first. Measure training time before requesting GPU infrastructure.

## Optimization
Keep inference stateless, load the model once, avoid unnecessary databases and scale only when traffic requires it.
