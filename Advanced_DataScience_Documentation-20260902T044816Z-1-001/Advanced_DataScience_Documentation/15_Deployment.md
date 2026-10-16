# Deployment Architecture
```text
User → React Static App → HTTPS → FastAPI → Preprocessing + AdaBoost Artifact
```
A single containerized FastAPI service is sufficient for the MVP. React can be deployed as a static application.

Package/version the preprocessing artifact, AdaBoost model, metadata and threshold together.

GPU is not required for the core AdaBoost workflow; use CPU-first deployment. Future scale may add a model registry, object storage, API gateway, autoscaling and drift monitoring only when justified.

Maintain previous model/application versions for rollback.
