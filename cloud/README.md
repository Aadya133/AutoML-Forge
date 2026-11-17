# Cloud deployment notes

The included Compose stack is the portable baseline. For AWS/Azure:
1. Build and push `backend` and `frontend` images to ECR/ACR.
2. Use managed PostgreSQL (RDS/Azure Database for PostgreSQL).
3. Run MLflow tracking server as a private service; use object storage (S3/Azure Blob) for artifacts.
4. Store database credentials and MLflow secrets in Secrets Manager/Key Vault.
5. Put the frontend and API behind HTTPS ingress / ALB / Application Gateway.
6. Mount durable storage or use object storage for uploaded CSVs and model artifacts.
7. Configure DVC remote storage and least-privilege credentials.
8. Optionally swap the built-in JWT login for an OIDC provider and enforce tenant-level authorization. (The old demo login stub has already been replaced by JWT authentication.)
9. Run training as a separate worker service/queue for autoscaling and retry support.

This repository supplies cloud-oriented container images and environment boundaries, not an automatically provisioned cloud account or Terraform deployment.
