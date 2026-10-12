$ErrorActionPreference = "Stop"

Write-Host "Starting SupportMind MLflow server..." -ForegroundColor Cyan

# Activate the SupportMind conda environment
conda activate supportmind

# Start MLflow using the existing database and artifact store.
# Security middleware is disabled because this is the local Minikube development setup.
mlflow server `
    --host 0.0.0.0 `
    --port 5000 `
    --disable-security-middleware `
    --backend-store-uri "sqlite:///C:/Users/danie/supportmind-ai/backend/mlflow/mlflow.db" `
    --default-artifact-root "file:///C:/Users/danie/supportmind-ai/backend/mlflow/artifacts"