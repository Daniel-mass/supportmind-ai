# Containerization, Observability & API Validation

This section documents how SupportMind AI was containerized, connected to its supporting infrastructure, and validated end-to-end before moving to Kubernetes.

The goal was to make the SupportMind backend reproducible and portable while keeping infrastructure services such as Phoenix, Qdrant, Neo4j, Ollama, and MLflow properly separated.

---

## 1. Containerization Architecture

SupportMind uses Docker for the application layer.

The main application container is:

```text
supportmind-api
```

It runs the FastAPI backend and exposes:

```text
http://localhost:8000
```

The supporting services are:

```text
                    ┌─────────────────────┐
                    │   SupportMind API   │
                    │    FastAPI/Docker    │
                    │      :8000          │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
        ┌───────────┐    ┌───────────┐    ┌───────────┐
        │  Phoenix  │    │  Qdrant   │    │   Neo4j   │
        │   :6006   │    │   :6333   │    │   :7687   │
        └───────────┘    └───────────┘    └───────────┘
              │
              │
              ▼
        Agent Tracing


        ┌───────────┐                 ┌───────────┐
        │  Ollama   │                 │  MLflow   │
        │   :11434  │                 │   :5000   │
        └───────────┘                 └───────────┘
             ▲                              ▲
             │                              │
             └──── Windows Host ───────────┘
```

The SupportMind API is containerized, while Ollama and MLflow remain available on the Windows host during local development.

Phoenix, Qdrant, and Neo4j run as Docker services on the shared Docker network.

---

# 2. Backend Dockerfile

The SupportMind API uses a Python 3.11 slim image.

File:

```text
backend/Dockerfile
```

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### What this does

* Uses Python 3.11.
* Creates `/app` as the working directory.
* Installs the Python dependencies.
* Copies the FastAPI application into the image.
* Exposes port `8000`.
* Starts FastAPI using Uvicorn.

The application listens on `0.0.0.0` inside the container so Docker can expose it to the host machine.

---

# 3. Docker Network

SupportMind uses a shared external Docker network:

```text
supportmind-net
```

Create it once:

```powershell
docker network create --driver bridge --subnet 172.20.0.0/16 supportmind-net
```

Check it:

```powershell
docker network ls
```

Inspect it:

```powershell
docker network inspect supportmind-net
```

The shared network allows containers to communicate using Docker service/container names instead of localhost.

For example:

```text
phoenix:6006
sharebot-qdrant:6333
neo4j-sentinel:7687
```

---

# 4. Docker Compose

The root project contains:

```text
docker-compose.yml
```

The Compose file defines the SupportMind API and Phoenix.

## SupportMind API

```yaml
supportmind-api:
  build:
    context: ./backend
    dockerfile: Dockerfile

  container_name: supportmind-api

  ports:
    - "8000:8000"
```

This means:

```text
Windows localhost:8000
        │
        ▼
Docker container port 8000
        │
        ▼
FastAPI
```

---

# 5. Environment Configuration

The API receives infrastructure locations through environment variables.

```yaml
environment:
  OLLAMA_HOST: http://host.docker.internal:11434
  SUPPORTMIND_LLM_MODEL: qwen3:8b
  SUPPORTMIND_GRAPH_RAG_MODEL: qwen3:8b

  PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006/v1/traces

  MLFLOW_TRACKING_URI: http://host.docker.internal:5000

  QDRANT_URL: http://sharebot-qdrant:6333
  QDRANT_HOST: sharebot-qdrant
  QDRANT_PORT: "6333"
  QDRANT_HTTPS: "false"

  NEO4J_URI: bolt://neo4j-sentinel:7687
```

## Why the addresses are different

Inside Docker:

```text
127.0.0.1
```

means the current container.

Therefore, Docker cannot use:

```text
http://127.0.0.1:5000
```

to reach MLflow running on Windows.

Docker Desktop provides:

```text
host.docker.internal
```

which allows a container to reach the Windows host.

Therefore:

```text
MLFLOW_TRACKING_URI=http://host.docker.internal:5000
```

is used.

For services running inside Docker, service names are used instead:

```text
http://phoenix:6006
http://sharebot-qdrant:6333
bolt://neo4j-sentinel:7687
```

---

# 6. Phoenix Observability

Phoenix is used for tracing the SupportMind agent workflow.

Phoenix configuration:

```yaml
phoenix:
  image: arizephoenix/phoenix:version-20.14.0

  container_name: supportmind-phoenix

  ports:
    - "6006:6006"
    - "4317:4317"
```

The web UI is available at:

```text
http://localhost:6006
```

The SupportMind API sends traces to:

```text
http://phoenix:6006/v1/traces
```

because `phoenix` is the Docker service name.

---

# 7. Phoenix Healthcheck

The Phoenix container originally used a `curl` healthcheck.

The Phoenix image did not contain `curl`, so the healthcheck was changed to Python:

```yaml
healthcheck:
  test:
    [
      "CMD",
      "python",
      "-c",
      "import urllib.request; urllib.request.urlopen('http://localhost:6006/health')"
    ]
  interval: 10s
  timeout: 5s
  retries: 10
  start_period: 20s
```

This allows Docker Compose to determine when Phoenix is ready.

The API depends on Phoenix being healthy:

```yaml
depends_on:
  phoenix:
    condition: service_healthy
```

---

# 8. Raw SupportMind Data Mount

The API container needs access to the Help Desk dataset.

The project directory is mounted read-only:

```yaml
volumes:
  - ./rawsupportminddata:/data/rawsupportminddata:ro
```

Inside the container the files are available under:

```text
/data/rawsupportminddata
```

The read-only flag:

```text
:ro
```

prevents the application container from modifying the source dataset.

---

# 9. Build the SupportMind API Image

From the project root:

```powershell
cd C:\Users\danie\supportmind-ai

conda activate supportmind
```

Build and start the API:

```powershell
docker compose up -d --build supportmind-api
```

Explanation:

```text
up
```

starts the service.

```text
-d
```

runs it in detached/background mode.

```text
--build
```

rebuilds the Docker image if application files or dependencies changed.

---

# 10. Check Running Containers

```powershell
docker ps
```

Expected important containers include:

```text
supportmind-api
supportmind-phoenix
sharebot-qdrant
neo4j-sentinel
```

Check the API logs:

```powershell
docker logs supportmind-api
```

Follow logs live:

```powershell
docker logs -f supportmind-api
```

Check Phoenix logs:

```powershell
docker logs supportmind-phoenix
```

---

# 11. Verify Phoenix

Test Phoenix from Windows:

```powershell
curl.exe -s http://localhost:6006
```

Phoenix should return its web application HTML.

Check the container:

```powershell
docker ps
```

The Phoenix container should show:

```text
(healthy)
```

---

# 12. Verify the API

SupportMind API:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

Basic API health:

```text
GET /api/v1/health
```

---

# 13. Verify MLflow

MLflow runs on the Windows host during local development.

Start MLflow:

```powershell
cd C:\Users\danie\supportmind-ai

conda activate supportmind

mlflow server `
  --backend-store-uri "sqlite:///C:/Users/danie/supportmind-ai/backend/mlflow/mlflow.db" `
  --host 0.0.0.0 `
  --port 5000 `
  --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*,0.0.0.0:*"
```

The MLflow UI is:

```text
http://127.0.0.1:5000
```

The backend uses:

```text
http://host.docker.internal:5000
```

---

# 14. Why `--allowed-hosts` Is Required

Modern MLflow versions include security middleware that validates incoming Host headers.

A request from Docker arrives using:

```text
Host: host.docker.internal:5000
```

Therefore MLflow must explicitly allow that host.

The required option is:

```powershell
--allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*,0.0.0.0:*"
```

Without this configuration, the Docker container can reach MLflow but MLflow responds with:

```text
403 Forbidden
```

After adding the allowed hosts, Docker can communicate with MLflow successfully.

---

# 15. Verify Docker → MLflow Connectivity

Check the configured tracking URI:

```powershell
docker exec supportmind-api python -c "import os; print(os.getenv('MLFLOW_TRACKING_URI'))"
```

Expected:

```text
http://host.docker.internal:5000
```

Test the MLflow server:

```powershell
docker exec supportmind-api python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:5000').status)"
```

Expected:

```text
200
```

Test the MLflow API:

```powershell
docker exec supportmind-api python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:5000/api/2.0/mlflow/experiments/search?max_results=10').status)"
```

Expected:

```text
200
```

---

# 16. Verify SupportMind MLflow Health

Open:

```text
http://localhost:8000/docs
```

Run:

```text
GET /api/v1/mlflow/health
```

The expected result is:

```text
200 OK
```

This confirms:

```text
SupportMind API
      │
      ▼
Docker networking
      │
      ▼
host.docker.internal
      │
      ▼
MLflow Tracking Server
      │
      ▼
mlflow.db
```

---

# 17. Verify MLflow Metrics and Reports

Open:

```text
http://127.0.0.1:5000
```

The MLflow UI should expose the existing SupportMind experiments and runs.

For each relevant run, verify:

### Parameters

Examples include:

```text
model
provider
dataset
configuration
```

### Metrics

Depending on the experiment:

```text
accuracy
precision
recall
F1
evaluation scores
latency
other recorded metrics
```

### Artifacts

Depending on the run:

```text
evaluation reports
JSON results
CSV results
plots
other generated artifacts
```

The existing MLflow tracking data is stored in:

```text
backend/mlflow/mlflow.db
```

---

# 18. Verify Phoenix Traces

Open:

```text
http://localhost:6006
```

The SupportMind agent workflow should produce traces for the orchestration process.

Phoenix was also verified to preserve Unicode correctly.

For example, the refund response displayed the correct:

```text
5–10 business days
```

inside Phoenix.

This confirmed that the earlier:

```text
5â10
```

appearance was a Windows/terminal rendering issue rather than corruption of the underlying SupportMind trace data.

---

# 19. Support Assist End-to-End Test

The main AI support endpoint is:

```text
POST /api/v1/support/assist
```

Example request:

```json
{
  "request": "I need help with a refund that has not arrived."
}
```

PowerShell:

```powershell
$body = @{
    request = "I need help with a refund that has not arrived."
} | ConvertTo-Json -Compress

$response = Invoke-RestMethod `
    -Uri "http://localhost:8000/api/v1/support/assist" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$response | ConvertTo-Json -Depth 20
```

A successful response should contain:

```text
response
intelligence
predictions
retrieval
evidence
agents
workflow
```

The workflow validates the complete SupportMind orchestration path:

```text
User Request
     │
     ▼
Classification
     │
     ▼
Retrieval / Graph-RAG
     │
     ▼
Resolution
     │
     ▼
Escalation
     │
     ▼
Response Composition
     │
     ▼
Final Support Response
```

---

# 20. Final Containerization Validation

Before moving to Kubernetes, the following components were validated:

| Component                 | Status   |
| ------------------------- | -------- |
| SupportMind API container | Working  |
| FastAPI endpoints         | Working  |
| Phoenix container         | Healthy  |
| Phoenix traces            | Working  |
| Qdrant connection         | Working  |
| Neo4j connection          | Working  |
| Ollama connection         | Working  |
| Graph-RAG                 | Working  |
| ML intelligence           | Working  |
| Support Assist E2E        | Working  |
| MLflow server             | Working  |
| Docker → MLflow           | Working  |
| MLflow health API         | Working  |
| MLflow metrics            | Verified |
| MLflow reports/artifacts  | Verified |
| Phoenix Unicode output    | Verified |

At this stage, the containerized SupportMind backend and its local observability stack have been validated end-to-end.

The next infrastructure phase is Kubernetes deployment.

---

# 21. Useful Docker Commands

### List containers

```powershell
docker ps
```

### List all containers

```powershell
docker ps -a
```

### View API logs

```powershell
docker logs supportmind-api
```

### Follow API logs

```powershell
docker logs -f supportmind-api
```

### View Phoenix logs

```powershell
docker logs supportmind-phoenix
```

### Restart API

```powershell
docker restart supportmind-api
```

### Stop Compose services

```powershell
docker compose down
```

### Rebuild API

```powershell
docker compose up -d --build supportmind-api
```

### Check Docker network

```powershell
docker network inspect supportmind-net
```

### Check API environment variable

```powershell
docker exec supportmind-api python -c "import os; print(os.getenv('MLFLOW_TRACKING_URI'))"
```

### Open a shell inside the API container

```powershell
docker exec -it supportmind-api /bin/bash
```

### Check API container health/logs after rebuilding

```powershell
docker ps
docker logs --tail 100 supportmind-api
```

---

# 22. Important Docker Networking Rule

When troubleshooting SupportMind locally, remember:

```text
localhost
```

always refers to the machine/process where the request originates.

Therefore:

### Windows → Windows service

```text
http://127.0.0.1:5000
```

### Docker container → Windows host

```text
http://host.docker.internal:5000
```

### Docker container → Docker service

```text
http://phoenix:6006
http://sharebot-qdrant:6333
bolt://neo4j-sentinel:7687
```

This distinction is essential when debugging containerized applications.

---

# 23. Next Phase — Kubernetes

After completing and validating Docker containerization, the next phase is Kubernetes.

The planned progression is:

```text
Docker
  │
  ▼
Kubernetes Manifests
  │
  ▼
Minikube
  │
  ▼
Deploy SupportMind API
  │
  ▼
Deploy Supporting Services
  │
  ▼
Services / Networking
  │
  ▼
ConfigMaps / Secrets
  │
  ▼
Persistent Volumes
  │
  ▼
Health Checks
  │
  ▼
Scaling
  │
  ▼
Helm
  │
  ▼
Kubernetes Validation
```

The Docker implementation should remain the foundation for the Kubernetes deployment rather than creating a second application architecture.
