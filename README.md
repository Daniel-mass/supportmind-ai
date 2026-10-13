# SupportMind AI
### Enterprise Customer & Issue Intelligence Platform

**SupportMind AI** is an AI-powered customer-support and issue-intelligence platform that transforms historical support data, organizational knowledge, and operational workflows into actionable insights.

Rather than functioning only as a chatbot, SupportMind combines NLP/ML, hybrid retrieval, knowledge graphs, Graph-RAG, multi-agent orchestration, deterministic operational analytics, and evaluation/observability to support both customer assistance and enterprise issue analysis.

> **Core principle:** Ground AI-generated assistance in retrieved evidence, preserve provenance, validate outputs, and use structured operational data to understand how support work actually flows.

---

## Table of Contents

- [Why SupportMind?](#why-supportmind)
- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [NLP and Machine Learning](#nlp-and-machine-learning)
- [Retrieval and Graph-RAG](#retrieval-and-graph-rag)
- [Operational Intelligence](#operational-intelligence)
- [Agentic AI](#agentic-ai)
- [Evaluation and Observability](#evaluation-and-observability)
- [Frontend Dashboard](#frontend-dashboard)
- [API Overview](#api-overview)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Containerization and Deployment](#containerization-and-deployment)
- [CI with GitHub Actions](#ci-with-github-actions)
- [Datasets and Knowledge Sources](#datasets-and-knowledge-sources)
- [Engineering Decisions](#engineering-decisions)
- [Limitations and Scope](#limitations-and-scope)
- [Future Directions](#future-directions)

---

## Why SupportMind?

Customer-support organizations accumulate information across tickets, knowledge articles, historical resolutions, and operational workflows. When this information remains fragmented, teams struggle to retrieve relevant evidence, identify recurring problems, and understand process bottlenecks.

SupportMind addresses these challenges through a unified intelligence layer.

| Challenge | SupportMind approach |
|---|---|
| Manual ticket triage | ML-based ticket category and priority predictions |
| Repeated support issues | Semantic retrieval and historical issue discovery |
| Fragmented knowledge | Document ingestion, embeddings, and vector search |
| Relationships hidden in text | Neo4j knowledge graphs and graph retrieval |
| Unsupported AI responses | Evidence provenance, citation checks, and groundedness validation |
| Complex support workflows | AutoGen orchestration and agent-level tracing |
| Limited operational visibility | Ticket, agent, team, workflow, and organization analytics |
| Difficult model evaluation | DeepEval and MLflow |
| Difficult debugging of agent execution | Phoenix and OpenTelemetry |

## Key Capabilities

### 1. Data ingestion and knowledge preparation

- Modular loaders for supported file formats, including PDF, DOCX, TXT, CSV, JSON, and ticket/email data.
- Parsing, text normalization, preprocessing, and metadata handling.
- Configurable chunking strategies, including recursive and semantic approaches, plus CSV row chunking.
- Embedding providers and vector-store registries.
- Qdrant integration for vector storage and retrieval.
- A synthetic NovaCare Technologies knowledge base for development and demonstration.

### 2. NLP and machine learning

- Ticket-category classification.
- Priority prediction.
- Conversation-complexity classification.
- Semantic similarity and historical-ticket retrieval.
- BERTopic-based discovery of recurring issues in historical tickets.
- Sentiment as a supporting signal where available.

### 3. Retrieval and knowledge graphs

- Dense vector retrieval.
- Sparse retrieval using BM25.
- Hybrid retrieval combining dense and sparse results.
- Reciprocal Rank Fusion (RRF).
- Candidate deduplication and reranking.
- ColBERT late-interaction reranking and a Cross-Encoder baseline.
- Neo4j-backed retrieval graphs and operational relationship graphs.
- Graph-RAG evidence retrieval and fusion.

### 4. Enterprise intelligence

- Ticket intelligence.
- Agent and team analytics.
- Workflow metrics and lifecycle analysis.
- Agent handoff analysis.
- Workflow variant discovery.
- Bottleneck analysis.
- Organization-level operational insights and emerging-issue analysis.
- Evidence records with provenance for supported intelligence responses.

### 5. AI-powered support

The Support Assist workflow brings together classification signals, retrieval, Graph-RAG, agent orchestration, evidence validation, and response composition through a unified API.

The system is designed to provide grounded assistance rather than treat a language model's unsupported output as authoritative.

---

## Architecture

```text
                 SUPPORT DATA & KNOWLEDGE
        Tickets | Documents | Historical Records
                         |
                         v
                  INGESTION LAYER
          Loaders | Parsing | Normalization
             Chunking | Embeddings
                         |
                         v
                NLP / ML INTELLIGENCE
       Category | Priority | Complexity
        Semantic Similarity | BERTopic
                         |
             +-----------+-----------+
             |                       |
             v                       v
       QDRANT VECTOR STORE      NEO4J GRAPH
       Dense / BM25 / Hybrid    Relationships
       Retrieval / Reranking   Graph Retrieval
             |                       |
             +-----------+-----------+
                         |
                         v
                     GRAPH-RAG
        Evidence Fusion | Provenance | Validation
                         |
                         v
                 AUTOGEN AGENTS
       Coordinator | Classification | Retrieval
          Resolution | Escalation | Composition
                         |
                         v
                  FASTAPI SERVICES
       Support Assist | Intelligence | Evaluation
                         |
                         v
                 NEXT.JS DASHBOARD
       Support | Tickets | Agents | Workflows
       Organization | Evaluation | Observability

       PHOENIX  -> Agent execution tracing
       DEEPEVAL -> LLM evaluation
       MLFLOW   -> Experiment and metric tracking
```

**Architecture principle:** Retrieval, operational intelligence, generation, evaluation, and observability are separate concerns connected through defined interfaces. Qdrant, Neo4j, Ollama, and individual agents are internal components, not separate public SupportMind application APIs.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | Next.js App Router, TypeScript |
| UI and styling | Tailwind CSS, shadcn |
| Data fetching and state | TanStack Query, Zustand |
| Visualization | Recharts, React Flow, Lucide |
| Backend API | Python 3.11, FastAPI, Pydantic |
| NLP and ML | scikit-learn, Transformers, DistilBERT, BERTopic |
| Embeddings | FastEmbed and provider-based embedding integrations |
| Vector database | Qdrant |
| Graph database | Neo4j |
| Retrieval | Dense search, BM25, hybrid retrieval, RRF, reranking |
| LLM inference | Ollama with Qwen3:8b |
| Agent orchestration | AutoGen |
| LLM evaluation | DeepEval |
| Tracing and observability | Phoenix, OpenTelemetry |
| Experiment tracking | MLflow |
| Containers | Docker, Docker Compose |
| Orchestration | Kubernetes, Minikube, Helm |
| Continuous integration | GitHub Actions |

---

## NLP and Machine Learning

SupportMind uses different datasets for different learning objectives rather than forcing every dataset into a single prediction pipeline.

| Capability | Approach | Purpose |
|---|---|---|
| Ticket-category classification | Word and character TF-IDF with class-balanced Logistic Regression | Predict a ticket category from subject and description |
| Priority detection | ExtraTrees with class balancing and engineered ticket metadata | Predict ticket priority from information available at creation time |
| Conversation complexity | DistilBERT classification pipeline | Predict single-turn versus multi-turn conversation complexity |
| Semantic similarity | Embedding-based retrieval | Find semantically related historical support cases |
| Issue discovery | BERTopic with BGE-base embeddings, UMAP, and HDBSCAN | Discover recurring themes without predefined class labels |
| Sentiment | Supporting signal where available | Add context to support analysis |

### Selected model results

The following are results recorded during model development, not a claim that every model has been independently retrained in the current environment.

| Model | Recorded evaluation |
|---|---|
| Ticket-category classifier | 77.55% accuracy, 69.02% macro-F1, 77.56% weighted-F1 on a 49-example test split |
| Priority detector | 73.26% accuracy, 39.66% macro-F1, 76.35% weighted-F1 on a held-out split |
| BERTopic | 8,460 documents, 59 discovered topics, 83.70% assigned to topics, 16.30% outliers |
| BERTopic coherence | C_V coherence of 0.7972; topic diversity of 0.5983 |

**Interpretation matters.** Priority detection has weaker performance on minority classes than on the dominant Medium class. The ticket-category evaluation set is small, and the source data is synthetic. BERTopic discovers clusters; it does not automatically establish business root causes.

The conversation-complexity documentation records a leakage-aware, text-grouped train/validation/test preparation pipeline and DistilBERT setup. Full training and final evaluation must be confirmed against the current model artifacts before reporting a final score.

### Dataset-driven scope

The initial plan considered additional intent classification, named entity recognition, urgency models, escalation prediction, and other NLP tasks. These were removed from the supported ML roadmap when the available data did not justify reliable supervised implementations.

This avoids presenting unvalidated predictions as implemented intelligence.

---

## Retrieval and Graph-RAG

SupportMind combines complementary retrieval strategies to improve the relevance of evidence supplied to the AI workflow.

### Retrieval pipeline

```text
User Request
     |
     v
Dense Retrieval (Qdrant) + Sparse Retrieval (BM25)
     |
     v
Hybrid Fusion (RRF)
     |
     v
Candidate Deduplication
     |
     v
Reranking (ColBERT / Cross-Encoder baseline)
     |
     v
Context Construction
     |
     v
Evidence with Provenance
```

### Graph-RAG

The Graph-RAG layer combines vector-retrieved evidence, graph-derived context, and operational intelligence through dedicated services.

Implemented design elements include:

- Typed request, context, evidence, and response models.
- Vector and graph retrieval integration.
- Evidence fusion, ranking, and context budgeting.
- Provenance preservation.
- Prompt construction and prompt-injection resistance measures.
- Ollama-backed Qwen3:8b generation.
- Citation and groundedness validation.
- A dedicated Graph-RAG service and API.

The purpose is to improve context and traceability. Graph-RAG does not guarantee that every generated statement is correct, so evaluation and validation remain essential.

---

## Operational Intelligence

SupportMind's intelligence layer converts raw support records and workflow history into structured analytics.

### Workflow

```text
Raw Help Desk Data
        |
        v
Canonical Ticket and Agent Schemas
        |
        v
Workflow Event Construction
        |
        v
Lifecycle / Waiting / Handoff Metrics
        |
        v
Ticket, Agent, Team and Organization Analytics
        |
        v
Intelligence APIs and Dashboard
```

Key capabilities include:

- Canonical schemas for tickets, workflow events, agents, evidence, and overview data.
- Ticket and agent analytics.
- Workflow transition and lifecycle metrics.
- Waiting-time analysis and bottleneck identification.
- Agent handoff detection.
- Workflow variant aggregation.
- Operational graph construction and Neo4j querying.
- Organization intelligence and emerging-issue analysis.

The intelligence layer uses deterministic analytics where possible, keeping operational measurements separate from LLM-generated explanations. Bottlenecks and correlations should be interpreted as analytical signals, not automatic proof of causation.

---

## Agentic AI

SupportMind uses AutoGen to coordinate a multi-stage support-assistance workflow.

```text
Customer Request
       |
       v
Coordinator / Orchestrator
       |
       v
Classification Agent
       |
       v
Retrieval Agent
       |
       v
Resolution Agent
       |
       v
Escalation Agent
       |
       v
Response Composition
       |
       v
Validated Support Response
```

The orchestration layer coordinates execution and manages structured state, intermediate results, evidence references, iteration handling, and error paths.

- **Classification:** uses available classification and intelligence signals.
- **Retrieval:** obtains relevant evidence from retrieval services.
- **Resolution:** prepares a structured recommendation grounded in supplied evidence.
- **Escalation:** evaluates the need for human intervention.
- **Response composition:** communicates the resulting assistance while preserving upstream decisions.
- **Coordinator:** manages the workflow and final response.

Pydantic models and semantic validation help maintain structured outputs and verify evidence references. The architecture does not mean that every agent is an independently trained model or that every predicted field is ML-backed.

A unified Support Assist API exposes the workflow rather than creating a separate public endpoint for each internal agent.

---

## Evaluation and Observability

SupportMind separates **answer-quality evaluation**, **execution tracing**, and **ML experiment tracking**.

| Component | Responsibility |
|---|---|
| DeepEval | Evaluate LLM/RAG outputs using configured evaluation metrics |
| Phoenix | Trace agent execution, spans, inputs/outputs, and workflow behavior |
| OpenTelemetry | Provide instrumentation and trace transport |
| MLflow | Track experiments, parameters, metrics, artifacts, and evaluation runs |

### DeepEval

The evaluation integration includes result loading, evaluation-related API routes, and dashboard integration. Results must be interpreted according to the actual evaluation dataset, metrics, and run configuration.

### Phoenix

Custom tracing records the support orchestration and agent workflow. This makes it easier to inspect the execution path and investigate failures or unexpected behavior.

### MLflow

MLflow provides experiment and evaluation tracking. Its recorded local setup uses a SQLite backend and an artifact directory.

These tools complement one another: Phoenix helps explain *what happened during execution*, DeepEval evaluates *the quality of outputs*, and MLflow records *experiments and their results*.

---

## Frontend Dashboard

The Next.js application provides a centralized interface for interacting with the platform and inspecting its intelligence.

The implemented dashboard structure includes:

- **Overview:** platform-level operational summary.
- **AI Support:** unified support-assistance interface.
- **Tickets:** ticket and ticket-intelligence views.
- **Intelligence:** ticket, ML, knowledge/retrieval, and graph intelligence.
- **Agent Operations:** agent intelligence and workflow views.
- **Organization:** trends, emerging issues, and operational insights.
- **Evaluation:** configured evaluation results and related views.
- **Observability:** Phoenix, MLflow, and system-health views.

The dashboard consumes backend APIs and presents structured data through charts, tables, and graph-oriented visualizations.

It should not be confused with a claim that all proposed customer, agent, manager, and administrator portals, authentication flows, or third-party connector-management screens have been implemented.

---

## API Overview

FastAPI exposes the application's capabilities through versioned routes.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Application health |
| POST | `/api/v1/support/assist` | Unified AI support workflow |
| GET | `/api/v1/tickets/{ticket_id}/intelligence` | Ticket intelligence |
| GET | `/api/v1/workflow/{ticket_id}/intelligence` | Workflow intelligence |
| GET | `/api/v1/agents/{agent_id}/intelligence` | Agent intelligence |
| GET | `/api/v1/organization/intelligence` | Organization intelligence |
| POST | `/api/v1/graph-rag/query` | Graph-RAG query |
| GET | `/api/graph-rag/health` | Graph-RAG health |
| GET | `/api/v1/mlflow/health` | MLflow connectivity/status |
| GET | `/docs` | Interactive FastAPI documentation |

Evaluation and other observability routes are also part of the application API surface. Consult the live OpenAPI specification for their exact current paths and request/response schemas.

Example Support Assist request:

```http
POST /api/v1/support/assist
Content-Type: application/json
```

```json
{
  "request": "I need help with a refund that has not arrived."
}
```

The API accepts a single request string for this workflow; it should not be assumed to provide persistent conversational memory across requests.

**For the authoritative endpoint list, open `/docs` on the running backend.**

---

## Project Structure

The following is a high-level guide to the principal areas of the repository. It is not intended to be an exhaustive file listing.

```text
supportmind-ai/
├── backend/
│   ├── app/
│   │   ├── agents/                 # Agent orchestration
│   │   ├── api/                    # Application routes
│   │   ├── embeddings/              # Embedding providers and pipeline
│   │   ├── evaluation/              # DeepEval and evaluation workflows
│   │   ├── ingestion/               # Loaders, parsers, preprocessing, chunking
│   │   ├── intelligence/            # Structured operational intelligence
│   │   ├── nlp/                     # Classification and topic modeling
│   │   ├── observability/            # Phoenix / MLflow integration
│   │   ├── retrieval/               # Dense, sparse, hybrid, Graph-RAG
│   │   └── vector_store/             # Qdrant integration
│   ├── Dockerfile
│   ├── requirements.txt
│   └── mlflow/                      # Local MLflow data/artifacts
├── frontend/
│   ├── app/                         # Next.js application
│   └── ...
├── helm/
│   └── supportmind/                 # Helm chart
├── k8s/                             # Kubernetes manifests
├── scripts/                         # Development/deployment helpers
├── docker-compose.yml
└── README.md
```

Some model weights, raw datasets, and generated artifacts are intentionally not stored in Git. A fresh clone may therefore require separate artifact preparation before every ML-dependent capability can run.

---

## Run Locally

### Prerequisites

Install or prepare:

- Git
- Python 3.11
- Node.js and npm compatible with the frontend dependencies
- Conda or another Python environment manager
- Ollama with the required local model
- Access to Qdrant and Neo4j
- The configured data files and model artifacts required by the selected features

Docker is recommended for the configured backend services. Kubernetes and Helm are optional if you only want to explore local application development.

### 1. Clone the repository

```powershell
git clone https://github.com/Daniel-mass/supportmind-ai.git
cd supportmind-ai
```

### 2. Prepare the backend environment

```powershell
conda create -n supportmind python=3.11 -y
conda activate supportmind

cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Return to the repository root when running root-level commands:

```powershell
cd ..
```

### 3. Configure environment variables

Create the appropriate local environment configuration using the variable names expected by the application.

Typical configuration includes:

```dotenv
OLLAMA_HOST=http://localhost:11434
SUPPORTMIND_LLM_MODEL=qwen3:8b
SUPPORTMIND_GRAPH_RAG_MODEL=qwen3:8b

QDRANT_URL=http://localhost:6333

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-local-password>
NEO4J_DATABASE=neo4j

MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Additional variables may be required for data paths, embedding models, collections, and Phoenix. Use the application's configuration code and existing environment templates as the source of truth.

**Never commit real credentials, `.env` files, local database files, or private datasets.**

### 4. Start the local dependencies

Start Ollama and ensure the required model is available:

```powershell
ollama pull qwen3:8b
ollama list
```

Start Qdrant and Neo4j using your configured local services or container setup. Confirm that their endpoints, credentials, database names, and collection configuration match the backend environment.

### 5. Start the backend

```powershell
conda activate supportmind
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check the API:

- Health: `http://127.0.0.1:8000/api/v1/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`

### 6. Start the frontend

Open another PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

Configure the frontend's API base URL using the environment variable expected by the current frontend API client. Ensure the backend is running and accessible.

> **Setup note:** These are the main development commands. They do not automatically provision the required datasets, trained model weights, Qdrant collections, Neo4j graph contents, or evaluation results. Follow the relevant configuration and data-preparation code when setting up a clean environment.

---

## Containerization and Deployment

### Docker

The backend uses a Python 3.11 slim image, installs dependencies from `requirements.txt`, copies the application source, and starts Uvicorn on port `8000`.

Build and start the configured API service from the repository root:

```powershell
docker compose up -d --build supportmind-api
```

Inspect running containers and API logs:

```powershell
docker ps
docker logs supportmind-api
```

The configured Docker environment connects to services such as Qdrant and Neo4j through the shared `supportmind-net` network. Ollama and MLflow have also been configured to run on the Windows host during local development.

When a container needs to access a service on the Windows host, `host.docker.internal` may be required instead of `localhost`.

### Kubernetes and Minikube

The local Kubernetes setup includes:

- A `supportmind` namespace.
- A backend API Deployment.
- A ClusterIP Service.
- Configuration through a ConfigMap and secret references.
- Health verification from within the running API pod.
- A local image workflow for Minikube.

The development setup uses local dataset access through a mounted host directory. That mount may require a separate running `minikube mount` process, so it is not equivalent to portable production storage.

Useful commands:

```powershell
minikube status
minikube kubectl -- get pods -n supportmind
minikube kubectl -- get services -n supportmind
minikube kubectl -- get deployments -n supportmind
```

### Helm

The chart under `helm/supportmind` packages the Kubernetes resources and configurable deployment values.

```powershell
helm lint .\helm\supportmind
helm template supportmind .\helm\supportmind
helm status supportmind -n supportmind
helm list -n supportmind
```

The recorded local installation used:

```powershell
helm upgrade --install supportmind .\helm\supportmind -n supportmind --take-ownership
```

Use `--take-ownership` only when intentionally adopting existing Kubernetes resources and after confirming the implications for the target cluster.

**Deployment scope:** The documented deployment work targets local development and validation using Docker and Minikube. A production cloud deployment, managed database architecture, external connector deployment, autoscaling strategy, and production-grade secret/storage management should not be inferred from this setup.

---

## CI with GitHub Actions

The repository includes a GitHub Actions CI workflow.

The recorded workflow covers:

- Frontend lint/build checks.
- Backend Python syntax compilation.
- Backend Docker image build.

The repository's frontend lint step has historically allowed the workflow to continue despite existing lint errors. A passing workflow therefore does not necessarily mean that every lint issue is resolved or that every backend integration and end-to-end test has run in CI.

CI currently demonstrates automated code and build checks. It should not be described as a complete continuous-delivery pipeline that deploys to production.

Review the latest workflow and run results in the repository's **Actions** tab for the current status.

---

## Datasets and Knowledge Sources

SupportMind uses separate data sources for model development, operational analytics, and retrieval.

| Dataset/source | Primary use |
|---|---|
| Customer Support Tickets Dataset | Semantic similarity and issue-topic discovery |
| Customer Support on Twitter | Conversation-complexity task development |
| IT Helpdesk Synthetic Tickets | Ticket-category classification |
| Help Desk Tickets | Priority modelling and operational workflow intelligence |
| NovaCare knowledge-base documents | Retrieval and support-assistance demonstrations |
| Enron Email Dataset | Potential email-intelligence development source; live connector status is separate |

The ingestion code supports multiple file formats. That does **not** mean Gmail, Outlook, Zendesk, Freshdesk, Jira Service Management, Confluence, Notion, Slack, and Microsoft Teams have all been integrated as live, authenticated connectors.

Third-party connector support should be claimed only after the corresponding integration has been implemented and validated.

---

## Engineering Decisions

### Data-driven modelling instead of feature inflation

The initial roadmap considered a wider set of NLP tasks. The implemented scope was narrowed to tasks supported by the available data and useful to the platform. Unsupported supervised tasks were removed rather than represented as completed.

### Separate retrieval from generation

Retrieval services locate evidence; Graph-RAG combines relevant graph and vector context; the generation layer composes an answer. This separation supports independent testing and troubleshooting.

### Preserve evidence provenance

Evidence identifiers, source metadata, and validation help make generated responses more traceable. Grounding validation is a safeguard, not a mathematical guarantee that hallucinations are impossible.

### Separate operational analytics from LLM reasoning

Ticket and workflow measurements are derived through structured analytics where possible. This avoids asking an LLM to invent counts, waiting times, or workflow statistics.

### Use the right tool for each operational concern

- Neo4j models graph relationships.
- Qdrant stores vectors for retrieval.
- AutoGen coordinates agents.
- Phoenix traces agent execution.
- DeepEval evaluates LLM/RAG outputs.
- MLflow tracks experiments and evaluation runs.

### Treat reproducibility as an engineering requirement

Datasets, model weights, local environment configuration, database state, and evaluation artifacts can all affect results. A repository clone alone may not reproduce every local development result unless the necessary artifacts and configuration are available.

---

## Limitations and Scope

SupportMind is a portfolio-grade AI platform with a local containerized deployment workflow. The following boundaries are important:

- The ticket-category and priority models were developed on limited or synthetic data; their recorded metrics do not establish general enterprise performance.
- Priority performance is weaker on rare classes, so predictions should not be treated as definitive.
- BERTopic produces topic clusters and outliers, not guaranteed root-cause diagnoses.
- Sentiment is a supporting signal, not a claim that a dedicated, production-trained sentiment model exists for every source.
- Intent classification, NER, ML-based urgency prediction, and ML-based escalation prediction were removed from the supported ML scope.
- No live third-party email, ticketing, or collaboration connector should be assumed without separate implementation evidence.
- Evidence validation reduces risk but does not guarantee that all generated answers are correct.
- Local Minikube deployment is not equivalent to production cloud deployment.
- Model weights and local datasets may be excluded from Git and may need to be prepared separately.
- Historical test reports should not be interpreted as a fresh, complete test-suite pass.

---

## Future Directions

Potential extensions, subject to validation and prioritization, include:

- More comprehensive automated regression and end-to-end tests.
- Reproducible model-artifact preparation and retrieval benchmarks.
- Stronger evaluation datasets and minority-class model analysis.
- Production-ready external connectors with authentication, incremental ingestion, deduplication, and monitoring.
- Cloud Kubernetes deployment and managed persistence.
- Additional security controls, access management, resource monitoring, and alerting.
- A formal CD workflow after deployment and rollback requirements are defined.

These are future directions, not claims about currently implemented functionality.

---

## Project Repository

**GitHub:** [Daniel-mass/supportmind-ai](https://github.com/Daniel-mass/supportmind-ai)

SupportMind AI brings together NLP/ML, hybrid retrieval, knowledge graphs, Graph-RAG, agentic workflows, structured operational intelligence, evaluation, observability, and modern application deployment in one integrated customer-support intelligence platform.
