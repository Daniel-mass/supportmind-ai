# Intelligence Layer

The Intelligence Layer transforms canonical SupportMind support data into operational intelligence for customers, support agents, managers, operations teams, APIs, and downstream agents.

It does not act as a chatbot or primary reasoning layer. Instead, it produces deterministic, structured intelligence from tickets, workflow events, agents, teams, handoffs, bottlenecks, workflow variants, and operational graph relationships.

## Intelligence Layer Workflow

```text
Raw Support Data
      ↓
Ingestion Layer
      ↓
Canonical Schemas
      ↓
Workflow Intelligence
      ↓
Operational Analytics
      ↓
Operational Graph
      ↓
Intelligence Services
      ↓
FastAPI
      ↓
Next.js Dashboards
      ↓
Agents / Copilot Workflows
```

### Core responsibilities

```text
Ticket Understanding
        ↓
Workflow Understanding
        ↓
Operational Analytics
        ↓
Relationship / Graph Intelligence
        ↓
Evidence Generation
        ↓
API-ready Intelligence
```

The frontend is responsible for visualization using Next.js, Recharts, React Flow, and Cytoscape. The Intelligence Layer returns structured data rather than chart components.

---

# Project Structure

```text
backend/app/intelligence/
│
├── schemas/
│   ├── ticket.py
│   ├── workflow.py
│   ├── agent.py
│   ├── evidence.py
│   └── overview.py
│
├── ingestion/
│   ├── helpdesk_loader.py
│   └── normalizer.py
│
├── workflow/
│   ├── event_builder.py
│   ├── metrics.py
│   ├── bottlenecks.py
│   ├── handoffs.py
│   └── variants.py
│
├── analytics/
│   ├── ticket_analytics.py
│   ├── agent_analytics.py
│   ├── team_analytics.py
│   └── organization.py
│
├── graph/
│   ├── graph_builder.py
│   ├── graph_queries.py
│   └── operational_graph.py
│
├── services/
│   ├── ticket_intelligence.py
│   ├── workflow_intelligence.py
│   ├── organization_intelligence.py
│   └── evidence_service.py
│
└── api/
    ├── tickets.py
    ├── workflow.py
    ├── agents.py
    └── organization.py
```

---

# Schemas

## `schemas/ticket.py`

Defines the canonical `Ticket` contract used throughout the Intelligence Layer.

Normalizes ticket identity, type, priority, status, timestamps, assignee, reporter, and source metadata.

Acts as the common ticket representation independent of the original Help Desk schema.

## `schemas/workflow.py`

Defines the canonical `WorkflowEvent` contract for ticket lifecycle events.

Represents timestamped status, assignee, team, priority, and category changes.

Provides a consistent workflow-event representation for analytics and graph construction.

## `schemas/agent.py`

Defines the canonical support `Agent` identity model.

Stores agent identity, optional name/team, active state, source, and metadata.

Performance metrics are intentionally kept outside the identity schema.

## `schemas/evidence.py`

Defines the canonical `IntelligenceEvidence` contract.

Tracks scope, evidence type, metric, value, source dataset, source fields, and provenance metadata.

Provides traceable operational evidence for dashboards and downstream agents.

## `schemas/overview.py`

Defines reusable dashboard response contracts such as distributions and metric summaries.

Provides clean API-level structures for organization and operational summaries.

Contains response contracts only; business calculations happen in the analytics layer.

---

# Ingestion

## `ingestion/helpdesk_loader.py`

Loads the real Help Desk source files including `issues.csv` and `issues_change_history.csv`.

Validates source paths and exposes the datasets to the Intelligence Layer.

Preserves source data; downstream modules decide what information is operationally relevant.

## `ingestion/normalizer.py`

Maps raw Help Desk fields into canonical `Ticket` objects.

Handles real Help Desk field names and mixed timestamp formats.

Creates a stable canonical ticket representation for all downstream Intelligence components.

---

# Workflow Intelligence

## `workflow/event_builder.py`

Transforms canonical workflow events into ordered workflow intervals.

Groups events by ticket and pairs consecutive events chronologically.

Provides the structural workflow timeline without calculating higher-level analytics.

## `workflow/metrics.py`

Calculates lifecycle, active, waiting, transition, status, and assignee metrics.

Separates explicit waiting states from active lifecycle time.

Produces deterministic `WorkflowMetrics` for ticket and organizational analytics.

## `workflow/bottlenecks.py`

Identifies recurring workflow states that combine volume and duration characteristics.

Produces deterministic bottleneck candidates rather than claiming causal root causes.

Provides bottleneck evidence for manager and operations dashboards.

## `workflow/handoffs.py`

Detects agent-to-agent handoffs from consecutive workflow intervals.

Captures source agent, destination agent, teams, timestamps, and surrounding workflow timing.

Provides the foundation for workload and operational handoff intelligence.

## `workflow/variants.py`

Identifies recurring workflow status paths across tickets.

Collapses consecutive duplicate states and groups tickets by their resulting workflow path.

Provides workflow-pattern intelligence for operations analysis.

---

# Analytics

## `analytics/ticket_analytics.py`

Transforms ticket and workflow outputs into ticket-level operational metrics.

Calculates lifecycle ratios, transitions, handoffs, priority exposure, and waiting-state indicators.

Produces dashboard-ready `TicketAnalytics`.

## `analytics/agent_analytics.py`

Aggregates assigned-ticket workload and workflow activity for individual agents.

Tracks open/closed work, priority exposure, waiting work, time metrics, transitions, and handoffs.

Describes operational workload and activity without ranking agent performance.

## `analytics/team_analytics.py`

Aggregates agent-level intelligence into team-level operational metrics.

Provides workload, priority exposure, waiting workload, lifecycle, transition, and handoff totals.

Produces team dashboard data without creating team-performance scores.

## `analytics/organization.py`

Aggregates ticket, agent, team, workflow, bottleneck, variant, and emerging-issue signals.

Produces organization-wide distributions, workload summaries, lifecycle metrics, and operational signals.

Acts as the top-level deterministic analytics contract for Manager and Operations dashboards.

---

# Operational Graph

## `graph/graph_builder.py`

Converts canonical tickets, agents, workflow events, and handoffs into graph nodes and relationships.

Defines deterministic operational relationships such as `HAS_STATUS`, `ASSIGNED_TO`, `NEXT`, and `HANDOFF_TO`.

Remains independent of Neo4j so graph construction can be unit-tested separately.

## `graph/graph_queries.py`

Defines parameterized Cypher queries for operational graph retrieval.

Provides ticket, workflow, agent, team, transition, priority, and status queries.

Contains no database credentials or connection logic.

## `graph/operational_graph.py`

Owns the Neo4j driver, connection lifecycle, constraints, graph persistence, query execution, and health checks.

Uses the existing `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` environment configuration.

Provides the persistence and retrieval boundary between SupportMind and Neo4j.

---

# Intelligence Services

## `services/ticket_intelligence.py`

Composes ticket, workflow, and ticket analytics into one unified ticket-intelligence response.

Provides a single structured contract for API and dashboard consumers.

Does not perform new ML, NLP, or analytics calculations.

## `services/workflow_intelligence.py`

Composes workflow events, metrics, handoffs, bottlenecks, and variants.

Provides a unified workflow-intelligence response for a ticket.

Acts as an orchestration layer over already-computed workflow outputs.

## `services/organization_intelligence.py`

Composes organization analytics with agent, team, bottleneck, variant, and emerging-issue information.

Provides a single organization-level intelligence contract.

Designed primarily for Manager and Operations dashboard consumption.

## `services/evidence_service.py`

Converts deterministic analytics into traceable `IntelligenceEvidence` objects.

Records source dataset, source fields, metric, scope, and provenance.

Provides factual evidence for dashboards and future evidence-grounded agent workflows.

---

# FastAPI

## `api/tickets.py`

Exposes ticket-level intelligence through FastAPI.

Endpoint:

```text
GET /api/intelligence/tickets/{ticket_id}/intelligence
```

Delegates retrieval to the ticket-intelligence service through dependency injection.

## `api/workflow.py`

Exposes workflow intelligence for a specific ticket.

Endpoint:

```text
GET /api/intelligence/workflow/{ticket_id}/intelligence
```

Returns workflow events, metrics, handoffs, bottlenecks, and variants.

## `api/agents.py`

Exposes operational analytics for an individual support agent.

Endpoint:

```text
GET /api/intelligence/agents/{agent_id}/intelligence
```

Returns workload, workflow activity, priority exposure, and handoff metrics.

## `api/organization.py`

Exposes organization-level intelligence for Manager and Operations dashboards.

Endpoint:

```text
GET /api/intelligence/organization/intelligence
```

Returns organization metrics, team/agent intelligence, bottlenecks, workflow variants, and emerging issues.

---

# Data Flow to Frontend

The Intelligence Layer never renders charts itself.

```text
FastAPI
   ↓
JSON
   ↓
Next.js
   ↓
TanStack Query / application state
   ↓
┌───────────────────────┬───────────────────────┐
│ Recharts              │ React Flow/Cytoscape  │
│ KPI / trends /        │ operational graph /   │
│ distributions         │ workflow relationships│
└───────────────────────┴───────────────────────┘
```

Examples:

```text
OrganizationAnalytics.priority_distribution
        ↓
Recharts BarChart
```

```text
OrganizationAnalytics.team_workload
        ↓
Recharts BarChart
```

```text
Neo4j graph query
        ↓
nodes + relationships
        ↓
React Flow / Cytoscape
```

---

# Neo4j + Qdrant Responsibilities

The systems have different responsibilities.

```text
Qdrant
→ semantic similarity
→ dense retrieval
→ similar tickets / documents
```

```text
Neo4j
→ operational relationships
→ workflow structure
→ ticket ↔ agent ↔ status
→ handoffs
→ graph exploration
```

They can later be combined by retrieval and agent workflows without merging their responsibilities.

---

# Validation Status

The Intelligence Layer has been validated with unit and real-data integration tests.

```text
Schemas
✅ Complete

Ingestion
✅ Complete

Workflow Intelligence
✅ Complete

Analytics
✅ Complete

Neo4j Graph
✅ Complete

Services
✅ Complete

FastAPI
✅ Complete
```

Real Help Desk data used during validation:

```text
issues.csv
66,691 rows

issues_change_history.csv
257,508 rows
```

Real integration tests successfully exercised the Help Desk workflow, analytics, graph, services, and API layers.

---

# Scope Boundary

The Intelligence Layer intentionally does not contain:

* model training
* transformer inference
* embedding generation
* Qdrant implementation
* AutoGen orchestration
* LLM response generation
* frontend visualization code
* Kubernetes deployment logic

Those responsibilities belong to the NLP/ML, Retrieval, Agent, Frontend, and Infrastructure layers respectively.

---

# Next Layer

The next major layer is the **Agent Layer**.

The Agent Layer will consume the structured outputs created here:

```text
Ticket Intelligence
Workflow Intelligence
Organization Intelligence
Evidence
Neo4j Graph
Qdrant Retrieval
ML/NLP Predictions
        ↓
     AutoGen
        ↓
Coordinator
Classification
Retrieval
Resolution
Escalation
Evaluation
```

This allows SupportMind to evolve from deterministic operational intelligence into an evidence-grounded autonomous support system.
