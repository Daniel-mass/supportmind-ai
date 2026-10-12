# SupportMind AI — Agentic Layer

## 1. Overview

The **Agentic Layer** is the intelligence and decision-orchestration layer of SupportMind AI.

It converts an incoming customer-support request into a structured workflow where multiple specialized AI agents work together to:

1. Understand the customer request
2. Classify the issue
3. Retrieve relevant knowledge and evidence
4. Determine a possible resolution
5. Decide whether escalation is required
6. Compose the final customer/support response
7. Track the complete workflow and its execution

Instead of using one large LLM prompt to perform everything, SupportMind separates responsibilities across specialized agents.

```text
Customer Request
       │
       ▼
 Coordinator
       │
       ├──► Classification Agent
       │
       ├──► Retrieval Agent
       │       ├── Qdrant
       │       └── Graph-RAG / Neo4j
       │
       ├──► Resolution Agent
       │
       ├──► Escalation Agent
       │
       └──► Response Composer Agent
                    │
                    ▼
             Final Response
```

---

# 2. Why an Agentic Layer?

Traditional customer-support systems often follow a fixed pipeline:

```text
Input → Search → LLM → Response
```

This can make it difficult to:

* separate responsibilities
* validate decisions
* trace how an answer was produced
* handle escalation independently
* prevent unsupported answers
* introduce human review
* debug individual stages

SupportMind uses an **agent-based architecture** where each major responsibility has its own controlled component.

For example:

* The **Retrieval Agent** finds evidence.
* The **Resolution Agent** reasons over that evidence.
* The **Escalation Agent** determines whether escalation rules apply.
* The **Response Composer Agent** communicates the result.

This makes the workflow easier to understand, test, monitor, and extend.

---

# 3. Agentic Architecture

## Coordinator

The Coordinator is responsible for controlling the overall workflow.

It determines which capabilities are required and coordinates execution between the specialized agents.

```text
Request
   ↓
Coordinator
   ↓
Agents
   ↓
Validated Results
   ↓
Final Response
```

The Coordinator does **not** replace the specialized agents. Its primary responsibility is orchestration.

---

## Classification Agent

The Classification Agent analyzes the incoming support request and produces structured signals such as:

* Category
* Priority
* Sentiment
* Complexity
* Topic

Example:

```text
Category    → Billing
Priority    → High
Sentiment   → Negative
Complexity  → Multi-turn
```

These signals become input for downstream agents.

---

## Retrieval Agent

The Retrieval Agent searches for relevant information needed to answer the request.

It can use:

* **Qdrant** for vector similarity search
* **Neo4j** for graph-based relationships
* Graph-RAG for connecting related knowledge

The important principle is:

> Agents should reason using retrieved evidence rather than relying only on the LLM's internal knowledge.

Retrieved information is assigned canonical evidence IDs such as:

```text
E1
E2
E3
E4
E5
```

These IDs allow downstream agents to reference the same evidence consistently.

---

## Resolution Agent

The Resolution Agent determines what can be done based on:

* Customer request
* Classification signals
* Retrieved evidence

It produces a structured resolution recommendation.

Example:

```json
{
  "resolution_type": "policy_based",
  "recommendation": "...",
  "rationale": "...",
  "evidence_ids": ["E1", "E2"],
  "confidence": 0.75,
  "requires_human_review": true
}
```

The agent is not allowed to invent evidence.

Evidence IDs are validated against the actual evidence supplied to the agent.

---

## Escalation Agent

The Escalation Agent determines whether the case requires escalation.

It considers:

* Retrieved evidence
* Resolution result
* Escalation policies
* Confidence
* Human-review requirements

It produces a structured result such as:

```json
{
  "escalation_type": "no_escalation",
  "should_escalate": false,
  "reason": "...",
  "evidence_ids": ["E4"],
  "confidence": 0.95
}
```

This separates **resolution reasoning** from **escalation reasoning**.

---

## Response Composer Agent

The Response Composer Agent converts the validated upstream results into a final response.

It receives:

* Classification signals
* Retrieved evidence
* Resolution
* Escalation result

Its responsibility is communication, not making new decisions.

The agent is explicitly instructed to:

* use only supplied evidence
* preserve uncertainty
* avoid inventing information
* avoid changing the resolution
* avoid independently deciding escalation
* adapt the response to the required audience and tone

This separation helps prevent the final response generator from silently changing an upstream decision.

---

# 4. Structured State Between Agents

Agents communicate through structured Pydantic state rather than passing uncontrolled text wherever possible.

Conceptually:

```text
Request
   ↓
Coordinator State
   ↓
Classification State
   ↓
Evidence State
   ↓
Resolution State
   ↓
Escalation State
   ↓
Response Composer State
   ↓
Final Response
```

This provides:

* predictable data contracts
* validation
* easier debugging
* clearer agent boundaries
* easier testing

---

# 5. Guardrails and Validation

A major part of the Agentic Layer is that LLM output is **not automatically trusted**.

The general pattern is:

```text
LLM
 ↓
Structured Output
 ↓
Pydantic Validation
 ↓
Semantic Validation
 ↓
Accepted Agent Result
```

For example, if an agent returns:

```text
evidence_ids = ["E1", "E99"]
```

but the available evidence only contains:

```text
E1, E2, E3, E4, E5
```

the output is rejected because `E99` does not exist.

This provides a basic but important boundary between probabilistic LLM generation and deterministic application logic.

---

# 6. Evidence Provenance

SupportMind maintains evidence provenance throughout the workflow.

Each evidence item can contain:

```text
Evidence ID
Source
Content
Source Reference
Score
Metadata
```

Example:

```text
E1
Source: refund_policy
Reference: node-refund-001
Score: 0.91
```

This allows downstream agents to work with the same evidence identity instead of independently retrieving or inventing references.

The final response can therefore be traced back to the evidence used during reasoning.

---

# 7. Phoenix — Agent Observability

**Phoenix** is used for observability of the Agentic Layer.

Phoenix does not make decisions and does not replace the agents.

Its role is to help us understand **what happened during an agentic workflow**.

Conceptually:

```text
SupportMind Workflow
        │
        ├── Classification
        ├── Retrieval
        ├── Resolution
        ├── Escalation
        └── Generation
                │
                ▼
             Phoenix
                │
                ▼
        Trace / Span Visualization
```

The expected trace structure is:

```text
supportmind.agent-orchestrator.run
│
├── agent.classification
├── agent.retrieval
├── agent.resolution
├── agent.escalation
└── agent.generation
```

This allows developers to investigate:

* Which agents executed?
* In what order?
* How long did each stage take?
* Where did an error occur?
* Which agent produced a particular result?
* Did the complete workflow execute as expected?

### Important distinction

Phoenix is primarily an **observability/tracing layer**.

It does not evaluate whether the final answer is accurate by itself.

Later, evaluation and experiment tracking are handled separately using tools such as **MLflow**.

---

# 8. Technology Stack

| Technology        | Role                                   |
| ----------------- | -------------------------------------- |
| **Python**        | Core backend and agent implementation  |
| **AutoGen**       | Multi-agent orchestration              |
| **Ollama**        | Local LLM execution                    |
| **Qwen3 8B**      | Local reasoning/generation model       |
| **Pydantic**      | Structured state and output validation |
| **Qdrant**        | Vector retrieval                       |
| **Neo4j**         | Knowledge graph / Graph-RAG            |
| **FastAPI**       | Backend API layer                      |
| **Phoenix**       | Agent tracing and observability        |
| **OpenTelemetry** | Trace instrumentation/export           |
| **Pytest**        | Unit and integration testing           |

---

# 9. What Makes Our Agentic Layer Different?

The goal is not simply to create multiple AI agents.

The main contribution is the **controlled cooperation between specialized agents**.

### 1. Evidence-first reasoning

Agents receive retrieved evidence and are constrained to use it rather than freely inventing information.

### 2. Canonical evidence identity

Evidence is assigned stable IDs such as `E1`, `E2`, and `E3`, allowing multiple agents to refer to the same evidence consistently.

### 3. Structured agent contracts

Agents communicate through validated Pydantic models rather than relying entirely on free-form LLM text.

### 4. Separation of responsibilities

Resolution, escalation, and response generation are intentionally separated.

The response generator does not independently rewrite the resolution decision.

### 5. Semantic validation

Validation goes beyond checking whether JSON is syntactically correct.

For example:

```text
LLM says E99
        ↓
Does E99 exist?
        ↓
No
        ↓
Reject output
```

### 6. Human-review awareness

Resolution and escalation results can explicitly indicate:

```text
requires_human_review = true
```

This allows uncertainty and operational risk to move through the workflow instead of being hidden.

### 7. End-to-end observability

Phoenix provides visibility into the complete multi-agent execution rather than treating the LLM as a black box.

---

# 10. Testing Strategy

The Agentic Layer is tested at multiple levels.

### Unit Tests

Individual components are tested independently.

Examples:

```text
Output parsers
State builders
Evidence validation
Agent utilities
```

### Focused Real-LLM Tests

Individual agents can be tested against the local Ollama model.

Examples:

```text
Resolution Agent
Escalation Agent
Response Composer Agent
```

### Real Multi-Agent Integration Test

The complete workflow is executed using:

```text
Classification
      ↓
Retrieval
      ↓
Resolution
      ↓
Escalation
      ↓
Response Generation
```

The test verifies that the agents can operate together and produce a completed workflow.

Phoenix is used separately to verify the execution trace.

---

# 11. Current Workflow

The current Agentic Layer follows this high-level flow:

```text
Customer Request
      │
      ▼
Coordinator
      │
      ▼
Classification
      │
      ▼
Retrieval
 ┌────┴────┐
 ▼         ▼
Qdrant    Neo4j
 │         │
 └────┬────┘
      ▼
 Evidence
      │
      ▼
Resolution
      │
      ▼
Escalation
      │
      ▼
Response Composer
      │
      ▼
Final Response
      │
      ▼
Phoenix Trace
```

---

# 12. Next Layer: MLflow

Phoenix and MLflow serve different purposes.

```text
Phoenix
   ↓
"What happened during the agent workflow?"

MLflow
   ↓
"How well did the system perform?"
```

MLflow will be used for areas such as:

* Experiment tracking
* Model/version tracking
* Evaluation results
* Metrics
* Runs
* Comparing system changes
* Reproducibility

Together:

```text
Agentic Layer
      │
      ├── Phoenix
      │      └── Execution observability
      │
      └── MLflow
             └── Evaluation & experiment tracking
```

This separation keeps **workflow observability** and **system evaluation** as two related but distinct concerns.
