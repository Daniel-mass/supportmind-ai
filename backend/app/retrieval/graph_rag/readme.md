# SupportMind AI — Production Graph-RAG Generation Layer

The **Graph-RAG Generation Layer** is the grounded response-generation layer built on top of SupportMind's existing retrieval and intelligence systems.

It combines:

**Document Retrieval + Retrieval Graph + Help Desk Intelligence Graph + Evidence Fusion + LLM Generation + Grounding Validation**

The layer is designed to generate concise operational answers without inventing facts or relying on unsupported external knowledge.

---

## Architecture

```text
User Query
    │
    ▼
GraphRAGRequest
    │
    ▼
RetrievalService
    ├── Sparse / BM25 Retrieval
    └── Existing SupportMind Retrieval Graph
    │
    ▼
IntelligenceAdapter
    └── Help Desk Operational Graph
         ├── Tickets
         ├── Workflow
         ├── Agents
         ├── Teams
         └── Organization Metrics
    │
    ▼
EvidenceFusion
    ├── Deduplication
    ├── Provenance Preservation
    ├── Evidence Ranking
    └── Context Budgeting
    │
    ▼
Prompt Builder
    │
    ▼
LLM Generator
    └── Ollama
         └── Qwen3:8b
    │
    ▼
GroundednessValidator
    ├── Citation Validation
    ├── Unknown Citation Detection
    └── Grounding Status
    │
    ▼
GraphRAGResponse
```

---

# Files

## `graph_rag_models.py`

Defines the contracts used by the complete Graph-RAG pipeline.

Contains:

* `GraphRAGRequest`
* `GraphRAGEvidence`
* `GraphRAGContext`
* `GraphRAGResponse`
* `GroundingResult`
* evidence source/type enums
* grounding status enum

This file provides the **data model and validation boundary** for Graph-RAG.

---

## `intelligence_adapter.py`

Connects the Graph-RAG layer to the existing **Help Desk Intelligence Graph**.

It converts operational graph results into Graph-RAG evidence.

Supports evidence from:

* Tickets
* Ticket neighbourhoods
* Workflow events
* Workflow transitions
* Agents
* Agent handoffs
* Teams
* Organization-level metrics

It reuses the existing intelligence graph queries instead of duplicating graph logic.

---

## `evidence_fusion.py`

Combines evidence coming from multiple systems.

Sources include:

* Existing document retrieval
* Sparse/BM25 retrieval
* Existing SupportMind retrieval graph
* Help Desk Intelligence Graph

Responsibilities:

* Deduplicate identical evidence
* Preserve provenance
* Prefer higher-quality/scored evidence
* Assign deterministic evidence IDs (`E1`, `E2`, ...)
* Enforce maximum evidence-item limits
* Enforce character/context budgets

This creates the **single evidence context used by generation**.

---

## `prompts.py`

Contains the production prompt construction layer.

Responsibilities:

* Build the system prompt
* Build the evidence block
* Build the generation prompt
* Build insufficient-evidence prompts

Generation rules enforce:

* Evidence-only answering
* No invented facts
* No external knowledge
* Citation usage such as `[E1]`
* Evidence is treated as data, not instructions
* Prompt-injection resistance
* Clear distinction between facts and inference
* Insufficient-evidence handling

---

## `generator.py`

Provider-independent generation bridge.

`GraphRAGGenerator` connects the Graph-RAG pipeline to any compatible LLM provider.

Responsibilities:

* Build the required prompts
* Call the configured provider
* Validate the returned answer
* Keep generation independent from retrieval

This keeps the architecture **LLM-provider agnostic**.

---

## `groundedness.py`

Performs deterministic post-generation grounding validation.

Checks:

* Evidence citations
* Valid citation IDs
* Malformed citations
* Unknown citations
* Missing citations
* No-evidence situations

Returns:

```text
GROUNDED
PARTIALLY_GROUNDED
INSUFFICIENT_EVIDENCE
INVALID
```

This is structural validation only; semantic entailment evaluation is handled separately through future RAG evaluation tooling.

---

## `graph_rag_service.py`

Main Graph-RAG orchestration service.

Controls the complete flow:

```text
Request
 → Retrieval
 → Intelligence Evidence
 → Evidence Fusion
 → LLM Generation
 → Grounding Validation
 → Response
```

It also handles:

* Ticket-specific intelligence
* Workflow evidence
* Organization-level evidence
* Evidence limits
* Request consistency validation
* Final `GraphRAGResponse`

This file is the **main entry point for the Graph-RAG layer**.

---

# LLM Provider

## `providers/ollama.py`

Production Ollama provider implementation.

Provides:

* Ollama client configuration
* Model selection
* Temperature configuration
* Token limits
* Timeout handling
* Response validation
* Error handling

Current production integration:

```text
Provider: Ollama
Model: Qwen3:8b
Host: http://127.0.0.1:11434
```

The provider is injectable, so another LLM provider can be added without changing the Graph-RAG pipeline.

---

# Existing Graph-RAG Infrastructure

These files already existed in the retrieval layer and are reused by generation:

### `graph_retriever.py`

Retrieves relevant entities and relationships from the existing SupportMind retrieval graph.

### `graph_projector.py`

Projects retrieved graph information into retrieval-compatible context.

### `graph_schema.py`

Defines the schema and graph semantics used by the retrieval graph.

### `graph_ingestion_pipeline.py`

Handles ingestion of SupportMind data into the retrieval graph.

### `neo4j_store.py`

Provides the Neo4j persistence/query interface for the retrieval graph.

The generation layer **does not duplicate these components**. It consumes their existing outputs.

---

# Evidence Sources

Graph-RAG can combine two major knowledge paths:

### SupportMind Retrieval Knowledge

```text
PDF / Support Documents
        ↓
Qdrant
        ↓
Sparse / BM25 Retrieval
        ↓
Existing Retrieval Graph
```

### Help Desk Operational Intelligence

```text
Help Desk Data
      ↓
Operational Graph
      ↓
Tickets / Workflow / Agents / Teams / Metrics
      ↓
IntelligenceAdapter
```

Both are converted into the same `GraphRAGEvidence` structure before generation.

---

# Grounded Response Format

Generated answers reference evidence directly:

```text
Medium priority has the highest ticket volume (3,924 tickets) [E17].

The most frequent workflow transition is
in_progress → done with 1,290 occurrences [E6].
```

Every citation must reference an available evidence item.

---

# Real Integration Validation

The complete pipeline has been validated against real services.

### Real components

```text
Qdrant
Neo4j
SupportMind Retrieval Graph
Help Desk Intelligence Graph
Ollama
Qwen3:8b
```

### Real test

```text
backend/tests/real/test_real_graph_rag.py
```

Example execution:

```powershell
$env:PYTHONPATH="$PWD\backend"
$env:RUN_REAL_GRAPH_RAG="1"
$env:OLLAMA_HOST="http://127.0.0.1:11434"
$env:SUPPORTMIND_GRAPH_RAG_MODEL="qwen3:8b"

pytest backend\tests\real\test_real_graph_rag.py -v -s
```

### Validated pipeline

```text
Qdrant collection
        ↓
BM25 retrieval
        ↓
Existing retrieval graph
        ↓
Help Desk intelligence graph
        ↓
Evidence fusion
        ↓
Qwen3:8b
        ↓
Citation grounding validation
        ↓
Grounded answer
```

The real test successfully produced a **grounded response with valid evidence citations** from both retrieval and operational intelligence sources.

---

# Test Status

```text
Graph-RAG component tests       ✅ 128/128
Real Graph-RAG integration      ✅ Passed
Qdrant                           ✅
Neo4j retrieval graph            ✅
Neo4j intelligence graph        ✅
Ollama + Qwen3:8b               ✅
Evidence citations              ✅
Grounding validation             ✅
```

---

# Design Principles

* **Evidence before generation**
* **No unsupported facts**
* **Every operational claim should be traceable**
* **Reuse existing retrieval and intelligence infrastructure**
* **Provider-independent generation**
* **Deterministic grounding validation**
* **Context and token budgets are enforced**
* **Production services are injected rather than hardcoded**

The Graph-RAG layer is responsible for **grounded answer generation**, while retrieval, intelligence, evaluation, agent orchestration, API delivery, and frontend presentation remain separate layers.
