# SupportMind AI — Ingestion & Embedding Architecture

## Overview

The ingestion and embedding modules form the foundation of SupportMind AI's Retrieval-Augmented Generation (RAG) pipeline.

Their responsibility is to transform raw documents into high-quality vector representations that can later be indexed inside a vector database for semantic retrieval.

The architecture is intentionally provider-agnostic and modular. Every component has a single responsibility, making it easy to replace or extend individual parts without affecting the rest of the pipeline.

---

# Overall Architecture

```
Raw Document
      │
      ▼
Document Loader
      │
      ▼
Document Parser
      │
      ▼
IngestedDocument
      │
      ▼
Chunking Pipeline
      │
      ▼
TextNodes
      │
      ▼
Embedding Pipeline
      │
      ▼
EmbeddedNodes
      │
      ▼
Vector Database (Next Stage)
```

---

# Module Structure

```
backend/app/

ingestion/
│
├── loaders/
├── parsers/
├── chunking/
├── schemas/
└── tests/

embeddings/
│
├── base_embedder.py
├── embedding_schema.py
├── huggingface_embedder.py
├── fastembed_embedder.py
├── ollama_embedder.py
├── embedding_registry.py
├── embedding_pipeline.py
└── tests/
```

---

# Stage 1 — Document Loading

## Purpose

Load documents from different file formats into memory.

Supported loaders include:

* PDF
* DOCX
* TXT
* Markdown
* CSV
* Future custom loaders

Each loader is responsible only for reading the file.

No parsing.

No cleaning.

No chunking.

---

# Stage 2 — Document Parsing

## Purpose

Convert raw file contents into clean text.

Parsers normalize formatting while preserving document meaning.

Responsibilities include:

* extracting text
* cleaning whitespace
* preserving metadata
* preparing documents for chunking

Output:

```
IngestedDocument
```

---

# Stage 3 — Chunking

Large documents cannot be embedded directly.

Chunking divides a document into smaller semantic units.

SupportMind currently supports multiple strategies.

## Basic Chunker

Simple fixed-size chunking.

Useful for debugging.

---

## Recursive Chunker

Hierarchical splitting using separators.

Preserves natural text boundaries.

---

## LangChain Chunker

Uses LangChain's RecursiveCharacterTextSplitter.

Useful for compatibility with LangChain workflows.

---

## LlamaIndex Chunker

Uses LlamaIndex's sentence-based parsing.

Designed for general semantic chunking.

---

## Semantic Chunker

Groups semantically related content.

Produces higher-quality retrieval chunks.

---

## Sentence Window Chunker

Current default strategy.

Each node stores:

* current sentence
* neighboring sentence window

Advantages:

* preserves local context
* excellent retrieval quality
* lightweight
* compatible with LlamaIndex metadata replacement

---

# Chunker Registry

```
ChunkerRegistry
```

Responsibilities:

* register available chunkers
* instantiate chunkers
* validate strategies
* expose supported chunkers

The registry hides implementation details from the rest of the application.

---

# Chunking Pipeline

```
ChunkingPipeline
```

Responsibilities:

* choose chunker
* execute chunking
* enrich metadata
* return production-ready TextNodes

Additional metadata added includes:

* document_id
* source
* source_file
* chunk_index
* total_chunks
* chunking_strategy

Output:

```
List[TextNode]
```

---

# Stage 4 — Embeddings

Chunk text is converted into dense vector representations.

These vectors capture semantic meaning instead of exact keywords.

Embedding providers are completely interchangeable.

---

# EmbeddedNode Schema

Every provider returns the same object.

```
EmbeddedNode
```

Contains:

* node_id
* document_id
* text
* embedding
* embedding_model
* embedding_dimension
* metadata
* created_at

Using a common schema keeps downstream systems provider-agnostic.

---

# BaseEmbedder

Abstract base class defining the common embedding interface.

Responsibilities:

* validate inputs
* validate outputs
* expose provider name
* standardize embedding operations

Implemented methods:

```
embed()

embed_batch()
```

Implemented internally by:

```
_embed()

_embed_batch()
```

---

# HuggingFace Embedder

Built using SentenceTransformers.

Current default model:

```
BAAI/bge-base-en-v1.5
```

Characteristics:

* high semantic accuracy
* GPU support
* CPU compatible
* ideal for enterprise RAG

---

# FastEmbed Embedder

Built on ONNX Runtime.

Characteristics:

* extremely fast CPU inference
* lightweight deployment
* optimized memory usage
* ideal for production servers

---

# Ollama Embedder

Uses locally running Ollama models.

Current embedding model:

```
bge-m3
```

Advantages:

* completely offline
* privacy preserving
* no API cost
* enterprise deployment friendly

---

# Embedding Registry

```
EmbeddingRegistry
```

Responsibilities:

* register embedding providers
* instantiate providers
* validate provider names
* expose supported providers

Current providers:

* huggingface
* fastembed
* ollama

---

# Embedding Pipeline

```
EmbeddingPipeline
```

Responsibilities:

* select embedding provider
* embed one node
* embed multiple nodes
* return standardized EmbeddedNodes

This acts as the single entry point into the embedding subsystem.

Application code never communicates directly with provider implementations.

---

# Design Principles

The architecture follows several engineering principles.

## Provider Agnostic

Every provider returns the same schema.

Changing providers requires no downstream code changes.

---

## Single Responsibility

Each component performs exactly one task.

Examples:

* Loader → reads files
* Parser → extracts text
* Chunker → creates nodes
* Embedder → creates vectors

---

## Extensibility

Adding a new provider requires:

* implementing BaseEmbedder
* registering it

Nothing else changes.

---

## Testability

Every module has dedicated unit tests.

Providers are tested independently from pipelines.

Pipelines are tested independently from providers.

This enables reliable refactoring.

---

# Current Pipeline

```
Raw Document

↓

Loader

↓

Parser

↓

IngestedDocument

↓

Chunking Pipeline

↓

TextNodes

↓

Embedding Pipeline

↓

EmbeddedNodes
```

---

# Next Development Stage

The next milestone is implementing vector storage.

Planned modules:

```
vectorstore/

base_vectorstore.py

qdrant_vectorstore.py

collection_manager.py

vector_registry.py

vector_pipeline.py
```

After vector storage is completed, an indexing pipeline will orchestrate:

```
Chunking

↓

Embedding

↓

Caching

↓

Vector Storage
```

This will complete the end-to-end document indexing workflow.
