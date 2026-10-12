"""
vector_search_result.py

Provider-agnostic schema representing a retrieved vector search result.

Every vector database implementation (Qdrant, Pinecone, Milvus,
Weaviate, etc.) must convert its native search response into this
standardized representation.

This ensures downstream retrieval, Graph-RAG, agent workflows, and
evaluation components remain independent of the underlying vector
database.

SupportMind AI
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    """
    Represents a single retrieved document chunk returned by a
    vector database search.
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Unique vector identifier."
    )

    node_id: str = Field(
        ...,
        min_length=1,
        description="Original chunk identifier."
    )

    document_id: str = Field(
        ...,
        min_length=1,
        description="Parent document identifier."
    )

    text: str = Field(
        ...,
        description="Retrieved chunk text."
    )

    score: float = Field(
        ...,
        ge=0.0,
        description="Similarity score returned by the vector database."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata."
    )

    embedding_model: str = Field(
        ...,
        description="Embedding model used to generate the stored vector."
    )

    provider: str = Field(
        ...,
        description="Vector database provider."
    )

    collection_name: str = Field(
        ...,
        description="Collection searched."
    )

    retrieved_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Retrieval timestamp."
    )