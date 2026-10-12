# backend/app/embeddings/embedding_schema.py

"""
Embedding Schema

Standard representation of an embedded document chunk.

Every embedding provider returns this schema,
allowing downstream components (Vector DB,
Retrieval, Reranking) to remain provider-agnostic.

SupportMind AI
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EmbeddedNode(BaseModel):
    """
    Represents a vectorized document chunk.
    """

    node_id: str = Field(
        ...,
        description="Unique identifier of the original TextNode."
    )

    document_id: str = Field(
        ...,
        description="Parent document identifier."
    )

    text: str = Field(
        ...,
        description="Original chunk text."
    )

    embedding: list[float] = Field(
        ...,
        description="Dense embedding vector."
    )

    embedding_model: str = Field(
        ...,
        description="Embedding model used to generate the vector."
    )

    embedding_dimension: int = Field(
        ...,
        description="Dimension of the embedding vector."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata."
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Embedding creation timestamp."
    )