"""
document_Schema.py

Standard document representation used across
the SupportMind AI ingestion pipeline.

Responsibilities
----------------
1. Represent a normalized ingested document.
2. Provide a stable document identifier.
3. Preserve source and source-file information.
4. Preserve extracted document content and metadata.

SupportMind AI
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def generate_document_id(
    source: str,
    source_file: str,
) -> str:
    """
    Generate a deterministic document ID.

    The same source + source_file will always produce
    the same document ID.

    This is important for idempotent ingestion.

    Example
    -------
    DOC-8F4A2C91D5E7B123
    """

    normalized_source = (
        str(source)
        .strip()
        .lower()
    )

    normalized_file = str(
        Path(source_file)
    ).strip().lower()

    identity = (
        f"{normalized_source}:"
        f"{normalized_file}"
    )

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16].upper()

    return f"DOC-{digest}"


class IngestedDocument(BaseModel):
    """
    Standard document representation used across
    the SupportMind AI pipeline.

    Every loader must return an IngestedDocument
    so downstream services can operate on one
    unified representation.

    The document ID is deterministic based on:

        source
        +
        source_file

    Therefore, re-ingesting the same source file
    produces the same document ID.
    """

    document_id: str | None = Field(
        default=None,
        description=(
            "Stable unique identifier for the document."
        ),
    )

    source: str = Field(
        ...,
        description=(
            "Origin of the document "
            "(pdf, csv, email, ticket, api, etc.)."
        ),
    )

    source_file: str = Field(
        ...,
        description=(
            "Original file name or source identifier."
        ),
    )

    content: str = Field(
        ...,
        description="Extracted textual content.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional metadata about the document."
        ),
    )

    def model_post_init(
        self,
        __context: Any,
    ) -> None:
        """
        Assign a deterministic document ID after
        source and source_file have been validated.

        Existing explicitly supplied document IDs
        are preserved.
        """

        if self.document_id is None:

            self.document_id = generate_document_id(
                source=self.source,
                source_file=self.source_file,
            )