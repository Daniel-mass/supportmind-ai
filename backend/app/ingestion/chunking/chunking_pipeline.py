"""
Chunking Pipeline

Production-grade chunking pipeline.

Responsibilities
----------------
1. Select chunking strategy.
2. Execute chunker.
3. Assign deterministic chunk/node IDs.
4. Enrich node metadata.
5. Return production-ready nodes.

SupportMind AI
"""

from __future__ import annotations

import uuid

from llama_index.core.schema import Document

from app.ingestion.chunking.chunker_registry import ChunkerRegistry
from app.ingestion.schemas.document_schema import IngestedDocument


class ChunkingPipeline:
    """
    Executes one selected chunking strategy.

    Chunk IDs are deterministic so that re-ingesting
    the same document with the same chunking strategy
    produces the same Qdrant point IDs.

    This makes Qdrant upsert operations idempotent.
    """

    # Fixed namespace used for deterministic UUID generation.

    _NODE_NAMESPACE = uuid.UUID(
        "7f7c5d2b-7b1e-4b89-9d76-9b9c2f4d2a11"
    )

    def __init__(
        self,
        default_strategy: str = "sentence_window",
    ) -> None:

        self.registry = ChunkerRegistry()

        self.default_strategy = (
            default_strategy
        )

    def chunk(
        self,
        document: IngestedDocument,
        strategy: str | None = None,
    ):
        """
        Chunk an ingested document.

        Parameters
        ----------
        document:
            Normalized ingested document.

        strategy:
            Chunking strategy.

            If omitted, the configured default strategy
            is used.

        Returns
        -------
        list[TextNode]
            Chunked and metadata-enriched nodes.
        """

        if document is None:

            raise ValueError(
                "IngestedDocument cannot be None."
            )

        strategy = (
            strategy
            or self.default_strategy
        )

        strategy = strategy.strip().lower()

        chunker = (
            self.registry.get_chunker(
                strategy
            )
        )

        llama_document = Document(
            text=document.content,
            metadata=document.metadata.copy(),
        )

        nodes = chunker.split(
            llama_document
        )

        total_chunks = len(nodes)

        for index, node in enumerate(nodes):

            # --------------------------------------------------
            # Deterministic node ID
            # --------------------------------------------------
            #
            # Same:
            #   document
            #   strategy
            #   chunk index
            #
            # produces the same UUID.
            #
            # Therefore:
            #
            # ingestion run 1
            #      ↓
            # node UUID A
            #
            # ingestion run 2
            #      ↓
            # node UUID A
            #
            # Qdrant upsert => UPDATE
            # instead of INSERT.
            #

            node_identity = (
                f"{document.document_id}:"
                f"{strategy}:"
                f"{index}"
            )

            deterministic_node_id = str(
                uuid.uuid5(
                    self._NODE_NAMESPACE,
                    node_identity,
                )
            )

            node.node_id = (
                deterministic_node_id
            )

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------

            node.metadata.update(
                {
                    "document_id": (
                        document.document_id
                    ),
                    "source": document.source,
                    "source_file": (
                        document.source_file
                    ),
                    "chunking_strategy": strategy,
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                }
            )

        return nodes