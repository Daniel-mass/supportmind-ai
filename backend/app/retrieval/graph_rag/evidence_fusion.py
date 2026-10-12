"""
evidence_fusion.py

Cross-system evidence fusion for SupportMind Graph-RAG.

This module combines:

    1. BuiltContext from the existing RetrievalService
       - dense retrieval
       - BM25
       - hybrid RRF
       - existing retrieval Graph-RAG
       - reranking

    2. GraphRAGEvidence from the IntelligenceAdapter
       - Help Desk operational graph
       - workflow evidence
       - organizational evidence

into one bounded GraphRAGContext.

This module does not:
    - perform retrieval
    - perform graph traversal
    - call Neo4j directly
    - call Qdrant
    - call an LLM
    - perform LLM generation
"""

from __future__ import annotations

import re
from typing import Iterable

from app.retrieval.context.context_models import (
    BuiltContext,
    ContextItem,
    ContextSource,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGContext,
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
)


# ============================================================================
# Constants
# ============================================================================

_WHITESPACE_PATTERN = re.compile(r"\s+")


# ============================================================================
# Evidence Fusion
# ============================================================================


class EvidenceFusion:
    """
    Normalize and combine retrieval-layer and intelligence-layer evidence.

    The class is deliberately deterministic. Relevance ranking remains the
    responsibility of the existing retrieval/reranking layer.
    """

    def __init__(
        self,
        *,
        default_max_items: int = 20,
        default_max_characters: int = 16000,
    ) -> None:
        self._validate_positive_integer(
            default_max_items,
            "default_max_items",
        )

        self._validate_positive_integer(
            default_max_characters,
            "default_max_characters",
        )

        self.default_max_items = default_max_items
        self.default_max_characters = (
            default_max_characters
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def fuse(
        self,
        *,
        built_context: BuiltContext,
        intelligence_evidence: Iterable[
            GraphRAGEvidence
        ] | None = None,
        max_items: int | None = None,
        max_characters: int | None = None,
    ) -> GraphRAGContext:
        """
        Combine retrieval-layer and intelligence-layer evidence.

        Ordering:
            1. Existing retrieval context order is preserved.
            2. Intelligence evidence order is preserved.
            3. Exact duplicate evidence is removed.
            4. Global evidence IDs are assigned deterministically.
            5. Evidence budgets are applied.

        Parameters
        ----------
        built_context:
            Existing generation-ready context from RetrievalService.

        intelligence_evidence:
            Evidence returned by IntelligenceAdapter.

        max_items:
            Optional maximum number of final evidence items.

        max_characters:
            Optional maximum total evidence characters.

        Returns
        -------
        GraphRAGContext
            Unified bounded evidence context.
        """

        if not isinstance(
            built_context,
            BuiltContext,
        ):
            raise TypeError(
                "built_context must be a BuiltContext instance."
            )

        effective_max_items = (
            self.default_max_items
            if max_items is None
            else max_items
        )

        effective_max_characters = (
            self.default_max_characters
            if max_characters is None
            else max_characters
        )

        self._validate_positive_integer(
            effective_max_items,
            "max_items",
        )

        self._validate_positive_integer(
            effective_max_characters,
            "max_characters",
        )

        intelligence_items = list(
            intelligence_evidence or []
        )

        self._validate_intelligence_evidence(
            intelligence_items
        )

        # ------------------------------------------------------------------
        # 1. Convert existing retrieval context into GraphRAGEvidence.
        # ------------------------------------------------------------------

        retrieval_evidence = [
            self._context_item_to_evidence(
                item
            )
            for item in built_context.selected_items
        ]

        # ------------------------------------------------------------------
        # 2. Combine both evidence streams.
        # ------------------------------------------------------------------

        combined = (
            retrieval_evidence
            + intelligence_items
        )

        # ------------------------------------------------------------------
        # 3. Exact semantic/text duplicate removal.
        # ------------------------------------------------------------------

        deduplicated = self._deduplicate(
            combined
        )

        # ------------------------------------------------------------------
        # 4. Assign final deterministic evidence IDs.
        # ------------------------------------------------------------------

        numbered = self._assign_evidence_ids(
            deduplicated
        )

        # ------------------------------------------------------------------
        # 5. Apply final context budgets.
        # ------------------------------------------------------------------

        selected, truncated, character_count = (
            self._apply_budget(
                numbered,
                max_items=effective_max_items,
                max_characters=effective_max_characters,
            )
        )

        return GraphRAGContext(
            query=built_context.query,
            evidence=selected,
            selected_count=len(selected),
            character_count=character_count,
            truncated=truncated,
            metadata={
                "retrieval_evidence_count": len(
                    retrieval_evidence
                ),
                "intelligence_evidence_count": len(
                    intelligence_items
                ),
                "deduplicated_evidence_count": len(
                    deduplicated
                ),
                "max_items": effective_max_items,
                "max_characters": (
                    effective_max_characters
                ),
            },
        )

    # ========================================================================
    # Context conversion
    # ========================================================================

    @classmethod
    def _context_item_to_evidence(
        cls,
        item: ContextItem,
    ) -> GraphRAGEvidence:
        """
        Convert an existing retrieval ContextItem into GraphRAG evidence.
        """

        source, evidence_type = (
            cls._map_context_source(
                item.source
            )
        )

        metadata = dict(
            item.metadata or {}
        )

        metadata.update(
            {
                "context_item_id": item.id,
                "context_rank": item.rank,
                "context_source": item.source.value,
            }
        )

        source_reference = cls._source_reference(
            item
        )

        return GraphRAGEvidence(
            # Temporary ID.
            # Final global numbering happens during fusion.
            evidence_id="E1",
            source=source,
            evidence_type=evidence_type,
            text=item.text,
            source_reference=source_reference,
            score=(
                float(item.score)
                if item.score is not None
                else None
            ),
            metadata=metadata,
        )

    @staticmethod
    def _map_context_source(
        source: ContextSource,
    ) -> tuple[
        GraphRAGEvidenceSource,
        GraphRAGEvidenceType,
    ]:
        """
        Map an existing retrieval ContextSource to Graph-RAG evidence.
        """

        if source == ContextSource.DENSE:
            return (
                GraphRAGEvidenceSource.RETRIEVAL_DENSE,
                GraphRAGEvidenceType.DOCUMENT,
            )

        if source in (
            ContextSource.BM25,
            ContextSource.SPARSE,
        ):
            return (
                GraphRAGEvidenceSource.RETRIEVAL_SPARSE,
                GraphRAGEvidenceType.DOCUMENT,
            )

        if source == ContextSource.HYBRID:
            return (
                GraphRAGEvidenceSource.RETRIEVAL_HYBRID,
                GraphRAGEvidenceType.DOCUMENT,
            )

        if source == ContextSource.GRAPH:
            return (
                GraphRAGEvidenceSource.RETRIEVAL_GRAPH,
                GraphRAGEvidenceType.GRAPH_FACT,
            )

        if source == ContextSource.RERANKED:
            # Reranked results preserve the upstream retrieval provenance
            # inside metadata. The generation layer treats them as document
            # evidence rather than introducing a new retrieval source enum.
            return (
                GraphRAGEvidenceSource.RETRIEVAL_HYBRID,
                GraphRAGEvidenceType.DOCUMENT,
            )

        raise ValueError(
            f"Unsupported ContextSource: {source}"
        )

    @staticmethod
    def _source_reference(
        item: ContextItem,
    ) -> str:
        """
        Build a stable provenance reference for a retrieval item.
        """

        metadata = item.metadata or {}

        node_id = metadata.get(
            "node_id"
        )

        if isinstance(
            node_id,
            str,
        ) and node_id.strip():
            return f"node:{node_id.strip()}"

        document_id = metadata.get(
            "document_id"
        )

        if isinstance(
            document_id,
            str,
        ) and document_id.strip():
            return (
                f"document:{document_id.strip()}"
            )

        return f"context:{item.id}"

    # ========================================================================
    # Deduplication
    # ========================================================================

    def _deduplicate(
        self,
        evidence: list[GraphRAGEvidence],
    ) -> list[GraphRAGEvidence]:
        """
        Remove exact textual duplicates.

        If duplicate evidence exists, prefer the version with:
            1. a score over an unscored item
            2. the higher score
            3. the earlier item when otherwise tied
        """

        selected: dict[
            str,
            GraphRAGEvidence,
        ] = {}

        for item in evidence:
            key = self._normalize_text(
                item.text
            )

            existing = selected.get(
                key
            )

            if existing is None:
                selected[key] = item
                continue

            if self._should_replace(
                existing,
                item,
            ):
                selected[key] = item

        return list(
            selected.values()
        )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize text for deterministic exact-duplicate detection.
        """

        return _WHITESPACE_PATTERN.sub(
            " ",
            text.strip().lower(),
        )

    @staticmethod
    def _should_replace(
        existing: GraphRAGEvidence,
        candidate: GraphRAGEvidence,
    ) -> bool:
        """
        Decide whether a duplicate candidate should replace the existing one.
        """

        if (
            existing.score is None
            and candidate.score is not None
        ):
            return True

        if (
            existing.score is not None
            and candidate.score is None
        ):
            return False

        if (
            existing.score is not None
            and candidate.score is not None
        ):
            return (
                candidate.score
                > existing.score
            )

        return False

    # ========================================================================
    # Evidence numbering
    # ========================================================================

    @staticmethod
    def _assign_evidence_ids(
        evidence: list[GraphRAGEvidence],
    ) -> list[GraphRAGEvidence]:
        """
        Assign one deterministic global citation sequence.
        """

        return [
            item.model_copy(
                update={
                    "evidence_id": (
                        f"E{index}"
                    )
                }
            )
            for index, item in enumerate(
                evidence,
                start=1,
            )
        ]

    # ========================================================================
    # Budgeting
    # ========================================================================

    @staticmethod
    def _apply_budget(
        evidence: list[GraphRAGEvidence],
        *,
        max_items: int,
        max_characters: int,
    ) -> tuple[
        list[GraphRAGEvidence],
        bool,
        int,
    ]:
        """
        Select evidence without splitting individual evidence items.

        Items are consumed in the already-established deterministic order.
        """

        selected: list[
            GraphRAGEvidence
        ] = []

        character_count = 0
        truncated = False

        for item in evidence:
            if len(selected) >= max_items:
                truncated = True
                break

            item_length = len(
                item.text
            )

            if (
                character_count
                + item_length
                > max_characters
            ):
                truncated = True
                break

            selected.append(
                item
            )

            character_count += (
                item_length
            )

        if len(selected) < len(evidence):
            truncated = True

        return (
            selected,
            truncated,
            character_count,
        )

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_intelligence_evidence(
        evidence: list[GraphRAGEvidence],
    ) -> None:
        """
        Validate incoming intelligence evidence.
        """

        for index, item in enumerate(
            evidence
        ):
            if not isinstance(
                item,
                GraphRAGEvidence,
            ):
                raise TypeError(
                    "intelligence_evidence["
                    f"{index}"
                    "] must be a "
                    "GraphRAGEvidence instance."
                )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        field_name: str,
    ) -> None:
        if isinstance(
            value,
            bool,
        ) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{field_name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than zero."
            )