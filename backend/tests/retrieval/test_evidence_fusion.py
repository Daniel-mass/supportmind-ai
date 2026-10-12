"""
Unit tests for Graph-RAG EvidenceFusion.
"""

from __future__ import annotations

import pytest

from app.retrieval.context.context_models import (
    BuiltContext,
    ContextItem,
    ContextSource,
)
from app.retrieval.graph_rag.evidence_fusion import (
    EvidenceFusion,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def built_context() -> BuiltContext:
    return BuiltContext(
        query="Why was the ticket delayed?",
        selected_items=[
            ContextItem(
                id="dense-1",
                text="Ticket ABC-123 was delayed during approval.",
                source=ContextSource.DENSE,
                rank=1,
                score=0.91,
                metadata={
                    "node_id": "node-abc-123",
                    "document_id": "doc-001",
                },
            ),
            ContextItem(
                id="graph-1",
                text="Ticket ABC-123 is assigned to Payments.",
                source=ContextSource.GRAPH,
                rank=2,
                score=0.82,
                metadata={
                    "node_id": "ticket:ABC-123",
                    "document_id": "doc-001",
                },
            ),
        ],
        total_candidates=4,
        selected_count=2,
        character_count=98,
    )


@pytest.fixture
def intelligence_evidence() -> list[GraphRAGEvidence]:
    return [
        GraphRAGEvidence(
            evidence_id="E1",
            source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
            evidence_type=GraphRAGEvidenceType.WORKFLOW_EVENT,
            text=(
                "Ticket ABC-123 spent 18 hours "
                "waiting for customer approval."
            ),
            source_reference=(
                "workflow:ticket:ABC-123:1"
            ),
        ),
        GraphRAGEvidence(
            evidence_id="E2",
            source=(
                GraphRAGEvidenceSource
                .ORGANIZATION_INTELLIGENCE
            ),
            evidence_type=GraphRAGEvidenceType.METRIC,
            text=(
                "High priority tickets have "
                "an elevated approval wait time."
            ),
            source_reference=(
                "organization:approval-wait"
            ),
        ),
    ]


# ============================================================================
# Constructor
# ============================================================================


def test_constructor_uses_defaults() -> None:
    fusion = EvidenceFusion()

    assert fusion.default_max_items == 20
    assert fusion.default_max_characters == 16000


def test_constructor_rejects_invalid_defaults() -> None:
    with pytest.raises(ValueError):
        EvidenceFusion(
            default_max_items=0
        )

    with pytest.raises(ValueError):
        EvidenceFusion(
            default_max_characters=0
        )


# ============================================================================
# Conversion
# ============================================================================


def test_fuse_converts_dense_and_graph_context_items(
    built_context: BuiltContext,
) -> None:
    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context
    )

    assert result.selected_count == 2

    assert (
        result.evidence[0].source
        == GraphRAGEvidenceSource.RETRIEVAL_DENSE
    )

    assert (
        result.evidence[0].evidence_type
        == GraphRAGEvidenceType.DOCUMENT
    )

    assert (
        result.evidence[1].source
        == GraphRAGEvidenceSource.RETRIEVAL_GRAPH
    )

    assert (
        result.evidence[1].evidence_type
        == GraphRAGEvidenceType.GRAPH_FACT
    )


def test_fuse_preserves_retrieval_provenance(
    built_context: BuiltContext,
) -> None:
    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context
    )

    first = result.evidence[0]

    assert first.source_reference == (
        "node:node-abc-123"
    )

    assert first.metadata[
        "context_item_id"
    ] == "dense-1"

    assert first.metadata[
        "context_source"
    ] == "dense"


# ============================================================================
# Intelligence evidence
# ============================================================================


def test_fuse_adds_intelligence_evidence(
    built_context: BuiltContext,
    intelligence_evidence: list[
        GraphRAGEvidence
    ],
) -> None:
    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context,
        intelligence_evidence=(
            intelligence_evidence
        ),
    )

    assert result.selected_count == 4

    assert result.evidence[2].source == (
        GraphRAGEvidenceSource.HELPDESK_GRAPH
    )

    assert result.evidence[3].source == (
        GraphRAGEvidenceSource
        .ORGANIZATION_INTELLIGENCE
    )


def test_fuse_assigns_global_evidence_ids(
    built_context: BuiltContext,
    intelligence_evidence: list[
        GraphRAGEvidence
    ],
) -> None:
    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context,
        intelligence_evidence=(
            intelligence_evidence
        ),
    )

    assert [
        item.evidence_id
        for item in result.evidence
    ] == [
        "E1",
        "E2",
        "E3",
        "E4",
    ]


# ============================================================================
# Deduplication
# ============================================================================


def test_fuse_removes_exact_duplicate_text(
    built_context: BuiltContext,
) -> None:
    duplicate = GraphRAGEvidence(
        evidence_id="E90",
        source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
        evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
        text=(
            " Ticket ABC-123 was delayed during approval. "
        ),
        source_reference="helpdesk:duplicate",
    )

    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context,
        intelligence_evidence=[duplicate],
    )

    assert result.selected_count == 2


def test_duplicate_with_score_replaces_unscored_item(
    built_context: BuiltContext,
) -> None:
    # Make retrieval item unscored.
    context = built_context.model_copy(
        update={
            "selected_items": [
                built_context.selected_items[0].model_copy(
                    update={
                        "score": None
                    }
                )
            ],
            "selected_count": 1,
        }
    )

    scored_duplicate = GraphRAGEvidence(
        evidence_id="E2",
        source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
        evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
        text=(
            "Ticket ABC-123 was delayed during approval."
        ),
        source_reference="helpdesk:duplicate",
        score=0.95,
    )

    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=context,
        intelligence_evidence=[
            scored_duplicate
        ],
    )

    assert result.selected_count == 1
    assert result.evidence[0].score == pytest.approx(
        0.95
    )


# ============================================================================
# Budgeting
# ============================================================================


def test_fuse_respects_max_items(
    built_context: BuiltContext,
    intelligence_evidence: list[
        GraphRAGEvidence
    ],
) -> None:
    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context,
        intelligence_evidence=(
            intelligence_evidence
        ),
        max_items=2,
    )

    assert result.selected_count == 2
    assert result.truncated is True


def test_fuse_respects_character_budget(
    built_context: BuiltContext,
    intelligence_evidence: list[
        GraphRAGEvidence
    ],
) -> None:
    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context,
        intelligence_evidence=(
            intelligence_evidence
        ),
        max_characters=60,
    )

    assert result.character_count <= 60
    assert result.truncated is True


def test_fuse_marks_untruncated_context(
    built_context: BuiltContext,
) -> None:
    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context,
        max_items=10,
        max_characters=10000,
    )

    assert result.truncated is False


# ============================================================================
# Validation
# ============================================================================


def test_fuse_rejects_invalid_built_context() -> None:
    fusion = EvidenceFusion()

    with pytest.raises(TypeError):
        fusion.fuse(
            built_context="invalid"  # type: ignore[arg-type]
        )


def test_fuse_rejects_invalid_intelligence_evidence(
    built_context: BuiltContext,
) -> None:
    fusion = EvidenceFusion()

    with pytest.raises(TypeError):
        fusion.fuse(
            built_context=built_context,
            intelligence_evidence=[
                "invalid"  # type: ignore[list-item]
            ],
        )


def test_fuse_rejects_invalid_budget(
    built_context: BuiltContext,
) -> None:
    fusion = EvidenceFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            built_context=built_context,
            max_items=0,
        )

    with pytest.raises(ValueError):
        fusion.fuse(
            built_context=built_context,
            max_characters=0,
        )


# ============================================================================
# Empty intelligence evidence
# ============================================================================


def test_fuse_works_without_intelligence_evidence(
    built_context: BuiltContext,
) -> None:
    fusion = EvidenceFusion()

    result = fusion.fuse(
        built_context=built_context,
        intelligence_evidence=None,
    )

    assert result.selected_count == 2
    assert result.metadata[
        "intelligence_evidence_count"
    ] == 0