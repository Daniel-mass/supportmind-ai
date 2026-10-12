import pytest

from app.retrieval.context.context_builder import ContextBuilder
from app.retrieval.context.context_models import (
    ContextBudget,
    ContextItem,
    ContextRequest,
    ContextSource,
)


def make_item(
    item_id,
    text,
    source,
    rank,
    score=None,
    metadata=None,
):
    return ContextItem(
        id=item_id,
        text=text,
        source=source,
        rank=rank,
        score=score,
        metadata=metadata or {},
    )


@pytest.fixture
def builder():
    return ContextBuilder()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_builder_accepts_valid_request(builder):
    request = ContextRequest(
        query="refund issue",
        results=[
            make_item(
                "doc-1",
                "Refunds are processed within five days.",
                ContextSource.DENSE,
                1,
                0.9,
            )
        ],
    )

    result = builder.build(request)

    assert result.query == "refund issue"
    assert result.selected_count == 1


def test_builder_rejects_none_request(builder):
    with pytest.raises(TypeError):
        builder.build(None)


def test_builder_rejects_invalid_request_type(builder):
    with pytest.raises(TypeError):
        builder.build("refund issue")


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_builder_handles_empty_results(builder):
    request = ContextRequest(
        query="refund issue",
        results=[],
    )

    result = builder.build(request)

    assert result.total_candidates == 0
    assert result.selected_count == 0
    assert result.selected_items == []
    assert result.truncated is False


def test_builder_creates_no_evidence_message_for_empty_results(builder):
    request = ContextRequest(
        query="refund issue",
        results=[],
    )

    result = builder.build(request)

    evidence_section = next(
        section
        for section in result.sections
        if section.title == "Evidence"
    )

    assert "No supporting evidence" in evidence_section.content


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_builder_deduplicates_same_document(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-1",
                "Refund policy.",
                ContextSource.DENSE,
                1,
                0.80,
            ),
            make_item(
                "doc-1",
                "Refund policy duplicate.",
                ContextSource.SPARSE,
                1,
                0.70,
            ),
        ],
    )

    result = builder.build(request)

    assert result.selected_count == 1
    assert result.selected_items[0].id == "doc-1"


def test_builder_keeps_different_documents(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-1",
                "Refund policy.",
                ContextSource.DENSE,
                1,
                0.90,
            ),
            make_item(
                "doc-2",
                "Refund escalation.",
                ContextSource.DENSE,
                2,
                0.80,
            ),
        ],
    )

    result = builder.build(request)

    assert result.selected_count == 2


# ---------------------------------------------------------------------------
# Source priority
# ---------------------------------------------------------------------------

def test_reranked_source_is_prioritized(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "dense-1",
                "Dense evidence.",
                ContextSource.DENSE,
                1,
                0.99,
            ),
            make_item(
                "reranked-1",
                "Reranked evidence.",
                ContextSource.RERANKED,
                4,
                0.70,
            ),
        ],
    )

    result = builder.build(request)

    assert result.selected_items[0].source == ContextSource.RERANKED


def test_graph_evidence_is_preserved(builder):
    request = ContextRequest(
        query="payment failure",
        results=[
            make_item(
                "graph-1",
                "Payment failure caused by gateway timeout.",
                ContextSource.GRAPH,
                1,
                0.80,
            ),
            make_item(
                "dense-1",
                "Payment troubleshooting information.",
                ContextSource.DENSE,
                1,
                0.90,
            ),
        ],
    )

    result = builder.build(request)

    sources = {
        item.source
        for item in result.selected_items
    }

    assert ContextSource.GRAPH in sources


# ---------------------------------------------------------------------------
# Source diversity
# ---------------------------------------------------------------------------

def test_builder_attempts_source_diversity(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "reranked-1",
                "Reranked evidence.",
                ContextSource.RERANKED,
                1,
                0.95,
            ),
            make_item(
                "graph-1",
                "Graph evidence.",
                ContextSource.GRAPH,
                1,
                0.80,
            ),
            make_item(
                "dense-1",
                "Dense evidence.",
                ContextSource.DENSE,
                1,
                0.75,
            ),
        ],
        budget=ContextBudget(
            max_items=3,
            max_characters=1000,
        ),
    )

    result = builder.build(request)

    sources = {
        item.source
        for item in result.selected_items
    }

    assert ContextSource.RERANKED in sources
    assert ContextSource.GRAPH in sources


# ---------------------------------------------------------------------------
# Top-k
# ---------------------------------------------------------------------------

def test_builder_respects_max_items(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                f"doc-{index}",
                f"Evidence {index}",
                ContextSource.DENSE,
                index,
                1.0 / index,
            )
            for index in range(1, 10)
        ],
        budget=ContextBudget(
            max_items=3,
            max_characters=10000,
        ),
    )

    result = builder.build(request)

    assert result.selected_count == 3
    assert len(result.selected_items) == 3


def test_builder_allows_budget_larger_than_results(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-1",
                "Evidence",
                ContextSource.DENSE,
                1,
                0.9,
            )
        ],
        budget=ContextBudget(
            max_items=10,
            max_characters=10000,
        ),
    )

    result = builder.build(request)

    assert result.selected_count == 1
    assert result.truncated is False


# ---------------------------------------------------------------------------
# Character budget
# ---------------------------------------------------------------------------

def test_builder_respects_character_budget(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-1",
                "A" * 100,
                ContextSource.DENSE,
                1,
                0.9,
            ),
            make_item(
                "doc-2",
                "B" * 100,
                ContextSource.DENSE,
                2,
                0.8,
            ),
        ],
        budget=ContextBudget(
            max_items=10,
            max_characters=100,
        ),
    )

    result = builder.build(request)

    assert result.selected_count == 1
    assert result.character_count <= 100


def test_builder_marks_context_as_truncated_when_items_are_excluded(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-1",
                "A" * 100,
                ContextSource.DENSE,
                1,
                0.9,
            ),
            make_item(
                "doc-2",
                "B" * 100,
                ContextSource.DENSE,
                2,
                0.8,
            ),
        ],
        budget=ContextBudget(
            max_items=10,
            max_characters=100,
        ),
    )

    result = builder.build(request)

    assert result.truncated is True


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_builder_orders_by_priority(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "low",
                "Low evidence",
                ContextSource.SPARSE,
                1,
                0.99,
            ),
            make_item(
                "high",
                "High evidence",
                ContextSource.RERANKED,
                5,
                0.50,
            ),
        ],
    )

    result = builder.build(request)

    assert result.selected_items[0].id == "high"


def test_builder_is_deterministic(builder):
    results = [
        make_item(
            "doc-2",
            "Evidence two",
            ContextSource.DENSE,
            2,
            0.8,
        ),
        make_item(
            "doc-1",
            "Evidence one",
            ContextSource.DENSE,
            1,
            0.8,
        ),
    ]

    request_one = ContextRequest(
        query="refund",
        results=results,
    )

    request_two = ContextRequest(
        query="refund",
        results=list(reversed(results)),
    )

    result_one = builder.build(request_one)
    result_two = builder.build(request_two)

    assert [
        item.id for item in result_one.selected_items
    ] == [
        item.id for item in result_two.selected_items
    ]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def test_builder_creates_query_section(builder):
    request = ContextRequest(
        query="Why is my refund delayed?",
    )

    result = builder.build(request)

    query_section = next(
        section
        for section in result.sections
        if section.title == "User Query"
    )

    assert query_section.content == "Why is my refund delayed?"


def test_builder_creates_evidence_section(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-1",
                "Refunds take five days.",
                ContextSource.DENSE,
                1,
                0.9,
            )
        ],
    )

    result = builder.build(request)

    evidence_section = next(
        section
        for section in result.sections
        if section.title == "Evidence"
    )

    assert "Refunds take five days." in evidence_section.content


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_builder_includes_provenance_by_default(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-1",
                "Refund policy.",
                ContextSource.GRAPH,
                2,
                0.8,
            )
        ],
    )

    result = builder.build(request)

    evidence_section = next(
        section
        for section in result.sections
        if section.title == "Evidence"
    )

    assert "Source: graph" in evidence_section.content
    assert "Rank: 2" in evidence_section.content


def test_builder_can_disable_provenance(builder):
    request = ContextRequest(
        query="refund",
        include_provenance=False,
        results=[
            make_item(
                "doc-1",
                "Refund policy.",
                ContextSource.GRAPH,
                2,
                0.8,
            )
        ],
    )

    result = builder.build(request)

    evidence_section = next(
        section
        for section in result.sections
        if section.title == "Evidence"
    )

    assert "Source: graph" not in evidence_section.content
    assert "Rank: 2" not in evidence_section.content


def test_builder_includes_metadata_by_default(builder):
    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-1",
                "Refund policy.",
                ContextSource.DENSE,
                1,
                0.8,
                metadata={
                    "document_id": "refund-policy",
                    "category": "billing",
                },
            )
        ],
    )

    result = builder.build(request)

    evidence_section = next(
        section
        for section in result.sections
        if section.title == "Evidence"
    )

    assert "document_id=refund-policy" in evidence_section.content
    assert "category=billing" in evidence_section.content


def test_builder_can_disable_metadata(builder):
    request = ContextRequest(
        query="refund",
        include_metadata=False,
        results=[
            make_item(
                "doc-1",
                "Refund policy.",
                ContextSource.DENSE,
                1,
                0.8,
                metadata={
                    "document_id": "refund-policy",
                },
            )
        ],
    )

    result = builder.build(request)

    evidence_section = next(
        section
        for section in result.sections
        if section.title == "Evidence"
    )

    assert "document_id=refund-policy" not in evidence_section.content


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

def test_builder_preserves_original_metadata(builder):
    metadata = {
        "document_id": "doc-42",
        "ticket_id": "5847",
        "category": "billing",
    }

    request = ContextRequest(
        query="refund",
        results=[
            make_item(
                "doc-42",
                "Refund evidence.",
                ContextSource.RERANKED,
                1,
                0.99,
                metadata=metadata,
            )
        ],
    )

    result = builder.build(request)

    assert result.selected_items[0].metadata == metadata


# ---------------------------------------------------------------------------
# Original input immutability
# ---------------------------------------------------------------------------

def test_builder_does_not_mutate_original_results(builder):
    original = make_item(
        "doc-1",
        "Refund evidence.",
        ContextSource.DENSE,
        1,
        0.9,
        metadata={"category": "billing"},
    )

    original_text = original.text
    original_metadata = dict(original.metadata)

    request = ContextRequest(
        query="refund",
        results=[original],
    )

    builder.build(request)

    assert original.text == original_text
    assert original.metadata == original_metadata