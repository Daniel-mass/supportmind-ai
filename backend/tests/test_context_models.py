import pytest
from pydantic import ValidationError

from app.retrieval.context.context_models import (
    BuiltContext,
    ContextBudget,
    ContextItem,
    ContextRequest,
    ContextSection,
    ContextSource,
)


def make_item(
    *,
    item_id="doc-1",
    text="Refund policy information.",
    source=ContextSource.DENSE,
    rank=1,
    score=0.91,
):
    return ContextItem(
        id=item_id,
        text=text,
        source=source,
        rank=rank,
        score=score,
        metadata={"document_id": "doc-1"},
    )


# ---------------------------------------------------------------------------
# ContextSource
# ---------------------------------------------------------------------------

def test_context_source_contains_expected_retrieval_sources():
    assert ContextSource.DENSE.value == "dense"
    assert ContextSource.SPARSE.value == "sparse"
    assert ContextSource.HYBRID.value == "hybrid"
    assert ContextSource.GRAPH.value == "graph"
    assert ContextSource.RERANKED.value == "reranked"


# ---------------------------------------------------------------------------
# ContextItem
# ---------------------------------------------------------------------------

def test_context_item_accepts_valid_data():
    item = make_item()

    assert item.id == "doc-1"
    assert item.text == "Refund policy information."
    assert item.source == ContextSource.DENSE
    assert item.rank == 1
    assert item.score == 0.91


def test_context_item_strips_id_and_text():
    item = ContextItem(
        id="  doc-1  ",
        text="  refund policy  ",
        source="dense",
        rank=1,
    )

    assert item.id == "doc-1"
    assert item.text == "refund policy"


def test_context_item_rejects_empty_id():
    with pytest.raises(ValidationError):
        ContextItem(
            id="",
            text="some text",
            source="dense",
            rank=1,
        )


def test_context_item_rejects_whitespace_id():
    with pytest.raises(ValidationError):
        ContextItem(
            id="   ",
            text="some text",
            source="dense",
            rank=1,
        )


def test_context_item_rejects_empty_text():
    with pytest.raises(ValidationError):
        ContextItem(
            id="doc-1",
            text="",
            source="dense",
            rank=1,
        )


def test_context_item_rejects_whitespace_text():
    with pytest.raises(ValidationError):
        ContextItem(
            id="doc-1",
            text="   ",
            source="dense",
            rank=1,
        )


def test_context_item_rejects_invalid_rank():
    with pytest.raises(ValidationError):
        ContextItem(
            id="doc-1",
            text="text",
            source="dense",
            rank=0,
        )


def test_context_item_rejects_invalid_source():
    with pytest.raises(ValidationError):
        ContextItem(
            id="doc-1",
            text="text",
            source="invalid",
            rank=1,
        )


def test_context_item_accepts_missing_score():
    item = make_item(score=None)

    assert item.score is None


def test_context_item_rejects_boolean_score():
    with pytest.raises(ValidationError):
        make_item(score=True)


def test_context_item_converts_numeric_score_to_float():
    item = make_item(score=1)

    assert item.score == 1.0
    assert isinstance(item.score, float)


def test_context_item_preserves_metadata():
    item = make_item()

    assert item.metadata["document_id"] == "doc-1"


def test_context_item_defaults_metadata_to_empty_dict():
    item = ContextItem(
        id="doc-1",
        text="text",
        source="dense",
        rank=1,
    )

    assert item.metadata == {}


def test_context_item_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ContextItem(
            id="doc-1",
            text="text",
            source="dense",
            rank=1,
            unknown_field="bad",
        )


# ---------------------------------------------------------------------------
# ContextBudget
# ---------------------------------------------------------------------------

def test_context_budget_uses_defaults():
    budget = ContextBudget()

    assert budget.max_items == 8
    assert budget.max_characters == 12000


def test_context_budget_accepts_custom_values():
    budget = ContextBudget(
        max_items=5,
        max_characters=5000,
    )

    assert budget.max_items == 5
    assert budget.max_characters == 5000


def test_context_budget_rejects_zero_max_items():
    with pytest.raises(ValidationError):
        ContextBudget(max_items=0)


def test_context_budget_rejects_negative_max_items():
    with pytest.raises(ValidationError):
        ContextBudget(max_items=-1)


def test_context_budget_rejects_zero_max_characters():
    with pytest.raises(ValidationError):
        ContextBudget(max_characters=0)


def test_context_budget_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ContextBudget(unknown_field=True)


# ---------------------------------------------------------------------------
# ContextRequest
# ---------------------------------------------------------------------------

def test_context_request_accepts_valid_request():
    request = ContextRequest(
        query="Why has my refund not arrived?",
        results=[make_item()],
    )

    assert request.query == "Why has my refund not arrived?"
    assert len(request.results) == 1


def test_context_request_strips_query():
    request = ContextRequest(query="  refund status  ")

    assert request.query == "refund status"


def test_context_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        ContextRequest(query="")


def test_context_request_rejects_whitespace_query():
    with pytest.raises(ValidationError):
        ContextRequest(query="   ")


def test_context_request_defaults_results_to_empty():
    request = ContextRequest(query="refund")

    assert request.results == []


def test_context_request_defaults_budget():
    request = ContextRequest(query="refund")

    assert request.budget.max_items == 8
    assert request.budget.max_characters == 12000


def test_context_request_defaults_metadata_flags():
    request = ContextRequest(query="refund")

    assert request.include_metadata is True
    assert request.include_provenance is True


def test_context_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ContextRequest(
            query="refund",
            unknown_field=True,
        )


# ---------------------------------------------------------------------------
# ContextSection
# ---------------------------------------------------------------------------

def test_context_section_accepts_valid_data():
    section = ContextSection(
        title="Evidence",
        content="Refunds are processed within five business days.",
    )

    assert section.title == "Evidence"
    assert section.content.startswith("Refunds")


def test_context_section_strips_strings():
    section = ContextSection(
        title="  Evidence  ",
        content="  Some evidence  ",
    )

    assert section.title == "Evidence"
    assert section.content == "Some evidence"


def test_context_section_rejects_empty_title():
    with pytest.raises(ValidationError):
        ContextSection(
            title="",
            content="content",
        )


def test_context_section_rejects_empty_content():
    with pytest.raises(ValidationError):
        ContextSection(
            title="Evidence",
            content="",
        )


def test_context_section_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ContextSection(
            title="Evidence",
            content="content",
            unknown=True,
        )


# ---------------------------------------------------------------------------
# BuiltContext
# ---------------------------------------------------------------------------

def test_built_context_accepts_valid_context():
    item = make_item()

    context = BuiltContext(
        query="refund",
        sections=[
            ContextSection(
                title="Evidence",
                content="Refund information.",
            )
        ],
        selected_items=[item],
        total_candidates=3,
        selected_count=1,
        character_count=100,
    )

    assert context.query == "refund"
    assert context.total_candidates == 3
    assert context.selected_count == 1
    assert context.character_count == 100
    assert context.truncated is False


def test_built_context_defaults_sections_and_items():
    context = BuiltContext(
        query="refund",
        total_candidates=0,
        selected_count=0,
        character_count=0,
    )

    assert context.sections == []
    assert context.selected_items == []
    assert context.truncated is False


def test_built_context_validates_selected_count():
    item = make_item()

    with pytest.raises(ValidationError):
        BuiltContext(
            query="refund",
            selected_items=[item],
            total_candidates=1,
            selected_count=0,
            character_count=100,
        )


def test_built_context_accepts_zero_selected_items():
    context = BuiltContext(
        query="refund",
        selected_items=[],
        total_candidates=5,
        selected_count=0,
        character_count=0,
    )

    assert context.selected_count == 0


def test_built_context_rejects_negative_total_candidates():
    with pytest.raises(ValidationError):
        BuiltContext(
            query="refund",
            total_candidates=-1,
            selected_count=0,
            character_count=0,
        )


def test_built_context_rejects_negative_character_count():
    with pytest.raises(ValidationError):
        BuiltContext(
            query="refund",
            total_candidates=0,
            selected_count=0,
            character_count=-1,
        )


def test_built_context_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        BuiltContext(
            query="refund",
            total_candidates=0,
            selected_count=0,
            character_count=0,
            unknown_field=True,
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_context_models_are_json_serializable():
    item = make_item()

    request = ContextRequest(
        query="refund",
        results=[item],
    )

    payload = request.model_dump_json()

    assert "refund" in payload
    assert "doc-1" in payload
    assert "dense" in payload


def test_built_context_is_json_serializable():
    item = make_item()

    context = BuiltContext(
        query="refund",
        selected_items=[item],
        total_candidates=1,
        selected_count=1,
        character_count=len(item.text),
    )

    payload = context.model_dump_json()

    assert "refund" in payload
    assert "doc-1" in payload
    assert "dense" in payload