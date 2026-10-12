"""
Unit tests for GraphRAGService.
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
from app.retrieval.graph_rag.generator import (
    GraphRAGGenerator,
)
from app.retrieval.graph_rag.groundedness import (
    GroundednessValidator,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
    GraphRAGRequest,
)
from app.retrieval.graph_rag.graph_rag_service import (
    GraphRAGService,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)


# ============================================================================
# Fakes
# ============================================================================


class FakeRetrievalService:
    def __init__(
        self,
        built_context: BuiltContext,
    ) -> None:
        self.built_context = built_context
        self.requests = []

    def retrieve(
        self,
        request,
    ) -> BuiltContext:
        self.requests.append(
            request
        )
        return self.built_context


class FakeIntelligenceAdapter:
    def __init__(
        self,
        ticket_evidence=None,
        organization_evidence=None,
    ) -> None:
        self.ticket_evidence = (
            ticket_evidence or []
        )

        self.organization_evidence = (
            organization_evidence or []
        )

        self.ticket_calls = []
        self.organization_calls = 0
        self.relevant_organization_calls = []

    def get_ticket_evidence(
        self,
        ticket_id: str,
        *,
        include_neighborhood: bool = True,
        include_workflow: bool = True,
        include_transitions: bool = True,
    ):
        self.ticket_calls.append(
            {
                "ticket_id": ticket_id,
                "include_neighborhood": (
                    include_neighborhood
                ),
                "include_workflow": (
                    include_workflow
                ),
                "include_transitions": (
                    include_transitions
                ),
            }
        )

        return list(
            self.ticket_evidence
        )

    def get_organization_evidence(
        self,
    ):
        self.organization_calls += 1

        return list(
            self.organization_evidence
        )

    def get_relevant_organization_evidence(
        self,
        query: str,
    ):
        self.relevant_organization_calls.append(
            query
        )

        return list(
            self.organization_evidence
        )


class FakeGenerator:
    def __init__(
        self,
        answer: str = (
            "The ticket spent 18 hours "
            "waiting for approval. [E2]"
        ),
        provider_name: str = "ollama",
        model_name: str = "qwen2.5:7b",
    ) -> None:
        self.answer = answer
        self.provider_name = provider_name
        self.model_name = model_name
        self.contexts = []

    def generate(
        self,
        context,
    ) -> str:
        self.contexts.append(
            context
        )
        return self.answer


class FakeGroundednessValidator:
    def __init__(
        self,
        result,
    ) -> None:
        self.result = result
        self.calls = []

    def validate(
        self,
        *,
        answer: str,
        context,
    ):
        self.calls.append(
            {
                "answer": answer,
                "context": context,
            }
        )

        return self.result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def retrieval_request() -> RetrievalRequest:
    return RetrievalRequest(
        query="Why was ticket ABC-123 delayed?",
        top_k=5,
        collection_name="supportmind",
        query_vector=[0.1, 0.2, 0.3],
        include_graph=True,
        rerank=True,
        rerank_top_k=5,
    )


@pytest.fixture
def built_context() -> BuiltContext:
    return BuiltContext(
        query="Why was ticket ABC-123 delayed?",
        selected_items=[
            ContextItem(
                id="retrieval-1",
                text=(
                    "Ticket ABC-123 was delayed "
                    "during approval."
                ),
                source=ContextSource.HYBRID,
                rank=1,
                score=0.92,
                metadata={
                    "node_id": "ticket:ABC-123",
                    "document_id": "doc-123",
                },
            )
        ],
        total_candidates=1,
        selected_count=1,
        character_count=45,
    )


@pytest.fixture
def intelligence_evidence() -> list[
    GraphRAGEvidence
]:
    return [
        GraphRAGEvidence(
            evidence_id="E99",
            source=(
                GraphRAGEvidenceSource
                .HELPDESK_GRAPH
            ),
            evidence_type=(
                GraphRAGEvidenceType
                .WORKFLOW_EVENT
            ),
            text=(
                "Ticket ABC-123 spent 18 hours "
                "waiting for approval."
            ),
            source_reference=(
                "workflow:ABC-123:1"
            ),
        )
    ]


@pytest.fixture
def grounding_result():
    from app.retrieval.graph_rag.graph_rag_models import (
        GroundingResult,
        GroundingStatus,
    )

    return GroundingResult(
        status=GroundingStatus.GROUNDED,
        valid=True,
        cited_evidence_ids=["E2"],
    )


# ============================================================================
# Helper
# ============================================================================


def make_service(
    *,
    retrieval_service,
    intelligence_adapter,
    generator,
    grounding_validator,
) -> GraphRAGService:
    return GraphRAGService(
        retrieval_service=retrieval_service,
        intelligence_adapter=intelligence_adapter,
        evidence_fusion=EvidenceFusion(),
        generator=generator,
        groundedness_validator=(
            grounding_validator
        ),
    )


# ============================================================================
# Constructor
# ============================================================================


def test_constructor_accepts_valid_dependencies(
    built_context: BuiltContext,
    grounding_result,
) -> None:
    service = make_service(
        retrieval_service=FakeRetrievalService(
            built_context
        ),
        intelligence_adapter=FakeIntelligenceAdapter(),
        generator=FakeGenerator(),
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    assert service.generator.provider_name == (
        "ollama"
    )


@pytest.mark.parametrize(
    "dependency_name",
    [
        "retrieval_service",
        "intelligence_adapter",
        "evidence_fusion",
        "generator",
        "groundedness_validator",
    ],
)
def test_constructor_rejects_none_dependency(
    dependency_name: str,
    built_context: BuiltContext,
    grounding_result,
) -> None:
    dependencies = {
        "retrieval_service": (
            FakeRetrievalService(
                built_context
            )
        ),
        "intelligence_adapter": (
            FakeIntelligenceAdapter()
        ),
        "evidence_fusion": EvidenceFusion(),
        "generator": FakeGenerator(),
        "groundedness_validator": (
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    }

    dependencies[
        dependency_name
    ] = None

    with pytest.raises(
        ValueError,
        match="cannot be None",
    ):
        GraphRAGService(
            **dependencies
        )


# ============================================================================
# Full orchestration
# ============================================================================


def test_answer_orchestrates_ticket_graph_rag(
    retrieval_request: RetrievalRequest,
    built_context: BuiltContext,
    intelligence_evidence: list[
        GraphRAGEvidence
    ],
    grounding_result,
) -> None:
    retrieval = FakeRetrievalService(
        built_context
    )

    intelligence = (
        FakeIntelligenceAdapter(
            ticket_evidence=(
                intelligence_evidence
            )
        )
    )

    generator = FakeGenerator()

    grounding = FakeGroundednessValidator(
        grounding_result
    )

    service = make_service(
        retrieval_service=retrieval,
        intelligence_adapter=intelligence,
        generator=generator,
        grounding_validator=grounding,
    )

    request = GraphRAGRequest(
        query=retrieval_request.query,
        retrieval_request=retrieval_request,
        ticket_id="ABC-123",
    )

    response = service.answer(
        request
    )

    assert response.query == (
        retrieval_request.query
    )

    assert response.answer.startswith(
        "The ticket spent"
    )

    assert response.provider == "ollama"
    assert response.model_name == (
        "qwen2.5:7b"
    )

    assert len(
        retrieval.requests
    ) == 1

    assert retrieval.requests[0] is (
        retrieval_request
    )

    assert len(
        intelligence.ticket_calls
    ) == 1

    assert (
        intelligence.ticket_calls[0][
            "ticket_id"
        ]
        == "ABC-123"
    )

    assert len(
        generator.contexts
    ) == 1

    assert len(
        grounding.calls
    ) == 1

    assert response.grounding is (
        grounding_result
    )

    assert response.metadata[
        "retrieval_context_count"
    ] == 1

    assert response.metadata[
        "intelligence_evidence_count"
    ] == 1


# ============================================================================
# Organization routing
# ============================================================================


def test_answer_uses_query_relevant_organization_evidence_without_ticket_id(
    retrieval_request: RetrievalRequest,
    built_context: BuiltContext,
    grounding_result,
) -> None:
    organization_evidence = [
        GraphRAGEvidence(
            evidence_id="E1",
            source=(
                GraphRAGEvidenceSource
                .ORGANIZATION_INTELLIGENCE
            ),
            evidence_type=(
                GraphRAGEvidenceType.METRIC
            ),
            text=(
                "High-priority tickets represent "
                "18 percent of open tickets."
            ),
            source_reference=(
                "organization:priority"
            ),
        )
    ]

    intelligence = (
        FakeIntelligenceAdapter(
            organization_evidence=(
                organization_evidence
            )
        )
    )

    service = make_service(
        retrieval_service=(
            FakeRetrievalService(
                built_context
            )
        ),
        intelligence_adapter=intelligence,
        generator=FakeGenerator(),
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    request = GraphRAGRequest(
        query=retrieval_request.query,
        retrieval_request=retrieval_request,
    )

    service.answer(
        request
    )

    assert (
        intelligence.organization_calls
        == 0
    )

    assert (
        intelligence.relevant_organization_calls
        == [retrieval_request.query]
    )

    assert (
        len(
            intelligence.ticket_calls
        )
        == 0
    )


# ============================================================================
# Intelligence disabled
# ============================================================================


def test_answer_skips_intelligence_when_disabled(
    retrieval_request: RetrievalRequest,
    built_context: BuiltContext,
    grounding_result,
) -> None:
    intelligence = (
        FakeIntelligenceAdapter()
    )

    service = make_service(
        retrieval_service=(
            FakeRetrievalService(
                built_context
            )
        ),
        intelligence_adapter=intelligence,
        generator=FakeGenerator(),
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    request = GraphRAGRequest(
        query=retrieval_request.query,
        retrieval_request=retrieval_request,
        include_intelligence=False,
        ticket_id="ABC-123",
    )

    response = service.answer(
        request
    )

    assert (
        len(
            intelligence.ticket_calls
        )
        == 0
    )

    assert response.metadata[
        "intelligence_evidence_count"
    ] == 0


# ============================================================================
# Workflow controls
# ============================================================================


def test_answer_passes_workflow_flags_to_adapter(
    retrieval_request: RetrievalRequest,
    built_context: BuiltContext,
    grounding_result,
) -> None:
    intelligence = (
        FakeIntelligenceAdapter()
    )

    service = make_service(
        retrieval_service=(
            FakeRetrievalService(
                built_context
            )
        ),
        intelligence_adapter=intelligence,
        generator=FakeGenerator(),
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    request = GraphRAGRequest(
        query=retrieval_request.query,
        retrieval_request=retrieval_request,
        ticket_id="ABC-123",
        include_workflow=False,
    )

    service.answer(
        request
    )

    call = (
        intelligence.ticket_calls[0]
    )

    assert call[
        "include_workflow"
    ] is False

    assert call[
        "include_transitions"
    ] is False


# ============================================================================
# Query consistency
# ============================================================================


def test_answer_rejects_query_mismatch(
    retrieval_request: RetrievalRequest,
    built_context: BuiltContext,
    grounding_result,
) -> None:
    service = make_service(
        retrieval_service=(
            FakeRetrievalService(
                built_context
            )
        ),
        intelligence_adapter=(
            FakeIntelligenceAdapter()
        ),
        generator=FakeGenerator(),
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    request = GraphRAGRequest(
        query="Different question",
        retrieval_request=retrieval_request,
    )

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        service.answer(
            request
        )


# ============================================================================
# Type validation
# ============================================================================


def test_answer_rejects_invalid_request(
    built_context: BuiltContext,
    grounding_result,
) -> None:
    service = make_service(
        retrieval_service=(
            FakeRetrievalService(
                built_context
            )
        ),
        intelligence_adapter=(
            FakeIntelligenceAdapter()
        ),
        generator=FakeGenerator(),
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    with pytest.raises(
        TypeError,
        match="GraphRAGRequest",
    ):
        service.answer(
            "invalid"  # type: ignore[arg-type]
        )


# ============================================================================
# Dependency output validation
# ============================================================================


def test_answer_rejects_invalid_retrieval_output(
    retrieval_request: RetrievalRequest,
    grounding_result,
) -> None:
    class BadRetrieval:
        def retrieve(self, request):
            return "invalid"

    service = make_service(
        retrieval_service=BadRetrieval(),
        intelligence_adapter=(
            FakeIntelligenceAdapter()
        ),
        generator=FakeGenerator(),
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    request = GraphRAGRequest(
        query=retrieval_request.query,
        retrieval_request=retrieval_request,
    )

    with pytest.raises(
        TypeError,
        match="BuiltContext",
    ):
        service.answer(
            request
        )


# ============================================================================
# Grounding propagation
# ============================================================================


def test_answer_propagates_grounding_result(
    retrieval_request: RetrievalRequest,
    built_context: BuiltContext,
) -> None:
    from app.retrieval.graph_rag.graph_rag_models import (
        GroundingResult,
        GroundingStatus,
    )

    grounding_result = GroundingResult(
        status=GroundingStatus.INVALID,
        valid=False,
        cited_evidence_ids=["E99"],
        invalid_citation_ids=["E99"],
        issues=[
            "Unknown evidence citation."
        ],
    )

    service = make_service(
        retrieval_service=(
            FakeRetrievalService(
                built_context
            )
        ),
        intelligence_adapter=(
            FakeIntelligenceAdapter()
        ),
        generator=FakeGenerator(),
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    request = GraphRAGRequest(
        query=retrieval_request.query,
        retrieval_request=retrieval_request,
    )

    response = service.answer(
        request
    )

    assert response.grounding.status == (
        GroundingStatus.INVALID
    )

    assert response.grounding.valid is False


# ============================================================================
# Lifecycle
# ============================================================================


def test_close_closes_closeable_dependencies(
    built_context: BuiltContext,
    grounding_result,
) -> None:
    class CloseableRetrieval(
        FakeRetrievalService
    ):
        def __init__(
            self,
            context,
        ):
            super().__init__(
                context
            )
            self.closed = False

        def close(self):
            self.closed = True

    class CloseableIntelligence(
        FakeIntelligenceAdapter
    ):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    class CloseableGenerator(
        FakeGenerator
    ):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    retrieval = CloseableRetrieval(
        built_context
    )

    intelligence = (
        CloseableIntelligence()
    )

    generator = CloseableGenerator()

    service = make_service(
        retrieval_service=retrieval,
        intelligence_adapter=intelligence,
        generator=generator,
        grounding_validator=(
            FakeGroundednessValidator(
                grounding_result
            )
        ),
    )

    service.close()

    assert retrieval.closed is True
    assert intelligence.closed is True
    assert generator.closed is True