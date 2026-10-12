"""
Standalone tests for QueryRouter.

No Qdrant.
No Neo4j.
No embeddings.
No BM25.
No external services.

SupportMind AI
"""

import pytest

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalSource,
)
from app.retrieval.query.query_router import QueryRouter


# ==============================================================
# Fixtures
# ==============================================================


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


def make_request(
    query: str,
    include_graph: bool = False,
) -> RetrievalRequest:
    return RetrievalRequest(
        query=query,
        collection_name="supportmind_documents",
        top_k=5,
        include_graph=include_graph,
    )


# ==============================================================
# Default routing
# ==============================================================


def test_default_route_is_hybrid(router):
    request = make_request(
        "What is the refund policy?"
    )

    result = router.route(request)

    assert result == RetrievalSource.HYBRID


def test_normal_natural_language_query_uses_hybrid(router):
    request = make_request(
        "How can a customer request a refund?"
    )

    result = router.route(request)

    assert result == RetrievalSource.HYBRID


# ==============================================================
# Graph routing
# ==============================================================


def test_explicit_graph_request_routes_to_graph(router):
    request = make_request(
        "Show relationships between payment failure and refunds",
        include_graph=True,
    )

    result = router.route(request)

    assert result == RetrievalSource.GRAPH


@pytest.mark.parametrize(
    "query",
    [
        "What is related to payment failure?",
        "What is connected to account suspension?",
        "What are the relationships between refund and payment?",
        "Which policies are associated with cancellation?",
        "What depends on the payment service?",
        "Show dependencies between services",
    ],
)
def test_relationship_queries_route_to_graph(
    router,
    query,
):
    request = make_request(query)

    result = router.route(request)

    assert result == RetrievalSource.GRAPH


# ==============================================================
# Sparse routing
# ==============================================================


@pytest.mark.parametrize(
    "query",
    [
        "Find ticket INC-1024",
        "Search error code E102",
        "Find incident 4521",
        "Find document ID DOC-100",
        "Search case CASE-22",
    ],
)
def test_keyword_queries_route_to_sparse(
    router,
    query,
):
    request = make_request(query)

    result = router.route(request)

    assert result == RetrievalSource.SPARSE


# ==============================================================
# Routing priority
# ==============================================================


def test_graph_has_priority_over_sparse(router):
    request = make_request(
        "What error codes are related to payment failure?"
    )

    result = router.route(request)

    assert result == RetrievalSource.GRAPH


def test_explicit_graph_flag_has_priority(router):
    request = make_request(
        "Find error code E102",
        include_graph=True,
    )

    result = router.route(request)

    assert result == RetrievalSource.GRAPH


# ==============================================================
# Input validation
# ==============================================================


def test_route_rejects_invalid_request(router):
    with pytest.raises(ValueError):
        router.route("refund policy")


def test_route_query_creates_valid_request(router):
    result = router.route_query(
        query="What is the refund policy?",
        collection_name="supportmind_documents",
    )

    assert result == RetrievalSource.HYBRID


def test_route_query_rejects_invalid_query():
    router = QueryRouter()

    with pytest.raises(Exception):
        router.route_query(
            query="",
            collection_name="supportmind_documents",
        )


# ==============================================================
# Normalization
# ==============================================================


def test_normalize_query_lowercases_text():
    result = QueryRouter._normalize_query(
        "PAYMENT FAILURE"
    )

    assert result == "payment failure"


def test_normalize_query_removes_punctuation():
    result = QueryRouter._normalize_query(
        "What is related to payment?"
    )

    assert result == "what is related to payment"


def test_normalize_query_collapses_whitespace():
    result = QueryRouter._normalize_query(
        "payment     failure"
    )

    assert result == "payment failure"


# ==============================================================
# Custom default
# ==============================================================


def test_custom_default_strategy():
    router = QueryRouter(
        default_strategy=RetrievalSource.DENSE
    )

    request = make_request(
        "Tell me about refunds"
    )

    result = router.route(request)

    assert result == RetrievalSource.DENSE


def test_reranked_cannot_be_primary_route():
    with pytest.raises(ValueError):
        QueryRouter(
            default_strategy=RetrievalSource.RERANKED
        )