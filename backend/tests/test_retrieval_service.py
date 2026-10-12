from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.retrieval.context.context_builder import ContextBuilder

from app.retrieval.context.context_models import (
    BuiltContext,
    ContextItem,
    ContextRequest,
    ContextSource,
)
from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)


# ============================================================================
# Helpers
# ============================================================================


def make_item(
    item_id: str,
    text: str,
    *,
    source: ContextSource = ContextSource.DENSE,
    score: float | None = 0.8,
    rank: int = 1,
    metadata: dict | None = None,
) -> ContextItem:
    return ContextItem(
        id=item_id,
        text=text,
        source=source,
        score=score,
        rank=rank,
        metadata=metadata or {},
    )


def make_request(
    query: str = "customer cannot login",
    *,
    results: list[ContextItem] | None = None,
) -> ContextRequest:
    return ContextRequest(
        query=query,
        results=results or [],
    )


def make_built_context(
    query: str = "customer cannot login",
    items: list[ContextItem] | None = None,
) -> BuiltContext:
    return BuiltContext(
        query=query,
        items=items or [],
        sections=[],
        total_candidates=len(items or []),
        selected_count=len(items or []),
        character_count=sum(len(item.text) for item in (items or [])),
        truncated=False,
    )


# ============================================================================
# Configuration tests
# ============================================================================


def test_config_uses_expected_defaults():
    config = RetrievalServiceConfig()

    assert config.use_dense is True
    assert config.use_bm25 is True
    assert config.use_graph is True
    assert config.use_reranker is True
    assert config.default_top_k == 5


def test_config_accepts_custom_values():
    config = RetrievalServiceConfig(
        use_dense=False,
        use_bm25=True,
        use_graph=False,
        use_reranker=False,
        default_top_k=10,
    )

    assert config.use_dense is False
    assert config.use_bm25 is True
    assert config.use_graph is False
    assert config.use_reranker is False
    assert config.default_top_k == 10


def test_config_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        RetrievalServiceConfig(default_top_k=0)

    with pytest.raises(ValueError):
        RetrievalServiceConfig(default_top_k=-1)


def test_config_rejects_boolean_top_k():
    with pytest.raises(TypeError):
        RetrievalServiceConfig(default_top_k=True)


def test_config_rejects_non_integer_top_k():
    with pytest.raises(TypeError):
        RetrievalServiceConfig(default_top_k=5.5)


# ============================================================================
# Service construction
# ============================================================================


def test_service_can_be_created_without_dependencies():
    service = RetrievalService()

    assert service.dense_retriever is None
    assert service.bm25_retriever is None
    assert service.graph_retriever is None
    assert service.reranker is None
    assert service.context_builder is not None


def test_service_accepts_dependency_injection():
    dense = Mock()
    bm25 = Mock()
    graph = Mock()
    reranker = Mock()
    builder = Mock()

    service = RetrievalService(
        dense_retriever=dense,
        bm25_retriever=bm25,
        graph_retriever=graph,
        reranker=reranker,
        context_builder=builder,
    )

    assert service.dense_retriever is dense
    assert service.bm25_retriever is bm25
    assert service.graph_retriever is graph
    assert service.reranker is reranker
    assert service.context_builder is builder


# ============================================================================
# Request validation
# ============================================================================


def test_retrieve_rejects_none_request():
    service = RetrievalService()

    with pytest.raises(ValueError):
        service.retrieve(None)


def test_retrieve_rejects_invalid_request_type():
    service = RetrievalService()

    with pytest.raises(TypeError):
        service.retrieve("customer cannot login")


from pydantic import ValidationError

def test_retrieve_rejects_empty_query():
    service = RetrievalService()

    with pytest.raises(ValidationError):
        make_request("   ")


# ============================================================================
# Enabled retriever diagnostics
# ============================================================================


def test_enabled_retrievers_returns_available_retrievers():
    service = RetrievalService(
        dense_retriever=Mock(),
        bm25_retriever=Mock(),
        graph_retriever=Mock(),
    )

    assert service.enabled_retrievers() == (
        "dense",
        "bm25",
        "graph",
    )


def test_enabled_retrievers_excludes_disabled_retrievers():
    config = RetrievalServiceConfig(
        use_dense=False,
        use_bm25=True,
        use_graph=False,
    )

    service = RetrievalService(
        dense_retriever=Mock(),
        bm25_retriever=Mock(),
        graph_retriever=Mock(),
        config=config,
    )

    assert service.enabled_retrievers() == ("bm25",)


def test_enabled_retrievers_excludes_missing_dependencies():
    service = RetrievalService(
        dense_retriever=Mock(),
        bm25_retriever=None,
        graph_retriever=None,
    )

    assert service.enabled_retrievers() == ("dense",)


def test_is_reranking_enabled_when_configured():
    service = RetrievalService(
        reranker=Mock(),
    )

    assert service.is_reranking_enabled() is True


def test_is_reranking_disabled_without_reranker():
    service = RetrievalService(
        reranker=None,
    )

    assert service.is_reranking_enabled() is False


def test_is_reranking_disabled_by_configuration():
    config = RetrievalServiceConfig(
        use_reranker=False,
    )

    service = RetrievalService(
        reranker=Mock(),
        config=config,
    )

    assert service.is_reranking_enabled() is False


# ============================================================================
# Individual retriever execution
# ============================================================================


def test_dense_retriever_is_called():
    dense = Mock(
        return_value=[
            make_item(
                "dense-1",
                "Login troubleshooting",
                source=ContextSource.DENSE,
            )
        ]
    )

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        bm25_retriever=None,
        graph_retriever=None,
        reranker=None,
        context_builder=builder,
    )

    service.retrieve(make_request())

    dense.assert_called_once()


def test_bm25_retriever_is_called():
    bm25 = Mock(
        return_value=[
            make_item(
                "bm25-1",
                "Login troubleshooting",
                source=ContextSource.BM25,
            )
        ]
    )

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=None,
        bm25_retriever=bm25,
        graph_retriever=None,
        reranker=None,
        context_builder=builder,
    )

    service.retrieve(make_request())

    bm25.assert_called_once()


def test_graph_retriever_is_called():
    graph = Mock(
        return_value=[
            make_item(
                "graph-1",
                "Customer linked to failed login ticket",
                source=ContextSource.GRAPH,
            )
        ]
    )

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=None,
        bm25_retriever=None,
        graph_retriever=graph,
        reranker=None,
        context_builder=builder,
    )

    service.retrieve(make_request())

    graph.assert_called_once()


# ============================================================================
# Disabled retrievers
# ============================================================================


def test_disabled_dense_retriever_is_not_called():
    config = RetrievalServiceConfig(
        use_dense=False,
        use_bm25=False,
        use_graph=False,
    )

    dense = Mock()
    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        context_builder=builder,
        config=config,
    )

    service.retrieve(make_request())

    dense.assert_not_called()


def test_disabled_bm25_retriever_is_not_called():
    config = RetrievalServiceConfig(
        use_dense=False,
        use_bm25=False,
        use_graph=False,
    )

    bm25 = Mock()
    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        bm25_retriever=bm25,
        context_builder=builder,
        config=config,
    )

    service.retrieve(make_request())

    bm25.assert_not_called()


def test_disabled_graph_retriever_is_not_called():
    config = RetrievalServiceConfig(
        use_dense=False,
        use_bm25=False,
        use_graph=False,
    )

    graph = Mock()
    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        graph_retriever=graph,
        context_builder=builder,
        config=config,
    )

    service.retrieve(make_request())

    graph.assert_not_called()


# ============================================================================
# Candidate collection
# ============================================================================


def test_service_combines_multiple_retriever_results():
    dense_item = make_item(
        "dense-1",
        "Dense result",
        source=ContextSource.DENSE,
    )

    bm25_item = make_item(
        "bm25-1",
        "BM25 result",
        source=ContextSource.BM25,
    )

    graph_item = make_item(
        "graph-1",
        "Graph result",
        source=ContextSource.GRAPH,
    )

    dense = Mock(return_value=[dense_item])
    bm25 = Mock(return_value=[bm25_item])
    graph = Mock(return_value=[graph_item])

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        bm25_retriever=bm25,
        graph_retriever=graph,
        reranker=None,
        context_builder=builder,
    )

    service.retrieve(make_request())

    context_request = builder.build.call_args.args[0]

    ids = [item.id for item in context_request.results]

    assert ids == [
        "dense-1",
        "bm25-1",
        "graph-1",
    ]


def test_service_deduplicates_same_document():
    dense_item = make_item(
        "doc-1",
        "Same document",
        source=ContextSource.DENSE,
    )

    bm25_item = make_item(
        "doc-1",
        "Same document",
        source=ContextSource.BM25,
    )

    dense = Mock(return_value=[dense_item])
    bm25 = Mock(return_value=[bm25_item])

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        bm25_retriever=bm25,
        context_builder=builder,
    )

    service.retrieve(make_request())

    context_request = builder.build.call_args.args[0]

    assert len(context_request.results) == 1
    assert context_request.results[0].id == "doc-1"


def test_service_keeps_different_documents():
    dense = Mock(
        return_value=[
            make_item("doc-1", "First"),
            make_item("doc-2", "Second"),
        ]
    )

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        context_builder=builder,
    )

    service.retrieve(make_request())

    context_request = builder.build.call_args.args[0]

    assert len(context_request.results) == 2


# ============================================================================
# Empty retrieval
# ============================================================================


def test_empty_retrieval_produces_valid_context():
    dense = Mock(return_value=[])
    bm25 = Mock(return_value=[])
    graph = Mock(return_value=[])

    builder = Mock()
    expected = make_built_context()
    builder.build.return_value = expected

    service = RetrievalService(
        dense_retriever=dense,
        bm25_retriever=bm25,
        graph_retriever=graph,
        context_builder=builder,
    )

    result = service.retrieve(make_request())

    assert result is expected

    context_request = builder.build.call_args.args[0]

    assert context_request.results == []


def test_none_retriever_result_is_treated_as_empty():
    dense = Mock(return_value=None)

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        context_builder=builder,
    )

    service.retrieve(make_request())

    context_request = builder.build.call_args.args[0]

    assert context_request.results == []


def test_string_retriever_result_is_rejected():
    dense = Mock(return_value="invalid result")

    builder = Mock()

    service = RetrievalService(
        dense_retriever=dense,
        context_builder=builder,
    )

    with pytest.raises(TypeError):
        service.retrieve(make_request())


# ============================================================================
# Reranking
# ============================================================================


def test_reranker_receives_combined_candidates():
    dense_item = make_item(
        "dense-1",
        "Dense",
        source=ContextSource.DENSE,
    )

    graph_item = make_item(
        "graph-1",
        "Graph",
        source=ContextSource.GRAPH,
    )

    dense = Mock(return_value=[dense_item])
    graph = Mock(return_value=[graph_item])

    reranker = Mock(
        return_value=[
            graph_item,
            dense_item,
        ]
    )

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        graph_retriever=graph,
        reranker=reranker,
        context_builder=builder,
    )

    request = make_request()

    service.retrieve(request)

    reranker.assert_called_once()

    args = reranker.call_args.args

    assert args[0] == request.query
    assert len(args[1]) == 2


def test_reranker_can_be_disabled():
    config = RetrievalServiceConfig(
        use_reranker=False,
    )

    dense = Mock(
        return_value=[
            make_item("doc-1", "Result"),
        ]
    )

    reranker = Mock()

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        reranker=reranker,
        context_builder=builder,
        config=config,
    )

    service.retrieve(make_request())

    reranker.assert_not_called()


def test_reranker_is_not_called_when_no_candidates():
    dense = Mock(return_value=[])
    reranker = Mock()

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        reranker=reranker,
        context_builder=builder,
    )

    service.retrieve(make_request())

    reranker.assert_not_called()


def test_object_reranker_interface_is_supported():
    dense_item = make_item(
        "doc-1",
        "Result",
    )

    dense = Mock(return_value=[dense_item])

    reranker = Mock()
    reranker.rerank.return_value = [dense_item]

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        reranker=reranker,
        context_builder=builder,
    )

    service.retrieve(make_request())

    reranker.rerank.assert_called_once()


def test_reranker_none_result_becomes_empty():
    dense = Mock(
        return_value=[
            make_item("doc-1", "Result"),
        ]
    )

    reranker = Mock(return_value=None)

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        reranker=reranker,
        context_builder=builder,
    )

    service.retrieve(make_request())

    context_request = builder.build.call_args.args[0]

    assert context_request.results == []


# ============================================================================
# Context Builder integration
# ============================================================================


def test_context_builder_is_called():
    dense = Mock(
        return_value=[
            make_item("doc-1", "Result"),
        ]
    )

    builder = Mock()
    expected = make_built_context()
    builder.build.return_value = expected

    service = RetrievalService(
        dense_retriever=dense,
        context_builder=builder,
    )

    result = service.retrieve(make_request())

    builder.build.assert_called_once()

    assert result is expected


def test_context_builder_receives_original_query():
    dense = Mock(
        return_value=[
            make_item("doc-1", "Result"),
        ]
    )

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        context_builder=builder,
    )

    request = make_request(
        "why did my payment fail?"
    )

    service.retrieve(request)

    context_request = builder.build.call_args.args[0]

    assert context_request.query == "why did my payment fail?"


def test_context_builder_receives_fused_results():
    dense = Mock(
        return_value=[
            make_item("dense-1", "Dense"),
        ]
    )

    graph = Mock(
        return_value=[
            make_item(
                "graph-1",
                "Graph",
                source=ContextSource.GRAPH,
            )
        ]
    )

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        graph_retriever=graph,
        context_builder=builder,
    )

    service.retrieve(make_request())

    context_request = builder.build.call_args.args[0]

    assert len(context_request.results) == 2
    assert context_request.results[0].id == "dense-1"
    assert context_request.results[1].id == "graph-1"


# ============================================================================
# Metadata / provenance
# ============================================================================


def test_service_preserves_retrieval_metadata():
    item = make_item(
        "doc-1",
        "Payment troubleshooting",
        source=ContextSource.GRAPH,
        score=0.91,
        metadata={
            "product": "Premium Subscription",
            "ticket_id": "T-100",
            "graph_hops": 2,
        },
    )

    graph = Mock(return_value=[item])

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        graph_retriever=graph,
        context_builder=builder,
    )

    service.retrieve(make_request())

    context_request = builder.build.call_args.args[0]

    result = context_request.results[0]

    assert result.metadata["product"] == "Premium Subscription"
    assert result.metadata["ticket_id"] == "T-100"
    assert result.metadata["graph_hops"] == 2


def test_graph_source_is_preserved():
    item = make_item(
        "graph-1",
        "Graph evidence",
        source=ContextSource.GRAPH,
    )

    graph = Mock(return_value=[item])

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        graph_retriever=graph,
        context_builder=builder,
    )

    service.retrieve(make_request())

    context_request = builder.build.call_args.args[0]

    assert context_request.results[0].source == ContextSource.GRAPH


# ============================================================================
# Determinism
# ============================================================================


def test_candidate_order_is_deterministic():
    dense = Mock(
        return_value=[
            make_item("doc-1", "One"),
            make_item("doc-2", "Two"),
        ]
    )

    bm25 = Mock(
        return_value=[
            make_item("doc-3", "Three"),
        ]
    )

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        bm25_retriever=bm25,
        context_builder=builder,
    )

    service.retrieve(make_request())

    first_request = builder.build.call_args.args[0]

    builder.reset_mock()

    service.retrieve(make_request())

    second_request = builder.build.call_args.args[0]

    first_ids = [item.id for item in first_request.results]
    second_ids = [item.id for item in second_request.results]

    assert first_ids == second_ids


# ============================================================================
# Non-mutation
# ============================================================================


def test_service_does_not_mutate_original_results():
    item = make_item(
        "doc-1",
        "Original",
        metadata={
            "category": "billing",
        },
    )

    original = deepcopy(item)

    dense = Mock(return_value=[item])

    builder = Mock()
    builder.build.return_value = make_built_context()

    service = RetrievalService(
        dense_retriever=dense,
        context_builder=builder,
    )

    service.retrieve(make_request())

    assert item.id == original.id
    assert item.text == original.text
    assert item.metadata == original.metadata
    assert item.source == original.source
    assert item.score == original.score
    assert item.rank == original.rank


# ============================================================================
# Lifecycle
# ============================================================================


def test_close_closes_supported_components():
    dense = Mock()
    bm25 = Mock()
    graph = Mock()
    reranker = Mock()
    builder = Mock()

    service = RetrievalService(
        dense_retriever=dense,
        bm25_retriever=bm25,
        graph_retriever=graph,
        reranker=reranker,
        context_builder=builder,
    )

    service.close()

    dense.close.assert_called_once()
    bm25.close.assert_called_once()
    graph.close.assert_called_once()
    reranker.close.assert_called_once()
    builder.close.assert_called_once()


def test_close_does_not_fail_for_components_without_close():
    class SimpleComponent:
        pass

    service = RetrievalService(
        dense_retriever=SimpleComponent(),
        bm25_retriever=SimpleComponent(),
        graph_retriever=SimpleComponent(),
        reranker=SimpleComponent(),
        context_builder=SimpleComponent(),
    )

    service.close()


def test_context_manager_closes_components():
    dense = Mock()

    builder = Mock()
    builder.build.return_value = make_built_context()

    with RetrievalService(
        dense_retriever=dense,
        context_builder=builder,
    ) as service:
        assert service is not None

    dense.close.assert_called_once()
    builder.close.assert_called_once()