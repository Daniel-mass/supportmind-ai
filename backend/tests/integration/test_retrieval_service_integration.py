from unittest.mock import Mock

from app.retrieval.context.context_builder import ContextBuilder
from app.retrieval.context.context_models import (
    ContextItem,
    ContextRequest,
    ContextSource,
)
from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)


def make_item(
    item_id: str,
    text: str,
    source: ContextSource,
    rank: int,
    score: float,
    document_id: str,
):
    return ContextItem(
        id=item_id,
        text=text,
        source=source,
        rank=rank,
        score=score,
        metadata={
            "document_id": document_id,
        },
    )


def test_full_retrieval_service_pipeline():

    # ---------------------------------------------------------
    # Dense results
    # ---------------------------------------------------------

    dense_results = [
        make_item(
            item_id="doc-1",
            text="Password reset requires email verification.",
            source=ContextSource.DENSE,
            rank=1,
            score=0.91,
            document_id="doc-1",
        ),
        make_item(
            item_id="doc-2",
            text="Users can update their account email.",
            source=ContextSource.DENSE,
            rank=2,
            score=0.84,
            document_id="doc-2",
        ),
    ]

    # ---------------------------------------------------------
    # BM25 results
    # ---------------------------------------------------------

    bm25_results = [
        make_item(
            item_id="doc-1",
            text="Password reset requires email verification.",
            source=ContextSource.BM25,
            rank=1,
            score=8.4,
            document_id="doc-1",
        ),
        make_item(
            item_id="doc-3",
            text="Account recovery requests may require verification.",
            source=ContextSource.BM25,
            rank=2,
            score=7.2,
            document_id="doc-3",
        ),
    ]

    # ---------------------------------------------------------
    # Graph results
    # ---------------------------------------------------------

    graph_results = [
        make_item(
            item_id="doc-4",
            text="Password reset is associated with account recovery.",
            source=ContextSource.GRAPH,
            rank=1,
            score=2.1,
            document_id="doc-4",
        )
    ]

    # ---------------------------------------------------------
    # Mock retrievers
    #
    # We are NOT testing the retrievers themselves here.
    # Their standalone tests already exist.
    # ---------------------------------------------------------

    dense_retriever = Mock(
        return_value=dense_results
    )

    bm25_retriever = Mock(
        return_value=bm25_results
    )

    graph_retriever = Mock(
        return_value=graph_results
    )

    # ---------------------------------------------------------
    # Mock reranker
    #
    # Return candidates with reranked source.
    # ---------------------------------------------------------

    reranker = Mock()

    reranker.return_value = [
        make_item(
            item_id="doc-1",
            text="Password reset requires email verification.",
            source=ContextSource.RERANKED,
            rank=1,
            score=0.98,
            document_id="doc-1",
        ),
        make_item(
            item_id="doc-4",
            text="Password reset is associated with account recovery.",
            source=ContextSource.RERANKED,
            rank=2,
            score=0.87,
            document_id="doc-4",
        ),
        make_item(
            item_id="doc-3",
            text="Account recovery requests may require verification.",
            source=ContextSource.RERANKED,
            rank=3,
            score=0.79,
            document_id="doc-3",
        ),
    ]

    # ---------------------------------------------------------
    # REAL ContextBuilder
    # ---------------------------------------------------------

    context_builder = ContextBuilder()

    # ---------------------------------------------------------
    # REAL RetrievalService
    # ---------------------------------------------------------

    service = RetrievalService(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
        context_builder=context_builder,
        config=RetrievalServiceConfig(
            use_dense=True,
            use_bm25=True,
            use_graph=True,
            use_reranker=True,
            default_top_k=5,
        ),
    )

    # ---------------------------------------------------------
    # Real request
    # ---------------------------------------------------------

    request = ContextRequest(
        query="How do I reset my password?",
    )

    # ---------------------------------------------------------
    # Execute complete service
    # ---------------------------------------------------------

    result = service.retrieve(request)

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert result.query == "How do I reset my password?"

    assert result.total_candidates == 3

    assert result.selected_count > 0

    assert result.selected_count <= 8

    assert len(result.selected_items) == result.selected_count

    assert len(result.sections) >= 2

    assert result.sections[0].title == "User Query"

    assert result.sections[1].title == "Evidence"

    # Context should contain actual evidence.
    assert "Password reset" in result.sections[1].content

    # ---------------------------------------------------------
    # Verify all retrieval branches were called
    # ---------------------------------------------------------

    dense_retriever.assert_called_once_with(request)

    bm25_retriever.assert_called_once_with(request)

    graph_retriever.assert_called_once_with(request)

    # ---------------------------------------------------------
    # Verify reranking received the query
    # ---------------------------------------------------------

    reranker.assert_called_once()

    rerank_args = reranker.call_args

    assert rerank_args.args[0] == request.query

    reranked_candidates = rerank_args.args[1]

    assert len(reranked_candidates) == 4

    # ---------------------------------------------------------
    # Verify deduplication
    #
    # doc-1 appeared in both dense and BM25.
    # Therefore only 3 unique candidates should reach reranking.
    # ---------------------------------------------------------

    candidate_ids = {
        candidate.id
        for candidate in reranked_candidates
    }

    assert candidate_ids == {
        "doc-1",
        "doc-2",
        "doc-3",
        "doc-4",
    }