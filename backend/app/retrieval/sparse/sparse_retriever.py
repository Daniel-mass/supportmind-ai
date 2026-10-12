"""
app/retrieval/sparse/sparse_retriever.py

Production sparse retrieval implementation for SupportMind AI.

Responsibilities
----------------
1. Build and maintain a reusable BM25 index.
2. Execute lexical retrieval against indexed documents.
3. Accept a validated RetrievalRequest.
4. Convert sparse matches into RetrievalResult objects.
5. Preserve BM25 scores and document metadata.
6. Assign deterministic retrieval ranks.
7. Provide application factories for constructing the retriever
   from supported document sources.
8. Provide optional Qdrant-based index bootstrap.

Architecture
------------
Qdrant is used only as a document source while constructing the
BM25 index through ``from_qdrant()``.

After the BM25 index has been built, sparse retrieval itself is
completely independent of Qdrant.

Qdrant
   |
   | payload: text, node_id, document_id, metadata
   v
SparseDocument
   |
   v
BM25Okapi
   |
   v
SparseRetriever
   |
   v
RetrievalResult

Locked Technology Stack
-----------------------
Sparse Retrieval : BM25
BM25 Library     : rank-bm25
Contracts        : Pydantic retrieval models
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from rank_bm25 import BM25Okapi

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalSource,
)


# ==============================================================
# Sparse Document
# ==============================================================


@dataclass(frozen=True)
class SparseDocument:
    """
    Provider-independent document representation used by BM25.
    """

    id: str
    node_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate sparse document fields."""

        if (
            not isinstance(self.id, str)
            or not self.id.strip()
        ):
            raise ValueError(
                "SparseDocument.id must be a non-empty string."
            )

        if (
            not isinstance(self.node_id, str)
            or not self.node_id.strip()
        ):
            raise ValueError(
                "SparseDocument.node_id must be a non-empty string."
            )

        if (
            not isinstance(self.document_id, str)
            or not self.document_id.strip()
        ):
            raise ValueError(
                "SparseDocument.document_id must be a non-empty string."
            )

        if not isinstance(self.text, str):
            raise ValueError(
                "SparseDocument.text must be a string."
            )

        if not isinstance(self.metadata, dict):
            raise ValueError(
                "SparseDocument.metadata must be a dictionary."
            )


# ==============================================================
# Sparse Retriever
# ==============================================================


class SparseRetriever:
    """
    Production BM25-based sparse retriever.

    The BM25 index is built once and reused across queries.

    BM25 score sign is NOT used as the lexical relevance
    criterion. A document must share at least one query token
    with the query to be considered a lexical match.
    """

    SOURCE = RetrievalSource.SPARSE

    _WORD_PATTERN = re.compile(
        r"\b\w+\b",
        re.UNICODE,
    )

    def __init__(
        self,
        index: BM25Okapi | None = None,
        documents: Sequence[SparseDocument] | None = None,
    ) -> None:
        """
        Initialize the sparse retriever.

        If documents are supplied without an index, the BM25
        index is built automatically.

        If an index is supplied, documents must also be supplied
        so result positions can be mapped back to documents.
        """

        if (
            index is not None
            and documents is None
        ):
            raise ValueError(
                "Documents are required when providing "
                "a pre-built BM25 index."
            )

        self._index: BM25Okapi | None = None

        self._documents: tuple[
            SparseDocument,
            ...
        ] = ()

        if documents is not None:
            self.build_index(
                documents
            )

        if index is not None:
            self._index = index

    # ==========================================================
    # Qdrant Factory
    # ==========================================================

    @classmethod
    def from_qdrant(
        cls,
        *,
        qdrant_store: Any,
        collection_name: str,
        batch_size: int = 100,
    ) -> "SparseRetriever":
        """
        Build a BM25 sparse retriever from a Qdrant collection.

        Qdrant is used only during index construction.

        The Qdrant collection must contain payload fields:

            text
            node_id
            document_id

        ``metadata`` is optional.

        Vector data is not loaded.

        After construction, the returned SparseRetriever performs
        retrieval entirely from its in-memory BM25 index.
        """

        documents = (
            _load_sparse_documents_from_qdrant(
                qdrant_store=qdrant_store,
                collection_name=collection_name,
                batch_size=batch_size,
            )
        )

        return cls(
            documents=documents
        )

    # ==========================================================
    # Index Management
    # ==========================================================

    @property
    def is_indexed(self) -> bool:
        """
        Return True when a BM25 index exists.
        """

        return self._index is not None

    @property
    def document_count(self) -> int:
        """
        Return the number of indexed documents.
        """

        return len(
            self._documents
        )

    def build_index(
        self,
        documents: Sequence[SparseDocument],
    ) -> None:
        """
        Build or replace the BM25 index.
        """

        validated_documents = (
            self._validate_documents(
                documents
            )
        )

        if not validated_documents:
            raise ValueError(
                "Cannot build BM25 index from "
                "an empty document collection."
            )

        tokenized_documents = [
            self._tokenize(
                document.text
            )
            for document in validated_documents
        ]

        if any(
            not tokens
            for tokens in tokenized_documents
        ):
            raise ValueError(
                "Indexed documents must contain "
                "at least one token."
            )

        self._index = BM25Okapi(
            tokenized_documents
        )

        self._documents = tuple(
            validated_documents
        )

    # ==========================================================
    # Retrieval
    # ==========================================================

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        """
        Execute BM25 sparse retrieval.
        """

        self._validate_request(
            request
        )

        if not self.is_indexed:
            raise ValueError(
                "BM25 index has not been built."
            )

        query_tokens = self._tokenize(
            request.query
        )

        if not query_tokens:
            return []

        scores = self._index.get_scores(
            query_tokens
        )

        # ------------------------------------------------------
        # Determine lexical overlap independently from BM25
        # score.
        #
        # This prevents negative/zero BM25 scores from being
        # interpreted as evidence that a document is irrelevant
        # when it actually shares query tokens.
        # ------------------------------------------------------

        query_token_set = set(
            query_tokens
        )

        matched_indices = [
            index
            for index, document in enumerate(
                self._documents
            )
            if query_token_set.intersection(
                self._tokenize(
                    document.text
                )
            )
        ]

        if not matched_indices:
            return []

        # ------------------------------------------------------
        # Rank by BM25 score.
        #
        # Original document index is used as a deterministic
        # tie-breaker.
        # ------------------------------------------------------

        ranked_indices = sorted(
            matched_indices,
            key=lambda index: (
                -float(
                    scores[index]
                ),
                index,
            ),
        )

        results: list[
            RetrievalResult
        ] = []

        for index in ranked_indices:

            score = float(
                scores[index]
            )

            document = self._documents[
                index
            ]

            results.append(
                self._convert_result(
                    document=document,
                    score=score,
                    rank=len(results) + 1,
                )
            )

            if len(results) >= request.top_k:
                break

        return results

    # ==========================================================
    # Result Conversion
    # ==========================================================

    @classmethod
    def _convert_result(
        cls,
        document: SparseDocument,
        score: float,
        rank: int,
    ) -> RetrievalResult:
        """
        Convert SparseDocument into RetrievalResult.
        """

        return RetrievalResult(
            id=document.id,
            node_id=document.node_id,
            document_id=document.document_id,
            text=document.text,
            score=score,
            rank=rank,
            source=cls.SOURCE,
            metadata=dict(
                document.metadata
            ),
            retrieval_scores={
                "bm25_score": score,
            },
        )

    # ==========================================================
    # Tokenization
    # ==========================================================

    @classmethod
    def _tokenize(
        cls,
        text: str,
    ) -> list[str]:
        """
        Deterministic case-insensitive tokenization.
        """

        if not isinstance(
            text,
            str,
        ):
            raise ValueError(
                "Text must be a string."
            )

        return [
            token.lower()
            for token in cls._WORD_PATTERN.findall(
                text
            )
        ]

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_request(
        request: RetrievalRequest,
    ) -> None:
        """
        Validate retrieval request.
        """

        if request is None:
            raise ValueError(
                "Retrieval request cannot be None."
            )

        if not isinstance(
            request,
            RetrievalRequest,
        ):
            raise ValueError(
                "request must be a RetrievalRequest instance."
            )

    @staticmethod
    def _validate_documents(
        documents: Sequence[SparseDocument],
    ) -> list[SparseDocument]:
        """
        Validate sparse documents.
        """

        if documents is None:
            raise ValueError(
                "Documents cannot be None."
            )

        if isinstance(
            documents,
            (str, bytes),
        ):
            raise ValueError(
                "Documents must be a sequence of SparseDocument objects."
            )

        try:
            document_list = list(
                documents
            )
        except TypeError as exc:
            raise ValueError(
                "Documents must be an iterable collection."
            ) from exc

        for document in document_list:

            if not isinstance(
                document,
                SparseDocument,
            ):
                raise ValueError(
                    "All indexed documents must be "
                    "SparseDocument instances."
                )

        return document_list


# ==============================================================
# Qdrant Document Loading
# ==============================================================


def _load_sparse_documents_from_qdrant(
    *,
    qdrant_store: Any,
    collection_name: str,
    batch_size: int = 100,
) -> list[SparseDocument]:
    """
    Load sparse documents from a Qdrant collection.

    Only payload data required by BM25 is loaded.

    Required payload fields
    -----------------------
    text
    node_id
    document_id

    Optional payload field
    ----------------------
    metadata

    Vector data is never loaded.
    """

    if qdrant_store is None:
        raise ValueError(
            "qdrant_store cannot be None."
        )

    if (
        not isinstance(
            collection_name,
            str,
        )
        or not collection_name.strip()
    ):
        raise ValueError(
            "collection_name must be a non-empty string."
        )

    if (
        not isinstance(
            batch_size,
            int,
        )
        or isinstance(
            batch_size,
            bool,
        )
        or batch_size <= 0
    ):
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if not hasattr(
        qdrant_store,
        "client",
    ):
        raise ValueError(
            "qdrant_store must expose a Qdrant client."
        )

    client = qdrant_store.client

    records: list[Any] = []

    offset = None

    while True:

        batch, next_offset = (
            client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        )

        records.extend(
            batch
        )

        if next_offset is None:
            break

        offset = next_offset

    if not records:
        raise ValueError(
            f"No documents found in Qdrant collection "
            f"'{collection_name}'."
        )

    sparse_documents: list[
        SparseDocument
    ] = []

    for record in records:

        payload = (
            getattr(
                record,
                "payload",
                None,
            )
            or {}
        )

        if not isinstance(
            payload,
            dict,
        ):
            continue

        text = payload.get(
            "text"
        )

        node_id = payload.get(
            "node_id"
        )

        document_id = payload.get(
            "document_id"
        )

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            continue

        if (
            not isinstance(
                node_id,
                str,
            )
            or not node_id.strip()
        ):
            continue

        if (
            not isinstance(
                document_id,
                str,
            )
            or not document_id.strip()
        ):
            continue

        metadata = payload.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        record_id = getattr(
            record,
            "id",
            None,
        )

        if record_id is None:
            continue

        sparse_documents.append(
            SparseDocument(
                id=str(
                    record_id
                ),
                node_id=node_id,
                document_id=document_id,
                text=text,
                metadata=dict(
                    metadata
                ),
            )
        )

    if not sparse_documents:
        raise ValueError(
            f"No usable documents could be created "
            f"from Qdrant collection "
            f"'{collection_name}'."
        )

    return sparse_documents


# ==============================================================
# Application Factories
# ==============================================================


def create_sparse_retriever(
    documents: Sequence[SparseDocument],
) -> SparseRetriever:
    """
    Create and initialize a sparse retriever from documents.
    """

    return SparseRetriever(
        documents=documents
    )


def create_sparse_retriever_from_qdrant(
    *,
    qdrant_store: Any,
    collection_name: str,
    batch_size: int = 100,
) -> SparseRetriever:
    """
    Create and initialize a sparse retriever from Qdrant.

    Qdrant is used only during index construction.

    Once the BM25 index has been created, retrieval itself
    is completely independent of Qdrant.
    """

    return SparseRetriever.from_qdrant(
        qdrant_store=qdrant_store,
        collection_name=collection_name,
        batch_size=batch_size,
    )