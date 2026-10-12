"""
Real BM25 retrieval test.

Uses:
    - Actual refund_policy.pdf
    - Actual pypdf extraction
    - Actual rank-bm25
    - Actual SparseRetriever
    - Actual RetrievalRequest

No mocks.
No synthetic documents.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pypdf import PdfReader

from app.retrieval.models.retrieval_models import RetrievalRequest
from app.retrieval.sparse.sparse_retriever import (
    SparseDocument,
    SparseRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PDF_PATH = Path(
    os.getenv(
        "REAL_TEST_PDF",
        "enterprise_knowledge/policies/refund_policy.pdf",
    )
)

if not PDF_PATH.is_absolute():
    PDF_PATH = PROJECT_ROOT / PDF_PATH


def _load_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from the real enterprise PDF.
    """

    if not pdf_path.exists():
        pytest.fail(
            f"Real test PDF does not exist: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))

    pages: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            pages.append(text)

    text = "\n".join(pages).strip()

    if not text:
        pytest.fail(
            f"No extractable text found in PDF: {pdf_path}"
        )

    return text


def _build_real_sparse_documents(
    pdf_text: str,
) -> list[SparseDocument]:
    """
    Convert the real PDF into paragraph-level sparse documents.

    We deliberately use the real PDF content rather than creating
    fake test documents.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in pdf_text.split("\n\n")
        if paragraph.strip()
    ]

    documents: list[SparseDocument] = []

    for index, paragraph in enumerate(
        paragraphs
    ):
        documents.append(
            SparseDocument(
                id=f"refund_policy_real_{index}",
                node_id=f"refund_policy_real_{index}",
                document_id="refund_policy.pdf",
                text=paragraph,
                metadata={
                    "source": "real_test",
                    "file": "refund_policy.pdf",
                    "path": str(PDF_PATH),
                    "paragraph_index": index,
                },
            )
        )

    if not documents:
        pytest.fail(
            "PDF produced zero usable paragraphs."
        )

    return documents


def test_real_bm25_refund_policy_retrieval() -> None:
    """
    Verify real BM25 retrieval against refund_policy.pdf.
    """

    pdf_text = _load_pdf_text(
        PDF_PATH
    )

    documents = _build_real_sparse_documents(
        pdf_text
    )

    retriever = SparseRetriever(
        documents=documents
    )

    assert retriever.is_indexed
    assert retriever.document_count == len(
        documents
    )

    request = RetrievalRequest.model_validate(
        {
            "query": "refund eligibility refund policy",
            "collection_name": "real_bm25_test",
            "top_k": 5,
        }
    )

    results = retriever.retrieve(
        request
    )

    assert results, (
        "BM25 returned no results for a query "
        "against refund_policy.pdf."
    )

    assert len(results) <= 5

    # ------------------------------------------------------
    # Contract checks
    # ------------------------------------------------------

    assert results[0].source.name.lower() == "sparse"

    assert results[0].rank == 1

    assert all(
        result.document_id == "refund_policy.pdf"
        for result in results
    )

    assert all(
        isinstance(result.score, float)
        for result in results
    )

    assert all(
        result.retrieval_scores.get(
            "bm25_score"
        )
        == result.score
        for result in results
    )

    # ------------------------------------------------------
    # Ranking should be deterministic.
    # ------------------------------------------------------

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # ------------------------------------------------------
    # Actual lexical evidence check.
    #
    # This prevents the test from passing merely because
    # BM25 returned arbitrary documents.
    # ------------------------------------------------------

    query_terms = {
        "refund",
        "eligibility",
        "policy",
    }

    combined_text = " ".join(
        result.text.lower()
        for result in results
    )

    assert any(
        term in combined_text
        for term in query_terms
    )