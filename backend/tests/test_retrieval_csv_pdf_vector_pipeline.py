"""
test_retrieval_csv_pdf_vector_pipeline.py

SupportMind AI
------------------------------------------------------------

FIRST-STAGE RETRIEVAL VALIDATION

This test intentionally does NOT use the future
app.retrieval pipeline.

It validates the retrieval capability that already
exists inside:

    VectorPipeline.search()

Current architecture:

    CSV / PDF
        ↓
    IngestionOrchestrator
        ↓
    Parser
        ↓
    Preprocessing
        ↓
    Chunking
        ↓
    EmbeddingPipeline
        ↓
    VectorPipeline
        ↓
    Qdrant
        ↓
    Query Embedding
        ↓
    VectorPipeline.search()
        ↓
    Similarity Results

The purpose of this test is to prove that:

1. CSV data can be ingested into Qdrant.
2. PDF data can be ingested into Qdrant.
3. Query text can be embedded using the same embedding model.
4. VectorPipeline.search() retrieves relevant chunks.
5. Retrieved chunks contain the expected information.
6. CSV and PDF retrieval work independently.

This is NOT yet the dedicated Retrieval Pipeline.

Future phase:

    app/retrieval/
        ├── retrieval_pipeline.py
        ├── hybrid_search/
        ├── reranker/
        └── graph_rag/

Those components will be developed after this test passes.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

from app.orchestrator.ingestion_orchestrator import (
    IngestionOrchestrator,
)

from app.orchestrator.orchestration_config import (
    OrchestrationConfig,
)

from app.embeddings.embedding_pipeline import (
    EmbeddingPipeline,
)

from app.vector_store.qdrant_store import (
    QdrantStore,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIRECTORY = Path(
    r"C:\Users\danie\supportmind-ai"
)


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

CSV_SOURCE_DIRECTORY = (
    BASE_DIRECTORY
    / "rawsupportminddata"
    / "final clean Customer Support Ticket Dataset"
)

CSV_SOURCE_FILE = (
    CSV_SOURCE_DIRECTORY
    / "supportmind_final_clean_dataset.csv"
)

CSV_TEST_DIRECTORY = (
    CSV_SOURCE_DIRECTORY
    / "_retrieval_test"
)

CSV_TEST_FILE = (
    CSV_TEST_DIRECTORY
    / "supportmind_retrieval_sample.csv"
)

CSV_COLLECTION = (
    "supportmind_retrieval_csv_test"
)


# ------------------------------------------------------------
# PDF
# ------------------------------------------------------------

PDF_SOURCE_DIRECTORY = (
    BASE_DIRECTORY
    / "enterprise_knowledge"
    / "policies"
)

PDF_COLLECTION = (
    "supportmind_retrieval_pdf_test"
)


# ------------------------------------------------------------
# TEST SIZE
# ------------------------------------------------------------

CSV_SAMPLE_ROWS = 200


# ------------------------------------------------------------
# Retrieval configuration
# ------------------------------------------------------------

EMBEDDING_PROVIDER = "fastembed"

TOP_K = 5


# ============================================================
# QUESTIONS
# ============================================================

"""
These questions are deliberately based on the sample data
provided for this test.

The CSV questions test:

    - customer identification
    - product identification
    - ticket problem
    - ticket status
    - ticket channel
    - customer satisfaction

The PDF questions test information explicitly contained
inside NovaCare Refund Policy.
"""


CSV_TEST_QUESTIONS = [

    {
        "question": (
            "What problem is Marisa Obrien having "
            "with her GoPro Hero?"
        ),
        "expected_terms": [
            "gopro",
            "product setup",
        ],
    },

    {
        "question": (
            "Which customer has a Dell XPS that "
            "is not turning on?"
        ),
        "expected_terms": [
            "christopher",
            "dell xps",
        ],
    },

    {
        "question": (
            "What is the ticket status for the "
            "Dell XPS customer?"
        ),
        "expected_terms": [
            "closed",
            "dell xps",
        ],
    },

    {
        "question": (
            "What problem is Jessica Rios experiencing "
            "with her LG Smart TV?"
        ),
        "expected_terms": [
            "lg smart tv",
            "peripheral compatibility",
        ],
    },

    {
        "question": (
            "Which customer is having an invalid "
            "credentials problem with Microsoft Surface?"
        ),
        "expected_terms": [
            "microsoft surface",
            "invalid credentials",
        ],
    },

]


PDF_TEST_QUESTIONS = [

    {
        "question": (
            "What types of payments are eligible "
            "for a refund at NovaCare?"
        ),
        "expected_terms": [
            "duplicate payment",
            "unauthorized",
            "failed",
        ],
    },

    {
        "question": (
            "What information must a support agent "
            "verify before processing a refund?"
        ),
        "expected_terms": [
            "transaction",
            "payment status",
        ],
    },

    {
        "question": (
            "How long is the standard NovaCare refund "
            "initiation SLA?"
        ),
        "expected_terms": [
            "two business days",
        ],
    },

    {
        "question": (
            "What should support do when a Payment "
            "Gateway timeout occurs?"
        ),
        "expected_terms": [
            "reconciliation",
            "transaction",
        ],
    },

    {
        "question": (
            "When should a refund case be escalated?"
        ),
        "expected_terms": [
            "unresolved",
            "reconciliation",
        ],
    },

]


# ============================================================
# UTILITY
# ============================================================

def print_section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CREATE CSV TEST SAMPLE
# ============================================================

def create_csv_test_sample() -> Path:

    print_section(
        "CREATING CSV RETRIEVAL TEST DATASET"
    )

    if not CSV_SOURCE_FILE.exists():

        raise FileNotFoundError(
            f"CSV source file not found:\n"
            f"{CSV_SOURCE_FILE}"
        )

    CSV_TEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[TEST] Source CSV : {CSV_SOURCE_FILE}"
    )

    print(
        f"[TEST] Sample rows: {CSV_SAMPLE_ROWS}"
    )

    df = pd.read_csv(
        CSV_SOURCE_FILE,
        nrows=CSV_SAMPLE_ROWS,
    )

    if df.empty:

        raise RuntimeError(
            "CSV source contains no rows."
        )

    df.to_csv(
        CSV_TEST_FILE,
        index=False,
    )

    print(
        f"[OK] Created: {CSV_TEST_FILE}"
    )

    print(
        f"[OK] Rows    : {len(df)}"
    )

    print(
        f"[OK] Columns : {len(df.columns)}"
    )

    return CSV_TEST_FILE


# ============================================================
# CLEAN CSV TEST DATA
# ============================================================

def cleanup_csv_test_sample() -> None:

    if CSV_TEST_DIRECTORY.exists():

        print(
            "\n[TEST] Removing temporary CSV test data..."
        )

        shutil.rmtree(
            CSV_TEST_DIRECTORY,
            ignore_errors=True,
        )

        print(
            "[OK] Temporary CSV test data removed."
        )


# ============================================================
# RESET COLLECTION
# ============================================================

def reset_collection(
    collection_name: str,
) -> None:

    qdrant_store = QdrantStore()

    if qdrant_store.collection_exists(
        collection_name
    ):

        print(
            f"[TEST] Deleting existing collection: "
            f"{collection_name}"
        )

        qdrant_store.delete_collection(
            collection_name
        )

        print(
            "[OK] Collection deleted."
        )

    else:

        print(
            f"[TEST] Collection does not exist: "
            f"{collection_name}"
        )


# ============================================================
# CREATE ORCHESTRATOR
# ============================================================

def create_orchestrator(
    collection_name: str,
) -> IngestionOrchestrator:

    config = OrchestrationConfig(

        embedding_provider=(
            EMBEDDING_PROVIDER
        ),

        vector_provider="qdrant",

        collection_name=(
            collection_name
        ),

        chunking_strategy=(
            "sentence_window"
        ),

        lowercase=True,

        remove_stopwords=False,

        spell_correction=True,

        lemmatize=True,

        stem=False,
    )

    return IngestionOrchestrator(
        config=config
    )


# ============================================================
# INGEST DATASET
# ============================================================

def ingest_dataset(
    source_directory: Path,
    collection_name: str,
    dataset_name: str,
):

    print_section(
        f"INGESTING {dataset_name}"
    )

    print(
        f"[TEST] Source directory:"
    )

    print(
        f"       {source_directory}"
    )

    print(
        f"[TEST] Collection:"
    )

    print(
        f"       {collection_name}"
    )

    orchestrator = create_orchestrator(
        collection_name
    )

    start = time.perf_counter()

    result = orchestrator.run(
        str(source_directory)
    )

    duration = (
        time.perf_counter()
        - start
    )

    print()
    print(
        f"[TEST] Ingestion completed in "
        f"{duration:.2f}s"
    )

    print(
        f"Success             : "
        f"{result.success}"
    )

    print(
        f"Documents discovered: "
        f"{result.documents_discovered}"
    )

    print(
        f"Documents processed : "
        f"{result.documents_processed}"
    )

    print(
        f"Documents failed    : "
        f"{result.documents_failed}"
    )

    print(
        f"Chunks created      : "
        f"{result.chunks_created}"
    )

    print(
        f"Vectors created     : "
        f"{result.vectors_created}"
    )

    print(
        f"Embedding dimension : "
        f"{result.embedding_dimension}"
    )

    print(
        f"Qdrant vector count : "
        f"{result.metadata.get('qdrant_vector_count')}"
    )

    if result.errors:

        print()

        print(
            "[INGESTION ERRORS]"
        )

        for error in result.errors:

            print(
                f"  {error}"
            )

    if not result.success:

        raise RuntimeError(
            f"{dataset_name} ingestion failed."
        )

    if result.vectors_created <= 0:

        raise RuntimeError(
            f"{dataset_name} ingestion produced "
            f"zero vectors."
        )

    print(
        f"[PASS] {dataset_name} ingestion successful."
    )

    return orchestrator


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(
    embedding_pipeline: EmbeddingPipeline,
    question: str,
) -> list[float]:

    embedded = embedding_pipeline.embed(
        _create_query_node(question),
        provider=EMBEDDING_PROVIDER,
    )

    return embedded.embedding


# ============================================================
# CREATE QUERY NODE
# ============================================================

def _create_query_node(
    question: str,
):
    """
    Create a temporary TextNode for query embedding.

    Query nodes are NOT inserted into Qdrant.

    They are used only to generate an embedding in the
    same embedding space as the stored document chunks.
    """

    from llama_index.core.schema import TextNode

    return TextNode(
        text=question
    )


# ============================================================
# SEARCH COLLECTION
# ============================================================

def search_collection(
    orchestrator: IngestionOrchestrator,
    embedding_pipeline: EmbeddingPipeline,
    collection_name: str,
    question: str,
    top_k: int = TOP_K,
):

    query_vector = embed_query(
        embedding_pipeline,
        question,
    )

    results = (
        orchestrator.vector_pipeline.search(

            collection_name=(
                collection_name
            ),

            query_vector=query_vector,

            top_k=top_k,

            provider_name="qdrant",
        )
    )

    return results


# ============================================================
# DISPLAY RETRIEVAL RESULTS
# ============================================================

def display_results(
    question: str,
    results,
) -> None:

    print()
    print(
        "-" * 80
    )

    print(
        f"QUESTION:"
    )

    print(
        question
    )

    print()

    if not results:

        print(
            "[NO RESULTS]"
        )

        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"RESULT #{index}"
        )

        print(
            f"Score      : "
            f"{getattr(result, 'score', None)}"
        )

        print(
            f"Node ID    : "
            f"{getattr(result, 'node_id', None)}"
        )

        print(
            f"Document ID: "
            f"{getattr(result, 'document_id', None)}"
        )

        print(
            "Text:"
        )

        print(
            getattr(
                result,
                "text",
                "",
            )
        )

        metadata = getattr(
            result,
            "metadata",
            None,
        )

        if metadata:

            print(
                "Metadata:"
            )

            print(
                metadata
            )

        print(
            "-" * 80
        )


# ============================================================
# CHECK EXPECTED TERMS
# ============================================================

def validate_results(
    results,
    expected_terms: list[str],
) -> tuple[bool, list[str]]:

    """
    Validate whether the retrieved text contains the
    information expected for the test question.

    This is NOT an ML evaluation metric.

    It is a basic retrieval sanity check.
    """

    combined_text = " ".join(

        str(
            getattr(
                result,
                "text",
                "",
            )
        )

        for result in results

    ).lower()

    missing_terms = []

    for term in expected_terms:

        if term.lower() not in combined_text:

            missing_terms.append(
                term
            )

    return (
        len(missing_terms) == 0,
        missing_terms,
    )


# ============================================================
# RUN RETRIEVAL TESTS
# ============================================================

def run_retrieval_tests(
    dataset_name: str,
    orchestrator: IngestionOrchestrator,
    collection_name: str,
    questions: list[dict],
    embedding_pipeline: EmbeddingPipeline,
) -> tuple[int, int]:

    print_section(
        f"{dataset_name} RETRIEVAL TEST"
    )

    passed = 0
    failed = 0

    for test_number, test_case in enumerate(
        questions,
        start=1,
    ):

        question = test_case[
            "question"
        ]

        expected_terms = test_case[
            "expected_terms"
        ]

        print()
        print(
            f"[TEST {test_number}] "
            f"{question}"
        )

        try:

            results = search_collection(

                orchestrator=orchestrator,

                embedding_pipeline=(
                    embedding_pipeline
                ),

                collection_name=(
                    collection_name
                ),

                question=question,

                top_k=TOP_K,
            )

            display_results(
                question,
                results,
            )

            success, missing = (
                validate_results(
                    results,
                    expected_terms,
                )
            )

            if success:

                print(
                    "[PASS] Expected information "
                    "was found in retrieved results."
                )

                passed += 1

            else:

                print(
                    "[FAIL] Expected information "
                    "was not found."
                )

                print(
                    f"Missing terms: {missing}"
                )

                failed += 1

        except Exception as exc:

            print(
                f"[FAIL] Retrieval crashed: "
                f"{type(exc).__name__}: {exc}"
            )

            failed += 1

    return passed, failed


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    csv_sample_created = False

    try:

        # ====================================================
        # 1. CREATE CSV SAMPLE
        # ====================================================

        create_csv_test_sample()

        csv_sample_created = True

        # ====================================================
        # 2. RESET QDRANT COLLECTIONS
        # ====================================================

        print_section(
            "RESETTING TEST COLLECTIONS"
        )

        reset_collection(
            CSV_COLLECTION
        )

        reset_collection(
            PDF_COLLECTION
        )

        # ====================================================
        # 3. CREATE SHARED QUERY EMBEDDING PIPELINE
        # ====================================================

        print_section(
            "INITIALIZING QUERY EMBEDDING PIPELINE"
        )

        embedding_pipeline = (
            EmbeddingPipeline(
                default_provider=(
                    EMBEDDING_PROVIDER
                )
            )
        )

        print(
            f"Embedding provider: "
            f"{EMBEDDING_PROVIDER}"
        )

        # ====================================================
        # 4. INGEST CSV
        # ====================================================

        csv_orchestrator = ingest_dataset(

            source_directory=(
                CSV_TEST_DIRECTORY
            ),

            collection_name=(
                CSV_COLLECTION
            ),

            dataset_name="CSV",
        )

        # ====================================================
        # 5. INGEST PDF
        # ====================================================

        pdf_orchestrator = ingest_dataset(

            source_directory=(
                PDF_SOURCE_DIRECTORY
            ),

            collection_name=(
                PDF_COLLECTION
            ),

            dataset_name="PDF",
        )

        # ====================================================
        # 6. CSV RETRIEVAL
        # ====================================================

        csv_passed, csv_failed = (
            run_retrieval_tests(

                dataset_name="CSV",

                orchestrator=(
                    csv_orchestrator
                ),

                collection_name=(
                    CSV_COLLECTION
                ),

                questions=(
                    CSV_TEST_QUESTIONS
                ),

                embedding_pipeline=(
                    embedding_pipeline
                ),
            )
        )

        # ====================================================
        # 7. PDF RETRIEVAL
        # ====================================================

        pdf_passed, pdf_failed = (
            run_retrieval_tests(

                dataset_name="PDF",

                orchestrator=(
                    pdf_orchestrator
                ),

                collection_name=(
                    PDF_COLLECTION
                ),

                questions=(
                    PDF_TEST_QUESTIONS
                ),

                embedding_pipeline=(
                    embedding_pipeline
                ),
            )
        )

        # ====================================================
        # 8. FINAL RESULTS
        # ====================================================

        total_passed = (
            csv_passed
            + pdf_passed
        )

        total_failed = (
            csv_failed
            + pdf_failed
        )

        total_tests = (
            total_passed
            + total_failed
        )

        total_duration = (
            time.perf_counter()
            - total_start
        )

        print_section(
            "FINAL RETRIEVAL VALIDATION"
        )

        print(
            f"CSV tests passed : "
            f"{csv_passed}"
        )

        print(
            f"CSV tests failed : "
            f"{csv_failed}"
        )

        print()

        print(
            f"PDF tests passed : "
            f"{pdf_passed}"
        )

        print(
            f"PDF tests failed : "
            f"{pdf_failed}"
        )

        print()

        print(
            f"Total tests      : "
            f"{total_tests}"
        )

        print(
            f"Total passed     : "
            f"{total_passed}"
        )

        print(
            f"Total failed     : "
            f"{total_failed}"
        )

        if total_tests > 0:

            accuracy = (
                total_passed
                / total_tests
                * 100
            )

        else:

            accuracy = 0.0

        print(
            f"Basic retrieval pass rate: "
            f"{accuracy:.2f}%"
        )

        print(
            f"Execution time: "
            f"{total_duration:.2f}s"
        )

        print()

        if total_failed == 0:

            print(
                "VECTOR-LEVEL RETRIEVAL "
                "VALIDATION: PASSED"
            )

            print()

            print(
                "Next phase:"
            )

            print(
                "Build the dedicated "
                "app/retrieval pipeline."
            )

        else:

            print(
                "VECTOR-LEVEL RETRIEVAL "
                "VALIDATION: FAILED"
            )

            print()

            print(
                "Do NOT move to the dedicated "
                "retrieval pipeline yet."
            )

            print(
                "Investigate embedding, chunking, "
                "Qdrant search, or query formulation."
            )

    except Exception as exc:

        print_section(
            "RETRIEVAL TEST CRASHED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    finally:

        if csv_sample_created:

            cleanup_csv_test_sample()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()