"""
test_ingestion_to_retrieval_csv_pdf.py

SupportMind AI
------------------------------------------------------------

END-TO-END INGESTION + RETRIEVAL VALIDATION

CSV:
    Original large dataset
        ↓
    200-row temporary sample
        ↓
    IngestionOrchestrator
        ↓
    Qdrant
        ↓
    SparseRetriever / BM25
        ↓
    RetrievalService
        ↓
    BuiltContext

PDF:
    Enterprise policy documents
        ↓
    IngestionOrchestrator
        ↓
    Qdrant
        ↓
    SparseRetriever / BM25
        ↓
    RetrievalService
        ↓
    BuiltContext

This validates the real architecture:

    Loader
      ↓
    Parser
      ↓
    Preprocessing
      ↓
    Chunking
      ↓
    Embedding
      ↓
    Qdrant
      ↓
    Query Embedding
      ↓
    Dense Retrieval
      ↓
    BM25 Retrieval
      ↓
    RRF
      ↓
    Reranking
      ↓
    ContextBuilder
      ↓
    BuiltContext

This is an integration/sanity test.

It is NOT a formal retrieval benchmark.

IMPORTANT:
    The original large CSV is NEVER ingested.

    Only the first 200 rows are copied to a temporary
    test CSV and ingested into a disposable Qdrant
    collection.

DIAGNOSTICS:
    CSV retrieval additionally prints:

        Final RetrievalResult
              ↓
        _to_context_item()
              ↓
        ContextItem
              ↓
        ContextBuilder._fit_item_to_budget()
              ↓
        BuiltContext

    The ContextBuilder production code is NOT modified.

    Instead, the test temporarily wraps
    ContextBuilder._fit_item_to_budget() so we can determine
    whether candidates are being rejected because metadata
    consumes the context budget.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.orchestrator.ingestion_orchestrator import (
    IngestionOrchestrator,
)

from app.orchestrator.orchestration_config import (
    OrchestrationConfig,
)

from app.retrieval.context.context_builder import (
    ContextBuilder,
)

from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)

from app.retrieval.dense.dense_retriever import (
    DenseRetriever,
)

from app.retrieval.sparse.sparse_retriever import (
    SparseRetriever,
)

from app.retrieval.reranker.cross_encoder_baseline import (
    CrossEncoderBaseline,
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


# ============================================================
# CSV
# ============================================================

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
    / "_retrieval_e2e_test"
)

CSV_TEST_FILE = (
    CSV_TEST_DIRECTORY
    / "supportmind_retrieval_sample.csv"
)

CSV_COLLECTION = (
    "supportmind_e2e_csv_test"
)


# ============================================================
# PDF
# ============================================================

PDF_SOURCE_DIRECTORY = (
    BASE_DIRECTORY
    / "enterprise_knowledge"
    / "policies"
)

PDF_COLLECTION = (
    "supportmind_e2e_pdf_test"
)


# ============================================================
# TEST SIZE
# ============================================================

CSV_SAMPLE_ROWS = 200


# ============================================================
# EMBEDDING
# ============================================================

EMBEDDING_PROVIDER = "fastembed"


# ============================================================
# VECTOR STORE
# ============================================================

VECTOR_PROVIDER = "qdrant"


# ============================================================
# RETRIEVAL
# ============================================================

TOP_K = 5

RERANK_TOP_K = 5


# ============================================================
# DIAGNOSTICS
# ============================================================

# ------------------------------------------------------------
# Keep this enabled while diagnosing the CSV retrieval issue.
#
# Once the problem is fixed, this can be changed to False.
# ------------------------------------------------------------

ENABLE_CSV_RETRIEVAL_DIAGNOSTICS = True


# ============================================================
# CSV QUESTIONS
# ============================================================

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


# ============================================================
# PDF QUESTIONS
# ============================================================

PDF_TEST_QUESTIONS = [

    {
        "question": (
            "What types of payments are eligible "
            "for a refund at NovaCare?"
        ),
        "expected_terms": [
            "duplicate",
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
            "payment gateway",
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
# HELPERS
# ============================================================

def print_section(
    title: str,
) -> None:

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# SAFE VALUE HELPER
# ============================================================

def safe_text(
    value: Any,
    max_length: int = 1000,
) -> str:

    if value is None:

        return ""

    text = str(value)

    if len(text) <= max_length:

        return text

    return (
        text[:max_length]
        + "... [TRUNCATED]"
    )


# ============================================================
# OBJECT DIAGNOSTICS
# ============================================================

def print_retrieval_result_diagnostic(
    result: Any,
    index: int,
) -> None:

    print()
    print(
        "-" * 90
    )

    print(
        f"FINAL RETRIEVAL RESULT #{index}"
    )

    print(
        f"id              : "
        f"{getattr(result, 'id', None)}"
    )

    print(
        f"node_id         : "
        f"{getattr(result, 'node_id', None)}"
    )

    print(
        f"document_id     : "
        f"{getattr(result, 'document_id', None)}"
    )

    print(
        f"source          : "
        f"{getattr(result, 'source', None)}"
    )

    text = getattr(
        result,
        "text",
        None,
    )

    print(
        f"text type       : "
        f"{type(text).__name__}"
    )

    print(
        f"text length     : "
        f"{len(text or '')}"
    )

    print(
        "text:"
    )

    print(
        safe_text(
            text,
            max_length=1500,
        )
    )

    metadata = getattr(
        result,
        "metadata",
        None,
    )

    print(
        "metadata:"
    )

    print(
        metadata
    )

    # --------------------------------------------------------
    # Explicit window diagnostic.
    # --------------------------------------------------------

    if isinstance(
        metadata,
        dict,
    ):

        if "window" in metadata:

            window = metadata.get(
                "window"
            )

            print()

            print(
                f"window length   : "
                f"{len(str(window))}"
            )

            print(
                "window:"
            )

            print(
                safe_text(
                    window,
                    max_length=2500,
                )
            )

        if "original_text" in metadata:

            original_text = metadata.get(
                "original_text"
            )

            print()

            print(
                f"original_text length: "
                f"{len(str(original_text))}"
            )

            print(
                "original_text:"
            )

            print(
                safe_text(
                    original_text,
                    max_length=1500,
                )
            )

    retrieval_scores = getattr(
        result,
        "retrieval_scores",
        None,
    )

    print(
        "retrieval_scores:"
    )

    print(
        retrieval_scores
    )


# ============================================================
# CONTEXT ITEM DIAGNOSTICS
# ============================================================

def print_context_item_diagnostic(
    item: Any,
    index: int,
) -> None:

    print()
    print(
        "-" * 90
    )

    print(
        f"CONTEXT ITEM #{index}"
    )

    print(
        f"id              : "
        f"{getattr(item, 'id', None)}"
    )

    print(
        f"rank            : "
        f"{getattr(item, 'rank', None)}"
    )

    print(
        f"score           : "
        f"{getattr(item, 'score', None)}"
    )

    print(
        f"source          : "
        f"{getattr(item, 'source', None)}"
    )

    text = getattr(
        item,
        "text",
        None,
    )

    print(
        f"text type       : "
        f"{type(text).__name__}"
    )

    print(
        f"text length     : "
        f"{len(text or '')}"
    )

    print(
        "text:"
    )

    print(
        safe_text(
            text,
            max_length=1500,
        )
    )

    metadata = getattr(
        item,
        "metadata",
        None,
    )

    print(
        "metadata:"
    )

    print(
        metadata
    )

    # --------------------------------------------------------
    # Explicit ContextItem metadata diagnostics.
    # --------------------------------------------------------

    if isinstance(
        metadata,
        dict,
    ):

        metadata_total_chars = sum(
            len(str(value))
            for value in metadata.values()
        )

        print()

        print(
            f"metadata total chars: "
            f"{metadata_total_chars}"
        )

        print(
            f"metadata keys: "
            f"{list(metadata.keys())}"
        )

        if "window" in metadata:

            print(
                f"window length: "
                f"{len(str(metadata['window']))}"
            )


# ============================================================
# CONTEXT BUILDER BUDGET DIAGNOSTICS
# ============================================================

def install_context_builder_budget_diagnostics() -> None:

    """
    Temporarily wraps ContextBuilder._fit_item_to_budget().

    This is test-only instrumentation.

    The production ContextBuilder implementation is NOT changed.

    The wrapper reproduces the calculations performed by
    ContextBuilder._fit_item_to_budget() and prints:

        - candidate text length
        - metadata size
        - remaining budget
        - complete evidence line size
        - fixed formatting overhead
        - available text budget
        - final accept/reject decision

    This directly determines whether ContextBuilder is rejecting
    CSV candidates because metadata such as `window` consumes the
    character budget.
    """

    if not ENABLE_CSV_RETRIEVAL_DIAGNOSTICS:

        return

    original_method = (
        ContextBuilder._fit_item_to_budget
    )

    def diagnostic_wrapper(
        self,
        item,
        remaining_budget,
        include_metadata,
        include_provenance,
    ):

        print_section(
            "DIAGNOSTIC: CONTEXT BUILDER _fit_item_to_budget"
        )

        # ----------------------------------------------------
        # Candidate information.
        # ----------------------------------------------------

        print(
            f"Candidate ID       : "
            f"{getattr(item, 'id', None)}"
        )

        print(
            f"Candidate source   : "
            f"{getattr(item, 'source', None)}"
        )

        text = getattr(
            item,
            "text",
            "",
        )

        print(
            f"Text length        : "
            f"{len(text or '')}"
        )

        metadata = getattr(
            item,
            "metadata",
            {}
        ) or {}

        print(
            f"Metadata keys      : "
            f"{list(metadata.keys())}"
        )

        metadata_total_chars = sum(
            len(str(value))
            for value in metadata.values()
        )

        print(
            f"Metadata total chars: "
            f"{metadata_total_chars}"
        )

        if "window" in metadata:

            print(
                f"Window length      : "
                f"{len(str(metadata['window']))}"
            )

        print(
            f"Remaining budget   : "
            f"{remaining_budget}"
        )

        print(
            f"Include metadata   : "
            f"{include_metadata}"
        )

        print(
            f"Include provenance : "
            f"{include_provenance}"
        )

        # ----------------------------------------------------
        # Reproduce the exact formatting calculation used by
        # ContextBuilder._fit_item_to_budget().
        #
        # We do this ONLY for diagnostics.
        # ----------------------------------------------------

        candidate_copy = item.model_copy(
            deep=True
        )

        full_line = (
            self._format_single_evidence(
                index=1,
                item=candidate_copy,
                include_metadata=include_metadata,
                include_provenance=include_provenance,
            )
        )

        full_line_length = len(
            full_line
        )

        print(
            f"Complete evidence line: "
            f"{full_line_length}"
        )

        # ----------------------------------------------------
        # Calculate the fixed formatting overhead exactly as
        # the production method does.
        # ----------------------------------------------------

        empty_text_item = (
            item.model_copy(
                deep=True
            )
        )

        empty_text_item.text = ""

        empty_text_line = (
            self._format_single_evidence(
                index=1,
                item=empty_text_item,
                include_metadata=include_metadata,
                include_provenance=include_provenance,
            )
        )

        overhead = len(
            empty_text_line
        )

        print(
            f"Fixed overhead      : "
            f"{overhead}"
        )

        available_text = (
            remaining_budget
            - overhead
        )

        print(
            f"Available text budget: "
            f"{available_text}"
        )

        # ----------------------------------------------------
        # This is the critical diagnostic.
        # ----------------------------------------------------

        if available_text <= 0:

            print()

            print(
                "[DIAGNOSTIC RESULT]"
            )

            print(
                "Candidate WILL be rejected."
            )

            print(
                "Reason: metadata/provenance formatting "
                "already consumes the remaining budget."
            )

        elif full_line_length <= remaining_budget:

            print()

            print(
                "[DIAGNOSTIC RESULT]"
            )

            print(
                "Candidate WILL fit without trimming."
            )

        else:

            trimmed_text = (
                candidate_copy.text[
                    :available_text
                ].rstrip()
            )

            final_trimmed_line = (
                self._format_single_evidence(
                    index=1,
                    item=(
                        candidate_copy.model_copy(
                            update={
                                "text": trimmed_text
                            }
                        )
                    ),
                    include_metadata=include_metadata,
                    include_provenance=include_provenance,
                )
            )

            print(
                f"Trimmed text length : "
                f"{len(trimmed_text)}"
            )

            print(
                f"Final trimmed line : "
                f"{len(final_trimmed_line)}"
            )

            if not trimmed_text:

                print()

                print(
                    "[DIAGNOSTIC RESULT]"
                )

                print(
                    "Candidate WILL be rejected."
                )

                print(
                    "Reason: no document text remains "
                    "after metadata/provenance overhead."
                )

            elif len(final_trimmed_line) <= remaining_budget:

                print()

                print(
                    "[DIAGNOSTIC RESULT]"
                )

                print(
                    "Candidate SHOULD be accepted after trimming."
                )

            else:

                print()

                print(
                    "[DIAGNOSTIC RESULT]"
                )

                print(
                    "Candidate WILL be rejected."
                )

                print(
                    "Reason: formatted evidence remains "
                    "larger than the remaining budget."
                )

        print(
            "-" * 90
        )

        # ----------------------------------------------------
        # Call the ORIGINAL production method unchanged.
        # ----------------------------------------------------

        result = original_method(
            self,
            item,
            remaining_budget,
            include_metadata,
            include_provenance,
        )

        # ----------------------------------------------------
        # Print actual production decision.
        # ----------------------------------------------------

        print()

        if result is None:

            print(
                "[ACTUAL PRODUCTION RESULT] "
                "REJECTED (None)"
            )

        else:

            print(
                "[ACTUAL PRODUCTION RESULT] "
                "ACCEPTED"
            )

            print(
                f"Accepted text length: "
                f"{len(result.text)}"
            )

        return result

    # --------------------------------------------------------
    # Replace the class method temporarily.
    #
    # pytest process ends after the test session, so this is
    # safe test-only instrumentation.
    # --------------------------------------------------------

    ContextBuilder._fit_item_to_budget = (
        diagnostic_wrapper
    )


# ============================================================
# INSTALL RETRIEVAL DIAGNOSTICS
# ============================================================

def install_retrieval_diagnostics(
    retrieval_service: RetrievalService,
) -> None:

    """
    Temporarily wraps RetrievalService._to_context_item().

    This lets us inspect the exact RetrievalResult that reaches
    the conversion stage.

    We intentionally do NOT modify RetrievalService logic.

    The original method is still executed unchanged.
    """

    if not ENABLE_CSV_RETRIEVAL_DIAGNOSTICS:

        return

    original_method = (
        retrieval_service._to_context_item
    )

    def diagnostic_wrapper(
        result,
    ):

        print_section(
            "DIAGNOSTIC: RETRIEVAL RESULT → CONTEXT ITEM"
        )

        print_retrieval_result_diagnostic(
            result=result,
            index=1,
        )

        item = original_method(
            result
        )

        print_context_item_diagnostic(
            item=item,
            index=1,
        )

        return item

    # --------------------------------------------------------
    # Replace only this instance method.
    #
    # The original retrieval implementation remains unchanged.
    # --------------------------------------------------------

    retrieval_service._to_context_item = (
        diagnostic_wrapper
    )


# ============================================================
# CSV SAMPLE
# ============================================================

def create_csv_test_sample() -> Path:

    print_section(
        "CREATING CSV TEST SAMPLE"
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
        f"Source : {CSV_SOURCE_FILE}"
    )

    print(
        f"Rows   : {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Created: {CSV_TEST_FILE}"
    )

    # --------------------------------------------------------
    # Print the first row.
    # --------------------------------------------------------

    print()
    print(
        "CSV SAMPLE COLUMNS:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    print()
    print(
        "FIRST CSV ROW:"
    )

    if not df.empty:

        print(
            df.iloc[0].to_dict()
        )

    # --------------------------------------------------------
    # Check expected entities.
    # --------------------------------------------------------

    print()
    print(
        "EXPECTED ENTITY PRESENCE CHECK:"
    )

    csv_text = (
        df.astype(str)
        .to_string()
        .lower()
    )

    expected_entities = [
        "marisa",
        "gopro",
        "christopher",
        "dell xps",
        "jessica",
        "lg smart tv",
        "microsoft surface",
        "invalid credentials",
    ]

    for entity in expected_entities:

        found = entity.lower() in csv_text

        print(
            f"  {entity:<25} : "
            f"{'FOUND' if found else 'NOT FOUND'}"
        )

    return CSV_TEST_FILE


# ============================================================
# CSV SAMPLE CLEANUP
# ============================================================

def cleanup_csv_test_sample() -> None:

    if CSV_TEST_DIRECTORY.exists():

        print(
            "\n[TEST] Removing temporary CSV sample..."
        )

        shutil.rmtree(
            CSV_TEST_DIRECTORY,
            ignore_errors=True,
        )

        print(
            "[OK] Temporary CSV sample removed."
        )


# ============================================================
# QDRANT COLLECTION RESET
# ============================================================

def reset_collection(
    collection_name: str,
) -> None:

    store = QdrantStore()

    if store.collection_exists(
        collection_name
    ):

        print(
            f"[TEST] Deleting collection: "
            f"{collection_name}"
        )

        store.delete_collection(
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
    chunking_strategy: str,
) -> IngestionOrchestrator:

    config = OrchestrationConfig(

        embedding_provider=(
            EMBEDDING_PROVIDER
        ),

        vector_provider=(
            VECTOR_PROVIDER
        ),

        collection_name=(
            collection_name
        ),

        chunking_strategy=(
            chunking_strategy
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
# CONFIGURE RETRIEVAL AFTER INGESTION
# ============================================================

def configure_retrieval(
    orchestrator: IngestionOrchestrator,
    collection_name: str,
) -> None:

    print_section(
        "CONFIGURING RETRIEVAL SERVICE"
    )

    # --------------------------------------------------------
    # Dense Retriever
    # --------------------------------------------------------

    dense_retriever = DenseRetriever(
        pipeline=(
            orchestrator.vector_pipeline
        )
    )

    print(
        "[OK] DenseRetriever created."
    )

    # --------------------------------------------------------
    # Qdrant Store
    # --------------------------------------------------------

    qdrant_store = QdrantStore()

    # --------------------------------------------------------
    # Sparse Retriever / BM25
    # --------------------------------------------------------

    sparse_retriever = (
        SparseRetriever.from_qdrant(

            qdrant_store=qdrant_store,

            collection_name=(
                collection_name
            ),

            batch_size=100,
        )
    )

    print(
        "[OK] SparseRetriever / BM25 created."
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    reranker = CrossEncoderBaseline()

    print(
        "[OK] CrossEncoder reranker created."
    )

    # --------------------------------------------------------
    # Retrieval configuration
    # --------------------------------------------------------

    retrieval_config = (
        RetrievalServiceConfig(

            use_dense=True,

            use_sparse=True,

            use_graph=False,

            use_reranker=True,

            candidate_multiplier=4,

            max_candidate_k=40,

            context_max_items=8,

            context_max_characters=12000,

            include_metadata=True,

            include_provenance=True,
        )
    )

    # --------------------------------------------------------
    # Create RetrievalService
    # --------------------------------------------------------

    retrieval_service = RetrievalService(

        dense_retriever=(
            dense_retriever
        ),

        sparse_retriever=(
            sparse_retriever
        ),

        reranker=(
            reranker
        ),

        config=(
            retrieval_config
        ),
    )

    # --------------------------------------------------------
    # Attach RetrievalService to orchestrator
    # --------------------------------------------------------

    orchestrator.retrieval_service = (
        retrieval_service
    )

    print(
        "[OK] RetrievalService attached "
        "to IngestionOrchestrator."
    )


# ============================================================
# INSTALL CSV DIAGNOSTICS AFTER CONFIGURATION
# ============================================================

def configure_csv_diagnostics(
    orchestrator: IngestionOrchestrator,
) -> None:

    if not ENABLE_CSV_RETRIEVAL_DIAGNOSTICS:

        return

    retrieval_service = (
        getattr(
            orchestrator,
            "retrieval_service",
            None,
        )
    )

    if retrieval_service is None:

        raise RuntimeError(
            "Cannot install diagnostics because "
            "RetrievalService is not configured."
        )

    # --------------------------------------------------------
    # RetrievalResult → ContextItem diagnostics.
    # --------------------------------------------------------

    install_retrieval_diagnostics(
        retrieval_service
    )

    # --------------------------------------------------------
    # ContextBuilder budget diagnostics.
    #
    # This instruments the actual ContextBuilder used by
    # RetrievalService without modifying production code.
    # --------------------------------------------------------

    install_context_builder_budget_diagnostics()

    print(
        "[OK] CSV retrieval diagnostics installed."
    )


# ============================================================
# INGESTION
# ============================================================

def ingest_dataset(
    source_directory: Path,
    collection_name: str,
    dataset_name: str,
    chunking_strategy: str,
) -> IngestionOrchestrator:

    print_section(
        f"INGESTING {dataset_name}"
    )

    orchestrator = create_orchestrator(
        collection_name=collection_name,
        chunking_strategy=chunking_strategy,
    )

    started = time.perf_counter()

    result = orchestrator.run(
        str(source_directory)
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if not result.success:

        raise RuntimeError(
            f"{dataset_name} ingestion failed."
        )

    if result.vectors_created <= 0:

        raise RuntimeError(
            f"{dataset_name} produced zero vectors."
        )

    print()

    print(
        f"Ingestion time       : "
        f"{elapsed:.2f}s"
    )

    print(
        f"Success              : "
        f"{result.success}"
    )

    print(
        f"Documents discovered : "
        f"{result.documents_discovered}"
    )

    print(
        f"Documents processed  : "
        f"{result.documents_processed}"
    )

    print(
        f"Documents failed     : "
        f"{result.documents_failed}"
    )

    print(
        f"Chunks created       : "
        f"{result.chunks_created}"
    )

    print(
        f"Vectors created      : "
        f"{result.vectors_created}"
    )

    print(
        f"Embedding dimension  : "
        f"{result.embedding_dimension}"
    )

    print(
        f"Qdrant vector count  : "
        f"{result.metadata.get('qdrant_vector_count')}"
    )

    print(
        f"Chunking strategy    : "
        f"{chunking_strategy}"
    )

    print(
        f"[PASS] {dataset_name} ingestion."
    )

    # --------------------------------------------------------
    # Retrieval configured AFTER ingestion.
    # --------------------------------------------------------

    configure_retrieval(
        orchestrator=orchestrator,
        collection_name=collection_name,
    )

    return orchestrator


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def get_source_file(
    item,
) -> str:

    metadata = (
        getattr(
            item,
            "metadata",
            None,
        )
        or {}
    )

    nested = metadata.get(
        "metadata"
    )

    if isinstance(
        nested,
        dict,
    ):

        source_file = nested.get(
            "source_file"
        )

        if source_file:

            return str(
                source_file
            )

    source_file = metadata.get(
        "source_file"
    )

    if source_file:

        return str(
            source_file
        )

    file_name = metadata.get(
        "file_name"
    )

    if file_name:

        return str(
            file_name
        )

    return str(
        getattr(
            item,
            "source",
            "unknown",
        )
    )


# ============================================================
# DISPLAY CONTEXT
# ============================================================

def display_context(
    question: str,
    built_context,
) -> None:

    print()
    print(
        "-" * 90
    )

    print(
        "QUESTION:"
    )

    print(
        question
    )

    print()

    items = (
        getattr(
            built_context,
            "items",
            None,
        )
        or []
    )

    print(
        f"Context items: "
        f"{len(items)}"
    )

    if not items:

        print(
            "[NO CONTEXT]"
        )

        return

    for index, item in enumerate(
        items,
        start=1,
    ):

        source_file = get_source_file(
            item
        )

        print()
        print(
            f"RESULT #{index}"
        )

        print(
            f"Rank   : "
            f"{getattr(item, 'rank', None)}"
        )

        print(
            f"Score  : "
            f"{getattr(item, 'score', None)}"
        )

        print(
            f"Source : "
            f"{source_file}"
        )

        print(
            "Text:"
        )

        print(
            getattr(
                item,
                "text",
                "",
            )
        )

    print(
        "-" * 90
    )


# ============================================================
# FINAL CONTEXT DIAGNOSTICS
# ============================================================

def diagnose_built_context(
    question: str,
    built_context,
) -> None:

    if not ENABLE_CSV_RETRIEVAL_DIAGNOSTICS:

        return

    print_section(
        "DIAGNOSTIC: FINAL BUILT CONTEXT"
    )

    items = (
        getattr(
            built_context,
            "items",
            None,
        )
        or []
    )

    print(
        f"BuiltContext item count: "
        f"{len(items)}"
    )

    if not items:

        print(
            "[DIAGNOSTIC] BuiltContext is EMPTY."
        )

        print(
            "[DIAGNOSTIC] RetrievalService produced "
            "final candidates, but ContextBuilder "
            "returned zero context items."
        )

        return

    for index, item in enumerate(
        items,
        start=1,
    ):

        print_context_item_diagnostic(
            item=item,
            index=index,
        )


# ============================================================
# VALIDATE CONTEXT
# ============================================================

def validate_context(
    built_context,
    expected_terms: list[str],
) -> tuple[bool, list[str]]:

    items = (
        getattr(
            built_context,
            "items",
            None,
        )
        or []
    )

    combined_text = " ".join(

        str(
            getattr(
                item,
                "text",
                "",
            )
        )

        for item in items

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
# RETRIEVAL TESTS
# ============================================================

def run_retrieval_tests(
    dataset_name: str,
    orchestrator: IngestionOrchestrator,
    questions: list[dict],
) -> tuple[int, int]:

    print_section(
        f"{dataset_name} RETRIEVAL SERVICE TEST"
    )

    passed = 0
    failed = 0

    retrieval_service = (
        getattr(
            orchestrator,
            "retrieval_service",
            None,
        )
    )

    if retrieval_service is None:

        raise RuntimeError(
            "RetrievalService is not configured "
            "on the orchestrator."
        )

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

            started = time.perf_counter()

            built_context = (
                orchestrator.retrieve(

                    question,

                    top_k=TOP_K,

                    include_graph=False,

                    rerank=True,

                    rerank_top_k=RERANK_TOP_K,
                )
            )

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            display_context(
                question,
                built_context,
            )

            # ------------------------------------------------
            # Additional diagnostics
            # ------------------------------------------------

            if dataset_name == "CSV":

                diagnose_built_context(
                    question=question,
                    built_context=built_context,
                )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            success, missing = (
                validate_context(
                    built_context,
                    expected_terms,
                )
            )

            print()

            print(
                f"Retrieval latency: "
                f"{latency_ms:.2f} ms"
            )

            print(
                "Retrieval diagnostics:"
            )

            print(
                retrieval_service
                .last_execution_metadata
            )

            if success:

                print(
                    "[PASS] Expected information "
                    "found in retrieved context."
                )

                passed += 1

            else:

                print(
                    "[FAIL] Expected information "
                    "not found."
                )

                print(
                    f"Missing terms: "
                    f"{missing}"
                )

                failed += 1

        except Exception as exc:

            print(
                f"[FAIL] Retrieval crashed: "
                f"{type(exc).__name__}: {exc}"
            )

            failed += 1

    return (
        passed,
        failed,
    )


# ============================================================
# CSV E2E TEST
# ============================================================

def test_csv_ingestion_to_retrieval():

    print_section(
        "CSV END-TO-END INGESTION → RETRIEVAL"
    )

    csv_orchestrator = None
    csv_sample_created = False

    try:

        # ----------------------------------------------------
        # 1. Create 200-row sample
        # ----------------------------------------------------

        create_csv_test_sample()

        csv_sample_created = True

        # ----------------------------------------------------
        # 2. Reset disposable E2E collection
        # ----------------------------------------------------

        reset_collection(
            CSV_COLLECTION
        )

        # ----------------------------------------------------
        # 3. Ingest CSV sample
        # ----------------------------------------------------

        csv_orchestrator = ingest_dataset(

            source_directory=(
                CSV_TEST_DIRECTORY
            ),

            collection_name=(
                CSV_COLLECTION
            ),

            dataset_name="CSV",

            chunking_strategy="csv_row",
        )

        # ----------------------------------------------------
        # 4. Install diagnostics
        # ----------------------------------------------------

        configure_csv_diagnostics(
            csv_orchestrator
        )

        # ----------------------------------------------------
        # 5. Retrieval
        # ----------------------------------------------------

        passed, failed = (
            run_retrieval_tests(

                dataset_name="CSV",

                orchestrator=(
                    csv_orchestrator
                ),

                questions=(
                    CSV_TEST_QUESTIONS
                ),
            )
        )

        total = (
            passed
            + failed
        )

        print_section(
            "CSV E2E RESULT"
        )

        print(
            f"Passed: {passed}"
        )

        print(
            f"Failed: {failed}"
        )

        print(
            f"Total : {total}"
        )

        if total:

            print(
                f"Pass rate: "
                f"{(passed / total) * 100:.2f}%"
            )

        assert total > 0, (
            "No CSV retrieval tests were executed."
        )

        assert failed == 0, (
            f"{failed} CSV retrieval tests failed."
        )

    finally:

        if csv_orchestrator is not None:

            csv_orchestrator.close()

        if csv_sample_created:

            cleanup_csv_test_sample()


# ============================================================
# PDF E2E TEST
# ============================================================

def test_pdf_ingestion_to_retrieval():

    print_section(
        "PDF END-TO-END INGESTION → RETRIEVAL"
    )

    pdf_orchestrator = None

    try:

        # ----------------------------------------------------
        # 1. Validate PDF source directory
        # ----------------------------------------------------

        if not PDF_SOURCE_DIRECTORY.exists():

            raise FileNotFoundError(
                f"PDF source directory not found:\n"
                f"{PDF_SOURCE_DIRECTORY}"
            )

        # ----------------------------------------------------
        # 2. Reset disposable E2E collection
        # ----------------------------------------------------

        reset_collection(
            PDF_COLLECTION
        )

        # ----------------------------------------------------
        # 3. Ingest PDFs
        # ----------------------------------------------------

        pdf_orchestrator = ingest_dataset(

            source_directory=(
                PDF_SOURCE_DIRECTORY
            ),

            collection_name=(
                PDF_COLLECTION
            ),

            dataset_name="PDF",

            chunking_strategy="sentence_window",
        )

        # ----------------------------------------------------
        # 4. Retrieval
        # ----------------------------------------------------

        passed, failed = (
            run_retrieval_tests(

                dataset_name="PDF",

                orchestrator=(
                    pdf_orchestrator
                ),

                questions=(
                    PDF_TEST_QUESTIONS
                ),
            )
        )

        total = (
            passed
            + failed
        )

        print_section(
            "PDF E2E RESULT"
        )

        print(
            f"Passed: {passed}"
        )

        print(
            f"Failed: {failed}"
        )

        print(
            f"Total : {total}"
        )

        if total:

            print(
                f"Pass rate: "
                f"{(passed / total) * 100:.2f}%"
            )

        assert total > 0, (
            "No PDF retrieval tests were executed."
        )

        assert failed == 0, (
            f"{failed} PDF retrieval tests failed."
        )

    finally:

        if pdf_orchestrator is not None:

            pdf_orchestrator.close()


# ============================================================
# ARCHITECTURE COLLECTION TEST
# ============================================================

def test_e2e_test_configuration():

    """
    Confirms that the E2E test configuration points to
    disposable collections and does not accidentally target
    production collections.
    """

    assert CSV_COLLECTION == (
        "supportmind_e2e_csv_test"
    )

    assert PDF_COLLECTION == (
        "supportmind_e2e_pdf_test"
    )

    assert CSV_SAMPLE_ROWS == 200

    assert EMBEDDING_PROVIDER == "fastembed"

    assert VECTOR_PROVIDER == "qdrant"


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        "\nThis file is designed to run with pytest.\n"
    )

    print(
        "Run:"
    )

    print(
        "python -m pytest "
        "backend/tests/real/"
        "test_ingestion_to_retrieval_csv_pdf.py "
        "-v -s"
    )