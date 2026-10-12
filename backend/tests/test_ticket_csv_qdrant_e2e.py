"""
test_ticket_csv_qdrant_e2e.py

End-to-end validation for SupportMind AI using a controlled
sample of the cleaned customer support ticket CSV dataset.

The production CSV contains thousands of tickets and several
million characters.

This E2E test intentionally creates a small sample CSV so the
complete ingestion pipeline can be validated safely.

Pipeline:

CSV
    ↓
Filesystem Connector
    ↓
Loader Registry
    ↓
CSV Loader
    ↓
ONE IngestedDocument
    ↓
Parser Pipeline
    ↓
NLP Preprocessing
    ↓
Chunking Pipeline
    ↓
Embedding Pipeline
    ↓
Vector Pipeline
    ↓
Qdrant
    ↓
Verification

SupportMind AI
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

from app.vector_store.qdrant_store import (
    QdrantStore,
)


# ==============================================================
# CONFIGURATION
# ==============================================================

SOURCE_DIRECTORY = Path(
    r"C:\Users\danie\supportmind-ai"
    r"\rawsupportminddata"
    r"\final clean Customer Support Ticket Dataset"
)

SOURCE_CSV = (
    SOURCE_DIRECTORY
    / "supportmind_final_clean_dataset.csv"
)

# --------------------------------------------------------------
# IMPORTANT:
# Start small.
#
# 25 rows  -> first pipeline validation
# 100 rows -> second validation
# 500 rows -> larger validation
# Full CSV -> only after everything passes
# --------------------------------------------------------------

SAMPLE_ROWS = 200

TEST_DATA_DIRECTORY = (
    SOURCE_DIRECTORY
    / "_e2e_test"
)

TEST_CSV = (
    TEST_DATA_DIRECTORY
    / "supportmind_e2e_sample.csv"
)

TEST_COLLECTION = (
    "supportmind_customer_tickets_test"
)


# ==============================================================
# UTILITY FUNCTIONS
# ==============================================================

def print_section(title: str) -> None:
    """Print a clearly separated console section."""

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ==============================================================
# CREATE TEST DATASET
# ==============================================================

def create_test_sample() -> Path:
    """
    Create a small temporary CSV from the production dataset.

    The original production CSV is never modified.
    """

    print_section(
        "CREATING CONTROLLED E2E TEST DATASET"
    )

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Source CSV not found:\n{SOURCE_CSV}"
        )

    print(
        f"[TEST] Source CSV : {SOURCE_CSV}"
    )

    print(
        f"[TEST] Sample rows: {SAMPLE_ROWS}"
    )

    TEST_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[TEST] Reading sample rows..."
    )

    df = pd.read_csv(
        SOURCE_CSV,
        nrows=SAMPLE_ROWS,
    )

    if df.empty:
        raise RuntimeError(
            "The source CSV contains no rows."
        )

    df.to_csv(
        TEST_CSV,
        index=False,
    )

    print(
        f"[OK] Sample created: {TEST_CSV.name}"
    )

    print(
        f"[OK] Rows         : {len(df)}"
    )

    print(
        f"[OK] Columns      : {len(df.columns)}"
    )

    print(
        f"[OK] Characters   : "
        f"{df.to_csv(index=False).__len__():,}"
    )

    print(
        f"[OK] File size    : "
        f"{TEST_CSV.stat().st_size:,} bytes"
    )

    return TEST_CSV


# ==============================================================
# CLEANUP
# ==============================================================

def cleanup_test_sample() -> None:
    """
    Remove temporary E2E dataset.

    Production dataset remains untouched.
    """

    if TEST_DATA_DIRECTORY.exists():

        print(
            "\n[TEST] Removing temporary E2E dataset..."
        )

        shutil.rmtree(
            TEST_DATA_DIRECTORY,
            ignore_errors=True,
        )

        print(
            "[OK] Temporary E2E dataset removed."
        )


# ==============================================================
# RESET QDRANT
# ==============================================================

def reset_test_collection() -> None:
    """
    Delete the Qdrant test collection if it exists.

    Every E2E run therefore starts with a clean collection.
    """

    print_section(
        "RESETTING QDRANT TEST COLLECTION"
    )

    qdrant_store = QdrantStore()

    if qdrant_store.collection_exists(
        TEST_COLLECTION
    ):

        print(
            "[TEST] Existing test collection detected."
        )

        print(
            f"[TEST] Clearing: {TEST_COLLECTION}"
        )

        qdrant_store.delete_collection(
            TEST_COLLECTION
        )

        print(
            "[OK] Test collection cleared."
        )

    else:

        print(
            "[TEST] No existing test collection."
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    total_start = time.perf_counter()

    test_csv_created = False

    try:

        # ======================================================
        # 1. Create controlled dataset
        # ======================================================

        create_test_sample()

        test_csv_created = True

        # ======================================================
        # 2. Reset Qdrant
        # ======================================================

        reset_test_collection()

        # ======================================================
        # 3. Configuration
        # ======================================================

        print_section(
            "CONFIGURATION"
        )

        config = OrchestrationConfig(

            embedding_provider="fastembed",

            vector_provider="qdrant",

            collection_name=TEST_COLLECTION,

            chunking_strategy="sentence_window",

            # --------------------------------------------------
            # Disable expensive NLP preprocessing for E2E test.
            #
            # This is intentional.
            #
            # Embeddings should operate on the original/cleaned
            # ticket text.
            # --------------------------------------------------

            lowercase=True,

            remove_stopwords=False,

            spell_correction=True,

            lemmatize=True,

            stem=False,
        )

        print(
            f"Embedding provider : "
            f"{config.embedding_provider}"
        )

        print(
            f"Vector provider    : "
            f"{config.vector_provider}"
        )

        print(
            f"Collection         : "
            f"{config.collection_name}"
        )

        print(
            f"Chunking strategy  : "
            f"{config.chunking_strategy}"
        )

        print(
            f"Lowercase          : "
            f"{config.lowercase}"
        )

        print(
            f"Lemmatize          : "
            f"{config.lemmatize}"
        )

        # ======================================================
        # 4. Initialize orchestrator
        # ======================================================

        print_section(
            "INITIALIZING INGESTION ORCHESTRATOR"
        )

        orchestrator = IngestionOrchestrator(
            config=config
        )

        print(
            "[OK] Ingestion orchestrator initialized."
        )

        # ======================================================
        # 5. Run E2E pipeline
        # ======================================================

        print_section(
            "STARTING E2E INGESTION"
        )

        print(
            f"[TEST] Dataset : {TEST_CSV}"
        )

        print(
            f"[TEST] Rows    : {SAMPLE_ROWS}"
        )

        print()
        print(
            "[TEST] Pipeline running..."
        )

        print(
            "[TEST] Expected flow:"
        )

        print(
            "       Filesystem"
            " → Loader"
            " → Parser"
            " → Preprocessing"
            " → Chunking"
            " → Embedding"
            " → Qdrant"
        )

        pipeline_start = time.perf_counter()

        result = orchestrator.run(
            str(TEST_DATA_DIRECTORY)
        )

        pipeline_duration = (
            time.perf_counter()
            - pipeline_start
        )

        print()
        print(
            f"[TEST] Pipeline returned after "
            f"{pipeline_duration:.2f}s"
        )

        # ======================================================
        # 6. Extract metadata safely
        # ======================================================

        metadata = result.metadata or {}

        qdrant_before = metadata.get(
            "qdrant_vector_count_before"
        )

        qdrant_after = metadata.get(
            "qdrant_vector_count"
        )

        qdrant_delta = metadata.get(
            "qdrant_vector_delta"
        )

        generated_nodes = metadata.get(
            "generated_node_count"
        )

        unique_nodes = metadata.get(
            "unique_generated_node_count"
        )

        duplicate_nodes = metadata.get(
            "duplicate_generated_node_ids"
        )

        qdrant_healthy = metadata.get(
            "vector_store_healthy"
        )

        # ======================================================
        # 7. Results
        # ======================================================

        print_section(
            "E2E TEST RESULTS"
        )

        print(
            f"Success                : "
            f"{result.success}"
        )

        print(
            f"Documents discovered   : "
            f"{result.documents_discovered}"
        )

        print(
            f"Documents processed    : "
            f"{result.documents_processed}"
        )

        print(
            f"Documents failed       : "
            f"{result.documents_failed}"
        )

        print(
            f"Chunks created         : "
            f"{result.chunks_created}"
        )

        print(
            f"Vectors created        : "
            f"{result.vectors_created}"
        )

        print(
            f"Embedding provider     : "
            f"{result.embedding_provider}"
        )

        print(
            f"Embedding dimension    : "
            f"{result.embedding_dimension}"
        )

        print(
            f"Vector provider        : "
            f"{result.vector_provider}"
        )

        print(
            f"Collection             : "
            f"{result.collection_name}"
        )

        print(
            f"Qdrant count BEFORE    : "
            f"{qdrant_before}"
        )

        print(
            f"Qdrant count AFTER     : "
            f"{qdrant_after}"
        )

        print(
            f"Qdrant vector DELTA    : "
            f"{qdrant_delta}"
        )

        print(
            f"Generated node count   : "
            f"{generated_nodes}"
        )

        print(
            f"Unique node IDs        : "
            f"{unique_nodes}"
        )

        print(
            f"Duplicate node IDs     : "
            f"{duplicate_nodes}"
        )

        print(
            f"Qdrant healthy         : "
            f"{qdrant_healthy}"
        )

        # ======================================================
        # 8. Pipeline stages
        # ======================================================

        print_section(
            "PIPELINE STAGES"
        )

        if result.stages:

            for stage in result.stages:

                print(
                    f"{stage.stage:<35}"
                    f"{stage.status:<12}"
                    f"{stage.duration_seconds:>8.4f}s "
                    f"items_out={stage.items_out}"
                )

                if stage.error:

                    print(
                        f"    ERROR: "
                        f"{stage.error}"
                    )

        else:

            print(
                "[WARNING] No stage diagnostics returned."
            )

        # ======================================================
        # 9. Errors
        # ======================================================

        print_section(
            "ERRORS"
        )

        if result.errors:

            for error in result.errors:

                print(
                    f"[ERROR] {error}"
                )

        else:

            print(
                "None"
            )

        # ======================================================
        # 10. Basic sanity checks
        # ======================================================

        print_section(
            "SANITY CHECKS"
        )

        checks_passed = True

        if not result.success:

            print(
                "[FAIL] Orchestrator reported failure."
            )

            checks_passed = False

        else:

            print(
                "[PASS] Orchestrator reported success."
            )

        if result.documents_discovered != 1:

            print(
                "[FAIL] Expected exactly 1 CSV document."
            )

            checks_passed = False

        else:

            print(
                "[PASS] CSV loaded as exactly "
                "one IngestedDocument."
            )

        if result.chunks_created <= 0:

            print(
                "[FAIL] No chunks were created."
            )

            checks_passed = False

        else:

            print(
                f"[PASS] Chunks created: "
                f"{result.chunks_created}"
            )

        if result.vectors_created <= 0:

            print(
                "[FAIL] No vectors were created."
            )

            checks_passed = False

        else:

            print(
                f"[PASS] Vectors created: "
                f"{result.vectors_created}"
            )

        if duplicate_nodes not in (
            None,
            0,
        ):

            print(
                "[FAIL] Duplicate node IDs detected."
            )

            checks_passed = False

        else:

            print(
                "[PASS] No duplicate node IDs."
            )

        # ======================================================
        # 11. Final validation
        # ======================================================

        total_duration = (
            time.perf_counter()
            - total_start
        )

        print_section(
            "FINAL VALIDATION"
        )

        if result.success and checks_passed:

            print(
                "E2E PIPELINE VALIDATION: PASSED"
            )

        else:

            print(
                "E2E PIPELINE VALIDATION: FAILED"
            )

            print(
                "Check the pipeline stages and errors above."
            )

        print(
            f"Total execution time: "
            f"{total_duration:.2f}s"
        )

    except KeyboardInterrupt:

        print_section(
            "E2E TEST INTERRUPTED"
        )

        print(
            "[TEST] Process interrupted by user."
        )

        raise

    except Exception as exc:

        print_section(
            "E2E TEST CRASHED"
        )

        print(
            f"[ERROR] {type(exc).__name__}: {exc}"
        )

        raise

    finally:

        # ======================================================
        # 12. Cleanup
        # ======================================================

        if test_csv_created:

            cleanup_test_sample()


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()