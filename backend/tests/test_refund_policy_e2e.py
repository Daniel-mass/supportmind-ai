"""
End-to-end validation for SupportMind AI.

Tests:

enterprise_knowledge
        ↓
Filesystem Connector
        ↓
Loader Registry
        ↓
PDF Loader
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

from app.orchestrator.ingestion_orchestrator import (
    IngestionOrchestrator,
)

from app.orchestrator.orchestration_config import (
    OrchestrationConfig,
)

from app.vector_store.qdrant_store import (
    QdrantStore,
)


SOURCE_DIRECTORY = (
    r"C:\Users\danie\supportmind-ai"
    r"\enterprise_knowledge\policies"
)

TEST_COLLECTION = (
    "supportmind_refund_policy_test"
)


def reset_test_collection() -> None:
    """
    Delete the test collection if it already exists.

    E2E tests should start from a clean vector-store
    state so that vector counts represent the current
    ingestion only.
    """

    qdrant_store = QdrantStore()

    if qdrant_store.collection_exists(
        TEST_COLLECTION
    ):

        print(
            "\n[TEST] Existing test collection detected."
        )

        print(
            f"[TEST] Clearing collection: "
            f"{TEST_COLLECTION}"
        )

        qdrant_store.delete_collection(
            TEST_COLLECTION
        )

        print(
            "[TEST] Test collection cleared."
        )

    else:

        print(
            "\n[TEST] No existing test collection."
        )


def main():

    # ----------------------------------------------------------
    # Clean test state
    # ----------------------------------------------------------

    reset_test_collection()

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------

    config = OrchestrationConfig(
        embedding_provider="fastembed",
        vector_provider="qdrant",
        collection_name=TEST_COLLECTION,
        chunking_strategy="sentence_window",
    )

    # ----------------------------------------------------------
    # Orchestrator
    # ----------------------------------------------------------

    orchestrator = IngestionOrchestrator(
        config=config
    )

    result = orchestrator.run(
        SOURCE_DIRECTORY
    )

    # ----------------------------------------------------------
    # Results
    # ----------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "SUPPORTMIND AI — E2E INGESTION TEST"
    )
    print("=" * 70)

    print(
        f"Success: "
        f"{result.success}"
    )

    print(
        f"Documents discovered: "
        f"{result.documents_discovered}"
    )

    print(
        f"Documents processed: "
        f"{result.documents_processed}"
    )

    print(
        f"Documents failed: "
        f"{result.documents_failed}"
    )

    print(
        f"Chunks created: "
        f"{result.chunks_created}"
    )

    print(
        f"Vectors created: "
        f"{result.vectors_created}"
    )

    print(
        f"Embedding provider: "
        f"{result.embedding_provider}"
    )

    print(
        f"Embedding dimension: "
        f"{result.embedding_dimension}"
    )

    print(
        f"Vector provider: "
        f"{result.vector_provider}"
    )

    print(
        f"Collection: "
        f"{result.collection_name}"
    )

    print(
        f"Qdrant vector count: "
        f"{result.metadata.get('qdrant_vector_count')}"
    )

    print(
        f"Qdrant healthy: "
        f"{result.metadata.get('vector_store_healthy')}"
    )

    print("\nErrors:")

    if result.errors:

        for error in result.errors:

            print(
                f"  [ERROR] {error}"
            )

    else:

        print("  None")

    print("\n")
    print("=" * 70)

    if result.success:

        print(
            "E2E PIPELINE VALIDATION: PASSED"
        )

    else:

        print(
            "E2E PIPELINE VALIDATION: FAILED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()