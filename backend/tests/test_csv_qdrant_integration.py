from pathlib import Path

from app.ingestion.chunking.chunking_pipeline import ChunkingPipeline
from app.ingestion.loaders.csv_loader import CSVLoader
from app.embeddings.embedding_pipeline import EmbeddingPipeline
from app.vector_store.vector_pipeline import VectorPipeline
from app.vector_store.qdrant_store import QdrantStore
from app.vector_store.vector_registry import vector_registry
CSV_PATH = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\final clean Customer Support Ticket Dataset"
    r"\supportmind_final_clean_dataset.csv"
)

COLLECTION_NAME = "supportmind_csv_qdrant_test"


def test_csv_five_rows_qdrant_integration():

    # =========================================================
    # 1. Load CSV using production CSVLoader
    # =========================================================

    loader = CSVLoader()

    document = loader.load(CSV_PATH)

    assert document.metadata["rows"] == 8469

    # =========================================================
    # 2. Chunk using production ChunkingPipeline
    # =========================================================

    chunking_pipeline = ChunkingPipeline()

    nodes = chunking_pipeline.chunk(
        document=document,
        strategy="csv_row",
    )

    assert len(nodes) == 8469

    # Only use five rows for this Qdrant integration test.
    sample_nodes = nodes[:5]

    assert len(sample_nodes) == 5

    # =========================================================
    # 3. Embed five rows
    # =========================================================

    embedding_pipeline = EmbeddingPipeline(
        default_provider="fastembed",
    )

    embedded_nodes = embedding_pipeline.embed(
        sample_nodes
    )

    assert len(embedded_nodes) == 5

    assert all(
        len(node.embedding) == 768
        for node in embedded_nodes
    )

    print()
    print("CSV rows             :", document.metadata["rows"])
    print("Sample nodes         :", len(sample_nodes))
    print("Embedded nodes       :", len(embedded_nodes))
    print("Embedding dimension  :", embedded_nodes[0].embedding_dimension)

    # =========================================================
    # 4. Create VectorPipeline
    # =========================================================

    vector_pipeline = VectorPipeline()
    qdrant_store = QdrantStore(
    host="127.0.0.1",
    port=6333,
    https=False,
)

    vector_registry.register(qdrant_store)
    # =========================================================
    # 5. Clean up previous test collection
    # =========================================================

    if vector_pipeline.collection_exists(
        COLLECTION_NAME,
        provider_name="qdrant",
    ):
        vector_pipeline.delete_collection(
            COLLECTION_NAME,
            provider_name="qdrant",
        )

    try:

        # =====================================================
        # 6. Create 768-dimensional Qdrant collection
        # =====================================================

        vector_pipeline.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=768,
            provider_name="qdrant",
        )

        assert vector_pipeline.collection_exists(
            COLLECTION_NAME,
            provider_name="qdrant",
        )

        # =====================================================
        # 7. Insert five embedded nodes
        # =====================================================

        vector_pipeline.upsert_batch(
            collection_name=COLLECTION_NAME,
            nodes=embedded_nodes,
            provider_name="qdrant",
        )

        # =====================================================
        # 8. Verify Qdrant count
        # =====================================================

        count = vector_pipeline.count(
            COLLECTION_NAME,
            provider_name="qdrant",
        )

        print("Qdrant vector count :", count)

        assert count == 5

        # =====================================================
        # 9. Similarity search using first ticket's embedding
        # =====================================================

        query_vector = embedded_nodes[0].embedding

        results = vector_pipeline.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            top_k=5,
            provider_name="qdrant",
        )

        print("Search results       :", len(results))

        assert len(results) > 0

        # =====================================================
        # 10. The query ticket should be the top result
        # =====================================================

        top_result = results[0]

        print("Top result ID        :", top_result.id)
        print("Top result node ID   :", top_result.node_id)
        print("Top result score     :", top_result.score)
        print("Top result text      :", top_result.text[:200])

        assert top_result.node_id == embedded_nodes[0].node_id

        assert top_result.document_id == embedded_nodes[0].document_id

        assert top_result.text == embedded_nodes[0].text

        assert top_result.score > 0

    finally:

        # =====================================================
        # 11. Always remove temporary test collection
        # =====================================================

        if vector_pipeline.collection_exists(
            COLLECTION_NAME,
            provider_name="qdrant",
        ):
            vector_pipeline.delete_collection(
                COLLECTION_NAME,
                provider_name="qdrant",
            )