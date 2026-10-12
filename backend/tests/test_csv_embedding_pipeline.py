from pathlib import Path

from app.ingestion.chunking.chunking_pipeline import ChunkingPipeline
from app.ingestion.schemas.document_schema import IngestedDocument
from app.embeddings.embedding_pipeline import EmbeddingPipeline


CSV_PATH = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\final clean Customer Support Ticket Dataset"
    r"\supportmind_final_clean_dataset.csv"
)


def test_csv_row_embedding():

    assert CSV_PATH.exists()

    # ----------------------------------------------------------
    # 1. Create the standard SupportMind IngestedDocument
    # ----------------------------------------------------------

    document = IngestedDocument(
        source="csv",
        source_file=CSV_PATH.name,
        content=CSV_PATH.read_text(
            encoding="utf-8",
            errors="ignore",
        ),
    )

    # ----------------------------------------------------------
    # 2. Run the existing ChunkingPipeline
    #    using the new csv_row strategy
    # ----------------------------------------------------------

    chunking_pipeline = ChunkingPipeline()

    nodes = chunking_pipeline.chunk(
        document=document,
        strategy="csv_row",
    )

    assert len(nodes) == 8469

    # ----------------------------------------------------------
    # 3. Embed a small sample
    # ----------------------------------------------------------
    # We only embed 5 rows here.
    # Full 8,469-row embedding comes later.

    sample_nodes = nodes[:5]

    embedding_pipeline = EmbeddingPipeline(
        default_provider="fastembed",
    )

    embedded_nodes = embedding_pipeline.embed(
        sample_nodes,
    )

    # ----------------------------------------------------------
    # 4. Validate EmbeddingPipeline output
    # ----------------------------------------------------------

    assert isinstance(
        embedded_nodes,
        list,
    )

    assert len(embedded_nodes) == 5

    for original, embedded in zip(
        sample_nodes,
        embedded_nodes,
    ):

        # Same node identity
        assert embedded.node_id == original.node_id

        # Same document identity
        assert (
            embedded.document_id
            == document.document_id
        )

        # Original text preserved
        assert embedded.text == original.text

        # Embedding exists
        assert embedded.embedding

        # Dimension matches actual vector length
        assert (
            len(embedded.embedding)
            == embedded.embedding_dimension
        )

        # Expected provider/model
        assert (
            embedded.embedding_model
            == "BAAI/bge-base-en-v1.5"
        )

        # Metadata preserved
        assert (
            embedded.metadata["document_id"]
            == document.document_id
        )

        assert (
            embedded.metadata["source"]
            == "csv"
        )

        assert (
            embedded.metadata["chunking_strategy"]
            == "csv_row"
        )

    # ----------------------------------------------------------
    # 5. Output diagnostics
    # ----------------------------------------------------------

    print()
    print("Document ID        :", document.document_id)
    print("CSV rows           :", len(nodes))
    print("Sample nodes       :", len(sample_nodes))
    print("Embedded nodes     :", len(embedded_nodes))
    print(
        "Embedding model    :",
        embedded_nodes[0].embedding_model,
    )
    print(
        "Embedding dimension:",
        embedded_nodes[0].embedding_dimension,
    )
    print(
        "First node ID      :",
        embedded_nodes[0].node_id,
    )