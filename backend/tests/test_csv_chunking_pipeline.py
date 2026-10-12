from pathlib import Path

from app.ingestion.chunking.chunking_pipeline import ChunkingPipeline
from app.ingestion.schemas.document_schema import IngestedDocument


CSV_PATH = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\\final clean Customer Support Ticket Dataset"
    r"\supportmind_final_clean_dataset.csv"
)


def test_csv_row_chunking_pipeline():
    assert CSV_PATH.exists()

    document = IngestedDocument(
        source="csv",
        source_file=CSV_PATH.name,
        content=CSV_PATH.read_text(
            encoding="utf-8",
            errors="ignore",
        ),
    )

    pipeline = ChunkingPipeline()

    nodes = pipeline.chunk(
        document=document,
        strategy="csv_row",
    )

    assert len(nodes) == 8469

    assert all(node.text.strip() for node in nodes)

    assert all(
        node.metadata["document_id"] == document.document_id
        for node in nodes
    )

    assert all(
        node.metadata["source"] == "csv"
        for node in nodes
    )

    assert all(
        node.metadata["source_file"] == CSV_PATH.name
        for node in nodes
    )

    assert all(
        node.metadata["chunking_strategy"] == "csv_row"
        for node in nodes
    )

    assert nodes[0].metadata["chunk_index"] == 0
    assert nodes[-1].metadata["chunk_index"] == 8468

    assert nodes[0].metadata["total_chunks"] == 8469
    assert nodes[-1].metadata["total_chunks"] == 8469

    assert all(
        node.node_id
        for node in nodes
    )

    print()
    print("Document ID    :", document.document_id)
    print("CSV rows       :", 8469)
    print("Generated nodes:", len(nodes))
    print("Strategy       :", nodes[0].metadata["chunking_strategy"])
    print("First node ID  :", nodes[0].node_id)
    print("First node     :", nodes[0].text[:300])