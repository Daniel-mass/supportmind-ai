from pathlib import Path

from app.ingestion.chunking.chunking_pipeline import ChunkingPipeline
from app.ingestion.loaders.csv_loader import CSVLoader


CSV_PATH = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\final clean Customer Support Ticket Dataset"
    r"\supportmind_final_clean_dataset.csv"
)


def test_csv_chunking_deterministic_ids():

    assert CSV_PATH.exists()

    # ---------------------------------------------------------
    # Use the REAL production CSVLoader.
    # This ensures the test follows the actual ingestion path.
    # ---------------------------------------------------------

    loader = CSVLoader()

    document_1 = loader.load(CSV_PATH)
    document_2 = loader.load(CSV_PATH)

    # ---------------------------------------------------------
    # Same source + source_file must produce
    # the same deterministic document ID.
    # ---------------------------------------------------------

    assert document_1.document_id == document_2.document_id

    # ---------------------------------------------------------
    # Chunk both documents using the real pipeline.
    # ---------------------------------------------------------

    pipeline = ChunkingPipeline()

    nodes_1 = pipeline.chunk(
        document=document_1,
        strategy="csv_row",
    )

    nodes_2 = pipeline.chunk(
        document=document_2,
        strategy="csv_row",
    )

    # ---------------------------------------------------------
    # Both runs must produce exactly one node per CSV row.
    # ---------------------------------------------------------

    assert len(nodes_1) == 8469
    assert len(nodes_2) == 8469

    # ---------------------------------------------------------
    # Verify deterministic node IDs.
    # ---------------------------------------------------------

    for node_1, node_2 in zip(nodes_1, nodes_2):

        assert node_1.node_id == node_2.node_id

    # ---------------------------------------------------------
    # Verify content is also deterministic.
    # ---------------------------------------------------------

    for node_1, node_2 in zip(nodes_1, nodes_2):

        assert node_1.text == node_2.text

    # ---------------------------------------------------------
    # Explicit first/last checks.
    # ---------------------------------------------------------

    assert nodes_1[0].node_id == nodes_2[0].node_id
    assert nodes_1[-1].node_id == nodes_2[-1].node_id

    assert nodes_1[0].text == nodes_2[0].text
    assert nodes_1[-1].text == nodes_2[-1].text

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print()
    print("Document ID       :", document_1.document_id)
    print("Total nodes       :", len(nodes_1))
    print("First node ID     :", nodes_1[0].node_id)
    print(
        "First node same   :",
        nodes_1[0].node_id == nodes_2[0].node_id,
    )
    print("Last node ID      :", nodes_1[-1].node_id)
    print(
        "Last node same    :",
        nodes_1[-1].node_id == nodes_2[-1].node_id,
    )