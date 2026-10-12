from pathlib import Path

from app.ingestion.chunking.chunking_pipeline import ChunkingPipeline
from app.ingestion.loaders.csv_loader import CSVLoader


CSV_PATH = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata\final clean Customer Support Ticket Dataset\supportmind_final_clean_dataset.csv"
)


def test_csv_loader_to_csv_row_chunker():

    # ---------------------------------------------------------
    # Load CSV using the REAL production CSVLoader
    # ---------------------------------------------------------

    loader = CSVLoader()

    document = loader.load(CSV_PATH)

    # ---------------------------------------------------------
    # Chunk using the REAL ChunkingPipeline
    # ---------------------------------------------------------

    pipeline = ChunkingPipeline(
        default_strategy="csv_row"
    )

    nodes = pipeline.chunk(
        document,
        strategy="csv_row",
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    print()
    print(f"CSV loader rows : {document.metadata['rows']}")
    print(f"Generated nodes : {len(nodes)}")

    assert len(nodes) == document.metadata["rows"]

    assert all(
        node.text.strip()
        for node in nodes
    )

    # First ticket must survive the real pipeline.
    assert nodes[0].text.startswith(
        "Ticket ID: 1"
    )

    # Last ticket must also survive.
    assert nodes[-1].text.startswith(
        "Ticket ID: 8469"
    )

    print()
    print("First node:")
    print(nodes[0].text)

    print()
    print("Last node:")
    print(nodes[-1].text)