from pathlib import Path

import pandas as pd
from llama_index.core.schema import Document

from app.ingestion.chunking.csv_row_chunker import CSVRowChunker

# ------------------------------------------------------------
# Test CSV
# ------------------------------------------------------------

CSV_PATH = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\final clean Customer Support Ticket Dataset"
    r"\supportmind_final_clean_dataset.csv"
)

def test_csv_row_chunker():
    # --------------------------------------------------------
    # 1. Verify CSV exists
    # --------------------------------------------------------
    assert CSV_PATH.exists(), (
        f"CSV file not found: {CSV_PATH}"
    )

    # --------------------------------------------------------
    # 2. Load CSV
    # --------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    assert not df.empty, "CSV file is empty."

    # --------------------------------------------------------
    # 3. Convert each row into the same format
    #    used by CSVLoader
    # --------------------------------------------------------
    rows = []

    for _, row in df.iterrows():
        row_text = ", ".join(
            f"{column}: {value}"
            for column, value in row.items()
            if pd.notna(value)
        )

        if row_text.strip():
            rows.append(row_text)

    content = "\n".join(rows)

    # --------------------------------------------------------
    # 4. Create LlamaIndex Document
    # --------------------------------------------------------
    document = Document(
        text=content,
        metadata={
            "source": "csv",
            "source_file": CSV_PATH.name,
        },
    )

    # --------------------------------------------------------
    # 5. Run CSVRowChunker
    # --------------------------------------------------------
    chunker = CSVRowChunker()
    nodes = chunker.split(document)

    # --------------------------------------------------------
    # 6. Basic validation
    # --------------------------------------------------------
    assert nodes, "CSVRowChunker returned no nodes."

    assert len(nodes) == len(rows), (
        f"Expected {len(rows)} nodes, "
        f"but got {len(nodes)}."
    )

    # --------------------------------------------------------
    # 7. Validate every node
    # --------------------------------------------------------
    for index, node in enumerate(nodes):
        assert node.text.strip(), (
            f"Node {index} contains empty text."
        )

        assert node.metadata["chunk_index"] == index
        assert node.metadata["total_chunks"] == len(nodes)
        assert node.metadata["character_count"] == len(node.text)
        assert node.metadata["word_count"] == len(node.text.split())

    # --------------------------------------------------------
    # 8. Verify each row remains intact
    # --------------------------------------------------------
    for index, node in enumerate(nodes):
        assert node.text == rows[index], (
            f"Row {index} was modified or split."
        )

    # --------------------------------------------------------
    # 9. Print useful diagnostics
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("CSV ROW CHUNKER TEST")
    print("=" * 70)

    print(f"CSV file       : {CSV_PATH.name}")
    print(f"CSV rows       : {len(df):,}")
    print(f"Generated rows : {len(rows):,}")
    print(f"Generated nodes: {len(nodes):,}")

    print()
    print("First node:")
    print("-" * 70)
    print(nodes[0].text)

    if len(nodes) > 1:
        print()
        print("Second node:")
        print("-" * 70)
        print(nodes[1].text)

    print()
    print("=" * 70)
    print("CSV ROW CHUNKER TEST PASSED")
    print("=" * 70)
