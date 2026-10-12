"""
csv_loader.py

CSV document loader.

Reads CSV files and converts structured tabular data
into a standardized IngestedDocument.

SupportMind AI
"""

from pathlib import Path

import pandas as pd

from app.ingestion.loaders.base_loader import BaseLoader
from app.ingestion.schemas.document_schema import IngestedDocument


class CSVLoader(BaseLoader):
    """
    Loader for CSV documents.

    A CSV file is represented as ONE IngestedDocument.
    Individual rows are preserved inside the document
    content and metadata.
    """

    def load(
        self,
        file_path: Path,
    ) -> IngestedDocument:

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        rows = []

        for _, row in df.iterrows():

            row_text = ", ".join(
                f"{column}: {value}"
                for column, value in row.items()
                if pd.notna(value)
            )

            rows.append(row_text)

        content = "\n".join(rows)

        document = IngestedDocument(
            source="csv",
            source_file=file_path.name,
            content=content,
            metadata={
                "extension": file_path.suffix,
                "rows": len(df),
                "columns": list(df.columns),
                "file_size": file_path.stat().st_size,
            },
        )

        print(
            f"[OK] CSV parsed: {file_path.name} | "
            f"rows={len(df)} | "
            f"characters={len(content):,}"
        )

        return document