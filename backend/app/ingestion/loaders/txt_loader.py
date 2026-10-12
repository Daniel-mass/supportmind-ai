"""
TXT document loader.

This loader reads plain text (.txt) files and converts
them into the standardized IngestedDocument format used
throughout the SupportMind AI ingestion pipeline.
"""

from pathlib import Path

from app.ingestion.loaders.base_loader import BaseLoader
from app.ingestion.schemas.document_schema import IngestedDocument


class TXTLoader(BaseLoader):
    """
    Loader for plain text documents.
    """

    def load(self, file_path: Path) -> IngestedDocument:
        """
        Load a TXT document.

        Parameters
        ----------
        file_path : Path
            Path to the TXT file.

        Returns
        -------
        IngestedDocument
            Standardized document object.
        """

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        return IngestedDocument(
            source="txt",
            source_file=file_path.name,
            content=text,
            metadata={
                "extension": file_path.suffix,
                "encoding": "utf-8",
                "file_size": file_path.stat().st_size,
            },
        )