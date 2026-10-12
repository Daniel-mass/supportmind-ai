"""
PDF document loader.

Reads PDF documents and converts them into the
standardized IngestedDocument format.
"""

from pathlib import Path

from pypdf import PdfReader

from app.ingestion.loaders.base_loader import BaseLoader
from app.ingestion.schemas.document_schema import IngestedDocument


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    """

    def load(self, file_path: Path) -> IngestedDocument:
        """
        Load a PDF document.
        """

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(file_path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                pages.append(
                    f"--- Page {page_number} ---\n{text.strip()}"
                )

        content = "\n\n".join(pages)

        return IngestedDocument(
            source="pdf",
            source_file=file_path.name,
            content=content,
            metadata={
                "extension": file_path.suffix,
                "pages": len(reader.pages),
                "file_size": file_path.stat().st_size,
                "encrypted": reader.is_encrypted,
            },
        )