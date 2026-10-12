#backend/app/ingestion/parsers/docuemnt_parser.py
"""
Document Parser

Performs basic document normalization before advanced NLP.

Responsibilities
----------------
- Normalize whitespace
- Normalize newlines
- Remove excessive blank lines
- Trim document
- Preserve metadata

This parser intentionally avoids any semantic processing.
"""

import re

from app.ingestion.parsers.base_parser import BaseParser
from app.ingestion.schemas.document_schema import IngestedDocument


class DocumentParser(BaseParser):
    """
    Performs document-level normalization.

    Input:
        Raw IngestedDocument

    Output:
        Normalized IngestedDocument
    """

    def parse(self, document: IngestedDocument) -> IngestedDocument:
        """
        Normalize document content.
        """

        text = document.content

        # ----------------------------------------
        # Normalize Windows / Linux newlines
        # ----------------------------------------

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # ----------------------------------------
        # Replace tabs with spaces
        # ----------------------------------------

        text = text.replace("\t", " ")

        # ----------------------------------------
        # Collapse multiple spaces
        # ----------------------------------------

        text = re.sub(r"[ ]{2,}", " ", text)

        # ----------------------------------------
        # Collapse excessive blank lines
        # ----------------------------------------

        text = re.sub(r"\n{3,}", "\n\n", text)

        # ----------------------------------------
        # Strip whitespace from every line
        # ----------------------------------------

        lines = [line.strip() for line in text.split("\n")]

        text = "\n".join(lines)

        # ----------------------------------------
        # Trim entire document
        # ----------------------------------------

        text = text.strip()

        # ----------------------------------------
        # Copy metadata
        # ----------------------------------------

        metadata = document.metadata.copy()

        metadata["normalized"] = True
        metadata["character_count"] = len(text)
        metadata["line_count"] = len(text.splitlines())

        # ----------------------------------------
        # Return NEW document
        # ----------------------------------------

        return IngestedDocument(
            document_id=document.document_id,
            source=document.source,
            source_file=document.source_file,
            content=text,
            metadata=metadata,
        )