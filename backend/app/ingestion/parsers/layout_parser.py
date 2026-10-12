#backend/app/ingestion/parsers/layout_parser.py
"""
Layout Parser

Detects simple document structure.

Responsibilities
----------------
- Detect headings
- Count paragraphs
- Preserve document content
- Enrich metadata

Future versions may use:
- Docling
- LayoutParser
- LlamaParse
- OCR Layout Detection
"""

from app.ingestion.parsers.base_parser import BaseParser
from app.ingestion.schemas.document_schema import IngestedDocument


class LayoutParser(BaseParser):

    """
    Detects simple structural elements inside documents.
    """

    MAX_HEADING_WORDS = 8

    def parse(self, document: IngestedDocument) -> IngestedDocument:

        lines = document.content.splitlines()

        headings = []

        paragraph_count = 0

        previous_blank = True

        for line in lines:

            line = line.strip()

            if not line:
                previous_blank = True
                continue

            words = line.split()

            is_heading = (
                len(words) <= self.MAX_HEADING_WORDS
                and not line.endswith(".")
            )

            if is_heading:
                headings.append(line)

            if previous_blank:
                paragraph_count += 1

            previous_blank = False

        metadata = document.metadata.copy()

        metadata["layout"] = {

            "headings": headings,

            "heading_count": len(headings),

            "paragraph_count": paragraph_count,

            "total_lines": len(lines)

        }

        return IngestedDocument(

            document_id=document.document_id,

            source=document.source,

            source_file=document.source_file,

            content=document.content,

            metadata=metadata

        )