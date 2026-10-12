#backend/app/ingestion/parsers/base_parser.py
"""
Base Parser Interface

Every parser in SupportMind AI must inherit from this class.

Input:
    IngestedDocument

Output:
    IngestedDocument

This keeps every parser interchangeable inside the parsing pipeline.
"""

from abc import ABC, abstractmethod

from app.ingestion.schemas.document_schema import IngestedDocument


class BaseParser(ABC):
    """
    Abstract base parser.

    All parser implementations (Document, Semantic, Layout,
    Dependency, Output, etc.) inherit from this class.
    """

    @abstractmethod
    def parse(self, document: IngestedDocument) -> IngestedDocument:
        """
        Parse an ingested document.

        Parameters
        ----------
        document : IngestedDocument
            Document produced by the ingestion layer.

        Returns
        -------
        IngestedDocument
            Parsed / enriched document.
        """
        pass