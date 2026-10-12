"""
base_loader.py

Base loader for all SupportMind AI ingestion sources.

A loader may return:
    - one IngestedDocument
    - multiple IngestedDocument objects

This allows structured sources such as CSV files to expose
individual records while preserving a universal loader interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.schemas.document_schema import IngestedDocument


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    """

    @abstractmethod
    def load(
        self,
        file_path: Path,
    ) -> IngestedDocument | list[IngestedDocument]:
        """
        Load a source and return one or more standardized documents.

        Examples
        --------
        PDF:
            IngestedDocument

        DOCX:
            IngestedDocument

        CSV:
            list[IngestedDocument]

        Email:
            IngestedDocument
        """
        pass