"""
loader_registry.py

Central registry responsible for mapping file extensions
to their respective loader classes.

This follows the Factory Pattern so that the connector
does not need to know about individual loaders.

Author: SupportMind AI
"""
#backend/app/loaders/loader_registry.py
from pathlib import Path

from app.ingestion.loaders.txt_loader import TXTLoader
from app.ingestion.loaders.csv_loader import CSVLoader
from app.ingestion.loaders.json_loader import JSONLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.docx_loader import DOCXLoader
from app.ingestion.loaders.ticket_loader import TicketLoader
from app.ingestion.loaders.email_loader import EmailLoader


class LoaderRegistry:
    """
    Maps file extensions to loader instances.

    Example
    -------
    ".pdf"  -> PDFLoader()

    ".csv"  -> CSVLoader()

    ".txt"  -> TXTLoader()
    """

    def __init__(self):

        self._registry = {

            ".txt": TXTLoader(),

            ".csv": CSVLoader(),

            ".json": JSONLoader(),

            ".pdf": PDFLoader(),

            ".docx": DOCXLoader(),

            ".tickets": TicketLoader(),

            ".eml": EmailLoader(),

        }

    def get_loader(self, file_path: str):
        """
        Returns the appropriate loader based on file extension.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        BaseLoader | None
        """

        extension = Path(file_path).suffix.lower()

        return self._registry.get(extension)

    def supported_extensions(self):
        """
        Returns all supported extensions.
        """

        return list(self._registry.keys())

    def is_supported(self, file_path: str):
        """
        Checks whether a file is supported.
        """

        extension = Path(file_path).suffix.lower()

        return extension in self._registry