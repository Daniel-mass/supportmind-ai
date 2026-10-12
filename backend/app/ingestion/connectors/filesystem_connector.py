"""
filesystem_connector.py

Production-grade filesystem connector.

Supports loaders that return either:
    - one IngestedDocument
    - multiple IngestedDocument objects
"""

from pathlib import Path
from typing import List

from app.ingestion.loaders.loader_registry import LoaderRegistry
from app.ingestion.schemas.document_schema import IngestedDocument


class FilesystemConnector:
    """
    Connects the local filesystem with the ingestion framework.
    """

    def __init__(self):

        self.registry = LoaderRegistry()

    def ingest_directory(
        self,
        directory: str,
    ) -> List[IngestedDocument]:
        """
        Loads every supported file from a directory.

        A loader may return one document or multiple documents.
        The connector normalizes both cases into one flat list.
        """

        documents: list[IngestedDocument] = []

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        for file_path in directory.rglob("*"):

            if not file_path.is_file():
                continue

            if not self.registry.is_supported(file_path):

                print(
                    f"[SKIP] Unsupported: "
                    f"{file_path.name}"
                )

                continue

            loader = self.registry.get_loader(
                file_path
            )

            if loader is None:

                print(
                    f"[SKIP] No loader registered: "
                    f"{file_path.name}"
                )

                continue

            try:

                loaded = loader.load(file_path)

                # --------------------------------------------------
                # Normalize loader output
                # --------------------------------------------------

                if isinstance(
                    loaded,
                    list,
                ):

                    documents.extend(loaded)

                    print(
                        f"[OK] Loaded: "
                        f"{file_path.name} | "
                        f"documents={len(loaded)}"
                    )

                else:

                    documents.append(loaded)

                    print(
                        f"[OK] Loaded: "
                        f"{file_path.name}"
                    )

            except Exception as e:

                print(
                    f"[ERROR] {file_path.name}: {e}"
                )

                continue

        return documents