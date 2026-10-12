"""
JSON document loader.

Reads JSON files and converts structured JSON data
into a standardized IngestedDocument.
"""

import json
from pathlib import Path

from app.ingestion.loaders.base_loader import BaseLoader
from app.ingestion.schemas.document_schema import IngestedDocument


class JSONLoader(BaseLoader):
    """
    Loader for JSON documents.
    """

    def load(self, file_path: Path) -> IngestedDocument:
        """
        Load a JSON document and convert it into
        a readable text representation.
        """

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        content = self._json_to_text(data)

        return IngestedDocument(
            source="json",
            source_file=file_path.name,
            content=content,
            metadata={
                "extension": file_path.suffix,
                "file_size": file_path.stat().st_size,
                "root_type": type(data).__name__,
            },
        )

    def _json_to_text(self, obj, indent: int = 0) -> str:
        """
        Recursively converts JSON into
        a readable text format.
        """

        lines = []

        prefix = "  " * indent

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")

        elif isinstance(obj, list):
            for index, item in enumerate(obj, start=1):
                lines.append(f"{prefix}Item {index}:")
                lines.append(self._json_to_text(item, indent + 1))

        else:
            lines.append(f"{prefix}{obj}")

        return "\n".join(lines)