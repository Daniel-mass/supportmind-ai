##backend/app/ingestion/parsers/table_parser.py
from __future__ import annotations

import csv
import io


class TableParser:
    """
    Converts CSV-formatted text into structured Python dictionaries.
    """

    def parse(self, text: str) -> list[dict]:
        """
        Parse CSV text into structured rows.
        """

        if not text.strip():
            return []

        # Remove leading/trailing blank lines
        text = text.strip()

        stream = io.StringIO(text)

        reader = csv.DictReader(stream)

        rows = []

        for row in reader:

            cleaned = {}

            for key, value in row.items():

                if key is None:
                    continue

                cleaned[key.strip()] = value.strip() if value else ""

            rows.append(cleaned)

        return rows
