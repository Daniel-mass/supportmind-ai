##backend/app/ingestion/parsers/metadata_parser.py
"""
Metadata Parser

Extracts structured metadata from documents.

This parser identifies useful metadata such as:

- Emails
- Phone Numbers
- URLs
- Dates
- Version Numbers

These metadata fields are later useful for:

- Metadata Filtering
- Graph-RAG
- Search
- Knowledge Graph Construction
- Enterprise Search
"""

from __future__ import annotations

import re


class MetadataParser:
    """Extract structured metadata from document text."""

    # -----------------------------
    # Regular Expressions
    # -----------------------------

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    PHONE_PATTERN = re.compile(
        r"(?:\+?\d{1,3}[- ]?)?(?:\d{10})"
    )

    URL_PATTERN = re.compile(
        r"https?://[^\s]+"
    )

    DATE_PATTERN = re.compile(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b"
    )

    VERSION_PATTERN = re.compile(
        r"\b(?:v(?:ersion)?\s*)?(\d+\.\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )

    # -----------------------------

    def parse(self, text: str) -> dict:
        """
        Extract metadata from text.

        Parameters
        ----------
        text : str

        Returns
        -------
        dict
        """

        metadata = {
            "emails": sorted(set(self.EMAIL_PATTERN.findall(text))),
            "phones": sorted(set(self.PHONE_PATTERN.findall(text))),
            "urls": sorted(set(self.URL_PATTERN.findall(text))),
            "dates": sorted(set(self.DATE_PATTERN.findall(text))),
            "versions": sorted(set(self.VERSION_PATTERN.findall(text))),
        }

        metadata["email_count"] = len(metadata["emails"])
        metadata["phone_count"] = len(metadata["phones"])
        metadata["url_count"] = len(metadata["urls"])
        metadata["date_count"] = len(metadata["dates"])
        metadata["version_count"] = len(metadata["versions"])

        return metadata