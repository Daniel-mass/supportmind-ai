#backend/app/ingestion/preprocessors/text_cleaner.py
"""
Text Cleaner

Purpose
-------
Cleans raw extracted text while preserving its meaning.

This is the first preprocessing step after document parsing.

Operations:
- Remove tabs
- Remove extra spaces
- Remove excessive blank lines
- Remove invisible unicode characters
- Trim whitespace
"""

import re
import unicodedata


class TextCleaner:
    """
    Cleans raw text extracted from documents.

    This class intentionally performs only basic cleaning.
    It does NOT modify the semantic content.
    """

    def clean(self, text: str) -> str:
        """
        Clean extracted text.

        Parameters
        ----------
        text : str
            Raw extracted text.

        Returns
        -------
        str
            Cleaned text.
        """

        if not text:
            return ""

        # -----------------------------------------
        # Normalize unicode characters
        # -----------------------------------------
        text = unicodedata.normalize("NFKC", text)

        # -----------------------------------------
        # Replace tabs with spaces
        # -----------------------------------------
        text = text.replace("\t", " ")

        # -----------------------------------------
        # Remove non-breaking spaces
        # -----------------------------------------
        text = text.replace("\u00A0", " ")

        # -----------------------------------------
        # Remove zero-width spaces
        # -----------------------------------------
        text = text.replace("\u200B", "")

        # -----------------------------------------
        # Collapse multiple spaces
        # -----------------------------------------
        text = re.sub(r"[ ]{2,}", " ", text)

        # -----------------------------------------
        # Collapse excessive blank lines
        # Maximum: one empty line
        # -----------------------------------------
        text = re.sub(r"\n{3,}", "\n\n", text)

        # -----------------------------------------
        # Remove trailing spaces on each line
        # -----------------------------------------
        text = "\n".join(line.strip() for line in text.splitlines())

        # -----------------------------------------
        # Remove leading/trailing whitespace
        # -----------------------------------------
        text = text.strip()

        return text