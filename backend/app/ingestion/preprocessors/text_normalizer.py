#backend/app/ingestion/preprocessors/text_normalizer.py
"""
Text Normalizer

Purpose
-------
Standardizes cleaned text into a consistent format.

Unlike TextCleaner, this class changes text representation
without changing its semantic meaning.

Features
--------
- Unicode normalization
- Lowercase conversion (optional)
- Smart quote normalization
- Dash normalization
- Ellipsis normalization
- Optional punctuation removal
"""

import re
import string
import unicodedata


class TextNormalizer:
    """
    Standardizes text for downstream NLP pipelines.
    """

    def normalize(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = False,
    ) -> str:
        """
        Normalize text.

        Parameters
        ----------
        text : str
            Cleaned text.

        lowercase : bool
            Convert text to lowercase.

        remove_punctuation : bool
            Remove punctuation characters.

        Returns
        -------
        str
            Normalized text.
        """

        if not text:
            return ""

        # ---------------------------------------
        # Unicode normalization
        # ---------------------------------------
        text = unicodedata.normalize("NFKC", text)

        # ---------------------------------------
        # Normalize quotation marks
        # ---------------------------------------
        replacements = {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
            "…": "...",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # ---------------------------------------
        # Lowercase (optional)
        # ---------------------------------------
        if lowercase:
            text = text.lower()

        # ---------------------------------------
        # Remove punctuation (optional)
        # ---------------------------------------
        if remove_punctuation:

            punctuation = string.punctuation

            text = re.sub(
                f"[{re.escape(punctuation)}]",
                "",
                text,
            )

        # ---------------------------------------
        # Collapse multiple spaces
        # ---------------------------------------
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip()