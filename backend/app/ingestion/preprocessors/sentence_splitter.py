#backend/app/ingestion/preprocessors/sentence_splitter.py
"""
Sentence Splitter

Purpose
-------
Splits normalized text into individual sentences.

This implementation uses lightweight regex rules and preserves
sentence boundaries for downstream NLP tasks.

Later phases will replace the internal implementation with spaCy
without changing the public interface.
"""

import re


class SentenceSplitter:
    """
    Splits text into sentences.

    Returns a list of clean sentences.
    """

    def split(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Parameters
        ----------
        text : str

        Returns
        -------
        list[str]
        """

        if not text.strip():
            return []

        # Normalize line endings
        text = text.replace("\r\n", "\n")

        # Preserve paragraph boundaries
        paragraphs = text.split("\n\n")

        sentences = []

        # Sentence-ending punctuation followed by whitespace
        pattern = re.compile(r"(?<=[.!?])\s+")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            parts = pattern.split(paragraph)

            for sentence in parts:

                sentence = sentence.strip()

                if sentence:
                    sentences.append(sentence)

        return sentences