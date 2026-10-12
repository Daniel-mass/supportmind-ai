import re
from typing import List

from llama_index.core.schema import Document


class RecursiveChunker:

    def __init__(self, chunk_size: int = 300):

        self.chunk_size = chunk_size

    def split(self, document: Document) -> List[str]:

        text = document.text.strip()

        if not text:
            return []

        return self._split_recursive(text)

    def _split_recursive(self, text: str) -> List[str]:

        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if len(paragraphs) > 1:
            return self._merge_recursive(paragraphs)

        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) > 1:
            return self._merge_recursive(sentences)

        words = text.split()

        if len(words) > 1:
            return self._split_words(words)

        return [
            text[i:i+self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]

    def _merge_recursive(self, units: List[str]) -> List[str]:

        chunks = []

        current = ""

        for unit in units:

            candidate = (
                current + "\n\n" + unit
                if current
                else unit
            )

            if len(candidate) <= self.chunk_size:

                current = candidate

            else:

                if current:
                    chunks.extend(self._split_recursive(current))

                current = unit

        if current:
            chunks.extend(self._split_recursive(current))

        return chunks

    def _split_words(self, words: List[str]) -> List[str]:

        chunks = []

        current = ""

        for word in words:

            candidate = (
                current + " " + word
                if current
                else word
            )

            if len(candidate) <= self.chunk_size:

                current = candidate

            else:

                chunks.append(current)

                current = word

        if current:
            chunks.append(current)

        return chunks