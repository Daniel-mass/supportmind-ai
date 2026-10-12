from typing import List
from llama_index.core.schema import Document


class Chunker:

    def __init__(
        self,
        chunk_size: int = 100,
        overlap: int = 20,
    ):

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> List[str]:

        text = document.text

        words = text.split()

        if not words:
            return []

        chunks = []

        step = self.chunk_size - self.overlap

        for start in range(0, len(words), step):

            end = start + self.chunk_size

            chunk = words[start:end]

            if chunk:
                chunks.append(" ".join(chunk))

        return chunks