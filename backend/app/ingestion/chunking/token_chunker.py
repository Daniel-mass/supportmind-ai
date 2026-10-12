"""
token_chunker.py

Production Token Chunker using LlamaIndex.

Splits documents based on tokenizer token counts while
returning TextNode objects compatible with downstream
RAG pipelines.

SupportMind AI
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document, TextNode


class TokenChunker:
    """
    Production Token Chunker.

    Uses LlamaIndex TokenTextSplitter.

    Suitable for:

    • Embedding Pipelines
    • Vector Databases
    • RAG
    • Graph-RAG
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ):

        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[TextNode]:
        """
        Split text into token-based TextNodes.

        Parameters
        ----------
        text : str

        metadata : dict | None

        Returns
        -------
        List[TextNode]
        """

        text = text.strip()

        if not text:
            return []

        document = Document(
            text=text,
            metadata=metadata or {},
        )

        nodes = self.splitter.get_nodes_from_documents(
            [document]
        )

        total = len(nodes)

        for index, node in enumerate(nodes):

            node.metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": total,
                    "character_count": len(node.text),
                    "word_count": len(node.text.split()),
                }
            )

        return nodes