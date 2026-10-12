"""
chunker_registry.py

Central registry responsible for managing all available chunking
strategies used throughout SupportMind AI.

Unlike loaders and parsers, chunkers are registered as classes
instead of instantiated objects. This avoids unnecessarily loading
heavy NLP models during application startup.

Responsibilities
----------------
1. Register available chunkers.
2. Create chunker instances on demand.
3. Validate chunking strategies.
4. List supported chunkers.

Author: SupportMind AI
"""

from app.ingestion.chunking.basic_chunker import Chunker
from app.ingestion.chunking.recursive_chunker import RecursiveChunker
from app.ingestion.chunking.langchain_chunker import LangChainChunker
from app.ingestion.chunking.llamaindex_chunker import LlamaIndexChunker
from app.ingestion.chunking.semantic_chunker import SemanticChunker
from app.ingestion.chunking.sentence_window_chunker import SentenceWindowChunker
from app.ingestion.chunking.csv_row_chunker import CSVRowChunker
class ChunkerRegistry:
    """
    Factory responsible for creating chunkers.
    """

    def __init__(self):

        self._registry = {

            "basic": Chunker,

            "recursive": RecursiveChunker,

            "langchain": LangChainChunker,

            "llamaindex": LlamaIndexChunker,

            "semantic": SemanticChunker,

            "sentence_window": SentenceWindowChunker,
            "csv_row": CSVRowChunker,

        }

    def get_chunker(self, strategy: str):

        strategy = strategy.lower()

        if strategy not in self._registry:

            available = ", ".join(sorted(self._registry.keys()))

            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Available strategies: {available}"
            )

        return self._registry[strategy]()

    def supported_chunkers(self):

        return sorted(self._registry.keys())

    def is_supported(self, strategy: str):

        return strategy.lower() in self._registry