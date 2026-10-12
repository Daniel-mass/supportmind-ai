from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode


class LlamaIndexChunker:

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(
        self,
        document: Document,
    ) -> list[TextNode]:

        if not document.text.strip():
            return []

        return self.splitter.get_nodes_from_documents([document])