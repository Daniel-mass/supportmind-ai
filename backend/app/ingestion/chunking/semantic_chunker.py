from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class SemanticChunker:

    def __init__(
        self,
        embed_model_name="BAAI/bge-small-en-v1.5",
        buffer_size=1,
        breakpoint_percentile_threshold=95,
    ):

        self.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name
        )

        self.parser = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        )

    def split(
        self,
        document: Document,
    ) -> list[TextNode]:

        if not document.text.strip():
            return []

        nodes = self.parser.get_nodes_from_documents([document])

        total = len(nodes)

        for index, node in enumerate(nodes):

            node.metadata["chunk_index"] = index
            node.metadata["total_chunks"] = total
            node.metadata["character_count"] = len(node.text)
            node.metadata["word_count"] = len(node.text.split())

        return nodes