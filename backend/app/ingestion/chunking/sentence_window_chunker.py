from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import Document, TextNode


class SentenceWindowChunker:

    def __init__(self, window_size=3):

        self.parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
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

            node.metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": total,
                    "character_count": len(node.text),
                    "word_count": len(node.text.split()),
                }
            )

        return nodes