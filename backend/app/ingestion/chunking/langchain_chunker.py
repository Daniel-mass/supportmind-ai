from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import Document


class LangChainChunker:
    """
    Wrapper around LangChain RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
            or [
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split(self, document: Document) -> list[str]:

        text = document.text.strip()

        if not text:
            return []

        chunks = self.splitter.split_text(text)

        return [chunk.strip() for chunk in chunks if chunk.strip()]