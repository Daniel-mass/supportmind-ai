from app.ingestion.chunking.langchain_chunker import LangChainChunker


def main():

    sample = """
SupportMind AI processes customer refunds efficiently.

Customers can request refunds within seven days of cancellation.

Premium plans are non-refundable unless approved by management.

Technical support is available 24x7 through chat and email.

Enterprise customers receive dedicated account managers.

Payment failures should be reported immediately.

SupportMind continuously improves customer experience using AI.
"""

    chunker = LangChainChunker(
        chunk_size=120,
        chunk_overlap=25,
    )

    chunks = chunker.split(sample)

    print("=" * 70)
    print("LangChain Chunker Test")
    print("=" * 70)
    print(f"Chunks Created : {len(chunks)}")
    print("-" * 70)

    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}")
        print(chunk)
        print("-" * 70)


if __name__ == "__main__":
    main()