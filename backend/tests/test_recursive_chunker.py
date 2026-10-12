from backend.app.ingestion.chunking.basic_chunker import RecursiveChunker


def main():

    sample = """
SupportMind AI

Refund Policy

Refunds are processed within seven days after cancellation.

Premium subscriptions are non-refundable unless approved by management.

Enterprise customers receive dedicated account managers.

Technical support is available 24x7 through chat and email.

Please contact support@supportmind.ai for additional information.

Thank you.
"""

    chunker = RecursiveChunker(chunk_size=120)

    chunks = chunker.split(sample)

    print("=" * 70)
    print("Recursive Chunker Test")
    print("=" * 70)

    print(f"Chunks Created : {len(chunks)}")

    print("-" * 70)

    for i, chunk in enumerate(chunks, start=1):

        print(f"Chunk {i}")
        print("-" * 30)
        print(chunk)
        print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()