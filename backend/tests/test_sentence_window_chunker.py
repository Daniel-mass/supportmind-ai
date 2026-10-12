from app.ingestion.chunking.sentence_window_chunker import (
    SentenceWindowChunker,
)


def main():

    text = """
SupportMind AI processes customer refunds efficiently.

Customers can request refunds within seven days of cancellation.

Premium plans are non-refundable unless approved by management.

Technical support is available 24x7 through chat and email.

Enterprise customers receive dedicated account managers.

Payment failures should be reported immediately.

SupportMind continuously improves customer experience using AI.

Refund investigations normally require payment verification.

Customers receive confirmation emails after successful processing.

Knowledge articles are automatically recommended to support agents.

Escalations are forwarded to Level 2 support.

Machine learning models classify incoming tickets.

Semantic search improves enterprise document retrieval.

Graph-RAG connects customers, products, tickets and resolutions.

Neo4j stores relationships between entities.

Qdrant stores vector embeddings.

FastAPI exposes enterprise APIs.

LangChain orchestrates retrieval pipelines.

LlamaIndex manages indexing and document parsing.
"""

    chunker = SentenceWindowChunker(
        window_size=2,
    )

    nodes = chunker.split(
        text=text,
        metadata={
            "source": "supportmind.txt",
            "document_type": "knowledge_base",
            "department": "Support",
        },
    )

    print("=" * 70)
    print("Sentence Window Chunker Test")
    print("=" * 70)

    print(f"Nodes Created : {len(nodes)}")
    print("-" * 70)

    for i, node in enumerate(nodes, start=1):

        print(f"Node {i}")
        print(f"Node ID    : {node.node_id}")
        print(f"Characters : {len(node.text)}")
        print(f"Metadata   : {node.metadata}")
        print()

        print("Sentence")
        print("-" * 20)
        print(node.text)

        print()

        print("Window Context")
        print("-" * 20)
        print(node.metadata.get("window"))

        print("-" * 70)


if __name__ == "__main__":
    main()