from app.ingestion.chunking.llamaindex_chunker import (
    LlamaIndexChunker,
)


def main():

    sample = """
SupportMind AI processes customer refunds efficiently.

Customers can request refunds within seven days of cancellation.

Premium plans are non-refundable unless approved by management.

Technical support is available 24x7 through chat and email.

Enterprise customers receive dedicated account managers.

Payment failures should be reported immediately.

SupportMind continuously improves customer experience using AI.

Refund investigations normally require payment verification.

Customers receive confirmation emails after successful processing.

Enterprise accounts receive priority routing.

Subscription changes become effective immediately.

Invoices remain available from the customer portal.

Knowledge articles are automatically recommended to support agents.

Escalations are forwarded to Level 2 support.

Recurring incidents are monitored continuously.

Machine learning models classify incoming tickets.

Semantic search improves document retrieval.

Graph-RAG connects customers, tickets, products and resolutions.

Qdrant stores vector embeddings.

Neo4j stores relationships between entities.

FastAPI exposes REST APIs.

LangChain coordinates retrieval pipelines.

LlamaIndex manages document indexing.
"""

    chunker = LlamaIndexChunker(
        chunk_size=80,
        chunk_overlap=20,
    )

    nodes = chunker.split(
        sample,
        metadata={
            "source": "sample.txt",
            "document_type": "knowledge_base",
            "department": "Support",
        },
    )

    print("=" * 70)
    print("LlamaIndex Chunker Test")
    print("=" * 70)

    print(f"Nodes Created : {len(nodes)}")
    print("-" * 70)

    for index, node in enumerate(nodes, start=1):

        print(f"Node {index}")
        print(f"Node ID    : {node.node_id}")
        print(f"Characters : {len(node.text)}")
        print(f"Metadata   : {node.metadata}")
        print()

        print(node.text)

        print("-" * 70)


if __name__ == "__main__":
    main()