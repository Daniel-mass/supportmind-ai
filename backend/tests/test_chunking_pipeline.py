# backend/tests/test_chunking_pipeline.py

from app.ingestion.chunking.chunking_pipeline import ChunkingPipeline
from app.ingestion.schemas.document_schema import IngestedDocument

document = IngestedDocument(

    source="txt",

    source_file="supportmind.txt",

    content="""
SupportMind AI processes customer refunds efficiently.

Customers can request refunds within seven days.

Premium plans require manager approval.

Knowledge articles improve agent productivity.

GraphRAG connects related tickets.

FastAPI serves enterprise APIs.

LlamaIndex powers document retrieval.
""",

    metadata={
        "department": "Support"
    }

)

pipeline = ChunkingPipeline()

nodes = pipeline.chunk(document)

print("=" * 70)

print("Chunking Pipeline Test")

print("=" * 70)

print("Strategy :", nodes[0].metadata["chunking_strategy"])

print("Chunks :", len(nodes))

print()

for node in nodes:

    print("-" * 60)

    print(node.metadata["chunk_index"])

    print(node.metadata["total_chunks"])

    print(node.metadata["document_id"])

    print(node.metadata["source_file"])

    print()

    print(node.text)

    print()