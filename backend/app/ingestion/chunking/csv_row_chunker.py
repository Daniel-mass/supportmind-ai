from __future__ import annotations

"""
csv_row_chunker.py

CSV row-level chunker.

Each CSV row is treated as one independent retrieval unit.

This is designed for structured support-ticket data where
customer, product, issue, status, resolution, and other fields
belong to the same record.

SupportMind AI
"""

from llama_index.core.schema import Document, TextNode


class CSVRowChunker:
    """
    Chunk CSV-derived document content at row boundaries.

    The CSVLoader converts the CSV into newline-separated text.
    The first line is the CSV header, so it is excluded from
    retrieval chunks.

    Each remaining line represents one support ticket and
    becomes one independent TextNode.

    Example
    -------
    Input:

        Ticket ID,Customer Name,Customer Email,...
        1,Marisa Obrien,...
        2,Jessica Rios,...

    Output:

        TextNode(ticket 1)
        TextNode(ticket 2)
    """

    def split(
        self,
        document: Document,
    ) -> list[TextNode]:

        if not document.text.strip():
            return []

        # Split the document into non-empty lines.
        lines = [
            row.strip()
            for row in document.text.splitlines()
            if row.strip()
        ]

        # The first line is the CSV header.
        # It describes the columns and is not a support ticket.
        rows = lines

        total = len(rows)

        nodes: list[TextNode] = []

        for index, row in enumerate(rows):

            node = TextNode(
                text=row,
                metadata=document.metadata.copy(),
            )

            node.metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": total,
                    "character_count": len(row),
                    "word_count": len(row.split()),
                }
            )

            nodes.append(node)

        return nodes