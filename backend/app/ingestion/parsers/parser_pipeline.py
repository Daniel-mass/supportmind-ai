"""
parser_pipeline.py

Production parsing pipeline.

Receives an IngestedDocument from the loaders and
runs every required parser in the correct order.

Author: SupportMind AI
"""

from app.ingestion.schemas.document_schema import IngestedDocument
from app.ingestion.parsers.parser_registry import ParserRegistry


class ParserPipeline:
    """
    Executes the complete parsing workflow.
    """

    def __init__(self):

        self.registry = ParserRegistry()

    def run(self, document: IngestedDocument) -> IngestedDocument:
        """
        Execute parser pipeline.
        """

        # -------------------------------------------------
        # Stage 1
        # Document normalization
        # -------------------------------------------------

        document = (
            self.registry
            .get_document_parser()
            .parse(document)
        )

        # -------------------------------------------------
        # Stage 2
        # Layout Analysis
        # -------------------------------------------------

        document = (
            self.registry
            .get_layout_parser()
            .parse(document)
        )

        # -------------------------------------------------
        # Stage 3
        # Metadata Extraction
        # -------------------------------------------------

        extracted = (
            self.registry
            .get_metadata_parser()
            .parse(document.content)
        )

        document.metadata.update(extracted)

        # -------------------------------------------------
        # Stage 4
        # Optional Parsers
        # -------------------------------------------------

        source = document.source.lower()

        if source == "csv":

            document.metadata["table"] = (
                self.registry
                .get_table_parser()
                .parse(document.content)
            )

        elif source == "ticket":

            document.metadata["ticket"] = (
                self.registry
                .get_ticket_parser()
                .parse(document.metadata)
            )

        elif source == "email":

            document.metadata["email"] = (
                self.registry
                .get_email_parser()
                .parse(document.content)
            )

        return document