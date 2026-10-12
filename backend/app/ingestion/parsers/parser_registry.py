"""
parser_registry.py

Central registry for all document parsers used by the
SupportMind AI ingestion pipeline.

Unlike LoaderRegistry (which selects ONE loader),
ParserRegistry maintains an ordered execution pipeline.

Author: SupportMind AI
"""

from app.ingestion.parsers.document_parser import DocumentParser
from app.ingestion.parsers.layout_parser import LayoutParser
from app.ingestion.parsers.metadata_parser import MetadataParser
from app.ingestion.parsers.table_parser import TableParser
from app.ingestion.parsers.ticket_parser import TicketParser
from app.ingestion.parsers.email_parser import EmailParser


class ParserRegistry:
    """
    Stores parser instances used by the parsing pipeline.
    """

    def __init__(self):

        self.document_parser = DocumentParser()

        self.layout_parser = LayoutParser()

        self.metadata_parser = MetadataParser()

        self.table_parser = TableParser()

        self.ticket_parser = TicketParser()

        self.email_parser = EmailParser()

    def get_document_parser(self):
        return self.document_parser

    def get_layout_parser(self):
        return self.layout_parser

    def get_metadata_parser(self):
        return self.metadata_parser

    def get_table_parser(self):
        return self.table_parser

    def get_ticket_parser(self):
        return self.ticket_parser

    def get_email_parser(self):
        return self.email_parser