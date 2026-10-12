"""
Ticket Loader

Loads customer support tickets and converts them
into standardized IngestedDocument objects.
"""

from pathlib import Path

import pandas as pd

from app.ingestion.loaders.base_loader import BaseLoader
from app.ingestion.schemas.document_schema import IngestedDocument


class TicketLoader(BaseLoader):
    """
    Loader specialized for customer support tickets.
    """

    def load(self, file_path: Path) -> list[IngestedDocument]:
        """
        Load all tickets from a CSV file.

        Returns
        -------
        List[IngestedDocument]
        """

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path)

        documents = []

        for _, row in df.iterrows():

            content = f"""
Ticket ID: {row['TicketID']}
Customer: {row['Customer']}
Category: {row['Category']}
Priority: {row['Priority']}
Status: {row['Status']}

Description:
{row['Description']}
""".strip()

            doc = IngestedDocument(
                source="ticket",
                source_file=file_path.name,
                content=content,
                metadata={
                    "ticket_id": row["TicketID"],
                    "customer": row["Customer"],
                    "category": row["Category"],
                    "priority": row["Priority"],
                    "status": row["Status"],
                },
            )

            documents.append(doc)

        return documents