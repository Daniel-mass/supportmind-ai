"""
Email Loader

Loads .eml email files and converts them
into standardized IngestedDocument objects.
"""

from pathlib import Path
from email import policy
from email.parser import BytesParser

from app.ingestion.loaders.base_loader import BaseLoader
from app.ingestion.schemas.document_schema import IngestedDocument


class EmailLoader(BaseLoader):
    """
    Loader for RFC822 email files (.eml).
    """

    def load(self, file_path: Path) -> IngestedDocument:

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with open(file_path, "rb") as f:
            message = BytesParser(policy=policy.default).parse(f)

        body = ""

        if message.is_multipart():

            for part in message.walk():

                if part.get_content_type() == "text/plain":

                    body += part.get_content()

        else:

            body = message.get_content()

        content = f"""
Subject: {message.get('Subject')}

From: {message.get('From')}

To: {message.get('To')}

Date: {message.get('Date')}

Message:

{body.strip()}
""".strip()

        return IngestedDocument(
            source="email",
            source_file=file_path.name,
            content=content,
            metadata={
                "from": message.get("From"),
                "to": message.get("To"),
                "subject": message.get("Subject"),
                "date": message.get("Date"),
            },
        )