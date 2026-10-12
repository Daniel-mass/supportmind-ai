from app.ingestion.parsers.document_parser import DocumentParser
from app.ingestion.parsers.layout_parser import LayoutParser
from app.ingestion.schemas.document_schema import IngestedDocument


def main():

    raw = IngestedDocument(

        source="txt",

        source_file="knowledge.txt",

        content="""
SupportMind AI

Enterprise Knowledge Base

Refund Policy

Refunds are processed within 7 days.

Technical Issues

Restart the application.

Restart your browser.

Contact Support

support@supportmind.ai
""",

        metadata={}
    )

    document_parser = DocumentParser()

    layout_parser = LayoutParser()

    normalized = document_parser.parse(raw)

    parsed = layout_parser.parse(normalized)

    print("=" * 70)
    print("Layout Parser Test")
    print("=" * 70)

    print()

    print("Detected Headings")

    print("-" * 70)

    for heading in parsed.metadata["layout"]["headings"]:

        print(heading)

    print()

    print("=" * 70)

    print("Layout Metadata")

    print("=" * 70)

    for key, value in parsed.metadata["layout"].items():

        print(f"{key}: {value}")

    print()

    print("=" * 70)


if __name__ == "__main__":
    main()