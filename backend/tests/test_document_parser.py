from app.ingestion.parsers.document_parser import DocumentParser
from app.ingestion.schemas.document_schema import IngestedDocument


def main():

    parser = DocumentParser()

    raw_document = IngestedDocument(

        source="txt",

        source_file="sample.txt",

        content="""
        
        
        SupportMind      AI


        Refund      Policy


\tRefunds are processed within     7 days.


Contact      Support.

        
        
        """,

        metadata={}

    )

    print("=" * 70)
    print("Before Parsing")
    print("=" * 70)

    print(raw_document.content)

    print()

    parsed = parser.parse(raw_document)

    print("=" * 70)
    print("After Parsing")
    print("=" * 70)

    print(parsed.content)

    print()

    print("=" * 70)
    print("Metadata")
    print("=" * 70)

    print(parsed.metadata)

    print()

    print("=" * 70)


if __name__ == "__main__":
    main()