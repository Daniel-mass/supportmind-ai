from app.ingestion.parsers.metadata_parser import MetadataParser


sample_document = """
SupportMind AI
Enterprise Knowledge Base

Version 2.4

Created:
2026-07-02

Support Email:
support@supportmind.ai

Finance Email:
finance@supportmind.ai

Website:
https://supportmind.ai

Customer Portal:
https://portal.supportmind.ai

Phone:
+919876543210

Backup Contact:
9876543211

"""


def main():

    parser = MetadataParser()

    metadata = parser.parse(sample_document)

    print("=" * 70)
    print("Metadata Parser Test")
    print("=" * 70)

    for key, value in metadata.items():

        print(f"{key}: {value}")

    print("=" * 70)


if __name__ == "__main__":
    main()