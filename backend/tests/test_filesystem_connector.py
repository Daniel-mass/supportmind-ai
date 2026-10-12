from app.ingestion.connectors.filesystem_connector import FilesystemConnector


def main():

    connector = FilesystemConnector()

    documents = connector.ingest_directory("C:/Users/danie/supportmind-ai/data/raw")

    print("=" * 70)
    print("Filesystem Connector Test")
    print("=" * 70)

    print(f"Documents Loaded: {len(documents)}")

    print()

    for doc in documents:

        print(f"Source : {doc.source}")

        print(f"File   : {doc.source_file}")

        print(f"Chars  : {len(doc.content)}")

        print(f"Metadata : {doc.metadata}")
        print(f"document_id:{doc.document_id}")

        print("-" * 40)

    print("=" * 70)


if __name__ == "__main__":
    main()