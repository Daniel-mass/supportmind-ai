from pathlib import Path

from app.ingestion.loaders.csv_loader import CSVLoader


def main():
    sample_file = Path("data/raw/documents/sample.csv")

    loader = CSVLoader()

    document = loader.load(sample_file)

    print("=" * 60)
    print("CSV Loader Test")
    print("=" * 60)
    print(f"Source      : {document.source}")
    print(f"File        : {document.source_file}")
    print(f"Metadata    : {document.metadata}")
    print("-" * 60)
    print("Content:")
    print(document.content)
    print("=" * 60)


if __name__ == "__main__":
    main()