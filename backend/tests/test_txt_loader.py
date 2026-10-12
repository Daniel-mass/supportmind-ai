from pathlib import Path

from app.ingestion.loaders.txt_loader import TXTLoader


def main():
    # Path to the sample text file
    sample_file = Path("C:/Users/danie/supportmind-ai/data/raw/documents/sample.txt")

    # Create loader
    loader = TXTLoader()

    # Load document
    document = loader.load(sample_file)

    # Display results
    print("=" * 60)
    print("TXT Loader Test")
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