from pathlib import Path

from app.ingestion.loaders.pdf_loader import PDFLoader


def main():
    sample_file = Path("C:/Users/danie/supportmind-ai/data/raw/documents/Overall Result Summary.pdf")

    loader = PDFLoader()

    document = loader.load(sample_file)

    print("=" * 60)
    print("PDF Loader Test")
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