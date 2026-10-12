from pathlib import Path

from app.ingestion.loaders.email_loader import EmailLoader


def main():

    loader = EmailLoader()

    document = loader.load(
        Path("data/raw/emails/sample.eml")
    )

    print("=" * 70)
    print("Email Loader Test")
    print("=" * 70)

    print(document.metadata)

    print("-" * 70)

    print(document.content)

    print("=" * 70)


if __name__ == "__main__":
    main()