from app.ingestion.loaders.loader_registry import LoaderRegistry


def main():

    registry = LoaderRegistry()

    print("=" * 60)
    print("Loader Registry Test")
    print("=" * 60)

    print("\nSupported Extensions:")

    for ext in registry.supported_extensions():
        print(ext)

    print("\n")

    files = [
        "policy.pdf",
        "tickets.csv",
        "notes.txt",
        "sample.json",
        "knowledge.docx",
        "mail.eml",
        "history.tickets",
        "image.png",
    ]

    for file in files:

        loader = registry.get_loader(file)

        if loader:
            print(f"{file:20} -> {loader.__class__.__name__}")
        else:
            print(f"{file:20} -> Unsupported")

    print("=" * 60)


if __name__ == "__main__":
    main()