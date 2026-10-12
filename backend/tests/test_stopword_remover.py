from app.ingestion.preprocessors.stopword_remover import StopwordRemover


def main():

    tokens = [
        "The",
        "customer",
        "has",
        "not",
        "received",
        "the",
        "refund",
        "."
    ]

    remover = StopwordRemover()

    result = remover.remove(tokens)

    print("=" * 70)
    print("Stopword Remover Test")
    print("=" * 70)

    print("Before")
    print("-" * 70)
    print(tokens)

    print()

    print("After")
    print("-" * 70)
    print(result)

    print()

    print(f"Original Tokens : {len(tokens)}")
    print(f"Remaining Tokens: {len(result)}")

    print("=" * 70)


if __name__ == "__main__":
    main()