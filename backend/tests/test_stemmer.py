from app.ingestion.preprocessors.stemmer import Stemmer


def main():

    tokens = [
        "Customers",
        "were",
        "running",
        "towards",
        "payments",
        "processed",
        "children",
        "better"
    ]

    stemmer = Stemmer()

    result = stemmer.stem(tokens)

    print("=" * 70)
    print("Snowball Stemmer Test")
    print("=" * 70)

    print("Original Tokens")
    print("-" * 70)

    for token in tokens:
        print(token)

    print()

    print("Stemmed Tokens")
    print("-" * 70)

    for token in result:
        print(token)

    print()

    print("Comparison")
    print("-" * 70)

    for original, stemmed in zip(tokens, result):
        print(f"{original:<15} -> {stemmed}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()