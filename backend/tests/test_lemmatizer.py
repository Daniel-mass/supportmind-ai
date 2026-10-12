from app.ingestion.preprocessors.lemmatizer import Lemmatizer


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

    lemmatizer = Lemmatizer()

    lemmas = lemmatizer.lemmatize(tokens)

    print("=" * 70)
    print("Lemmatizer Test")
    print("=" * 70)

    print("Original Tokens")
    print("-" * 70)

    for token in tokens:
        print(token)

    print()

    print("Lemmatized Tokens")
    print("-" * 70)

    for lemma in lemmas:
        print(lemma)

    print()

    print("Comparison")
    print("-" * 70)

    for original, lemma in zip(tokens, lemmas):
        print(f"{original:<15} -> {lemma}")

    print()

    print("=" * 70)


if __name__ == "__main__":
    main()