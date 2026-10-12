from app.ingestion.preprocessors.spell_corrector import SpellCorrector


def main():

    corrector = SpellCorrector()

    tokens = [
        "paymant",
        "failedd",
        "recievd",
        "custmer",
        "SupportMind",
        "Neo4j",
        "LangChain",
        "Qdrant",
        "FastAPI",
        "GraphRAG",
        "12345",
        ".",
        "refund"
    ]

    corrected = corrector.correct(tokens)

    print("=" * 70)
    print("Spell Corrector Test")
    print("=" * 70)

    print("Original Tokens")
    print("-" * 70)

    for token in tokens:
        print(token)

    print()

    print("Corrected Tokens")
    print("-" * 70)

    for token in corrected:
        print(token)

    print()

    print("Comparison")
    print("-" * 70)

    for original, new in zip(tokens, corrected):
        print(f"{original:<15} -> {new}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()