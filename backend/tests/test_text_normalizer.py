from app.ingestion.preprocessors.text_normalizer import TextNormalizer


def main():

    sample = """
“SupportMind AI” — REFUND Policy…

Customer’s refund has NOT arrived!!

Amount: ₹500.
"""

    normalizer = TextNormalizer()

    print("=" * 70)
    print("Original")
    print("=" * 70)
    print(sample)

    print()

    print("=" * 70)
    print("Normalized (Default)")
    print("=" * 70)

    result = normalizer.normalize(sample)

    print(result)

    print()

    print("=" * 70)
    print("Normalized (Remove Punctuation)")
    print("=" * 70)

    result = normalizer.normalize(
        sample,
        lowercase=True,
        remove_punctuation=True,
    )

    print(result)

    print("=" * 70)


if __name__ == "__main__":
    main()