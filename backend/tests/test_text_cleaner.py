from app.ingestion.preprocessors.text_cleaner import TextCleaner


def main():

    raw_text = """
        SupportMind\t\tAI


        Refund      Policy



        Refunds are processed within     7 days.


        Contact\u00A0Support.
    """

    cleaner = TextCleaner()

    cleaned = cleaner.clean(raw_text)

    print("=" * 70)
    print("Before Cleaning")
    print("=" * 70)
    print(raw_text)

    print()

    print("=" * 70)
    print("After Cleaning")
    print("=" * 70)
    print(cleaned)

    print()

    print("=" * 70)
    print("Character Count")
    print("=" * 70)
    print("Before :", len(raw_text))
    print("After  :", len(cleaned))


if __name__ == "__main__":
    main()