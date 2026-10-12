from app.ingestion.preprocessors.word_tokenizer import WordTokenizer


def main():

    tokenizer = WordTokenizer()

    sample = """
SupportMind AI processes refunds within 7 days.

Contact support@supportmind.ai.

Version 2.4.1 is now available!

Can I cancel my subscription?
"""

    tokens = tokenizer.tokenize(sample)

    print("=" * 70)
    print("Word Tokenizer Test")
    print("=" * 70)

    print(f"Total Tokens : {len(tokens)}")

    print("-" * 70)

    for i, token in enumerate(tokens, start=1):
        print(f"{i:02d}. {token}")

    print("=" * 70)


if __name__ == "__main__":
    main()