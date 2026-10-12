from app.ingestion.preprocessors.sentence_splitter import SentenceSplitter


def main():

    sample = """
SupportMind AI

Refunds are processed within 7 days.

Customers receive email notifications.

Please contact support@supportmind.ai.

Version 2.4.1 improves ticket routing!

Can I cancel my subscription?

Yes, you can.
"""

    splitter = SentenceSplitter()

    sentences = splitter.split(sample)

    print("=" * 70)
    print("Sentence Splitter Test")
    print("=" * 70)

    print(f"Sentences Found : {len(sentences)}")

    print("-" * 70)

    for i, sentence in enumerate(sentences, start=1):

        print(f"{i}. {sentence}")

    print("=" * 70)


if __name__ == "__main__":
    main()