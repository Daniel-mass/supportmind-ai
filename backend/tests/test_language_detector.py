from app.ingestion.preprocessors.language_detector import LanguageDetector


def main():

    detector = LanguageDetector()

    samples = {
        "English":
            "My refund has not arrived yet.",

        "French":
            "Je n'ai pas reçu mon remboursement.",

        "German":
            "Ich habe meine Rückerstattung nicht erhalten.",

        "Spanish":
            "Todavía no he recibido mi reembolso.",

        "Hindi":
            "मुझे अभी तक मेरा रिफंड नहीं मिला।",

        "Japanese":
            "まだ返金を受け取っていません。",

        "Empty":
            ""
    }

    print("=" * 70)
    print("Language Detector Test")
    print("=" * 70)
    print()

    for expected, text in samples.items():

        result = detector.detect(text)

        print(f"Expected    : {expected}")
        print(f"Detected    : {result['language']}")
        print(f"Code        : {result['code']}")
        print(f"Confidence  : {result['confidence']:.4f}")

        print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()