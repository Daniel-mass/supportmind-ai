#backend/app/ingestion/preprocessors/language_dector.py
from lingua import Language, LanguageDetectorBuilder


class LanguageDetector:
    """
    Detects the language of a document.

    Uses Lingua because it provides excellent accuracy
    even for short enterprise support tickets.
    """

    def __init__(self):

        self.detector = (
            LanguageDetectorBuilder
            .from_all_languages()
            .build()
        )

    def detect(self, text: str) -> dict:
        """
        Detect language.

        Returns
        -------
        {
            "language": "...",
            "code": "...",
            "confidence": 0.xx
        }
        """

        if not text.strip():

            return {
                "language": "Unknown",
                "code": "unknown",
                "confidence": 0.0
            }

        language = self.detector.detect_language_of(text)

        if language is None:

            return {
                "language": "Unknown",
                "code": "unknown",
                "confidence": 0.0
            }

        confidence = (
            self.detector
            .compute_language_confidence_values(text)
        )

        score = 0.0

        for item in confidence:

            if item.language == language:
                score = round(item.value, 4)
                break

        return {
            "language": language.name.title(),
            "code": language.iso_code_639_1.name.lower(),
            "confidence": score
        }