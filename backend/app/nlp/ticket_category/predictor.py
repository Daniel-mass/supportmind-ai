"""
Ticket Category Predictor.

Loads the trained SupportMind Ticket Category model artifacts and performs
inference on new ticket text.

Model artifacts:
- classifier.joblib
- word_tfidf_vectorizer.joblib
- char_tfidf_vectorizer.joblib
- label_encoder.joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from scipy.sparse import hstack


MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "supportmind_ticket_category"
)


class TicketCategoryPredictor:
    """Inference wrapper for the SupportMind ticket category classifier."""

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR

        classifier_path = self.model_dir / "classifier.joblib"
        word_vectorizer_path = self.model_dir / "word_tfidf_vectorizer.joblib"
        char_vectorizer_path = self.model_dir / "char_tfidf_vectorizer.joblib"
        label_encoder_path = self.model_dir / "label_encoder.joblib"

        self.classifier = joblib.load(classifier_path)
        self.word_vectorizer = joblib.load(word_vectorizer_path)
        self.char_vectorizer = joblib.load(char_vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)

    @property
    def classes(self) -> list[str]:
        """Return the classes known by the classifier."""
        return [str(value) for value in self.classifier.classes_]

    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict the ticket category for a single ticket.

        Returns:
            {
                "text": ...,
                "predicted_category": ...,
                "confidence": ...
            }
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text must not be empty.")

        word_features = self.word_vectorizer.transform([text])
        char_features = self.char_vectorizer.transform([text])

        combined_features = hstack(
            [word_features, char_features]
        ).tocsr()

        prediction = self.classifier.predict(combined_features)[0]
        probabilities = self.classifier.predict_proba(combined_features)[0]

        confidence = float(probabilities.max())

        return {
            "text": text,
            "predicted_category": str(prediction),
            "confidence": confidence,
        }