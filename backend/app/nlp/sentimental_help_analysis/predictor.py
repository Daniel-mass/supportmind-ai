"""
Sentiment Predictor.

Uses the pretrained CardiffNLP Twitter RoBERTa sentiment model.

Model:
cardiffnlp/twitter-roberta-base-sentiment-latest
"""

from __future__ import annotations

from typing import Any

from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SentimentPredictor:
    """Inference wrapper for the pretrained sentiment model."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.model_name = model_name

        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name,
        )

    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict sentiment for a single text.

        Returns:
            {
                "text": ...,
                "sentiment_label": ...,
                "sentiment_confidence": ...
            }
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text must not be empty.")

        result = self.pipeline(text)[0]

        return {
            "text": text,
            "sentiment_label": str(result["label"]),
            "sentiment_confidence": float(result["score"]),
        }