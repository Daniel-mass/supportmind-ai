"""
BERTopic Predictor.

Loads the trained SupportMind BERTopic model and performs topic inference.

Embedding model:
BAAI/bge-base-en-v1.5

Embedding provider:
FastEmbed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import numpy as np
from bertopic import BERTopic
from fastembed import TextEmbedding


MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "supportmind_bertopic"
)

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


class TopicModelPredictor:
    """Inference wrapper for the SupportMind BERTopic model."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR

        self.embedding_model_name = embedding_model_name

        self.embedding_model = TextEmbedding(
            model_name=self.embedding_model_name,
            cache_dir=os.getenv("FASTEMBED_CACHE_DIR"),
        )

        self.topic_model = BERTopic.load(
            str(self.model_dir)
        )

    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict the BERTopic topic for a single document.

        Returns:
            {
                "text": ...,
                "topic_id": ...,
                "topic_probability": ...,
                "keywords": [...]
            }
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text must not be empty.")

        embeddings = np.asarray(
            list(
                self.embedding_model.embed([text])
            )
        )

        topics, probabilities = self.topic_model.transform(
            [text],
            embeddings=embeddings,
        )

        topic_id = int(topics[0])

        topic_probability: float | None = None

        if probabilities is not None:
            probability_array = np.asarray(probabilities)

            if probability_array.ndim == 2:
                if probability_array.shape[0] > 0:
                    topic_probability = float(
                        np.max(probability_array[0])
                    )

            elif probability_array.ndim == 1:
                if probability_array.size > 0:
                    topic_probability = float(
                        np.max(probability_array)
                    )

        keywords: list[str] = []

        topic_words = self.topic_model.get_topic(topic_id)

        if topic_words:
            keywords = [
                str(word)
                for word, _ in topic_words[:10]
            ]

        return {
            "text": text,
            "topic_id": topic_id,
            "topic_probability": topic_probability,
            "keywords": keywords,
        }