"""
Conversation Complexity Predictor.

Loads the trained SupportMind conversation complexity DistilBERT model
and performs inference on ticket/conversation text.

Labels:
- SIMPLE
- MULTI_TURN
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "supportmind_conversation_complexity"
)


class ConversationComplexityPredictor:
    """Inference wrapper for the conversation complexity model."""

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    @property
    def labels(self) -> dict[int, str]:
        """Return the model's ID-to-label mapping."""
        return {
            int(label_id): str(label)
            for label_id, label in self.model.config.id2label.items()
        }

    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict conversation complexity for a single text.

        Returns:
            {
                "text": ...,
                "predicted_complexity": ...,
                "confidence": ...
            }
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text must not be empty.")

        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = self.model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=-1,
        ).item()

        confidence = float(
            probabilities[0, predicted_class].item()
        )

        label = self.model.config.id2label[
            predicted_class
        ]

        return {
            "text": text,
            "predicted_complexity": str(label),
            "confidence": confidence,
        }