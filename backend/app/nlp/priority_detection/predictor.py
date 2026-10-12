"""
Priority Detection Predictor.

Loads the trained Help Desk priority detection model and performs inference.

The trained model expects these features:

- issue_proj
- issue_type
- created_year
- created_month
- created_dayofweek
- created_hour
- issue_contr_count
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "priority_detection_model.joblib"
)

REQUIRED_FEATURES = (
    "issue_proj",
    "issue_type",
    "created_year",
    "created_month",
    "created_dayofweek",
    "created_hour",
    "issue_contr_count",
)


class PriorityDetectionPredictor:
    """Inference wrapper for the SupportMind priority detection model."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else MODEL_PATH

        self.model = joblib.load(self.model_path)

    @property
    def classes(self) -> list[str]:
        """Return the priority classes known by the classifier."""
        classifier = self.model.named_steps["classifier"]

        return [str(value) for value in classifier.classes_]

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Predict ticket priority from the required structured features.

        Args:
            features:
                Mapping containing the seven features used during training.

        Returns:
            {
                "predicted_priority": ...,
                "confidence": ...,
                "probabilities": {...}
            }
        """
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping.")

        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in features
        ]

        if missing_features:
            raise ValueError(
                "Missing required priority features: "
                + ", ".join(missing_features)
            )

        input_data = pd.DataFrame(
            [
                {
                    feature: features[feature]
                    for feature in REQUIRED_FEATURES
                }
            ]
        )

        prediction = self.model.predict(input_data)[0]

        probabilities = self.model.predict_proba(input_data)[0]

        classifier = self.model.named_steps["classifier"]
        classes = classifier.classes_

        probability_map = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }

        confidence = float(probabilities.max())

        return {
            "predicted_priority": str(prediction),
            "confidence": confidence,
            "probabilities": probability_map,
        }