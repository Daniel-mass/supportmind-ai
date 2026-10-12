"""
#tools.py
Classification Agent tools.

These tools are thin adapters around the existing SupportMind NLP
predictors.

Responsibilities:
- Validate tool input.
- Invoke the appropriate predictor.
- Convert predictor output into ClassificationPrediction.

These tools do NOT:
- perform LLM reasoning
- decide which predictor should run
- resolve prediction conflicts
- make escalation decisions
- execute customer-impacting actions
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from app.agents.classification_agent.contracts.state import (
    ClassificationPrediction,
    ClassificationType,
)


class TicketCategoryPredictorProtocol(Protocol):
    """Interface required by the ticket category tool."""

    def predict(self, text: str) -> dict[str, Any]:
        ...


class PriorityDetectionPredictorProtocol(Protocol):
    """Interface required by the priority detection tool."""

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> dict[str, Any]:
        ...


class ConversationComplexityPredictorProtocol(Protocol):
    """Interface required by the conversation complexity tool."""

    def predict(self, text: str) -> dict[str, Any]:
        ...


class SentimentPredictorProtocol(Protocol):
    """Interface required by the sentiment tool."""

    def predict(self, text: str) -> dict[str, Any]:
        ...


class TopicModelPredictorProtocol(Protocol):
    """Interface required by the topic model tool."""

    def predict(self, text: str) -> dict[str, Any]:
        ...


def _validate_text(text: str) -> str:
    """Validate and normalize text input."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalized = text.strip()

    if not normalized:
        raise ValueError("text must not be empty.")

    return normalized


def _validate_confidence(
    confidence: Any,
) -> float | None:
    """Validate an optional predictor confidence value."""

    if confidence is None:
        return None

    if not isinstance(confidence, (int, float)):
        raise TypeError(
            "confidence must be a numeric value or None."
        )

    confidence = float(confidence)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "confidence must be between 0.0 and 1.0."
        )

    return confidence


class TicketCategoryTool:
    """Tool wrapper for the ticket category predictor."""

    def __init__(
        self,
        predictor: TicketCategoryPredictorProtocol,
    ) -> None:
        self.predictor = predictor

    def predict(
        self,
        text: str,
    ) -> ClassificationPrediction:
        """Run ticket category classification."""

        text = _validate_text(text)

        result = self.predictor.predict(text)

        value = result.get("predicted_category")

        if value is None:
            raise ValueError(
                "Ticket category predictor returned no prediction."
            )

        confidence = _validate_confidence(
            result.get("confidence")
        )

        return ClassificationPrediction(
            classification_type=ClassificationType.CATEGORY,
            value=str(value),
            confidence=confidence,
            source="ticket_category_predictor",
            metadata={
                "input_length": len(text),
            },
        )


class PriorityDetectionTool:
    """Tool wrapper for the priority detection predictor."""

    def __init__(
        self,
        predictor: PriorityDetectionPredictorProtocol,
    ) -> None:
        self.predictor = predictor

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> ClassificationPrediction:
        """Run priority classification."""

        if not isinstance(features, Mapping):
            raise TypeError(
                "features must be a mapping."
            )

        result = self.predictor.predict(features)

        value = result.get("predicted_priority")

        if value is None:
            raise ValueError(
                "Priority predictor returned no prediction."
            )

        confidence = _validate_confidence(
            result.get("confidence")
        )

        probabilities = result.get(
            "probabilities",
            {},
        )

        return ClassificationPrediction(
            classification_type=ClassificationType.PRIORITY,
            value=str(value),
            confidence=confidence,
            source="priority_detection_predictor",
            metadata={
                "probabilities": probabilities,
            },
        )


class ConversationComplexityTool:
    """Tool wrapper for the conversation complexity predictor."""

    def __init__(
        self,
        predictor: ConversationComplexityPredictorProtocol,
    ) -> None:
        self.predictor = predictor

    def predict(
        self,
        text: str,
    ) -> ClassificationPrediction:
        """Run conversation complexity classification."""

        text = _validate_text(text)

        result = self.predictor.predict(text)

        value = result.get("predicted_complexity")

        if value is None:
            raise ValueError(
                "Conversation complexity predictor "
                "returned no prediction."
            )

        confidence = _validate_confidence(
            result.get("confidence")
        )

        return ClassificationPrediction(
            classification_type=ClassificationType.COMPLEXITY,
            value=str(value),
            confidence=confidence,
            source="conversation_complexity_predictor",
            metadata={
                "input_length": len(text),
            },
        )


class SentimentTool:
    """Tool wrapper for the sentiment predictor."""

    def __init__(
        self,
        predictor: SentimentPredictorProtocol,
    ) -> None:
        self.predictor = predictor

    def predict(
        self,
        text: str,
    ) -> ClassificationPrediction:
        """Run sentiment classification."""

        text = _validate_text(text)

        result = self.predictor.predict(text)

        value = result.get("sentiment_label")

        if value is None:
            raise ValueError(
                "Sentiment predictor returned no prediction."
            )

        confidence = _validate_confidence(
            result.get("sentiment_confidence")
        )

        return ClassificationPrediction(
            classification_type=ClassificationType.SENTIMENT,
            value=str(value),
            confidence=confidence,
            source="sentiment_predictor",
            metadata={
                "input_length": len(text),
            },
        )


class TopicModelTool:
    """Tool wrapper for the BERTopic predictor."""

    def __init__(
        self,
        predictor: TopicModelPredictorProtocol,
    ) -> None:
        self.predictor = predictor

    def predict(
        self,
        text: str,
    ) -> ClassificationPrediction:
        """Run topic classification."""

        text = _validate_text(text)

        result = self.predictor.predict(text)

        if "topic_id" not in result:
            raise ValueError(
                "Topic predictor returned no topic ID."
            )

        topic_id = result["topic_id"]

        confidence = _validate_confidence(
            result.get("topic_probability")
        )

        keywords = result.get(
            "keywords",
            [],
        )

        return ClassificationPrediction(
            classification_type=ClassificationType.TOPIC,
            value=str(topic_id),
            confidence=confidence,
            source="topic_model_predictor",
            metadata={
                "keywords": keywords,
            },
        )