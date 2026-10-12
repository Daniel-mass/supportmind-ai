"""
Unit tests for Classification Agent tools.

The tools are tested independently from the real ML models.
Real predictor loading/inference was already validated by the NLP
predictor integration test.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.classification_agent.contracts.state import (
    ClassificationType,
)
from app.agents.classification_agent.tools.classification_tools import (
    ConversationComplexityTool,
    PriorityDetectionTool,
    SentimentTool,
    TicketCategoryTool,
    TopicModelTool,
)


class FakeTicketCategoryPredictor:
    def predict(self, text: str) -> dict[str, Any]:
        return {
            "predicted_category": "Software",
            "confidence": 0.82,
        }


class FakePriorityPredictor:
    def predict(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "predicted_priority": "High",
            "confidence": 0.73,
            "probabilities": {
                "Blocker": 0.05,
                "High": 0.73,
                "Medium": 0.15,
                "Low": 0.07,
            },
        }


class FakeComplexityPredictor:
    def predict(self, text: str) -> dict[str, Any]:
        return {
            "predicted_complexity": "MULTI_TURN",
            "confidence": 0.91,
        }


class FakeSentimentPredictor:
    def predict(self, text: str) -> dict[str, Any]:
        return {
            "sentiment_label": "LABEL_0",
            "sentiment_confidence": 0.88,
        }


class FakeTopicPredictor:
    def predict(self, text: str) -> dict[str, Any]:
        return {
            "topic_id": 14,
            "topic_probability": 0.67,
            "keywords": [
                "support",
                "issue",
                "product",
            ],
        }


def test_ticket_category_tool() -> None:
    tool = TicketCategoryTool(
        FakeTicketCategoryPredictor()
    )

    result = tool.predict(
        "The application is not responding."
    )

    assert result.classification_type == (
        ClassificationType.CATEGORY
    )
    assert result.value == "Software"
    assert result.confidence == 0.82
    assert result.source == (
        "ticket_category_predictor"
    )


def test_priority_detection_tool() -> None:
    tool = PriorityDetectionTool(
        FakePriorityPredictor()
    )

    features = {
        "issue_proj": "IT",
        "issue_type": "Ticket",
        "created_year": 2026,
        "created_month": 9,
        "created_dayofweek": 6,
        "created_hour": 14,
        "issue_contr_count": 1,
    }

    result = tool.predict(features)

    assert result.classification_type == (
        ClassificationType.PRIORITY
    )
    assert result.value == "High"
    assert result.confidence == 0.73
    assert result.source == (
        "priority_detection_predictor"
    )

    assert result.metadata["probabilities"]["High"] == 0.73


def test_conversation_complexity_tool() -> None:
    tool = ConversationComplexityTool(
        FakeComplexityPredictor()
    )

    result = tool.predict(
        "I contacted support several times."
    )

    assert result.classification_type == (
        ClassificationType.COMPLEXITY
    )
    assert result.value == "MULTI_TURN"
    assert result.confidence == 0.91


def test_sentiment_tool() -> None:
    tool = SentimentTool(
        FakeSentimentPredictor()
    )

    result = tool.predict(
        "This issue is extremely frustrating."
    )

    assert result.classification_type == (
        ClassificationType.SENTIMENT
    )
    assert result.value == "LABEL_0"
    assert result.confidence == 0.88


def test_topic_model_tool() -> None:
    tool = TopicModelTool(
        FakeTopicPredictor()
    )

    result = tool.predict(
        "The product keeps failing."
    )

    assert result.classification_type == (
        ClassificationType.TOPIC
    )
    assert result.value == "14"
    assert result.confidence == 0.67

    assert result.metadata["keywords"] == [
        "support",
        "issue",
        "product",
    ]


@pytest.mark.parametrize(
    "tool_factory",
    [
        lambda: TicketCategoryTool(
            FakeTicketCategoryPredictor()
        ),
        lambda: ConversationComplexityTool(
            FakeComplexityPredictor()
        ),
        lambda: SentimentTool(
            FakeSentimentPredictor()
        ),
        lambda: TopicModelTool(
            FakeTopicPredictor()
        ),
    ],
)
def test_text_tools_reject_empty_text(
    tool_factory,
) -> None:
    tool = tool_factory()

    with pytest.raises(ValueError):
        tool.predict("   ")


def test_priority_tool_rejects_non_mapping() -> None:
    tool = PriorityDetectionTool(
        FakePriorityPredictor()
    )

    with pytest.raises(TypeError):
        tool.predict("invalid")


def test_confidence_validation() -> None:
    class InvalidConfidencePredictor:
        def predict(self, text: str) -> dict[str, Any]:
            return {
                "predicted_category": "Software",
                "confidence": 1.5,
            }

    tool = TicketCategoryTool(
        InvalidConfidencePredictor()
    )

    with pytest.raises(ValueError):
        tool.predict("Test ticket")


def test_missing_category_prediction() -> None:
    class MissingPrediction:
        def predict(self, text: str) -> dict[str, Any]:
            return {
                "confidence": 0.8,
            }

    tool = TicketCategoryTool(
        MissingPrediction()
    )

    with pytest.raises(ValueError):
        tool.predict("Test ticket")


def test_missing_topic_id() -> None:
    class MissingTopic:
        def predict(self, text: str) -> dict[str, Any]:
            return {
                "topic_probability": 0.8,
                "keywords": [],
            }

    tool = TopicModelTool(
        MissingTopic()
    )

    with pytest.raises(ValueError):
        tool.predict("Test ticket")