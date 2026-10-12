"""
Real integration test for SupportMind NLP predictors.

This test verifies that the actual trained/pretrained predictor artifacts
can be loaded and perform inference successfully.

Covered predictors:
- Ticket Category
- Priority Detection
- Conversation Complexity
- Sentiment
- BERTopic

This is intentionally NOT a unit test.
No mocks are used.
"""

from __future__ import annotations

import math

import pytest

from app.nlp.conversation_intelligence.predictor import (
    ConversationComplexityPredictor,
)
from app.nlp.priority_detection.predictor import (
    PriorityDetectionPredictor,
)
from app.nlp.sentimental_help_analysis.predictor import SentimentPredictor
from app.nlp.ticket_category.predictor import TicketCategoryPredictor
from app.nlp.topic_modeling.predictor import TopicModelPredictor


@pytest.fixture(scope="module")
def ticket_category_predictor() -> TicketCategoryPredictor:
    return TicketCategoryPredictor()


@pytest.fixture(scope="module")
def priority_predictor() -> PriorityDetectionPredictor:
    return PriorityDetectionPredictor()


@pytest.fixture(scope="module")
def complexity_predictor() -> ConversationComplexityPredictor:
    return ConversationComplexityPredictor()


@pytest.fixture(scope="module")
def sentiment_predictor() -> SentimentPredictor:
    return SentimentPredictor()


@pytest.fixture(scope="module")
def topic_predictor() -> TopicModelPredictor:
    return TopicModelPredictor()


def test_ticket_category_predictor(
    ticket_category_predictor: TicketCategoryPredictor,
) -> None:
    """Verify real ticket category inference."""

    result = ticket_category_predictor.predict(
        "My laptop application is extremely slow."
    )

    assert isinstance(result, dict)

    assert "predicted_category" in result
    assert "confidence" in result

    assert isinstance(result["predicted_category"], str)
    assert result["predicted_category"]

    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_priority_detection_predictor(
    priority_predictor: PriorityDetectionPredictor,
) -> None:
    """Verify real priority model inference."""

    features = {
        "issue_proj": "IT",
        "issue_type": "Ticket",
        "created_year": 2026,
        "created_month": 9,
        "created_dayofweek": 6,
        "created_hour": 14,
        "issue_contr_count": 1,
    }

    result = priority_predictor.predict(features)

    assert isinstance(result, dict)

    assert "predicted_priority" in result
    assert "confidence" in result
    assert "probabilities" in result

    assert isinstance(result["predicted_priority"], str)
    assert result["predicted_priority"]

    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0

    probabilities = result["probabilities"]

    assert isinstance(probabilities, dict)
    assert probabilities

    assert all(
        isinstance(label, str)
        for label in probabilities
    )

    assert all(
        isinstance(probability, float)
        for probability in probabilities.values()
    )

    assert all(
        0.0 <= probability <= 1.0
        for probability in probabilities.values()
    )

    assert math.isclose(
        sum(probabilities.values()),
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    )


def test_conversation_complexity_predictor(
    complexity_predictor: ConversationComplexityPredictor,
) -> None:
    """Verify real conversation complexity inference."""

    result = complexity_predictor.predict(
        "I have contacted support several times "
        "and the issue is still unresolved."
    )

    assert isinstance(result, dict)

    assert "predicted_complexity" in result
    assert "confidence" in result

    assert isinstance(result["predicted_complexity"], str)
    assert result["predicted_complexity"]

    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_sentiment_predictor(
    sentiment_predictor: SentimentPredictor,
) -> None:
    """Verify real pretrained sentiment inference."""

    result = sentiment_predictor.predict(
        "This issue has been frustrating "
        "and nobody has resolved it."
    )

    assert isinstance(result, dict)

    assert "sentiment_label" in result
    assert "sentiment_confidence" in result

    assert isinstance(result["sentiment_label"], str)
    assert result["sentiment_label"]

    assert isinstance(result["sentiment_confidence"], float)
    assert 0.0 <= result["sentiment_confidence"] <= 1.0


def test_topic_model_predictor(
    topic_predictor: TopicModelPredictor,
) -> None:
    """Verify real BERTopic + FastEmbed inference."""

    result = topic_predictor.predict(
        "The product keeps failing and support "
        "has not resolved the issue."
    )

    assert isinstance(result, dict)

    assert "topic_id" in result
    assert "keywords" in result

    assert isinstance(result["topic_id"], int)

    assert isinstance(result["keywords"], list)

    assert all(
        isinstance(keyword, str)
        for keyword in result["keywords"]
    )

    if result["topic_probability"] is not None:
        assert isinstance(
            result["topic_probability"],
            float,
        )

        assert 0.0 <= result["topic_probability"] <= 1.0