
"""
Real Classification Agent integration test.

This test connects the Classification Agent to the actual SupportMind
NLP predictors and model artifacts.

It is intentionally gated behind RUN_REAL_CLASSIFICATION=1 because
several predictors load large ML models during initialization.
"""

from __future__ import annotations

import os

import pytest

from app.agents.classification_agent.agents import ClassificationAgent
from app.agents.classification_agent.contracts.state import (
    ClassificationState,
    ClassificationStatus,
    ClassificationType,
    PriorityFeatures,
)
from app.agents.classification_agent.tools.classification_tools import (
    ConversationComplexityTool,
    PriorityDetectionTool,
    SentimentTool,
    TicketCategoryTool,
    TopicModelTool,
)
from app.nlp.conversation_intelligence.predictor import (
    ConversationComplexityPredictor,
)
from app.nlp.priority_detection.predictor import (
    PriorityDetectionPredictor,
)
from app.nlp.sentimental_help_analysis.predictor import (
    SentimentPredictor,
)
from app.nlp.ticket_category.predictor import (
    TicketCategoryPredictor,
)
from app.nlp.topic_modeling.predictor import (
    TopicModelPredictor,
)


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_CLASSIFICATION") != "1",
    reason="Real Classification Agent test disabled.",
)


def test_real_classification_agent() -> None:
    """
    Run the Classification Agent against all five real predictors.
    """

    # ------------------------------------------------------------------
    # 1. Load the real predictors
    # ------------------------------------------------------------------

    category_predictor = TicketCategoryPredictor()
    priority_predictor = PriorityDetectionPredictor()
    sentiment_predictor = SentimentPredictor()
    complexity_predictor = ConversationComplexityPredictor()
    topic_predictor = TopicModelPredictor()

    # ------------------------------------------------------------------
    # 2. Wrap the predictors with the actual classification tools
    # ------------------------------------------------------------------

    category_tool = TicketCategoryTool(
        predictor=category_predictor,
    )

    priority_tool = PriorityDetectionTool(
        predictor=priority_predictor,
    )

    sentiment_tool = SentimentTool(
        predictor=sentiment_predictor,
    )

    complexity_tool = ConversationComplexityTool(
        predictor=complexity_predictor,
    )

    topic_tool = TopicModelTool(
        predictor=topic_predictor,
    )

    # ------------------------------------------------------------------
    # 3. Create the real Classification Agent
    # ------------------------------------------------------------------

    agent = ClassificationAgent(
        category_tool=category_tool,
        priority_tool=priority_tool,
        sentiment_tool=sentiment_tool,
        complexity_tool=complexity_tool,
        topic_tool=topic_tool,
    )

    # ------------------------------------------------------------------
    # 4. Build a realistic classification request
    # ------------------------------------------------------------------

    state = ClassificationState(
        request_id="real-classification-test-001",
        ticket_text=(
            "I was charged twice for the same payment. "
            "The amount was deducted from my account twice and "
            "I need the duplicate charge resolved."
        ),
        priority_features=PriorityFeatures(
            issue_proj="billing",
            issue_type="payment",
            created_year=2026,
            created_month=9,
            created_dayofweek=6,
            created_hour=10,
            issue_contr_count=1,
        ),
    )

    # ------------------------------------------------------------------
    # 5. Execute the real Classification Agent
    # ------------------------------------------------------------------

    result = agent.run(state)

    # ------------------------------------------------------------------
    # 6. Validate workflow completion
    # ------------------------------------------------------------------

    assert result.status is ClassificationStatus.COMPLETED
    assert result.iteration_count == 1

    assert result.predictions.predictions_requested == 5
    assert result.predictions.predictions_completed == 5

    assert len(result.completed_predictions) == 5
    assert not result.errors

    # ------------------------------------------------------------------
    # 7. Validate every classification result exists
    # ------------------------------------------------------------------

    assert result.predictions.category is not None
    assert result.predictions.priority is not None
    assert result.predictions.sentiment is not None
    assert result.predictions.complexity is not None
    assert result.predictions.topic is not None

    # ------------------------------------------------------------------
    # 8. Validate prediction contracts
    # ------------------------------------------------------------------

    assert (
        result.predictions.category.classification_type
        is ClassificationType.CATEGORY
    )

    assert (
        result.predictions.priority.classification_type
        is ClassificationType.PRIORITY
    )

    assert (
        result.predictions.sentiment.classification_type
        is ClassificationType.SENTIMENT
    )

    assert (
        result.predictions.complexity.classification_type
        is ClassificationType.COMPLEXITY
    )

    assert (
        result.predictions.topic.classification_type
        is ClassificationType.TOPIC
    )

    # ------------------------------------------------------------------
    # 9. Validate prediction values are actually populated
    # ------------------------------------------------------------------

    assert result.predictions.category.value.strip()
    assert result.predictions.priority.value.strip()
    assert result.predictions.sentiment.value.strip()
    assert result.predictions.complexity.value.strip()
    assert result.predictions.topic.value.strip()

    # ------------------------------------------------------------------
    # 10. Validate confidence values where available
    # ------------------------------------------------------------------

    predictions = [
        result.predictions.category,
        result.predictions.priority,
        result.predictions.sentiment,
        result.predictions.complexity,
        result.predictions.topic,
    ]

    for prediction in predictions:
        assert prediction is not None

        if prediction.confidence is not None:
            assert 0.0 <= prediction.confidence <= 1.0

    # ------------------------------------------------------------------
    # 11. Validate provenance/source information
    # ------------------------------------------------------------------

    assert (
        result.predictions.category.source
        == "ticket_category_predictor"
    )

    assert (
        result.predictions.priority.source
        == "priority_detection_predictor"
    )

    assert (
        result.predictions.sentiment.source
        == "sentiment_predictor"
    )

    assert (
        result.predictions.complexity.source
        == "conversation_complexity_predictor"
    )

    assert (
        result.predictions.topic.source
        == "topic_model_predictor"
    )

    # ------------------------------------------------------------------
    # 12. Validate priority-specific metadata
    # ------------------------------------------------------------------

    priority_metadata = result.predictions.priority.metadata

    assert "probabilities" in priority_metadata

    probabilities = priority_metadata["probabilities"]

    assert isinstance(probabilities, dict)
    assert probabilities

    for label, probability in probabilities.items():
        assert isinstance(label, str)
        assert 0.0 <= probability <= 1.0

    # ------------------------------------------------------------------
    # 13. Validate topic metadata
    # ------------------------------------------------------------------

    topic_metadata = result.predictions.topic.metadata

    assert "keywords" in topic_metadata

    assert isinstance(
        topic_metadata["keywords"],
        list,
    )

    # ------------------------------------------------------------------
    # 14. Print useful real integration output
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("REAL CLASSIFICATION AGENT RESULT")
    print("=" * 80)

    print(f"Status: {result.status.value}")
    print(
        f"Predictions requested: "
        f"{result.predictions.predictions_requested}"
    )
    print(
        f"Predictions completed: "
        f"{result.predictions.predictions_completed}"
    )
    print(
        f"Uncertainty detected: "
        f"{result.predictions.uncertainty_detected}"
    )

    print("\nPredictions")
    print("-" * 80)

    print(
        "Category:",
        result.predictions.category.value,
        f"(confidence={result.predictions.category.confidence})",
    )

    print(
        "Priority:",
        result.predictions.priority.value,
        f"(confidence={result.predictions.priority.confidence})",
    )

    print(
        "Sentiment:",
        result.predictions.sentiment.value,
        f"(confidence={result.predictions.sentiment.confidence})",
    )

    print(
        "Complexity:",
        result.predictions.complexity.value,
        f"(confidence={result.predictions.complexity.confidence})",
    )

    print(
        "Topic:",
        result.predictions.topic.value,
        f"(confidence={result.predictions.topic.confidence})",
    )

    print("=" * 80)

