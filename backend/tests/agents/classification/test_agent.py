from __future__ import annotations

import pytest

from app.agents.classification_agent.agents import ClassificationAgent
from app.agents.classification_agent.contracts.state import (
    ClassificationPrediction,
    ClassificationState,
    ClassificationStatus,
    ClassificationType,
    PriorityFeatures,
)


class FakeCategoryTool:
    def predict(self, text: str) -> ClassificationPrediction:
        return ClassificationPrediction(
            classification_type=ClassificationType.CATEGORY,
            value="billing",
            confidence=0.91,
            source="fake_category_predictor",
        )


class FakePriorityTool:
    def predict(
        self,
        features: PriorityFeatures,
    ) -> ClassificationPrediction:
        return ClassificationPrediction(
            classification_type=ClassificationType.PRIORITY,
            value="high",
            confidence=0.88,
            source="fake_priority_predictor",
            metadata={"probabilities": {"high": 0.88, "low": 0.12}},
        )


class FakeSentimentTool:
    def predict(self, text: str) -> ClassificationPrediction:
        return ClassificationPrediction(
            classification_type=ClassificationType.SENTIMENT,
            value="negative",
            confidence=0.94,
            source="fake_sentiment_predictor",
        )


class FakeComplexityTool:
    def predict(self, text: str) -> ClassificationPrediction:
        return ClassificationPrediction(
            classification_type=ClassificationType.COMPLEXITY,
            value="complex",
            confidence=0.82,
            source="fake_complexity_predictor",
        )


class FakeTopicTool:
    def predict(self, text: str) -> ClassificationPrediction:
        return ClassificationPrediction(
            classification_type=ClassificationType.TOPIC,
            value="payment_issue",
            confidence=0.76,
            source="fake_topic_predictor",
        )


def build_agent() -> ClassificationAgent:
    return ClassificationAgent(
        category_tool=FakeCategoryTool(),
        priority_tool=FakePriorityTool(),
        sentiment_tool=FakeSentimentTool(),
        complexity_tool=FakeComplexityTool(),
        topic_tool=FakeTopicTool(),
    )


def build_priority_features() -> PriorityFeatures:
    return PriorityFeatures(
        issue_proj="billing",
        issue_type="payment",
        created_year=2026,
        created_month=9,
        created_dayofweek=6,
        created_hour=21,
        issue_contr_count=1,
    )


def test_agent_runs_all_predictions_by_default() -> None:
    agent = build_agent()

    state = ClassificationState(
        ticket_text="I was charged twice for the same payment.",
        priority_features=build_priority_features(),
    )

    result = agent.run(state)

    assert result.status is ClassificationStatus.COMPLETED
    assert result.predictions.predictions_requested == 5
    assert result.predictions.predictions_completed == 5
    assert len(result.completed_predictions) == 5
    assert not result.errors


def test_agent_runs_only_requested_predictions() -> None:
    agent = build_agent()

    state = ClassificationState(
        ticket_text="I was charged twice.",
        requested_predictions=[
            ClassificationType.CATEGORY,
            ClassificationType.SENTIMENT,
        ],
    )

    result = agent.run(state)

    assert result.predictions.predictions_requested == 2
    assert result.predictions.predictions_completed == 2
    assert result.predictions.category is not None
    assert result.predictions.sentiment is not None
    assert result.predictions.priority is None
    assert result.predictions.complexity is None
    assert result.predictions.topic is None


def test_agent_removes_duplicate_requested_predictions() -> None:
    agent = build_agent()

    state = ClassificationState(
        ticket_text="I was charged twice.",
        requested_predictions=[
            ClassificationType.CATEGORY,
            ClassificationType.CATEGORY,
            ClassificationType.SENTIMENT,
        ],
    )

    result = agent.run(state)

    assert result.predictions.predictions_requested == 2
    assert result.predictions.predictions_completed == 2


def test_agent_executes_priority_with_features() -> None:
    agent = build_agent()

    state = ClassificationState(
        ticket_text="My payment is wrong.",
        requested_predictions=[ClassificationType.PRIORITY],
        priority_features=build_priority_features(),
    )

    result = agent.run(state)

    assert result.predictions.priority is not None
    assert result.predictions.priority.value == "high"
    assert result.predictions.priority.confidence == 0.88
    assert result.predictions.predictions_completed == 1
    assert not result.errors


def test_agent_records_missing_priority_features() -> None:
    agent = build_agent()

    state = ClassificationState(
        ticket_text="My payment is wrong.",
        requested_predictions=[ClassificationType.PRIORITY],
    )

    result = agent.run(state)

    assert result.status is ClassificationStatus.COMPLETED
    assert result.predictions.priority is None
    assert result.predictions.predictions_completed == 0
    assert len(result.errors) == 1
    assert result.errors[0].recoverable is True
    assert "priority" in result.errors[0].message.lower()


def test_agent_continues_when_one_tool_fails() -> None:
    class FailingCategoryTool:
        def predict(self, text: str) -> ClassificationPrediction:
            raise RuntimeError("category model unavailable")

    agent = ClassificationAgent(
        category_tool=FailingCategoryTool(),
        priority_tool=FakePriorityTool(),
        sentiment_tool=FakeSentimentTool(),
        complexity_tool=FakeComplexityTool(),
        topic_tool=FakeTopicTool(),
    )

    state = ClassificationState(
        ticket_text="My payment is wrong.",
        requested_predictions=[
            ClassificationType.CATEGORY,
            ClassificationType.SENTIMENT,
        ],
    )

    result = agent.run(state)

    assert result.status is ClassificationStatus.COMPLETED
    assert result.predictions.category is None
    assert result.predictions.sentiment is not None
    assert result.predictions.predictions_completed == 1
    assert len(result.errors) == 1
    assert result.errors[0].recoverable is True


def test_agent_detects_low_confidence_prediction() -> None:
    class LowConfidenceSentimentTool:
        def predict(self, text: str) -> ClassificationPrediction:
            return ClassificationPrediction(
                classification_type=ClassificationType.SENTIMENT,
                value="negative",
                confidence=0.55,
                source="fake_sentiment_predictor",
            )

    agent = ClassificationAgent(
        category_tool=FakeCategoryTool(),
        priority_tool=FakePriorityTool(),
        sentiment_tool=LowConfidenceSentimentTool(),
        complexity_tool=FakeComplexityTool(),
        topic_tool=FakeTopicTool(),
    )

    state = ClassificationState(
        ticket_text="Something went wrong.",
        requested_predictions=[ClassificationType.SENTIMENT],
    )

    result = agent.run(state)

    assert result.predictions.uncertainty_detected is True


def test_agent_detects_missing_confidence() -> None:
    class NoConfidenceTool:
        def predict(self, text: str) -> ClassificationPrediction:
            return ClassificationPrediction(
                classification_type=ClassificationType.SENTIMENT,
                value="negative",
                confidence=None,
                source="fake_sentiment_predictor",
            )

    agent = ClassificationAgent(
        category_tool=FakeCategoryTool(),
        priority_tool=FakePriorityTool(),
        sentiment_tool=NoConfidenceTool(),
        complexity_tool=FakeComplexityTool(),
        topic_tool=FakeTopicTool(),
    )

    state = ClassificationState(
        ticket_text="Something went wrong.",
        requested_predictions=[ClassificationType.SENTIMENT],
    )

    result = agent.run(state)

    assert result.predictions.uncertainty_detected is True


def test_agent_preserves_prediction_metadata() -> None:
    agent = build_agent()

    state = ClassificationState(
        ticket_text="My payment is wrong.",
        requested_predictions=[ClassificationType.PRIORITY],
        priority_features=build_priority_features(),
    )

    result = agent.run(state)

    assert result.predictions.priority is not None
    assert (
        result.predictions.priority.metadata["probabilities"]["high"]
        == 0.88
    )


def test_agent_increments_iteration_count() -> None:
    agent = build_agent()

    state = ClassificationState(
        ticket_text="My payment is wrong.",
        requested_predictions=[ClassificationType.SENTIMENT],
    )

    assert state.iteration_count == 0

    result = agent.run(state)

    assert result.iteration_count == 1


def test_agent_rejects_invalid_state() -> None:
    agent = build_agent()

    with pytest.raises(TypeError):
        agent.run("invalid state")  # type: ignore[arg-type]


def test_agent_rejects_empty_ticket_text() -> None:
    agent = build_agent()

    state = ClassificationState.model_construct(
        ticket_text="   ",
    )

    with pytest.raises(ValueError, match="ticket_text"):
        agent.run(state)