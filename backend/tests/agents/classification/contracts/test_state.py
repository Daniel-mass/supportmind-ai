"""
Unit tests for Classification Agent contracts/state.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.classification_agent.contracts.state import (
    ClassificationError,
    ClassificationPrediction,
    ClassificationResult,
    ClassificationState,
    ClassificationStatus,
    ClassificationType,
    PriorityFeatures,
)


def test_classification_status_values() -> None:
    assert ClassificationStatus.PENDING.value == "pending"
    assert ClassificationStatus.RUNNING.value == "running"
    assert ClassificationStatus.COMPLETED.value == "completed"
    assert ClassificationStatus.FAILED.value == "failed"


def test_classification_type_values() -> None:
    assert ClassificationType.CATEGORY.value == "category"
    assert ClassificationType.PRIORITY.value == "priority"
    assert ClassificationType.SENTIMENT.value == "sentiment"
    assert ClassificationType.COMPLEXITY.value == "complexity"
    assert ClassificationType.TOPIC.value == "topic"


def test_priority_features_valid() -> None:
    features = PriorityFeatures(
        issue_proj="IT",
        issue_type="Ticket",
        created_year=2026,
        created_month=9,
        created_dayofweek=6,
        created_hour=14,
        issue_contr_count=1,
    )

    assert features.issue_proj == "IT"
    assert features.issue_type == "Ticket"
    assert features.created_year == 2026
    assert features.issue_contr_count == 1


def test_priority_features_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PriorityFeatures(
            issue_proj="IT",
            issue_type="Ticket",
            created_year=2026,
            created_month=9,
            created_dayofweek=6,
            created_hour=14,
            issue_contr_count=1,
            unexpected_field="invalid",
        )


def test_classification_prediction_valid() -> None:
    prediction = ClassificationPrediction(
        classification_type=ClassificationType.CATEGORY,
        value="Software",
        confidence=0.82,
        source="ticket_category_predictor",
    )

    assert prediction.classification_type == (
        ClassificationType.CATEGORY
    )
    assert prediction.value == "Software"
    assert prediction.confidence == 0.82
    assert prediction.source == "ticket_category_predictor"


def test_classification_prediction_confidence_validation() -> None:
    with pytest.raises(ValidationError):
        ClassificationPrediction(
            classification_type=ClassificationType.PRIORITY,
            value="High",
            confidence=1.5,
            source="priority_detection_predictor",
        )


def test_classification_result_defaults() -> None:
    result = ClassificationResult()

    assert result.category is None
    assert result.priority is None
    assert result.sentiment is None
    assert result.complexity is None
    assert result.topic is None

    assert result.predictions_completed == 0
    assert result.predictions_requested == 0
    assert result.uncertainty_detected is False


def test_classification_result_with_predictions() -> None:
    result = ClassificationResult(
        category=ClassificationPrediction(
            classification_type=ClassificationType.CATEGORY,
            value="Software",
            confidence=0.82,
            source="ticket_category_predictor",
        ),
        sentiment=ClassificationPrediction(
            classification_type=ClassificationType.SENTIMENT,
            value="LABEL_2",
            confidence=0.91,
            source="sentiment_predictor",
        ),
        predictions_completed=2,
        predictions_requested=5,
    )

    assert result.category is not None
    assert result.category.value == "Software"

    assert result.sentiment is not None
    assert result.sentiment.value == "LABEL_2"

    assert result.predictions_completed == 2
    assert result.predictions_requested == 5


def test_classification_error() -> None:
    error = ClassificationError(
        source="priority_detection_predictor",
        error_type="missing_features",
        message="Required priority features are missing.",
        recoverable=True,
    )

    assert error.source == "priority_detection_predictor"
    assert error.error_type == "missing_features"
    assert error.recoverable is True


def test_classification_state_defaults() -> None:
    state = ClassificationState(
        ticket_text="My application is extremely slow.",
    )

    assert state.status == ClassificationStatus.PENDING
    assert state.priority_features is None

    assert state.requested_predictions == []
    assert state.completed_predictions == []

    assert state.predictions.predictions_completed == 0
    assert state.errors == []
    assert state.iteration_count == 0


def test_classification_state_with_priority_features() -> None:
    state = ClassificationState(
        request_id="classification-test-001",
        ticket_text="The production deployment is blocked.",
        priority_features=PriorityFeatures(
            issue_proj="IT",
            issue_type="Deployment",
            created_year=2026,
            created_month=9,
            created_dayofweek=6,
            created_hour=14,
            issue_contr_count=1,
        ),
        requested_predictions=[
            ClassificationType.CATEGORY,
            ClassificationType.PRIORITY,
            ClassificationType.SENTIMENT,
            ClassificationType.COMPLEXITY,
            ClassificationType.TOPIC,
        ],
    )

    assert state.request_id == "classification-test-001"
    assert state.priority_features is not None
    assert state.priority_features.issue_type == "Deployment"

    assert len(state.requested_predictions) == 5


def test_classification_state_rejects_empty_ticket() -> None:
    with pytest.raises(ValidationError):
        ClassificationState(
            ticket_text="",
        )


def test_classification_state_rejects_negative_iteration() -> None:
    with pytest.raises(ValidationError):
        ClassificationState(
            ticket_text="Test ticket",
            iteration_count=-1,
        )


def test_classification_state_can_track_errors() -> None:
    state = ClassificationState(
        ticket_text="Test ticket",
        status=ClassificationStatus.FAILED,
        errors=[
            ClassificationError(
                source="sentiment_predictor",
                error_type="model_error",
                message="Sentiment model failed.",
                recoverable=False,
            )
        ],
    )

    assert state.status == ClassificationStatus.FAILED
    assert len(state.errors) == 1
    assert state.errors[0].source == "sentiment_predictor"