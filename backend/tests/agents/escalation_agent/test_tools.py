#test_tools.py
from __future__ import annotations

import pytest

from app.agents.classification_agent.contracts.state import (
    ClassificationPrediction,
    ClassificationResult,
    ClassificationType,
)
from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationClassificationSignal,
    EscalationEvidence,
    EscalationResolutionSignal,
    EscalationSignal,
)
from app.agents.escalation_agent.tools import EscalationTools
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
    ResolutionAgentStatus,
    ResolutionRecommendation,
    ResolutionType,
)


def build_classification_prediction(
    classification_type: ClassificationType,
    value: str,
    confidence: float,
    source: str,
) -> ClassificationPrediction:
    return ClassificationPrediction(
        classification_type=classification_type,
        value=value,
        confidence=confidence,
        source=source,
        metadata={"test": True},
    )


def test_validate_state() -> None:
    state = EscalationAgentState(
        request="Customer cannot log in."
    )

    result = EscalationTools.validate_state(state)

    assert result is state


def test_validate_state_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="EscalationAgentState"):
        EscalationTools.validate_state("invalid")  # type: ignore[arg-type]


def test_classification_signals() -> None:
    result = ClassificationResult(
        category=build_classification_prediction(
            ClassificationType.CATEGORY,
            "Account",
            0.31,
            "ticket_category_predictor",
        ),
        priority=build_classification_prediction(
            ClassificationType.PRIORITY,
            "Medium",
            0.32,
            "priority_detection_predictor",
        ),
        sentiment=build_classification_prediction(
            ClassificationType.SENTIMENT,
            "negative",
            0.85,
            "sentiment_predictor",
        ),
        complexity=build_classification_prediction(
            ClassificationType.COMPLEXITY,
            "MULTI_TURN",
            0.97,
            "conversation_complexity_predictor",
        ),
        topic=build_classification_prediction(
            ClassificationType.TOPIC,
            "24",
            0.73,
            "topic_model_predictor",
        ),
        predictions_completed=5,
        predictions_requested=5,
        uncertainty_detected=True,
    )

    signals = EscalationTools.classification_signals(result)

    assert len(signals) == 5

    assert all(
        isinstance(signal, EscalationClassificationSignal)
        for signal in signals
    )

    assert signals[0].name == ClassificationType.CATEGORY.value
    assert signals[0].value == "Account"
    assert signals[0].source == "ticket_category_predictor"


def test_classification_signals_skip_missing_predictions() -> None:
    result = ClassificationResult(
        category=None,
        priority=None,
        sentiment=build_classification_prediction(
            ClassificationType.SENTIMENT,
            "negative",
            0.85,
            "sentiment_predictor",
        ),
        complexity=None,
        topic=None,
        predictions_completed=1,
        predictions_requested=5,
        uncertainty_detected=True,
    )

    signals = EscalationTools.classification_signals(result)

    assert len(signals) == 1
    assert signals[0].name == ClassificationType.SENTIMENT.value


def test_validate_classification_result_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="ClassificationResult"):
        EscalationTools.validate_classification_result(
            "invalid"  # type: ignore[arg-type]
        )


def test_resolution_signal() -> None:
    recommendation = ResolutionRecommendation(
        resolution_type=ResolutionType.TROUBLESHOOTING,
        recommendation="Guide the customer through account recovery.",
        rationale="Evidence supports account recovery.",
        evidence_ids=["E1", "E2"],
        confidence=0.86,
        requires_human_review=False,
        proposed_actions=[
            "Verify recovery state",
            "Guide password reset",
        ],
    )

    state = ResolutionAgentState(
        request="Customer cannot log in.",
        recommendation=recommendation,
        status=ResolutionAgentStatus.COMPLETED,
        iteration_count=1,
    )

    signal = EscalationTools.resolution_signal(state)

    assert signal is not None
    assert isinstance(signal, EscalationResolutionSignal)

    assert signal.resolution_type == "troubleshooting"
    assert signal.recommendation == (
        "Guide the customer through account recovery."
    )
    assert signal.confidence == 0.86
    assert signal.requires_human_review is False
    assert signal.proposed_actions == [
        "Verify recovery state",
        "Guide password reset",
    ]

    assert signal.metadata["evidence_ids"] == ["E1", "E2"]
    assert signal.metadata["resolution_agent_status"] == "completed"
    assert signal.metadata["resolution_iteration_count"] == 1


def test_resolution_signal_returns_none_without_recommendation() -> None:
    state = ResolutionAgentState(
        request="Customer cannot log in.",
        status=ResolutionAgentStatus.COMPLETED,
    )

    signal = EscalationTools.resolution_signal(state)

    assert signal is None


def test_validate_resolution_state_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="ResolutionAgentState"):
        EscalationTools.validate_resolution_state(
            "invalid"  # type: ignore[arg-type]
        )


def test_validate_operational_signals() -> None:
    signals = [
        EscalationSignal(
            name="sla_breach_risk",
            value="high",
            confidence=0.94,
            source="workflow_intelligence",
        ),
        EscalationSignal(
            name="repeated_handoffs",
            value="true",
            confidence=0.91,
            source="workflow_intelligence",
        ),
    ]

    result = EscalationTools.validate_operational_signals(signals)

    assert result is not signals
    assert len(result) == 2
    assert result[0] is signals[0]
    assert result[1] is signals[1]


def test_validate_operational_signals_rejects_invalid_signal() -> None:
    with pytest.raises(TypeError, match="EscalationSignal"):
        EscalationTools.validate_operational_signals(
            ["invalid"]  # type: ignore[list-item]
        )


def test_validate_operational_signals_requires_list() -> None:
    with pytest.raises(TypeError, match="list"):
        EscalationTools.validate_operational_signals(
            "invalid"  # type: ignore[arg-type]
        )