"""
Classification Agent.

This module orchestrates the existing SupportMind classification tools.

Responsibilities:
- Validate ClassificationState.
- Execute requested classification capabilities.
- Aggregate prediction results.
- Track completed predictions.
- Preserve uncertainty.
- Record tool failures.

This module does NOT:
- load ML models directly
- implement prediction algorithms
- perform retrieval
- perform Graph-RAG
- recommend resolutions
- make escalation decisions
- execute customer-impacting actions
- fabricate predictions
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.agents.classification_agent.contracts.state import (
    ClassificationError,
    ClassificationPrediction,
    ClassificationResult,
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


CLASSIFICATION_AGENT_NAME = "classification_agent"


class ClassificationAgent:
    """
    Orchestrates the existing classification tools.

    The agent receives already-initialized tool instances. This keeps
    model loading outside the agent and makes the component easy to test.
    """

    def __init__(
        self,
        *,
        category_tool: TicketCategoryTool,
        priority_tool: PriorityDetectionTool,
        sentiment_tool: SentimentTool,
        complexity_tool: ConversationComplexityTool,
        topic_tool: TopicModelTool,
        name: str = CLASSIFICATION_AGENT_NAME,
    ) -> None:
        self.name = name

        self._category_tool = category_tool
        self._priority_tool = priority_tool
        self._sentiment_tool = sentiment_tool
        self._complexity_tool = complexity_tool
        self._topic_tool = topic_tool

    def run(self, state: ClassificationState) -> ClassificationState:
        """
        Execute the requested classification capabilities.

        Independent classification failures are recorded and do not
        prevent other requested capabilities from running.
        """

        self._validate_state(state)

        state.status = ClassificationStatus.RUNNING
        state.iteration_count += 1

        # Avoid stale completion state when the same state object is reused.
        state.completed_predictions = []
        state.errors = []

        requested_predictions = self._resolve_requested_predictions(state)

        state.predictions = ClassificationResult(
            predictions_requested=len(requested_predictions),
        )

        for classification_type in requested_predictions:
            self._execute_prediction(state, classification_type)

        state.predictions.predictions_completed = len(
            state.completed_predictions
        )

        state.predictions.uncertainty_detected = (
            self._detect_uncertainty(state)
        )

        state.status = ClassificationStatus.COMPLETED

        return state

    def _execute_prediction(
        self,
        state: ClassificationState,
        classification_type: ClassificationType,
    ) -> None:
        """
        Execute one classification capability.

        Tool failures are isolated so that an unavailable predictor does
        not unnecessarily prevent independent predictions.
        """

        try:
            prediction = self._run_tool(
                state=state,
                classification_type=classification_type,
            )

            self._store_prediction(
                state=state,
                classification_type=classification_type,
                prediction=prediction,
            )

            state.completed_predictions.append(classification_type)

        except Exception as exc:
            state.errors.append(
                ClassificationError(
                    source=self.name,
                    error_type=type(exc).__name__,
                    message=(
                        f"{classification_type.value} classification "
                        f"failed: {exc}"
                    ),
                    recoverable=True,
                )
            )

    def _run_tool(
        self,
        *,
        state: ClassificationState,
        classification_type: ClassificationType,
    ) -> ClassificationPrediction:
        """
        Dispatch one classification type to its corresponding tool.
        """

        if classification_type is ClassificationType.CATEGORY:
            return self._category_tool.predict(state.ticket_text)

        if classification_type is ClassificationType.PRIORITY:
            features = state.priority_features

            if features is None:
                raise ValueError(
                    "Priority classification requires priority_features."
                )

            return self._priority_tool.predict(
                features.model_dump())

        if classification_type is ClassificationType.SENTIMENT:
            return self._sentiment_tool.predict(state.ticket_text)

        if classification_type is ClassificationType.COMPLEXITY:
            return self._complexity_tool.predict(state.ticket_text)

        if classification_type is ClassificationType.TOPIC:
            return self._topic_tool.predict(state.ticket_text)

        raise ValueError(
            f"Unsupported classification type: {classification_type}"
        )

    @staticmethod
    def _store_prediction(
        *,
        state: ClassificationState,
        classification_type: ClassificationType,
        prediction: ClassificationPrediction,
    ) -> None:
        """
        Store a prediction in the appropriate ClassificationResult field.
        """

        predictions = state.predictions

        if classification_type is ClassificationType.CATEGORY:
            predictions.category = prediction

        elif classification_type is ClassificationType.PRIORITY:
            predictions.priority = prediction

        elif classification_type is ClassificationType.SENTIMENT:
            predictions.sentiment = prediction

        elif classification_type is ClassificationType.COMPLEXITY:
            predictions.complexity = prediction

        elif classification_type is ClassificationType.TOPIC:
            predictions.topic = prediction

        else:
            raise ValueError(
                f"Unsupported classification type: {classification_type}"
            )

    @staticmethod
    def _detect_uncertainty(state: ClassificationState) -> bool:
        """
        Detect whether any completed prediction has missing or low confidence.

        A confidence below 0.70 is treated as an uncertainty signal.

        This does NOT determine whether the prediction is correct.
        """

        predictions = (
            state.predictions.category,
            state.predictions.priority,
            state.predictions.sentiment,
            state.predictions.complexity,
            state.predictions.topic,
        )

        for prediction in predictions:
            if prediction is None:
                continue

            if prediction.confidence is None:
                return True

            if prediction.confidence < 0.70:
                return True

        return False

    @staticmethod
    def _resolve_requested_predictions(
        state: ClassificationState,
    ) -> list[ClassificationType]:
        """
        Resolve requested prediction types.

        When no explicit prediction list is provided, run all
        classification
        capabilities that can be executed with the currently available input.

        Priority classification is included only when legitimate structured
        priority features are available.

        No priority features are ever fabricated from free-form ticket text.

        Duplicates are removed while preserving request order.
        """

        if not state.requested_predictions:
            resolved = [
                ClassificationType.CATEGORY,
                ClassificationType.SENTIMENT,
                ClassificationType.COMPLEXITY,
                ClassificationType.TOPIC,
            ]

            if state.priority_features is not None:
                resolved.insert(
                    1,
                    ClassificationType.PRIORITY,
                )

            return resolved

        resolved: list[ClassificationType] = []

        for classification_type in state.requested_predictions:
            if classification_type is ClassificationType.PRIORITY:
                if state.priority_features is None:
                    continue

            if classification_type not in resolved:
                resolved.append(classification_type)

        return resolved

    @staticmethod
    def _validate_state(state: ClassificationState) -> None:
        if not isinstance(state, ClassificationState):
            raise TypeError(
                "state must be an instance of ClassificationState."
            )

        if not state.ticket_text.strip():
            raise ValueError("ticket_text must not be empty.")