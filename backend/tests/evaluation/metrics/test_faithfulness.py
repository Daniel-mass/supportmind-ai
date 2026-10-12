from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.evaluation.metrics.faithfulness import FaithfulnessEvaluator


def test_faithfulness_evaluator_creates_metric() -> None:
    """Verify that the evaluator configures DeepEval correctly."""

    mock_model = MagicMock()

    with patch(
        "app.evaluation.metrics.faithfulness.FaithfulnessMetric"
    ) as mock_metric:

        evaluator = FaithfulnessEvaluator(
            model=mock_model,
            threshold=0.7,
        )

        mock_metric.assert_called_once_with(
            threshold=0.7,
            model=mock_model,
            include_reason=True,
            async_mode=False,
        )

        assert evaluator.metric == mock_metric.return_value