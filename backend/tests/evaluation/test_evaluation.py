from app.evaluation.contracts.evaluation import (
    EvaluationCase,
    EvaluationMetricResult,
    RAGEvaluationResult,
)


def test_evaluation_case_defaults():
    case = EvaluationCase(
        case_id="RAG-001",
        query="How do I reset my password?",
        generated_answer="Use the password recovery option.",
    )

    assert case.case_id == "RAG-001"
    assert case.query == "How do I reset my password?"
    assert case.retrieved_context == []
    assert case.reference_answer is None
    assert case.metadata == {}


def test_evaluation_case_with_context():
    case = EvaluationCase(
        case_id="RAG-002",
        query="How do I reset my password?",
        retrieved_context=[
            "Use the password recovery option.",
            "Contact support if recovery fails.",
        ],
        generated_answer="Use the password recovery option.",
        reference_answer="Use the password recovery option.",
    )

    assert len(case.retrieved_context) == 2
    assert case.reference_answer is not None


def test_metric_result():
    metric = EvaluationMetricResult(
        metric_name="faithfulness",
        score=0.94,
        passed=True,
        reason="The answer is supported by the retrieved context.",
    )

    assert metric.metric_name == "faithfulness"
    assert metric.score == 0.94
    assert metric.passed is True


def test_metric_score_validation():
    try:
        EvaluationMetricResult(
            metric_name="faithfulness",
            score=1.5,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for score above 1.0")


def test_rag_evaluation_result():
    result = RAGEvaluationResult(
        case_id="RAG-001",
        metrics=[
            EvaluationMetricResult(
                metric_name="faithfulness",
                score=0.94,
                passed=True,
            ),
            EvaluationMetricResult(
                metric_name="answer_relevancy",
                score=0.91,
                passed=True,
            ),
        ],
        overall_passed=True,
    )

    assert result.case_id == "RAG-001"
    assert result.metric_count == 2
    assert result.evaluated is True
    assert result.overall_passed is True


def test_empty_rag_evaluation_result():
    result = RAGEvaluationResult(case_id="RAG-EMPTY")

    assert result.metric_count == 0
    assert result.evaluated is False