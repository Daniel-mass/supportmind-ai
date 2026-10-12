from __future__ import annotations

from typing import Any

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)
import instructor
from app.evaluation.contracts.evaluation import (
    EvaluationCase,
    EvaluationMetricResult,
    RAGEvaluationResult,
)


class RAGASEvaluator:
    """Evaluate SupportMind RAG responses using RAGAS metrics."""

    def __init__(
        self,
        llm: Any,
        embeddings: Any | None = None,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings

        self.faithfulness = Faithfulness(llm=llm)

        self.context_precision = ContextPrecisionWithoutReference(
            llm=llm
        )

        self.context_recall = ContextRecall(llm=llm)

        self.answer_relevancy = None

        if embeddings is not None:
            self.answer_relevancy = AnswerRelevancy(
                llm=llm,
                embeddings=embeddings,
            )

    async def evaluate_case(
        self,
        case: EvaluationCase,
    ) -> RAGEvaluationResult:
        """Evaluate one SupportMind RAG case."""

        metrics: list[EvaluationMetricResult] = []

        faithfulness_result = await self.faithfulness.ascore(
            user_input=case.query,
            response=case.generated_answer,
            retrieved_contexts=case.retrieved_context,
        )

        metrics.append(
            EvaluationMetricResult(
                metric_name="faithfulness",
                score=float(faithfulness_result.value),
                reason=faithfulness_result.reason,
            )
        )

        context_precision_result = (
            await self.context_precision.ascore(
                user_input=case.query,
                response=case.generated_answer,
                retrieved_contexts=case.retrieved_context,
            )
        )

        metrics.append(
            EvaluationMetricResult(
                metric_name="context_precision",
                score=float(context_precision_result.value),
                reason=context_precision_result.reason,
            )
        )

        if case.reference_answer is not None:
            context_recall_result = await self.context_recall.ascore(
                user_input=case.query,
                retrieved_contexts=case.retrieved_context,
                reference=case.reference_answer,
            )

            metrics.append(
                EvaluationMetricResult(
                    metric_name="context_recall",
                    score=float(context_recall_result.value),
                    reason=context_recall_result.reason,
                )
            )

        if self.answer_relevancy is not None:
            answer_relevancy_result = (
                await self.answer_relevancy.ascore(
                    user_input=case.query,
                    response=case.generated_answer,
                )
            )

            metrics.append(
                EvaluationMetricResult(
                    metric_name="answer_relevancy",
                    score=float(answer_relevancy_result.value),
                    reason=answer_relevancy_result.reason,
                )
            )

        scores = [
            metric.score
            for metric in metrics
            if metric.score is not None
        ]

        overall_passed = None

        if scores:
            overall_passed = all(score >= 0.7 for score in scores)

        return RAGEvaluationResult(
            case_id=case.case_id,
            metrics=metrics,
            overall_passed=overall_passed,
            metadata={
                "evaluator": "ragas",
                "metric_count": len(metrics),
            },
        )


def create_ragas_evaluator(
    model_name: str = "llama3.2:latest",
    base_url: str = "http://localhost:11434/v1",
    embeddings: Any | None = None,
) -> RAGASEvaluator:
    """Create a RAGAS evaluator backed by a local Ollama model."""

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
    api_key="ollama",
    base_url=base_url,
    )


    llm = llm_factory(
        model_name,
        provider="openai",
        client=client,
        mode=instructor.Mode.TOOLS,
    )

    return RAGASEvaluator(
        llm=llm,
        embeddings=embeddings,
    )