"""
SupportMind Agent Orchestrator.

The Orchestrator is the execution layer of the Agent system.

Architecture:

    Coordinator
        ↓
    Router
        ↓
    Orchestrator
        ↓
    Specialized Agent
        ↓
    Orchestrator
        ↓
    Coordinator
        ↓
    Router
        ↓
    ...

Responsibilities:
- Own the specialized agent registry.
- Execute the capability selected by CoordinatorRouter.
- Translate CoordinatorState into specialized agent state.
- Merge specialized-agent results back into CoordinatorState.
- Track completed capabilities.
- Preserve evidence and prediction provenance.
- Stop failed workflows.
- Prevent duplicate capability execution.
- Protect against infinite orchestration loops.

The Orchestrator does NOT:
- make routing decisions;
- perform classification itself;
- perform retrieval itself;
- generate resolutions itself;
- make escalation decisions itself;
- implement retrieval algorithms;
- implement ML models;
- fabricate agent results.
"""

from __future__ import annotations

import os
from typing import Any

from opentelemetry import trace

from app.agents.classification_agent.agents import (
    ClassificationAgent,
)
from app.agents.classification_agent.contracts.state import (
    ClassificationState,
    ClassificationType,
)
from app.agents.coordinator.agent import CoordinatorAgent
from app.agents.coordinator.router import (
    CoordinatorRouter,
)
from app.agents.coordinator.state import (
    AgentResult,
    Capability,
    ClassificationSignals,
    CoordinatorState,
    EvidenceReference,
    PredictionSignal,
    WorkflowError,
    WorkflowStatus,
    WorkflowStep,
)
from app.agents.escalation_agent.agent import (
    EscalationAgent,
)
from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationClassificationSignal,
    EscalationEvidence,
    EscalationResolutionSignal,
)
from app.agents.resolution_agent.agents import (
    ResolutionAgent,
)
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
    ResolutionClassificationSignal,
    ResolutionEvidence,
)
from app.agents.response_composer_agent.agent import (
    ResponseComposerAgent,
)
from app.agents.response_composer_agent.contracts.state import (
    ResponseClassificationSignal,
    ResponseComposerAgentState,
    ResponseEvidence,
    ResponseEscalationSignal,
    ResponseResolutionSignal,
)
from app.agents.retrieval_agent.agents import (
    RetrievalAgent,
)
from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentState,
)
from app.observability.phoenix import (
    get_supportmind_tracer,
    set_span_input,
    set_span_output,
    set_span_kind,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)


# ============================================================================
# Constants
# ============================================================================

ORCHESTRATOR_NAME = "agent_orchestrator"

DEFAULT_MAX_ITERATIONS = 10

# The Qdrant collection used by the SupportMind knowledge base.
#
# It can be overridden through:
#
# SUPPORTMIND_QDRANT_COLLECTION=your_collection_name
#
# This keeps the orchestrator independent from environment-specific
# infrastructure configuration.
DEFAULT_RETRIEVAL_COLLECTION = "supportmind_knowledge"


class AgentOrchestrator:
    """
    Execute the SupportMind multi-agent workflow.

    The Coordinator decides what should happen next.
    The Router selects the next capability.
    This class executes that capability.
    """

    def __init__(
        self,
        *,
        coordinator: CoordinatorAgent,
        router: CoordinatorRouter,
        classification_agent: ClassificationAgent,
        retrieval_agent: RetrievalAgent,
        resolution_agent: ResolutionAgent,
        escalation_agent: EscalationAgent,
        response_composer_agent: ResponseComposerAgent,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        name: str = ORCHESTRATOR_NAME,
        retrieval_collection_name: str | None = None,
    ) -> None:
        if coordinator is None:
            raise ValueError(
                "coordinator must not be None."
            )

        if router is None:
            raise ValueError(
                "router must not be None."
            )

        if classification_agent is None:
            raise ValueError(
                "classification_agent must not be None."
            )

        if retrieval_agent is None:
            raise ValueError(
                "retrieval_agent must not be None."
            )

        if resolution_agent is None:
            raise ValueError(
                "resolution_agent must not be None."
            )

        if escalation_agent is None:
            raise ValueError(
                "escalation_agent must not be None."
            )

        if response_composer_agent is None:
            raise ValueError(
                "response_composer_agent must not be None."
            )

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be greater than zero."
            )

        if not name.strip():
            raise ValueError(
                "name must not be empty."
            )

        collection_name = (
            retrieval_collection_name
            or os.getenv(
                "SUPPORTMIND_QDRANT_COLLECTION",
                DEFAULT_RETRIEVAL_COLLECTION,
            )
        )

        if not collection_name.strip():
            raise ValueError(
                "retrieval_collection_name must not be empty."
            )

        self.name = name
        self.max_iterations = max_iterations
        self._tracer = get_supportmind_tracer()
        self._retrieval_collection_name = collection_name

        self._coordinator = coordinator
        self._router = router

        self._classification_agent = (
            classification_agent
        )
        self._retrieval_agent = retrieval_agent
        self._resolution_agent = resolution_agent
        self._escalation_agent = escalation_agent
        self._response_composer_agent = (
            response_composer_agent
        )

        self._agent_registry = {
            Capability.CLASSIFICATION: (
                self._execute_classification
            ),
            Capability.RETRIEVAL: (
                self._execute_retrieval
            ),
            Capability.RESOLUTION: (
                self._execute_resolution
            ),
            Capability.ESCALATION: (
                self._execute_escalation
            ),
            Capability.GENERATION: (
                self._execute_generation
            ),
        }

    # ======================================================================
    # Public API
    # ======================================================================

    async def run(
        self,
        state: CoordinatorState,
    ) -> CoordinatorState:
        """
        Execute the complete multi-agent workflow.

        The supplied CoordinatorState is mutated in place and
        returned after the workflow terminates.

        Phoenix tracing:
            One parent span represents the complete orchestration
            lifecycle. Specialized capability spans are created inside
            this parent span.
        """

        self._validate_state(state)

        with self._tracer.start_as_current_span(
            "supportmind.agent-orchestrator.run"
        ) as workflow_span:

            set_span_kind(
                workflow_span,
                "AGENT",
            )

            set_span_input(
                workflow_span,
                {
                    "request": state.request,
                    "requested_capabilities": [
                        capability.value
                        for capability in state.requested_capabilities
                    ],
                },
            )

            workflow_span.set_attribute(
                "supportmind.orchestrator.name",
                self.name,
            )

            workflow_span.set_attribute(
                "supportmind.workflow.max_iterations",
                self.max_iterations,
            )

            if state.status == WorkflowStatus.COMPLETED:
                workflow_span.set_attribute(
                    "supportmind.workflow.status",
                    "already_completed",
                )
                return state

            if state.status == WorkflowStatus.FAILED:
                workflow_span.set_attribute(
                    "supportmind.workflow.status",
                    "already_failed",
                )
                return state

            state.status = WorkflowStatus.RUNNING

            while True:
                if state.iteration_count >= self.max_iterations:
                    self._fail_workflow(
                        state=state,
                        error_type="max_iterations_exceeded",
                        message=(
                            "Agent orchestration stopped because the "
                            "maximum number of iterations was reached."
                        ),
                    )

                    workflow_span.set_attribute(
                        "supportmind.workflow.status",
                        "failed",
                    )

                    workflow_span.set_attribute(
                        "supportmind.workflow.error_type",
                        "max_iterations_exceeded",
                    )

                    return state

                # ----------------------------------------------------------
                # Coordinator reasoning
                # ----------------------------------------------------------

                previous_iteration_count = (
                    state.iteration_count
                )

                state = await self._coordinator.run(
                    state
                )

                if (
                    state.iteration_count
                    == previous_iteration_count
                ):
                    state.iteration_count += 1

                workflow_span.set_attribute(
                    "supportmind.workflow.iteration",
                    state.iteration_count,
                )

                # ----------------------------------------------------------
                # Coordinator failure detection
                # ----------------------------------------------------------

                if state.status == WorkflowStatus.FAILED:
                    workflow_span.set_attribute(
                        "supportmind.workflow.status",
                        "failed",
                    )

                    return state

                # ----------------------------------------------------------
                # IMPORTANT STATE NORMALIZATION
                # ----------------------------------------------------------

                if state.status == WorkflowStatus.COMPLETED:
                    state.status = WorkflowStatus.RUNNING

                # ----------------------------------------------------------
                # Router decision
                # ----------------------------------------------------------

                decision = self._router.route(
                    state
                )

                workflow_span.set_attribute(
                    "supportmind.router.should_continue",
                    decision.should_continue,
                )

                workflow_span.set_attribute(
                    "supportmind.router.should_terminate",
                    decision.should_terminate,
                )

                if decision.should_terminate:
                    state.status = WorkflowStatus.COMPLETED
                    state.current_step = WorkflowStep.COMPLETED

                    workflow_span.set_attribute(
                        "supportmind.workflow.status",
                        "completed",
                    )

                    workflow_span.set_attribute(
                        "supportmind.workflow.completed_capabilities",
                        len(
                            state.completed_capabilities
                        ),
                    )

                    set_span_output(
                        workflow_span,
                        {
                            "final_response": state.final_response,
                            "status": state.status.value,
                            "completed_capabilities": [
                                capability.value
                                for capability in state.completed_capabilities
                            ],
                            "iteration_count": state.iteration_count,
                            "errors": state.errors,
                        },
                    )

                    return state

                if not decision.should_continue:
                    self._fail_workflow(
                        state=state,
                        error_type="invalid_routing_decision",
                        message=(
                            "CoordinatorRouter returned a decision that "
                            "was neither executable nor terminal."
                        ),
                    )

                    workflow_span.set_attribute(
                        "supportmind.workflow.status",
                        "failed",
                    )

                    return state

                capability = decision.capability

                if capability is None:
                    self._fail_workflow(
                        state=state,
                        error_type="missing_capability",
                        message=(
                            "CoordinatorRouter requested continuation "
                            "without specifying a capability."
                        ),
                    )

                    workflow_span.set_attribute(
                        "supportmind.workflow.status",
                        "failed",
                    )

                    return state

                if capability in state.completed_capabilities:
                    self._fail_workflow(
                        state=state,
                        error_type="duplicate_capability",
                        message=(
                            f"Capability '{capability.value}' was selected "
                            "although it is already completed."
                        ),
                    )

                    workflow_span.set_attribute(
                        "supportmind.workflow.status",
                        "failed",
                    )

                    return state

                executor = self._agent_registry.get(
                    capability
                )

                if executor is None:
                    self._fail_workflow(
                        state=state,
                        error_type="unsupported_capability",
                        message=(
                            "No executor is registered for "
                            f"capability '{capability.value}'."
                        ),
                    )

                    workflow_span.set_attribute(
                        "supportmind.workflow.status",
                        "failed",
                    )

                    return state

                # ----------------------------------------------------------
                # Specialized agent execution
                # ----------------------------------------------------------

                state.current_step = (
                    self._capability_to_step(
                        capability
                    )
                )

                agent_name = (
                    self._agent_name_for_capability(
                        capability
                    )
                )

                with self._tracer.start_as_current_span(
                    f"agent.{capability.value}"
                ) as agent_span:

                    set_span_kind(
                        agent_span,
                        "AGENT",
                    )

                    set_span_input(
                        agent_span,
                        {
                            "request": state.request,
                            "capability": capability.value,
                            "iteration": state.iteration_count,
                        },
                    )

                    agent_span.set_attribute(
                        "supportmind.agent.name",
                        agent_name,
                    )

                    agent_span.set_attribute(
                        "supportmind.agent.capability",
                        capability.value,
                    )

                    agent_span.set_attribute(
                        "supportmind.workflow.iteration",
                        state.iteration_count,
                    )

                    try:
                        await executor(state)

                    except Exception as exc:
                        agent_span.record_exception(
                            exc
                        )

                        agent_span.set_attribute(
                            "supportmind.agent.status",
                            "failed",
                        )

                        self._fail_workflow(
                            state=state,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )

                        workflow_span.set_attribute(
                            "supportmind.workflow.status",
                            "failed",
                        )

                        return state

                    # ------------------------------------------------------
                    # Agent failure detection
                    # ------------------------------------------------------

                    result = state.agent_results.get(
                        agent_name
                    )

                    if result is not None:
                        set_span_output(
                            agent_span,
                            {
                                "agent_name": result.agent_name,
                                "success": result.success,
                                "result": result.result,
                                "evidence_ids": result.evidence_ids,
                                "error": result.error,
                            },
                        )

                    if (
                        result is not None
                        and not result.success
                    ):
                        agent_span.set_attribute(
                            "supportmind.agent.status",
                            "failed",
                        )

                        self._fail_workflow(
                            state=state,
                            error_type="agent_execution_failed",
                            message=(
                                f"Agent '{result.agent_name}' failed."
                            ),
                        )

                        workflow_span.set_attribute(
                            "supportmind.workflow.status",
                            "failed",
                        )

                        return state

                    if (
                        capability
                        not in state.completed_capabilities
                    ):
                        agent_span.set_attribute(
                            "supportmind.agent.status",
                            "failed",
                        )

                        self._fail_workflow(
                            state=state,
                            error_type="capability_not_completed",
                            message=(
                                f"Agent execution for "
                                f"'{capability.value}' returned without "
                                "marking the capability as completed."
                            ),
                        )

                        workflow_span.set_attribute(
                            "supportmind.workflow.status",
                            "failed",
                        )

                        return state

                    agent_span.set_attribute(
                        "supportmind.agent.status",
                        "completed",
                    )

                    agent_span.set_attribute(
                        "supportmind.agent.success",
                        True,
                    )

                    agent_span.set_attribute(
                        "supportmind.agent.evidence_count",
                        len(state.evidence),
                    )

    # ======================================================================
    # Classification
    # ======================================================================

    async def _execute_classification(
        self,
        state: CoordinatorState,
    ) -> None:
        classification_state = ClassificationState(
            ticket_text=state.request,
            priority_features=state.priority_features,
            requested_predictions=list(
                ClassificationType
            ),
        )

        result = self._classification_agent.run(
            classification_state
        )

        self._merge_classification_result(
            state=state,
            result=result,
        )

    def _merge_classification_result(
        self,
        *,
        state: CoordinatorState,
        result: ClassificationState,
    ) -> None:
        success = (
            result.status.value == "completed"
            and len(result.errors) == 0
        )

        predictions = result.predictions

        state.predictions = ClassificationSignals(
            category=self._prediction_value(
                predictions.category
            ),
            priority=self._prediction_value(
                predictions.priority
            ),
            sentiment=self._prediction_value(
                predictions.sentiment
            ),
            complexity=self._prediction_value(
                predictions.complexity
            ),
            topic=self._prediction_value(
                predictions.topic
            ),
        )

        state.agent_results[
            "classification_agent"
        ] = AgentResult(
            agent_name="classification_agent",
            success=success,
            result={
                "predictions_completed": [
                    item.value
                    for item in result.completed_predictions
                ],
                "uncertainty_detected": (
                    result.predictions.uncertainty_detected
                ),
            },
            evidence_ids=[],
            error=(
                result.errors[0].message
                if result.errors
                else None
            ),
        )

        if success:
            self._mark_completed(
                state,
                Capability.CLASSIFICATION,
            )
        else:
            raise RuntimeError(
                "Classification Agent failed."
            )

    # ======================================================================
    # Retrieval
    # ======================================================================

    async def _execute_retrieval(
        self,
        state: CoordinatorState,
    ) -> None:
        """
        Build the retrieval request and execute the Retrieval Agent.
        """

        retrieval_request = RetrievalRequest(
            query=state.request,
            collection_name=(
                self._retrieval_collection_name
            ),
        )

        retrieval_state = RetrievalAgentState(
            request=retrieval_request,
            use_graph_rag=True,
        )

        result = await self._retrieval_agent.run(
            retrieval_state
        )

        self._merge_retrieval_result(
            state=state,
            result=result,
        )

    def _merge_retrieval_result(
        self,
        *,
        state: CoordinatorState,
        result: RetrievalAgentState,
    ) -> None:
        success = result.successful

        evidence_ids = list(
            result.evidence_ids
        )

        self._add_retrieval_evidence(
            state=state,
            result=result,
        )

        state.agent_results[
            "retrieval_agent"
        ] = AgentResult(
            agent_name="retrieval_agent",
            success=success,
            result={
                "strategy": result.strategy.value,
                "evidence_count": len(
                    evidence_ids
                ),
                "execution_metadata": (
                    result.execution_metadata
                ),
            },
            evidence_ids=evidence_ids,
            error=(
                result.errors[0].message
                if result.errors
                else None
            ),
        )

        if success:
            self._mark_completed(
                state,
                Capability.RETRIEVAL,
            )
        else:
            raise RuntimeError(
                "Retrieval Agent failed."
            )

    @staticmethod
    def _add_retrieval_evidence(
        *,
        state: CoordinatorState,
        result: RetrievalAgentState,
    ) -> None:
        """
        Copy actual retrieval evidence into CoordinatorState.

        Evidence is only copied when it exists in the retrieval
        result. No synthetic evidence is created.
        """

        if result.graph_rag_response is not None:
            for evidence in (
                result.graph_rag_response.evidence
            ):
                state.evidence.append(
                    EvidenceReference(
                        evidence_id=(
                            evidence.evidence_id
                        ),
                        source=evidence.source,
                        content=evidence.text,
                        provenance={
                            "source_reference": (
                                evidence.source_reference
                            ),
                            "evidence_type": (
                                evidence.evidence_type.value
                            ),
                            "score": evidence.score,
                            "metadata": evidence.metadata,
                        },
                    )
                )

        elif result.context is not None:
            for item in (
                result.context.selected_items
            ):
                if not item.id:
                    continue

                state.evidence.append(
                    EvidenceReference(
                        evidence_id=item.id,
                        source="retrieval",
                        content=item.content,
                        provenance=None,
                    )
                )

    # ======================================================================
    # Resolution
    # ======================================================================

    async def _execute_resolution(
        self,
        state: CoordinatorState,
    ) -> None:
        resolution_state = ResolutionAgentState(
            request=state.request,
            classification_signals=(
                self._build_resolution_classification_signals(
                    state
                )
            ),
            evidence=(
                self._build_resolution_evidence(
                    state
                )
            ),
        )

        result = await self._resolution_agent.run(
            resolution_state
        )

        # --------------------------------------------------------------
        # TEMPORARY RESOLUTION DIAGNOSTICS
        #
        # This diagnostic is intentionally placed AFTER the Resolution
        # Agent has finished and BEFORE _merge_resolution_result().
        #
        # The purpose is to identify exactly where the evidence ID
        # mismatch occurs:
        #
        #   E1/E2/E3/... passed into Resolution Agent?
        #               ↓
        #   What raw response did Qwen return?
        #               ↓
        #   What did ResolutionOutputParser report?
        #
        # No normalization, recovery, or mutation is performed here.
        # --------------------------------------------------------------

        if result.recommendation is None:
            print()
            print("=" * 80)
            print("RESOLUTION AGENT DEBUG")
            print("=" * 80)

            print()
            print("Resolution state evidence IDs:")
            print(
                [
                    evidence.evidence_id
                    for evidence in resolution_state.evidence
                ]
            )

            print()
            print("Resolution execution metadata:")
            print(
                result.execution_metadata
            )

            print()
            print("Resolution errors:")

            for error in result.errors:
                print(
                    error
                )

            print()
            print("=" * 80)

        self._merge_resolution_result(
            state=state,
            result=result,
        )

    def _merge_resolution_result(
        self,
        *,
        state: CoordinatorState,
        result: Any,
    ) -> None:
        """
        Merge a Resolution Agent result into CoordinatorState.

        The Resolution Agent must produce a validated
        ResolutionRecommendation before the resolution capability
        can be considered complete.
        """

        success = (
            result.status.value == "completed"
            and result.recommendation is not None
        )

        # --------------------------------------------------------------
        # No recommendation
        # --------------------------------------------------------------

        if result.recommendation is None:
            if result.errors:
                raise RuntimeError(
                    "Resolution Agent failed without a "
                    "structured recommendation: "
                    f"{result.errors[0].message}"
                )

            raise RuntimeError(
                "Resolution Agent completed without a "
                "structured recommendation."
            )

        # --------------------------------------------------------------
        # Valid structured recommendation
        # --------------------------------------------------------------

        recommendation = result.recommendation

        state.agent_results[
            "resolution_agent"
        ] = AgentResult(
            agent_name="resolution_agent",
            success=success,
            result=recommendation.model_dump(),
            evidence_ids=list(
                recommendation.evidence_ids
            ),
            error=(
                result.errors[0].message
                if result.errors
                else None
            ),
        )

        if success:
            self._mark_completed(
                state,
                Capability.RESOLUTION,
            )
        else:
            raise RuntimeError(
                "Resolution Agent failed."
            )

    # ======================================================================
    # Escalation
    # ======================================================================

    async def _execute_escalation(
        self,
        state: CoordinatorState,
    ) -> None:
        escalation_state = EscalationAgentState(
            request=state.request,
            classification_signals=(
                self._build_escalation_classification_signals(
                    state
                )
            ),
            resolution=(
                self._build_escalation_resolution(
                    state
                )
            ),
            evidence=(
                self._build_escalation_evidence(
                    state
                )
            ),
        )

        result = await self._escalation_agent.run(
            escalation_state
        )

        self._merge_escalation_result(
            state=state,
            result=result,
        )

    def _merge_escalation_result(
        self,
        *,
        state: CoordinatorState,
        result: Any,
    ) -> None:
        """
        Merge an Escalation Agent result into CoordinatorState.

        The Escalation Agent must produce a structured
        EscalationRecommendation before the escalation capability
        can be considered complete.

        IMPORTANT:
        This method intentionally exposes the complete Escalation Agent
        diagnostic information when the structured recommendation is
        missing.

        The purpose of this diagnostic is to identify the exact output
        returned by the Escalation Agent before introducing an output
        parser or changing the Escalation Agent contract.
        """

        success = (
            result.status.value == "completed"
            and result.recommendation is not None
        )

        # --------------------------------------------------------------
        # No structured recommendation
        # --------------------------------------------------------------

        if result.recommendation is None:
            diagnostic = (
                self._build_escalation_failure_diagnostic(
                    result
                )
            )

            raise RuntimeError(
                "Escalation Agent completed without a "
                "structured recommendation.\n\n"
                "ESCALATION AGENT DIAGNOSTIC:\n"
                f"{diagnostic}"
            )

        # --------------------------------------------------------------
        # Valid structured recommendation
        # --------------------------------------------------------------

        recommendation = result.recommendation

        state.agent_results[
            "escalation_agent"
        ] = AgentResult(
            agent_name="escalation_agent",
            success=success,
            result=recommendation.model_dump(),
            evidence_ids=list(
                recommendation.evidence_ids
            ),
            error=(
                result.errors[0].message
                if result.errors
                else None
            ),
        )

        if success:
            self._mark_completed(
                state,
                Capability.ESCALATION,
            )
        else:
            raise RuntimeError(
                "Escalation Agent failed."
            )

    @staticmethod
    def _build_escalation_failure_diagnostic(
        result: EscalationAgentState,
    ) -> str:
        """
        Expose the complete Escalation Agent state relevant to debugging.

        This diagnostic intentionally preserves the raw model response
        from execution_metadata when available.

        No parsing, normalization, or recovery is performed here.
        """

        diagnostic: dict[str, Any] = {
            "status": (
                result.status.value
                if hasattr(result.status, "value")
                else str(result.status)
            ),
            "recommendation": result.recommendation,
            "iteration_count": result.iteration_count,
            "request": result.request,
            "errors": [
                error.model_dump()
                for error in result.errors
            ],
            "execution_metadata": (
                result.execution_metadata
            ),
        }

        raw_response = result.execution_metadata.get(
            "agent_response"
        )

        diagnostic[
            "raw_agent_response"
        ] = raw_response

        return repr(diagnostic)

    # ======================================================================
    # Generation / Response Composer
    # ======================================================================

    async def _execute_generation(
        self,
        state: CoordinatorState,
    ) -> None:
        generation_state = (
            self._build_response_composer_state(
                state
            )
        )

        result = (
            await self._response_composer_agent.run(
                generation_state
            )
        )

        if not result.successful:
            raise RuntimeError(
                "Response Composer Agent failed."
            )

        if result.composed_response is None:
            raise RuntimeError(
                "Response Composer Agent completed "
                "without a composed response."
            )

        composed = result.composed_response

        state.final_response = composed.response

        state.agent_results[
            "response_composer_agent"
        ] = AgentResult(
            agent_name="response_composer_agent",
            success=True,
            result=composed.model_dump(),
            evidence_ids=list(
                composed.evidence_ids
            ),
            error=None,
        )

        self._mark_completed(
            state,
            Capability.GENERATION,
        )

    # ======================================================================
    # State Builders
    # ======================================================================

    @staticmethod
    def _build_resolution_classification_signals(
        state: CoordinatorState,
    ) -> list[ResolutionClassificationSignal]:
        signals = []

        values = {
            "category": state.predictions.category,
            "priority": state.predictions.priority,
            "sentiment": state.predictions.sentiment,
            "complexity": state.predictions.complexity,
            "topic": state.predictions.topic,
        }

        for name, signal in values.items():
            if signal is None:
                continue

            signals.append(
                ResolutionClassificationSignal(
                    name=name,
                    value=signal.value,
                    confidence=signal.confidence,
                    source=signal.source,
                    metadata=signal.metadata,
                )
            )

        return signals

    @staticmethod
    def _build_resolution_evidence(
        state: CoordinatorState,
    ) -> list[ResolutionEvidence]:
        return [
            ResolutionEvidence(
                evidence_id=item.evidence_id,
                source=item.source,
                content=item.content,
                source_reference=(
                    item.provenance.get(
                        "source_reference"
                    )
                    if item.provenance
                    else None
                ),
                score=(
                    item.provenance.get("score")
                    if item.provenance
                    else None
                ),
                metadata=(
                    item.provenance.get(
                        "metadata",
                        {},
                    )
                    if item.provenance
                    else {}
                ),
            )
            for item in state.evidence
        ]

    @staticmethod
    def _build_escalation_classification_signals(
        state: CoordinatorState,
    ) -> list[EscalationClassificationSignal]:
        signals = []

        values = {
            "category": state.predictions.category,
            "priority": state.predictions.priority,
            "sentiment": state.predictions.sentiment,
            "complexity": state.predictions.complexity,
            "topic": state.predictions.topic,
        }

        for name, signal in values.items():
            if signal is None:
                continue

            signals.append(
                EscalationClassificationSignal(
                    name=name,
                    value=signal.value,
                    confidence=signal.confidence,
                    source=signal.source,
                    metadata=signal.metadata,
                )
            )

        return signals

    @staticmethod
    def _build_escalation_resolution(
        state: CoordinatorState,
    ) -> EscalationResolutionSignal:
        result = state.agent_results.get(
            "resolution_agent"
        )

        if result is None or not result.success:
            raise RuntimeError(
                "Escalation requires a successful "
                "Resolution Agent result."
            )

        data = result.result

        return EscalationResolutionSignal(
            resolution_type=data[
                "resolution_type"
            ],
            recommendation=data[
                "recommendation"
            ],
            rationale=data[
                "rationale"
            ],
            confidence=data.get(
                "confidence"
            ),
            requires_human_review=data.get(
                "requires_human_review",
                False,
            ),
            proposed_actions=data.get(
                "proposed_actions",
                [],
            ),
        )

    @staticmethod
    def _build_escalation_evidence(
        state: CoordinatorState,
    ) -> list[EscalationEvidence]:
        evidence: list[EscalationEvidence] = []

        for item in state.evidence:
            source_reference = item.provenance

            if isinstance(source_reference, dict):
                source_reference = source_reference.get(
                    "source_reference"
                )

            if source_reference is not None and not isinstance(
                source_reference,
                str,
            ):
                source_reference = str(source_reference)

            evidence.append(
                EscalationEvidence(
                    evidence_id=item.evidence_id,
                    source=item.source,
                    content=item.content,
                    source_reference=source_reference,
                )
            )

        return evidence

    @staticmethod
    def _build_response_composer_state(
        state: CoordinatorState,
    ) -> ResponseComposerAgentState:
        """
        Build the complete state consumed by the Response Composer Agent.

        The Response Composer receives already-computed upstream signals.
        This method only translates CoordinatorState into the strict
        Response Composer contract.

        Evidence provenance is explicitly flattened:

            EvidenceReference.provenance
                ↓
            ResponseEvidence.source_reference
            ResponseEvidence.score
            ResponseEvidence.metadata

        The complete provenance dictionary must never be passed into
        source_reference because ResponseEvidence expects a string.
        """

        # ------------------------------------------------------------------
        # Classification signals
        # ------------------------------------------------------------------

        classification_signals = []

        values = {
            "category": state.predictions.category,
            "priority": state.predictions.priority,
            "sentiment": state.predictions.sentiment,
            "complexity": state.predictions.complexity,
            "topic": state.predictions.topic,
        }

        for name, signal in values.items():
            if signal is None:
                continue

            classification_signals.append(
                ResponseClassificationSignal(
                    name=name,
                    value=signal.value,
                    confidence=signal.confidence,
                    source=signal.source,
                    metadata=signal.metadata,
                )
            )

        # ------------------------------------------------------------------
        # Resolution signal
        # ------------------------------------------------------------------

        resolution_signal = None

        resolution_result = state.agent_results.get(
            "resolution_agent"
        )

        if resolution_result is not None:
            data = resolution_result.result

            resolution_signal = ResponseResolutionSignal(
                resolution_type=data["resolution_type"],
                recommendation=data["recommendation"],
                rationale=data["rationale"],
                confidence=data.get("confidence"),
                requires_human_review=data.get(
                    "requires_human_review",
                    False,
                ),
                proposed_actions=data.get(
                    "proposed_actions",
                    [],
                ),
            )

        # ------------------------------------------------------------------
        # Escalation signal
        # ------------------------------------------------------------------

        escalation_signal = None

        escalation_result = state.agent_results.get(
            "escalation_agent"
        )

        if escalation_result is not None:
            data = escalation_result.result

            escalation_signal = ResponseEscalationSignal(
                escalation_type=data["escalation_type"],
                should_escalate=data["should_escalate"],
                reason=data["reason"],
                rationale=data["rationale"],
                recommended_team=data.get(
                    "recommended_team"
                ),
                recommended_path=data.get(
                    "recommended_path"
                ),
                evidence_ids=data.get(
                    "evidence_ids",
                    [],
                ),
                confidence=data.get(
                    "confidence"
                ),
                requires_human_review=data.get(
                    "requires_human_review",
                    False,
                ),
                proposed_actions=data.get(
                    "proposed_actions",
                    [],
                ),
            )

        # ------------------------------------------------------------------
        # Evidence
        # ------------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # item.provenance is the complete provenance dictionary.
        #
        # It must NOT be assigned directly to:
        #
        #     ResponseEvidence.source_reference
        #
        # Instead, extract the individual fields explicitly.
        #

        evidence: list[ResponseEvidence] = []

        for item in state.evidence:
            source_reference = None
            score = None
            metadata: dict[str, Any] = {}

            if isinstance(item.provenance, dict):
                source_reference = item.provenance.get(
                    "source_reference"
                )

                score = item.provenance.get(
                    "score"
                )

                raw_metadata = item.provenance.get(
                    "metadata",
                    {},
                )

                if isinstance(raw_metadata, dict):
                    metadata = raw_metadata
                elif raw_metadata is not None:
                    metadata = {
                        "raw_metadata": raw_metadata
                    }

            # --------------------------------------------------------------
            # Contract normalization
            # --------------------------------------------------------------
            #
            # ResponseEvidence.source_reference is:
            #
            #     str | None
            #
            # Therefore, never pass arbitrary objects through.
            #

            if source_reference is not None and not isinstance(
                source_reference,
                str,
            ):
                source_reference = str(
                    source_reference
                )

            evidence.append(
                ResponseEvidence(
                    evidence_id=item.evidence_id,
                    source=item.source,
                    content=item.content,
                    source_reference=source_reference,
                    score=score,
                    metadata=metadata,
                )
            )

        # ------------------------------------------------------------------
        # Final Response Composer state
        # ------------------------------------------------------------------

        return ResponseComposerAgentState(
            request=state.request,
            classification_signals=classification_signals,
            evidence=evidence,
            resolution=resolution_signal,
            escalation=escalation_signal,
        )

    # ======================================================================
    # Helpers
    # ======================================================================

    @staticmethod
    def _prediction_value(
        prediction: Any,
    ) -> PredictionSignal | None:
        """
        Convert ClassificationPrediction into Coordinator
        PredictionSignal.

        None remains None.
        """

        if prediction is None:
            return None

        return PredictionSignal(
            value=str(prediction.value),
            confidence=prediction.confidence,
            source="classification_agent",
            metadata=(
                prediction.metadata
                if hasattr(
                    prediction,
                    "metadata",
                )
                else {}
            ),
        )

    @staticmethod
    def _mark_completed(
        state: CoordinatorState,
        capability: Capability,
    ) -> None:
        if capability not in state.completed_capabilities:
            state.completed_capabilities.append(
                capability
            )

    @staticmethod
    def _capability_to_step(
        capability: Capability,
    ) -> WorkflowStep:
        mapping = {
            Capability.CLASSIFICATION: (
                WorkflowStep.CLASSIFICATION
            ),
            Capability.RETRIEVAL: (
                WorkflowStep.RETRIEVAL
            ),
            Capability.RESOLUTION: (
                WorkflowStep.RESOLUTION
            ),
            Capability.ESCALATION: (
                WorkflowStep.ESCALATION
            ),
            Capability.GENERATION: (
                WorkflowStep.GENERATION
            ),
        }

        return mapping[capability]

    @staticmethod
    def _agent_name_for_capability(
        capability: Capability,
    ) -> str:
        mapping = {
            Capability.CLASSIFICATION: (
                "classification_agent"
            ),
            Capability.RETRIEVAL: (
                "retrieval_agent"
            ),
            Capability.RESOLUTION: (
                "resolution_agent"
            ),
            Capability.ESCALATION: (
                "escalation_agent"
            ),
            Capability.GENERATION: (
                "response_composer_agent"
            ),
        }

        return mapping[capability]

    @staticmethod
    def _validate_state(
        state: CoordinatorState,
    ) -> None:
        if not isinstance(
            state,
            CoordinatorState,
        ):
            raise TypeError(
                "state must be an instance of "
                "CoordinatorState."
            )

        if not state.request.strip():
            raise ValueError(
                "request must not be empty."
            )

    def _fail_workflow(
        self,
        *,
        state: CoordinatorState,
        error_type: str,
        message: str,
    ) -> None:
        state.status = WorkflowStatus.FAILED

        state.errors.append(
            WorkflowError(
                source=self.name,
                error_type=error_type,
                message=message,
                recoverable=False,
            )
        )