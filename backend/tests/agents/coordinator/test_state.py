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


def test_default_coordinator_state():
    state = CoordinatorState(
        request="Why are customers reporting payment failures?"
    )

    assert state.status == WorkflowStatus.PENDING
    assert state.current_step is None
    assert state.requested_capabilities == []
    assert state.completed_capabilities == []
    assert state.predictions == ClassificationSignals()
    assert state.agent_results == {}
    assert state.evidence == []
    assert state.errors == []
    assert state.iteration_count == 0
    assert state.final_response is None


def test_ml_predictions_are_stored_as_signals():
    state = CoordinatorState(
        request="My payment failed twice."
    )

    state.predictions.priority = PredictionSignal(
        value="High",
        confidence=0.87,
        source="priority_model",
    )

    assert state.predictions.priority is not None
    assert state.predictions.priority.value == "High"
    assert state.predictions.priority.confidence == 0.87
    assert state.predictions.priority.source == "priority_model"


def test_predictions_remain_separate_from_evidence():
    state = CoordinatorState(
        request="My payment failed twice."
    )

    state.predictions.priority = PredictionSignal(
        value="Critical",
        confidence=0.91,
        source="priority_model",
    )

    state.evidence.append(
        EvidenceReference(
            evidence_id="E1",
            source="payment_policy",
            content="Payment failures may require verification.",
        )
    )

    assert state.predictions.priority.value == "Critical"
    assert len(state.evidence) == 1
    assert state.evidence[0].evidence_id == "E1"


def test_requested_and_completed_capabilities():
    state = CoordinatorState(
        request="Find similar payment incidents."
    )

    state.requested_capabilities = [
        Capability.CLASSIFICATION,
        Capability.RETRIEVAL,
    ]

    state.completed_capabilities.append(
        Capability.CLASSIFICATION
    )

    assert Capability.CLASSIFICATION in state.requested_capabilities
    assert Capability.RETRIEVAL in state.requested_capabilities
    assert Capability.CLASSIFICATION in state.completed_capabilities
    assert Capability.RETRIEVAL not in state.completed_capabilities


def test_agent_result_storage():
    state = CoordinatorState(
        request="Explain this billing issue."
    )

    result = AgentResult(
        agent_name="classification_agent",
        success=True,
        result={
            "category": "Billing",
            "priority": "High",
        },
    )

    state.agent_results[result.agent_name] = result

    assert "classification_agent" in state.agent_results
    assert state.agent_results["classification_agent"].success is True
    assert state.agent_results["classification_agent"].result["category"] == "Billing"


def test_evidence_provenance_is_preserved():
    evidence = EvidenceReference(
        evidence_id="E10",
        source="neo4j_operational_graph",
        content="Multiple handoffs detected.",
        provenance={
            "ticket_id": "T123",
            "graph": "help_desk_operational",
        },
    )

    assert evidence.evidence_id == "E10"
    assert evidence.provenance["ticket_id"] == "T123"


def test_workflow_error_storage():
    state = CoordinatorState(
        request="Investigate unresolved tickets."
    )

    error = WorkflowError(
        source="retrieval_agent",
        error_type="timeout",
        message="Retrieval service timed out.",
        recoverable=True,
    )

    state.errors.append(error)

    assert len(state.errors) == 1
    assert state.errors[0].recoverable is True


def test_state_serialization_round_trip():
    state = CoordinatorState(
        request="Analyze payment complaints.",
        status=WorkflowStatus.RUNNING,
        current_step=WorkflowStep.CLASSIFICATION,
        requested_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
        ],
        predictions=ClassificationSignals(
            category=PredictionSignal(
                value="Billing",
                confidence=0.93,
                source="category_model",
            )
        ),
    )

    payload = state.model_dump()
    restored = CoordinatorState.model_validate(payload)

    assert restored.request == state.request
    assert restored.status == WorkflowStatus.RUNNING
    assert restored.current_step == WorkflowStep.CLASSIFICATION
    assert restored.predictions.category.value == "Billing"
    assert restored.predictions.category.confidence == 0.93