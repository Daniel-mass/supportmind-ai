from app.agents.coordinator.router import (
    CoordinatorRouter,
    RoutingDecision,
)
from app.agents.coordinator.state import (
    Capability,
    CoordinatorState,
    WorkflowStatus,
)


def test_router_selects_first_requested_capability():
    state = CoordinatorState(
        request="Analyze this customer issue.",
        requested_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
        ],
    )

    router = CoordinatorRouter()

    decision = router.route(state)

    assert isinstance(decision, RoutingDecision)
    assert decision.capability == Capability.CLASSIFICATION
    assert decision.should_continue is True
    assert decision.should_terminate is False


def test_router_selects_next_incomplete_capability():
    state = CoordinatorState(
        request="Analyze this customer issue.",
        requested_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
        ],
        completed_capabilities=[
            Capability.CLASSIFICATION,
        ],
    )

    router = CoordinatorRouter()

    decision = router.route(state)

    assert decision.capability == Capability.RETRIEVAL
    assert decision.should_continue is True
    assert decision.should_terminate is False


def test_router_selects_generation_after_requested_capabilities():
    state = CoordinatorState(
        request="Explain this issue.",
        requested_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
        ],
        completed_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
        ],
    )

    router = CoordinatorRouter()

    decision = router.route(state)

    assert decision.capability == Capability.GENERATION
    assert decision.should_continue is True
    assert decision.should_terminate is False


def test_router_terminates_after_generation():
    state = CoordinatorState(
        request="Explain this issue.",
        requested_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
        ],
        completed_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
            Capability.GENERATION,
        ],
    )

    router = CoordinatorRouter()

    decision = router.route(state)

    assert decision.capability is None
    assert decision.should_continue is False
    assert decision.should_terminate is True


def test_router_terminates_failed_workflow():
    state = CoordinatorState(
        request="Investigate this issue.",
        status=WorkflowStatus.FAILED,
        requested_capabilities=[
            Capability.RETRIEVAL,
        ],
    )

    router = CoordinatorRouter()

    decision = router.route(state)

    assert decision.capability is None
    assert decision.should_continue is False
    assert decision.should_terminate is True


def test_router_terminates_completed_workflow():
    state = CoordinatorState(
        request="Investigate this issue.",
        status=WorkflowStatus.COMPLETED,
    )

    router = CoordinatorRouter()

    decision = router.route(state)

    assert decision.capability is None
    assert decision.should_continue is False
    assert decision.should_terminate is True


def test_router_does_not_repeat_completed_capability():
    state = CoordinatorState(
        request="Investigate this issue.",
        requested_capabilities=[
            Capability.CLASSIFICATION,
        ],
        completed_capabilities=[
            Capability.CLASSIFICATION,
        ],
    )

    router = CoordinatorRouter()

    decision = router.route(state)

    assert decision.capability == Capability.GENERATION


def test_router_preserves_requested_order():
    state = CoordinatorState(
        request="Resolve this issue.",
        requested_capabilities=[
            Capability.RETRIEVAL,
            Capability.CLASSIFICATION,
            Capability.RESOLUTION,
        ],
    )

    router = CoordinatorRouter()

    first = router.route(state)

    assert first.capability == Capability.RETRIEVAL


def test_routing_decision_contains_reason():
    state = CoordinatorState(
        request="Classify this ticket.",
        requested_capabilities=[
            Capability.CLASSIFICATION,
        ],
    )

    router = CoordinatorRouter()

    decision = router.route(state)

    assert decision.reason
    assert "classification" in decision.reason.lower()