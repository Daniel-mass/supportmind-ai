from __future__ import annotations

from app.agents.coordinator.state import (
    CoordinatorState,
    EvidenceReference,
)
from app.agents.response_composer_agent.contracts.state import (
    ResponseComposerAgentState,
)
from app.agents.orchestrator.orchestrator import AgentOrchestrator


def test_build_response_composer_state_flattens_evidence_provenance() -> None:
    state = CoordinatorState(
        request="A customer wants a refund.",
    )

    state.evidence.append(
        EvidenceReference(
            evidence_id="E1",
            source="refund_policy",
            content="Refunds are available within 14 days.",
            provenance={
                "source_reference": "node-refund-001",
                "evidence_type": "policy",
                "score": 0.91,
                "metadata": {
                    "context_source": "bm25",
                },
            },
        )
    )

    response_state = (
        AgentOrchestrator._build_response_composer_state(
            state
        )
    )

    assert isinstance(
        response_state,
        ResponseComposerAgentState,
    )

    assert len(
        response_state.evidence
    ) == 1

    evidence = response_state.evidence[0]

    assert evidence.evidence_id == "E1"
    assert evidence.source == "refund_policy"
    assert (
        evidence.content
        == "Refunds are available within 14 days."
    )

    assert (
        evidence.source_reference
        == "node-refund-001"
    )

    assert evidence.score == 0.91

    assert (
        evidence.metadata["context_source"]
        == "bm25"
    )