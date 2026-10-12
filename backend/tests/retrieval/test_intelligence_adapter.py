"""
Unit tests for the Graph-RAG IntelligenceAdapter.
"""

from __future__ import annotations

import pytest

from app.retrieval.graph_rag.intelligence_adapter import (
    IntelligenceAdapter,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
)
from backend.app.intelligence.graph.graph_queries import (
    GET_AGENT_HANDOFFS,
    GET_AGENT_TICKETS,
    GET_PRIORITY_DISTRIBUTION,
    GET_STATUS_DISTRIBUTION,
    GET_STATUS_TRANSITIONS,
    GET_TEAM_MEMBERS,
    GET_TEAM_TICKET_WORKLOAD,
    GET_TICKET,
    GET_TICKET_NEIGHBORHOOD,
    GET_TICKET_WORKFLOW,
    GET_TICKET_WORKFLOW_TRANSITIONS,
    GET_WORKFLOW_EVENT_VOLUME,
)


class FakeOperationalGraph:
    """
    Deterministic fake implementation of the existing
    Neo4jOperationalGraph.execute_query() contract.
    """

    def __init__(
        self,
        responses: dict[Any, list[dict]],
    ) -> None:
        # Normalize response keys so queries can be matched either by query object or by name
        normalized_responses: dict[str, list[dict]] = {}
        for key, value in responses.items():
            query_name = getattr(key, "name", str(key))
            normalized_responses[query_name] = value

        self.responses = normalized_responses
        self.calls: list[tuple[str, dict]] = []

    def execute_query(
        self,
        query,
        parameters=None,
    ):
        parameters = parameters or {}
        query_name = getattr(query, "name", str(query))

        self.calls.append(
            (
                query_name,
                parameters,
            )
        )

        return list(
            self.responses.get(
                query_name,
                [],
            )
        )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ticket_graph():
    return FakeOperationalGraph(
        {
            GET_TICKET.name: [
                {
                    "t": {
                        "ticket_id": "ABC-123",
                        "project": "PAY",
                        "priority": "High",
                        "status": "Open",
                    }
                }
            ],
            GET_TICKET_NEIGHBORHOOD.name: [
                {
                    "t": {
                        "ticket_id": "ABC-123",
                        "priority": "High",
                    },
                    "paths": [],
                }
            ],
            GET_TICKET_WORKFLOW.name: [
                {
                    "e": {
                        "event_id": "E100",
                        "timestamp": "2026-01-01T10:00:00Z",
                        "status": "Open",
                        "activity": "created",
                    }
                }
            ],
            GET_TICKET_WORKFLOW_TRANSITIONS.name: [
                {
                    "e1": {
                        "event_id": "E100",
                        "status": "Open",
                    },
                    "e2": {
                        "event_id": "E101",
                        "status": "In Progress",
                    },
                }
            ],
        }
    )


# ============================================================================
# Constructor
# ============================================================================


def test_adapter_rejects_none_graph() -> None:
    with pytest.raises(ValueError):
        IntelligenceAdapter(None)


def test_adapter_rejects_invalid_graph() -> None:
    with pytest.raises(TypeError):
        IntelligenceAdapter(object())


# ============================================================================
# Ticket
# ============================================================================


def test_get_ticket_evidence_returns_ticket_and_workflow(
    ticket_graph,
) -> None:
    adapter = IntelligenceAdapter(
        ticket_graph
    )

    evidence = adapter.get_ticket_evidence(
        "ABC-123"
    )

    assert len(evidence) == 4

    assert evidence[0].source == (
        GraphRAGEvidenceSource.HELPDESK_GRAPH
    )

    assert any(
        item.evidence_type
        == GraphRAGEvidenceType.WORKFLOW_EVENT
        for item in evidence
    )

    assert ticket_graph.calls[0] == (
        GET_TICKET.name,
        {"ticket_id": "ABC-123"},
    )


def test_get_ticket_evidence_can_disable_optional_queries(
    ticket_graph,
) -> None:
    adapter = IntelligenceAdapter(
        ticket_graph
    )

    evidence = adapter.get_ticket_evidence(
        "ABC-123",
        include_neighborhood=False,
        include_workflow=False,
        include_transitions=False,
    )

    assert len(evidence) == 1

    assert [
        call[0]
        for call in ticket_graph.calls
    ] == [
        GET_TICKET.name
    ]


def test_get_ticket_evidence_rejects_empty_ticket_id(
    ticket_graph,
) -> None:
    adapter = IntelligenceAdapter(
        ticket_graph
    )

    with pytest.raises(ValueError):
        adapter.get_ticket_evidence("   ")


def test_ticket_evidence_ids_are_deterministic(
    ticket_graph,
) -> None:
    adapter = IntelligenceAdapter(
        ticket_graph
    )

    evidence = adapter.get_ticket_evidence(
        "ABC-123"
    )

    assert [
        item.evidence_id
        for item in evidence
    ] == [
        "E1",
        "E2",
        "E3",
        "E4",
    ]


# ============================================================================
# Agent
# ============================================================================


def test_get_agent_evidence() -> None:
    graph = FakeOperationalGraph(
        {
            GET_AGENT_HANDOFFS.name: [
                {
                    "a": {
                        "agent_id": "A1",
                    },
                    "r": {
                        "handoff_id": "H1",
                    },
                    "b": {
                        "agent_id": "A2",
                    },
                }
            ],
            GET_AGENT_TICKETS.name: [
                {
                    "t": {
                        "ticket_id": "ABC-123",
                        "status": "Open",
                    }
                }
            ],
        }
    )

    adapter = IntelligenceAdapter(
        graph
    )

    evidence = adapter.get_agent_evidence(
        "A1"
    )

    assert len(evidence) == 2

    assert any(
        item.evidence_type
        == GraphRAGEvidenceType.HANDOFF
        for item in evidence
    )


# ============================================================================
# Team
# ============================================================================


def test_get_team_evidence() -> None:
    graph = FakeOperationalGraph(
        {
            GET_TEAM_MEMBERS.name: [
                {
                    "a": {
                        "agent_id": "A1",
                        "name": "Agent One",
                    }
                }
            ],
            GET_TEAM_TICKET_WORKLOAD.name: [
                {
                    "status": "Open",
                    "ticket_count": 14,
                }
            ],
        }
    )

    adapter = IntelligenceAdapter(
        graph
    )

    evidence = adapter.get_team_evidence(
        "Payments"
    )

    assert len(evidence) == 2

    assert any(
        item.evidence_type
        == GraphRAGEvidenceType.METRIC
        for item in evidence
    )


# ============================================================================
# Organization
# ============================================================================


def test_get_organization_evidence() -> None:
    graph = FakeOperationalGraph(
        {
            GET_STATUS_TRANSITIONS.name: [
                {
                    "from_status": "Open",
                    "to_status": "In Progress",
                    "transition_count": 25,
                }
            ],
            GET_WORKFLOW_EVENT_VOLUME.name: [
                {
                    "activity": "comment",
                    "event_count": 80,
                }
            ],
            GET_PRIORITY_DISTRIBUTION.name: [
                {
                    "priority": "High",
                    "ticket_count": 14,
                }
            ],
            GET_STATUS_DISTRIBUTION.name: [
                {
                    "status": "Open",
                    "ticket_count": 20,
                }
            ],
        }
    )

    adapter = IntelligenceAdapter(
        graph
    )

    evidence = adapter.get_organization_evidence()

    assert len(evidence) == 4

    assert all(
        item.source
        == GraphRAGEvidenceSource.ORGANIZATION_INTELLIGENCE
        for item in evidence
    )


# ============================================================================
# Organization Query Filtering Tests
# ============================================================================


def test_get_relevant_organization_evidence_selects_priority_data():
    responses = {
        GET_PRIORITY_DISTRIBUTION: [
            {"priority": "High", "ticket_count": 14},
            {"priority": "Medium", "ticket_count": 20},
        ],
        GET_STATUS_DISTRIBUTION: [
            {"status": "Open", "ticket_count": 20},
        ],
        GET_WORKFLOW_EVENT_VOLUME: [
            {"activity": "comment", "event_count": 80},
        ],
        GET_STATUS_TRANSITIONS: [
            {
                "from_status": "Open",
                "to_status": "In Progress",
                "transition_count": 25,
            },
        ],
    }

    graph = FakeOperationalGraph(responses)
    adapter = IntelligenceAdapter(graph)

    evidence = adapter.get_relevant_organization_evidence(
        "What are the ticket priorities?"
    )

    assert len(evidence) == 2
    assert all(
        item.metadata["query_name"] == GET_PRIORITY_DISTRIBUTION.name
        for item in evidence
    )


def test_get_relevant_organization_evidence_selects_status_data():
    responses = {
        GET_PRIORITY_DISTRIBUTION: [
            {"priority": "High", "ticket_count": 14},
        ],
        GET_STATUS_DISTRIBUTION: [
            {"status": "Open", "ticket_count": 20},
            {"status": "Closed", "ticket_count": 15},
        ],
        GET_WORKFLOW_EVENT_VOLUME: [
            {"activity": "comment", "event_count": 80},
        ],
        GET_STATUS_TRANSITIONS: [
            {
                "from_status": "Open",
                "to_status": "In Progress",
                "transition_count": 25,
            },
        ],
    }

    graph = FakeOperationalGraph(responses)
    adapter = IntelligenceAdapter(graph)

    evidence = adapter.get_relevant_organization_evidence(
        "Show me the ticket statuses."
    )

    assert len(evidence) == 2
    assert all(
        item.metadata["query_name"] == GET_STATUS_DISTRIBUTION.name
        for item in evidence
    )


def test_get_relevant_organization_evidence_selects_workflow_data():
    responses = {
        GET_PRIORITY_DISTRIBUTION: [
            {"priority": "High", "ticket_count": 14},
        ],
        GET_STATUS_DISTRIBUTION: [
            {"status": "Open", "ticket_count": 20},
        ],
        GET_WORKFLOW_EVENT_VOLUME: [
            {"activity": "comment", "event_count": 80},
            {"activity": "assignment", "event_count": 40},
        ],
        GET_STATUS_TRANSITIONS: [
            {
                "from_status": "Open",
                "to_status": "In Progress",
                "transition_count": 25,
            },
        ],
    }

    graph = FakeOperationalGraph(responses)
    adapter = IntelligenceAdapter(graph)

    evidence = adapter.get_relevant_organization_evidence(
        "What workflow activities happen most often?"
    )

    assert len(evidence) == 2
    assert all(
        item.metadata["query_name"] == GET_WORKFLOW_EVENT_VOLUME.name
        for item in evidence
    )


def test_get_relevant_organization_evidence_selects_transition_data():
    responses = {
        GET_PRIORITY_DISTRIBUTION: [
            {"priority": "High", "ticket_count": 14},
        ],
        GET_STATUS_DISTRIBUTION: [
            {"status": "Open", "ticket_count": 20},
        ],
        GET_WORKFLOW_EVENT_VOLUME: [
            {"activity": "comment", "event_count": 80},
        ],
        GET_STATUS_TRANSITIONS: [
            {
                "from_status": "Open",
                "to_status": "In Progress",
                "transition_count": 25,
            },
            {
                "from_status": "In Progress",
                "to_status": "Resolved",
                "transition_count": 18,
            },
        ],
    }

    graph = FakeOperationalGraph(responses)
    adapter = IntelligenceAdapter(graph)

    evidence = adapter.get_relevant_organization_evidence(
        "What operational delays and status transitions are common?"
    )

    assert len(evidence) == 2
    assert all(
        item.metadata["query_name"] == GET_STATUS_TRANSITIONS.name
        for item in evidence
    )


def test_get_relevant_organization_evidence_selects_all_relevant_categories():
    responses = {
        GET_PRIORITY_DISTRIBUTION: [
            {"priority": "High", "ticket_count": 14},
        ],
        GET_STATUS_DISTRIBUTION: [
            {"status": "Open", "ticket_count": 20},
        ],
        GET_WORKFLOW_EVENT_VOLUME: [
            {"activity": "comment", "event_count": 80},
        ],
        GET_STATUS_TRANSITIONS: [
            {
                "from_status": "Open",
                "to_status": "In Progress",
                "transition_count": 25,
            },
        ],
    }

    graph = FakeOperationalGraph(responses)
    adapter = IntelligenceAdapter(graph)

    evidence = adapter.get_relevant_organization_evidence(
        "What does the support data show about ticket priorities, "
        "statuses, workflow activity, and common operational delays?"
    )

    assert len(evidence) == 4

    query_names = {
        item.metadata["query_name"]
        for item in evidence
    }

    assert query_names == {
        GET_PRIORITY_DISTRIBUTION.name,
        GET_STATUS_DISTRIBUTION.name,
        GET_WORKFLOW_EVENT_VOLUME.name,
        GET_STATUS_TRANSITIONS.name,
    }


# ============================================================================
# Serialization
# ============================================================================


def test_serialization_handles_nested_mapping(
    ticket_graph,
) -> None:
    adapter = IntelligenceAdapter(
        ticket_graph
    )

    serialized = adapter._serialize_value(
        {
            "ticket": {
                "id": "ABC-123",
                "values": [
                    1,
                    2,
                    3,
                ],
            }
        }
    )

    assert serialized == {
        "ticket": {
            "id": "ABC-123",
            "values": [
                1,
                2,
                3,
            ],
        }
    }


def test_formatting_ticket_evidence(
    ticket_graph,
) -> None:
    adapter = IntelligenceAdapter(
        ticket_graph
    )

    evidence = adapter.get_ticket_evidence(
        "ABC-123",
        include_neighborhood=False,
        include_workflow=False,
        include_transitions=False,
    )

    assert "ABC-123" in evidence[0].text
    assert "High" in evidence[0].text
    assert evidence[0].source_reference.startswith(
        "ticket:ABC-123"
    )