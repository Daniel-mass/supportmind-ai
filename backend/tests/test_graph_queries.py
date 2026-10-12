import pytest

from backend.app.intelligence.graph.graph_queries import (
    GRAPH_QUERIES,
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
    get_graph_query,
    list_graph_queries,
)


def test_all_queries_are_registered():
    expected = {
        "get_ticket",
        "get_ticket_neighborhood",
        "get_ticket_workflow",
        "get_ticket_workflow_transitions",
        "get_agent_handoffs",
        "get_agent_tickets",
        "get_team_members",
        "get_team_ticket_workload",
        "get_status_transitions",
        "get_workflow_event_volume",
        "get_priority_distribution",
        "get_status_distribution",
    }

    assert set(GRAPH_QUERIES) == expected


def test_all_queries_have_unique_names():
    names = [
        query.name
        for query in GRAPH_QUERIES.values()
    ]

    assert len(names) == len(set(names))


def test_get_graph_query_returns_expected_query():
    query = get_graph_query("get_ticket")

    assert query == GET_TICKET
    assert query.name == "get_ticket"


def test_unknown_query_raises_clear_error():
    with pytest.raises(KeyError, match="Unknown graph query"):
        get_graph_query("does_not_exist")


def test_list_graph_queries_is_sorted():
    names = list_graph_queries()

    assert names == tuple(
        sorted(names)
    )


def test_ticket_queries_use_parameterized_ticket_id():
    assert "$ticket_id" in GET_TICKET.cypher
    assert "$ticket_id" in (
        GET_TICKET_NEIGHBORHOOD.cypher
    )
    assert "$ticket_id" in (
        GET_TICKET_WORKFLOW.cypher
    )
    assert "$ticket_id" in (
        GET_TICKET_WORKFLOW_TRANSITIONS.cypher
    )


def test_agent_queries_use_parameterized_agent_id():
    assert "$agent_id" in (
        GET_AGENT_HANDOFFS.cypher
    )

    assert "$agent_id" in (
        GET_AGENT_TICKETS.cypher
    )


def test_team_queries_use_parameterized_team():
    assert "$team" in GET_TEAM_MEMBERS.cypher
    assert "$team" in (
        GET_TEAM_TICKET_WORKLOAD.cypher
    )


def test_ticket_workflow_query_uses_next_relationship():
    assert "HAS_WORKFLOW_EVENT" in (
        GET_TICKET_WORKFLOW.cypher
    )

    assert "HAS_WORKFLOW_EVENT" in (
        GET_TICKET_WORKFLOW_TRANSITIONS.cypher
    )

    assert "NEXT" in (
        GET_TICKET_WORKFLOW_TRANSITIONS.cypher
    )


def test_handoff_query_uses_handoff_relationship():
    assert "HANDOFF_TO" in (
        GET_AGENT_HANDOFFS.cypher
    )


def test_team_workload_query_uses_assignment_and_team_relationships():
    assert "ASSIGNED_TO" in (
        GET_TEAM_TICKET_WORKLOAD.cypher
    )

    assert "MEMBER_OF" in (
        GET_TEAM_TICKET_WORKLOAD.cypher
    )


def test_organization_distribution_queries_return_counts():
    assert "count(t)" in (
        GET_PRIORITY_DISTRIBUTION.cypher
    )

    assert "count(t)" in (
        GET_STATUS_DISTRIBUTION.cypher
    )

    assert "count(*)" in (
        GET_STATUS_TRANSITIONS.cypher
    )

    assert "count(e)" in (
        GET_WORKFLOW_EVENT_VOLUME.cypher
    )