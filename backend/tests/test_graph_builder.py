from datetime import datetime, timezone

from backend.app.intelligence.graph.graph_builder import (
    build_agent_team_relationship,
    build_handoff_relationships,
    build_operational_graph,
    build_ticket_node,
    build_ticket_relationships,
    build_workflow_event_node,
    build_workflow_relationships,
)
from backend.app.intelligence.schemas.agent import Agent
from backend.app.intelligence.schemas.ticket import Ticket
from backend.app.intelligence.schemas.workflow import WorkflowEvent
from backend.app.intelligence.workflow.handoffs import HandoffRecord


TIMESTAMP = datetime(
    2026,
    1,
    1,
    10,
    0,
    tzinfo=timezone.utc,
)


def make_ticket(
    *,
    ticket_id: str = "PROJ:100",
    priority: str = "High",
    status: str = "in_progress",
    assignee: str | None = "agent-001",
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        project="PROJ",
        issue_num=100,
        issue_type="Bug",
        priority=priority,
        status=status,
        created_at=TIMESTAMP,
        started_at=TIMESTAMP,
        ended_at=None,
        resolution_date=None,
        resolution=None,
        reporter="user@example.com",
        assignee=assignee,
        contributor_count=1,
        comments_count=2,
        source="helpdesk",
        metadata={},
    )


def make_agent(
    *,
    agent_id: str = "agent-001",
    team: str | None = "Team-A",
) -> Agent:
    return Agent(
        agent_id=agent_id,
        name="Agent One",
        team=team,
        active=True,
        source="helpdesk",
        metadata={},
    )


def make_event(
    *,
    event_id: str,
    ticket_id: str = "PROJ:100",
    timestamp: datetime = TIMESTAMP,
    activity: str = "status_change",
    status: str | None = "in_progress",
    assignee: str | None = "agent-001",
) -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        ticket_id=ticket_id,
        timestamp=timestamp,
        activity=activity,
        status=status,
        assignee=assignee,
        team=None,
        priority="High",
        category="Bug",
        source="helpdesk_change_history",
        metadata={},
    )


def make_handoff(
    *,
    handoff_id: str = "handoff-001",
    ticket_id: str = "PROJ:100",
    from_agent: str = "agent-001",
    to_agent: str = "agent-002",
) -> HandoffRecord:
    return HandoffRecord(
        handoff_id=handoff_id,
        ticket_id=ticket_id,
        from_agent=from_agent,
        to_agent=to_agent,
        handoff_timestamp=TIMESTAMP,
        from_team="Team-A",
        to_team="Team-B",
        status_before="in_progress",
        status_after="waiting",
        time_before_handoff_seconds=100,
        time_after_handoff_seconds=200,
    )


def test_build_ticket_node():
    ticket = make_ticket()

    node = build_ticket_node(ticket)

    assert node.node_id == "ticket:PROJ:100"
    assert node.label == "Ticket"

    assert node.properties["ticket_id"] == "PROJ:100"
    assert node.properties["project"] == "PROJ"
    assert node.properties["issue_num"] == 100
    assert node.properties["issue_type"] == "Bug"
    assert node.properties["priority"] == "High"
    assert node.properties["status"] == "in_progress"


def test_build_ticket_relationships():
    ticket = make_ticket()

    relationships = build_ticket_relationships(
        ticket
    )

    relationship_types = {
        relationship.relationship_type
        for relationship in relationships
    }

    assert "HAS_TYPE" in relationship_types
    assert "HAS_PRIORITY" in relationship_types
    assert "HAS_STATUS" in relationship_types
    assert "ASSIGNED_TO" in relationship_types


def test_unassigned_ticket_has_no_assigned_to_relationship():
    ticket = make_ticket(
        assignee=None
    )

    relationships = build_ticket_relationships(
        ticket
    )

    relationship_types = {
        relationship.relationship_type
        for relationship in relationships
    }

    assert "HAS_TYPE" in relationship_types
    assert "HAS_PRIORITY" in relationship_types
    assert "HAS_STATUS" in relationship_types
    assert "ASSIGNED_TO" not in relationship_types


def test_build_workflow_event_node():
    event = make_event(
        event_id="event-001"
    )

    node = build_workflow_event_node(
        event
    )

    assert (
        node.node_id
        == "workflow_event:event-001"
    )

    assert node.label == "WorkflowEvent"

    assert (
        node.properties["event_id"]
        == "event-001"
    )

    assert (
        node.properties["ticket_id"]
        == "PROJ:100"
    )


def test_workflow_relationships_include_ticket_event_edges():
    events = [
        make_event(
            event_id="event-001",
            timestamp=TIMESTAMP,
        ),
        make_event(
            event_id="event-002",
            timestamp=(
                TIMESTAMP.replace(
                    minute=10
                )
            ),
        ),
    ]

    relationships = (
        build_workflow_relationships(
            events
        )
    )

    relationship_types = [
        relationship.relationship_type
        for relationship in relationships
    ]

    assert relationship_types.count(
        "HAS_WORKFLOW_EVENT"
    ) == 2

    assert relationship_types.count(
        "NEXT"
    ) == 1


def test_workflow_events_are_sorted_before_next_relationships():
    events = [
        make_event(
            event_id="event-002",
            timestamp=(
                TIMESTAMP.replace(
                    minute=20
                )
            ),
        ),
        make_event(
            event_id="event-001",
            timestamp=TIMESTAMP,
        ),
    ]

    relationships = (
        build_workflow_relationships(
            events
        )
    )

    next_relationships = [
        relationship
        for relationship in relationships
        if relationship.relationship_type
        == "NEXT"
    ]

    assert len(
        next_relationships
    ) == 1

    assert (
        next_relationships[0].source_id
        == "workflow_event:event-001"
    )

    assert (
        next_relationships[0].target_id
        == "workflow_event:event-002"
    )


def test_agent_team_relationship():
    agent = make_agent()

    relationships = (
        build_agent_team_relationship(
            agent
        )
    )

    assert len(relationships) == 1

    relationship = relationships[0]

    assert (
        relationship.relationship_type
        == "MEMBER_OF"
    )

    assert (
        relationship.source_id
        == "agent:agent-001"
    )

    assert (
        relationship.target_id
        == "team:Team-A"
    )


def test_agent_without_team_has_no_team_relationship():
    agent = make_agent(
        team=None
    )

    relationships = (
        build_agent_team_relationship(
            agent
        )
    )

    assert relationships == []


def test_handoff_relationship():
    handoff = make_handoff()

    relationships = (
        build_handoff_relationships(
            [handoff]
        )
    )

    assert len(relationships) == 1

    relationship = relationships[0]

    assert (
        relationship.relationship_type
        == "HANDOFF_TO"
    )

    assert (
        relationship.source_id
        == "agent:agent-001"
    )

    assert (
        relationship.target_id
        == "agent:agent-002"
    )

    assert (
        relationship.properties[
            "handoff_id"
        ]
        == "handoff-001"
    )


def test_operational_graph_deduplicates_nodes():
    ticket = make_ticket()
    agent = make_agent()

    events = [
        make_event(
            event_id="event-001"
        ),
    ]

    graph = build_operational_graph(
        tickets=[ticket],
        agents=[agent],
        workflow_events=events,
        handoffs=[],
    )

    node_ids = [
        node.node_id
        for node in graph.nodes
    ]

    assert len(node_ids) == len(
        set(node_ids)
    )


def test_operational_graph_contains_expected_nodes_and_relationships():
    ticket = make_ticket()
    agent = make_agent()

    events = [
        make_event(
            event_id="event-001"
        ),
        make_event(
            event_id="event-002",
            timestamp=(
                TIMESTAMP.replace(
                    minute=5
                )
            ),
        ),
    ]

    handoff = make_handoff()

    graph = build_operational_graph(
        tickets=[ticket],
        agents=[agent],
        workflow_events=events,
        handoffs=[handoff],
    )

    labels = {
        node.label
        for node in graph.nodes
    }

    relationship_types = {
        relationship.relationship_type
        for relationship
        in graph.relationships
    }

    assert "Ticket" in labels
    assert "Agent" in labels
    assert "WorkflowEvent" in labels
    assert "IssueType" in labels
    assert "Priority" in labels
    assert "Status" in labels
    assert "Team" in labels

    assert "HAS_TYPE" in relationship_types
    assert "HAS_PRIORITY" in relationship_types
    assert "HAS_STATUS" in relationship_types
    assert "ASSIGNED_TO" in relationship_types
    assert "HAS_WORKFLOW_EVENT" in relationship_types
    assert "NEXT" in relationship_types
    assert "MEMBER_OF" in relationship_types
    assert "HANDOFF_TO" in relationship_types