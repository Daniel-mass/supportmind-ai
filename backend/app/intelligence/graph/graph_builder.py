from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.intelligence.schemas.agent import Agent
from app.intelligence.schemas.ticket import Ticket
from app.intelligence.schemas.workflow import WorkflowEvent
from app.intelligence.workflow.handoffs import HandoffRecord


@dataclass(frozen=True)
class GraphNode:
    """
    Neo4j-independent representation of a graph node.
    """

    node_id: str
    label: str
    properties: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class GraphRelationship:
    """
    Neo4j-independent representation of a graph relationship.
    """

    relationship_id: str
    relationship_type: str
    source_id: str
    target_id: str
    properties: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class OperationalGraph:
    """
    Complete graph payload before persistence to Neo4j.
    """

    nodes: list[GraphNode]
    relationships: list[GraphRelationship]


def _enum_value(value: object) -> str:
    """
    Return raw value for Enum-like values.
    """
    return str(
        getattr(value, "value", value)
    )


def _normalize_text(
    value: object,
) -> str | None:
    """
    Normalize optional text values.
    """
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def ticket_node_id(
    ticket_id: str,
) -> str:
    return f"ticket:{ticket_id}"


def agent_node_id(
    agent_id: str,
) -> str:
    return f"agent:{agent_id}"


def workflow_event_node_id(
    event_id: str,
) -> str:
    return f"workflow_event:{event_id}"


def issue_type_node_id(
    issue_type: str,
) -> str:
    return f"issue_type:{issue_type}"


def priority_node_id(
    priority: str,
) -> str:
    return f"priority:{priority}"


def status_node_id(
    status: str,
) -> str:
    return f"status:{status}"


def team_node_id(
    team: str,
) -> str:
    return f"team:{team}"


def build_ticket_node(
    ticket: Ticket,
) -> GraphNode:
    return GraphNode(
        node_id=ticket_node_id(
            ticket.ticket_id
        ),
        label="Ticket",
        properties={
            "ticket_id": ticket.ticket_id,
            "project": ticket.project,
            "issue_num": ticket.issue_num,
            "issue_type": ticket.issue_type,
            "priority": _enum_value(
                ticket.priority
            ),
            "status": _enum_value(
                ticket.status
            ),
            "source": ticket.source,
        },
    )


def build_agent_node(
    agent: Agent,
) -> GraphNode:
    return GraphNode(
        node_id=agent_node_id(
            agent.agent_id
        ),
        label="Agent",
        properties={
            "agent_id": agent.agent_id,
            "name": agent.name,
            "team": agent.team,
            "active": agent.active,
            "source": agent.source,
        },
    )


def build_workflow_event_node(
    event: WorkflowEvent,
) -> GraphNode:
    timestamp = event.timestamp

    return GraphNode(
        node_id=workflow_event_node_id(
            event.event_id
        ),
        label="WorkflowEvent",
        properties={
            "event_id": event.event_id,
            "ticket_id": event.ticket_id,
            "timestamp": timestamp,
            "activity": event.activity,
            "status": _normalize_text(
                event.status
            ),
            "assignee": _normalize_text(
                event.assignee
            ),
            "team": _normalize_text(
                event.team
            ),
            "priority": _normalize_text(
                event.priority
            ),
            "category": _normalize_text(
                event.category
            ),
            "source": event.source,
        },
    )


def build_issue_type_node(
    issue_type: str,
) -> GraphNode:
    issue_type = issue_type.strip()

    return GraphNode(
        node_id=issue_type_node_id(
            issue_type
        ),
        label="IssueType",
        properties={
            "name": issue_type,
        },
    )


def build_priority_node(
    priority: str,
) -> GraphNode:
    priority = priority.strip()

    return GraphNode(
        node_id=priority_node_id(
            priority
        ),
        label="Priority",
        properties={
            "name": priority,
        },
    )


def build_status_node(
    status: str,
) -> GraphNode:
    status = status.strip()

    return GraphNode(
        node_id=status_node_id(
            status
        ),
        label="Status",
        properties={
            "name": status,
        },
    )


def build_team_node(
    team: str,
) -> GraphNode:
    team = team.strip()

    return GraphNode(
        node_id=team_node_id(
            team
        ),
        label="Team",
        properties={
            "name": team,
        },
    )


def build_relationship(
    *,
    relationship_type: str,
    source_id: str,
    target_id: str,
    relationship_id: str | None = None,
    properties: dict[str, Any] | None = None,
) -> GraphRelationship:
    """
    Build a deterministic graph relationship.
    """

    if relationship_id is None:
        relationship_id = (
            f"{relationship_type.lower()}:"
            f"{source_id}:"
            f"{target_id}"
        )

    return GraphRelationship(
        relationship_id=relationship_id,
        relationship_type=relationship_type,
        source_id=source_id,
        target_id=target_id,
        properties=properties or {},
    )


def build_ticket_relationships(
    ticket: Ticket,
) -> list[GraphRelationship]:
    """
    Build structural relationships for a ticket.
    """

    ticket_id = ticket_node_id(
        ticket.ticket_id
    )

    issue_type = ticket.issue_type.strip()
    priority = _enum_value(
        ticket.priority
    ).strip()
    status = _enum_value(
        ticket.status
    ).strip()

    relationships = [
        build_relationship(
            relationship_type="HAS_TYPE",
            source_id=ticket_id,
            target_id=issue_type_node_id(
                issue_type
            ),
        ),
        build_relationship(
            relationship_type="HAS_PRIORITY",
            source_id=ticket_id,
            target_id=priority_node_id(
                priority
            ),
        ),
        build_relationship(
            relationship_type="HAS_STATUS",
            source_id=ticket_id,
            target_id=status_node_id(
                status
            ),
        ),
    ]

    assignee = _normalize_text(
        ticket.assignee
    )

    if assignee:
        relationships.append(
            build_relationship(
                relationship_type="ASSIGNED_TO",
                source_id=ticket_id,
                target_id=agent_node_id(
                    assignee
                ),
            )
        )

    return relationships


def build_workflow_relationships(
    events: Iterable[WorkflowEvent],
) -> list[GraphRelationship]:
    """
    Build ticket-to-event and sequential NEXT relationships.

    Events are grouped by ticket and ordered chronologically.
    """

    events_by_ticket: dict[
        str,
        list[WorkflowEvent],
    ] = {}

    for event in events:
        events_by_ticket.setdefault(
            event.ticket_id,
            [],
        ).append(event)

    relationships: list[
        GraphRelationship
    ] = []

    for ticket_id, ticket_events in (
        events_by_ticket.items()
    ):
        ordered_events = sorted(
            ticket_events,
            key=lambda item: (
                item.timestamp,
                item.event_id,
            ),
        )

        ticket_node = ticket_node_id(
            ticket_id
        )

        previous_event: WorkflowEvent | None = None

        for event in ordered_events:
            event_node = workflow_event_node_id(
                event.event_id
            )

            relationships.append(
                build_relationship(
                    relationship_type=(
                        "HAS_WORKFLOW_EVENT"
                    ),
                    source_id=ticket_node,
                    target_id=event_node,
                )
            )

            if previous_event is not None:
                relationships.append(
                    build_relationship(
                        relationship_type="NEXT",
                        source_id=(
                            workflow_event_node_id(
                                previous_event.event_id
                            )
                        ),
                        target_id=event_node,
                    )
                )

            previous_event = event

    return relationships


def build_agent_team_relationship(
    agent: Agent,
) -> list[GraphRelationship]:
    """
    Connect an agent to its known team.
    """

    team = _normalize_text(
        agent.team
    )

    if not team:
        return []

    return [
        build_relationship(
            relationship_type="MEMBER_OF",
            source_id=agent_node_id(
                agent.agent_id
            ),
            target_id=team_node_id(
                team
            ),
        )
    ]


def build_handoff_relationships(
    handoffs: Iterable[HandoffRecord],
) -> list[GraphRelationship]:
    """
    Convert workflow handoffs into Agent -> Agent edges.
    """

    relationships: list[
        GraphRelationship
    ] = []

    for handoff in handoffs:
        if not handoff.from_agent:
            continue

        if not handoff.to_agent:
            continue

        relationship_id = (
            f"handoff:"
            f"{handoff.handoff_id}"
        )

        relationships.append(
            build_relationship(
                relationship_type="HANDOFF_TO",
                source_id=agent_node_id(
                    handoff.from_agent
                ),
                target_id=agent_node_id(
                    handoff.to_agent
                ),
                relationship_id=relationship_id,
                properties={
                    "handoff_id": (
                        handoff.handoff_id
                    ),
                    "ticket_id": (
                        handoff.ticket_id
                    ),
                    "timestamp": (
                        handoff.handoff_timestamp
                    ),
                    "from_team": (
                        handoff.from_team
                    ),
                    "to_team": (
                        handoff.to_team
                    ),
                },
            )
        )

    return relationships


def build_operational_graph(
    *,
    tickets: Iterable[Ticket],
    agents: Iterable[Agent],
    workflow_events: Iterable[WorkflowEvent],
    handoffs: Iterable[HandoffRecord] | None = None,
) -> OperationalGraph:
    """
    Build the complete operational graph payload.

    This function does NOT connect to Neo4j.
    """

    ticket_list = list(tickets)
    agent_list = list(agents)
    event_list = list(workflow_events)
    handoff_list = list(handoffs or [])

    nodes_by_id: dict[
        str,
        GraphNode,
    ] = {}

    relationships_by_id: dict[
        str,
        GraphRelationship,
    ] = {}

    def add_node(node: GraphNode) -> None:
        nodes_by_id.setdefault(
            node.node_id,
            node,
        )

    def add_relationship(
        relationship: GraphRelationship,
    ) -> None:
        relationships_by_id.setdefault(
            relationship.relationship_id,
            relationship,
        )

    for ticket in ticket_list:
        add_node(
            build_ticket_node(ticket)
        )

        issue_type = (
            ticket.issue_type.strip()
        )

        priority = (
            _enum_value(
                ticket.priority
            ).strip()
        )

        status = (
            _enum_value(
                ticket.status
            ).strip()
        )

        add_node(
            build_issue_type_node(
                issue_type
            )
        )

        add_node(
            build_priority_node(
                priority
            )
        )

        add_node(
            build_status_node(
                status
            )
        )

        for relationship in (
            build_ticket_relationships(
                ticket
            )
        ):
            add_relationship(
                relationship
            )

    for agent in agent_list:
        add_node(
            build_agent_node(agent)
        )

        for relationship in (
            build_agent_team_relationship(
                agent
            )
        ):
            add_relationship(
                relationship
            )

        if agent.team:
            add_node(
                build_team_node(
                    agent.team
                )
            )

    for event in event_list:
        add_node(
            build_workflow_event_node(
                event
            )
        )

    for relationship in (
        build_workflow_relationships(
            event_list
        )
    ):
        add_relationship(
            relationship
        )

    for relationship in (
        build_handoff_relationships(
            handoff_list
        )
    ):
        add_relationship(
            relationship
        )

    return OperationalGraph(
        nodes=list(
            nodes_by_id.values()
        ),
        relationships=list(
            relationships_by_id.values()
        ),
    )
