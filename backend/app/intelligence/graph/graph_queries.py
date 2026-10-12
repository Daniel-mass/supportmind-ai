##intelligence/graph/graph_queries.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphQuery:
    """
    Named, parameterized Cypher query used by the operational graph layer.

    Values are always supplied separately as Cypher parameters.
    User-provided values must never be interpolated into the query string.
    """

    name: str
    cypher: str


# ---------------------------------------------------------------------
# Ticket queries
# ---------------------------------------------------------------------

GET_TICKET = GraphQuery(
    name="get_ticket",
    cypher="""
MATCH (t:Ticket {ticket_id: $ticket_id})
RETURN t
""".strip(),
)


GET_TICKET_NEIGHBORHOOD = GraphQuery(
    name="get_ticket_neighborhood",
    cypher="""
MATCH (t:Ticket {ticket_id: $ticket_id})
OPTIONAL MATCH p=(t)-[r]-(n)
RETURN
    t,
    collect(DISTINCT p) AS paths
""".strip(),
)


GET_TICKET_WORKFLOW = GraphQuery(
    name="get_ticket_workflow",
    cypher="""
MATCH (t:Ticket {ticket_id: $ticket_id})
      -[:HAS_WORKFLOW_EVENT]
      ->(e:WorkflowEvent)
RETURN e
ORDER BY e.timestamp ASC, e.event_id ASC
""".strip(),
)


GET_TICKET_WORKFLOW_TRANSITIONS = GraphQuery(
    name="get_ticket_workflow_transitions",
    cypher="""
MATCH (t:Ticket {ticket_id: $ticket_id})
      -[:HAS_WORKFLOW_EVENT]
      ->(e1:WorkflowEvent)
      -[:NEXT]
      ->(e2:WorkflowEvent)
RETURN
    e1,
    e2
ORDER BY e1.timestamp ASC, e1.event_id ASC
""".strip(),
)


# ---------------------------------------------------------------------
# Agent queries
# ---------------------------------------------------------------------

GET_AGENT_HANDOFFS = GraphQuery(
    name="get_agent_handoffs",
    cypher="""
MATCH (a:Agent {agent_id: $agent_id})
      -[r:HANDOFF_TO]
      ->(b:Agent)
RETURN
    a,
    r,
    b
ORDER BY
    r.timestamp ASC,
    r.handoff_id ASC
""".strip(),
)


GET_AGENT_TICKETS = GraphQuery(
    name="get_agent_tickets",
    cypher="""
MATCH (t:Ticket)
      -[:ASSIGNED_TO]
      ->(a:Agent {agent_id: $agent_id})
RETURN t
ORDER BY
    t.status ASC,
    t.priority ASC,
    t.ticket_id ASC
""".strip(),
)


# ---------------------------------------------------------------------
# Team queries
# ---------------------------------------------------------------------

GET_TEAM_MEMBERS = GraphQuery(
    name="get_team_members",
    cypher="""
MATCH (a:Agent)
      -[:MEMBER_OF]
      ->(team:Team {name: $team})
RETURN a
ORDER BY
    a.name ASC,
    a.agent_id ASC
""".strip(),
)


GET_TEAM_TICKET_WORKLOAD = GraphQuery(
    name="get_team_ticket_workload",
    cypher="""
MATCH (t:Ticket)
      -[:ASSIGNED_TO]
      ->(a:Agent)
      -[:MEMBER_OF]
      ->(:Team {name: $team})
RETURN
    t.status AS status,
    count(t) AS ticket_count
ORDER BY
    ticket_count DESC,
    status ASC
""".strip(),
)


# ---------------------------------------------------------------------
# Organization / workflow queries
# ---------------------------------------------------------------------

GET_STATUS_TRANSITIONS = GraphQuery(
    name="get_status_transitions",
    cypher="""
MATCH (e1:WorkflowEvent)
      -[:NEXT]
      ->(e2:WorkflowEvent)
WHERE
    e1.status IS NOT NULL
    AND e2.status IS NOT NULL
RETURN
    e1.status AS from_status,
    e2.status AS to_status,
    count(*) AS transition_count
ORDER BY
    transition_count DESC,
    from_status ASC,
    to_status ASC
""".strip(),
)


GET_WORKFLOW_EVENT_VOLUME = GraphQuery(
    name="get_workflow_event_volume",
    cypher="""
MATCH (e:WorkflowEvent)
RETURN
    e.activity AS activity,
    count(e) AS event_count
ORDER BY
    event_count DESC,
    activity ASC
""".strip(),
)


GET_PRIORITY_DISTRIBUTION = GraphQuery(
    name="get_priority_distribution",
    cypher="""
MATCH (t:Ticket)
RETURN
    t.priority AS priority,
    count(t) AS ticket_count
ORDER BY
    ticket_count DESC,
    priority ASC
""".strip(),
)


GET_STATUS_DISTRIBUTION = GraphQuery(
    name="get_status_distribution",
    cypher="""
MATCH (t:Ticket)
RETURN
    t.status AS status,
    count(t) AS ticket_count
ORDER BY
    ticket_count DESC,
    status ASC
""".strip(),
)


# ---------------------------------------------------------------------
# Query registry
# ---------------------------------------------------------------------

GRAPH_QUERIES: dict[str, GraphQuery] = {
    query.name: query
    for query in [
        GET_TICKET,
        GET_TICKET_NEIGHBORHOOD,
        GET_TICKET_WORKFLOW,
        GET_TICKET_WORKFLOW_TRANSITIONS,
        GET_AGENT_HANDOFFS,
        GET_AGENT_TICKETS,
        GET_TEAM_MEMBERS,
        GET_TEAM_TICKET_WORKLOAD,
        GET_STATUS_TRANSITIONS,
        GET_WORKFLOW_EVENT_VOLUME,
        GET_PRIORITY_DISTRIBUTION,
        GET_STATUS_DISTRIBUTION,
    ]
}


def get_graph_query(
    name: str,
) -> GraphQuery:
    """
    Return a registered operational graph query.
    """

    try:
        return GRAPH_QUERIES[name]
    except KeyError as exc:
        available = ", ".join(
            sorted(GRAPH_QUERIES)
        )

        raise KeyError(
            f"Unknown graph query '{name}'. "
            f"Available queries: {available}"
        ) from exc


def list_graph_queries() -> tuple[str, ...]:
    """
    Return all registered graph query names.
    """

    return tuple(
        sorted(GRAPH_QUERIES)
    )