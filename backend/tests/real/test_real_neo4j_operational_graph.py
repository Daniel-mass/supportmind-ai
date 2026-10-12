from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from backend.app.intelligence.graph.graph_builder import (
    build_operational_graph,
)
from backend.app.intelligence.graph.operational_graph import (
    Neo4jOperationalGraph,
)
from backend.app.intelligence.schemas.agent import Agent
from backend.app.intelligence.schemas.ticket import Ticket
from backend.app.intelligence.schemas.workflow import WorkflowEvent
from backend.app.intelligence.workflow.bottlenecks import (
    identify_bottlenecks,
)
from backend.app.intelligence.workflow.event_builder import (
    build_workflow_intervals,
)
from backend.app.intelligence.workflow.handoffs import (
    identify_handoffs,
)
from backend.app.intelligence.workflow.metrics import (
    calculate_workflow_metrics,
)
from backend.app.intelligence.workflow.variants import (
    identify_workflow_variants,
)


PROJECT_ROOT = Path(
    r"C:\Users\danie\supportmind-ai"
)

HELPDESK_ROOT = (
    PROJECT_ROOT
    / "rawsupportminddata"
    / "Help Desk Tickets"
    / "Help Desk Tickets"
)

ISSUES_CSV = (
    HELPDESK_ROOT
    / "issues.csv"
)

CHANGE_HISTORY_CSV = (
    HELPDESK_ROOT
    / "issues_change_history.csv"
)

WORKFLOW_SAMPLE_SIZE = 5_000


def _value_from_row(
    row: pd.Series,
    candidates: list[str],
    default=None,
):
    for column in candidates:
        if column not in row.index:
            continue

        value = row[column]

        if pd.notna(value):
            return value

    return default


def _parse_optional_timestamp(value):
    if value is None or pd.isna(value):
        return None

    parsed = pd.to_datetime(
        value,
        format="mixed",
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return None

    return parsed.to_pydatetime()


def _parse_required_timestamp(
    row: pd.Series,
    candidates: list[str],
    field_name: str,
):
    value = _value_from_row(
        row,
        candidates,
    )

    parsed = _parse_optional_timestamp(
        value
    )

    assert parsed is not None, (
        f"Unable to derive {field_name}. "
        f"Checked columns: {candidates}"
    )

    return parsed


def _build_ticket_from_row(
    row: pd.Series,
) -> Ticket:
    project = str(
        _value_from_row(
            row,
            [
                "issue_proj",
                "project",
            ],
            default="UNKNOWN",
        )
    ).strip()

    issue_num = int(
        float(
            _value_from_row(
                row,
                [
                    "issue_num",
                    "issue_number",
                ],
                default=0,
            )
        )
    )

    ticket_id = (
        f"{project}:{issue_num}"
    )

    issue_type = str(
        _value_from_row(
            row,
            [
                "issue_type",
                "type",
            ],
            default="unknown",
        )
    ).strip()

    priority = str(
        _value_from_row(
            row,
            [
                "issue_priority",
                "priority",
            ],
            default="unknown",
        )
    ).strip()

    status = str(
        _value_from_row(
            row,
            [
                "issue_status",
                "status",
            ],
            default="unknown",
        )
    ).strip()

    started_at = _parse_required_timestamp(
        row,
        [
            "issue_started_at",
            "started_at",
            "started",
        ],
        "started_at",
    )

    created_at = _parse_optional_timestamp(
        _value_from_row(
            row,
            [
                "issue_created_at",
                "created_at",
                "created",
            ],
        )
    )

    if created_at is None:
        created_at = started_at

    ended_at = _parse_optional_timestamp(
        _value_from_row(
            row,
            [
                "issue_ended_at",
                "ended_at",
                "ended",
            ],
        )
    )

    resolution_date = _parse_optional_timestamp(
        _value_from_row(
            row,
            [
                "issue_resolution_date",
                "resolution_date",
                "resolved",
            ],
        )
    )

    assignee = _value_from_row(
        row,
        [
            "issue_assignee",
            "assignee",
        ],
    )

    contributor_count = _value_from_row(
        row,
        [
            "issue_contr_count",
            "contributor_count",
        ],
        default=0,
    )

    comments_count = _value_from_row(
        row,
        [
            "issue_comment_count",
            "comments_count",
            "comment_count",
        ],
        default=0,
    )

    try:
        contributor_count = int(
            float(contributor_count)
        )
    except (
        TypeError,
        ValueError,
    ):
        contributor_count = 0

    try:
        comments_count = int(
            float(comments_count)
        )
    except (
        TypeError,
        ValueError,
    ):
        comments_count = 0

    return Ticket(
        ticket_id=ticket_id,
        project=project,
        issue_num=issue_num,
        issue_type=issue_type,
        priority=priority,
        status=status,
        created_at=created_at,
        started_at=started_at,
        ended_at=ended_at,
        resolution_date=resolution_date,
        resolution=None,
        reporter=None,
        assignee=(
            str(assignee).strip()
            if assignee is not None
            and pd.notna(assignee)
            else None
        ),
        contributor_count=max(
            contributor_count,
            0,
        ),
        comments_count=max(
            comments_count,
            0,
        ),
        source="helpdesk",
        metadata={
            "graph_integration_test": True,
        },
    )


def _build_workflow_events(
    issues_df: pd.DataFrame,
    change_history_df: pd.DataFrame,
):
    issue_context = issues_df[
        [
            "id",
            "issue_proj",
            "issue_num",
            "issue_type",
            "issue_priority",
        ]
    ].copy()

    issue_context = issue_context.rename(
        columns={
            "id": "source_issue_id"
        }
    )

    issue_context["ticket_id"] = (
        issue_context[
            "issue_proj"
        ]
        .astype(str)
        .str.strip()
        + ":"
        + issue_context[
            "issue_num"
        ]
        .astype(int)
        .astype(str)
    )

    workflow_raw = change_history_df[
        change_history_df[
            "field"
        ].isin(
            [
                "status",
                "assignee",
            ]
        )
    ].copy()

    workflow_raw = workflow_raw.merge(
        issue_context,
        left_on="issueid",
        right_on="source_issue_id",
        how="inner",
    )

    workflow_raw["created"] = (
        pd.to_datetime(
            workflow_raw["created"],
            format="mixed",
            errors="coerce",
            utc=True,
        )
    )

    workflow_raw = (
        workflow_raw
        .dropna(subset=["created"])
        .sort_values(
            [
                "created",
                "issueid",
                "change_group_id",
                "id",
            ],
            kind="mergesort",
        )
        .head(
            WORKFLOW_SAMPLE_SIZE
        )
    )

    state_by_ticket: dict[
        str,
        dict[str, str | None],
    ] = defaultdict(
        lambda: {
            "status": None,
            "assignee": None,
        }
    )

    events: list[
        WorkflowEvent
    ] = []

    for row in workflow_raw.itertuples(
        index=False
    ):
        ticket_id = str(
            row.ticket_id
        )

        field = str(
            row.field
        ).strip().lower()

        value = (
            None
            if pd.isna(row.value)
            else str(row.value).strip()
        )

        state = (
            state_by_ticket[
                ticket_id
            ]
        )

        if field == "status":
            state["status"] = value
            activity = "status_change"
        else:
            state["assignee"] = value
            activity = "assignee_change"

        events.append(
            WorkflowEvent(
                event_id=f"change-{row.id}",
                ticket_id=ticket_id,
                timestamp=(
                    row.created
                    .to_pydatetime()
                ),
                activity=activity,
                status=state["status"],
                assignee=state["assignee"],
                team=None,
                priority=str(
                    row.issue_priority
                ).strip(),
                category=str(
                    row.issue_type
                ).strip(),
                source=(
                    "helpdesk_change_history"
                ),
                metadata={
                    "source_change_id": row.id,
                    "source_issue_id": (
                        row.issueid
                    ),
                    "field": field,
                    "change_group_id": (
                        row.change_group_id
                    ),
                },
            )
        )

    return events


def _count_nodes(
    store: Neo4jOperationalGraph,
    label: str,
) -> int:
    result = store.execute_cypher(
        f"""
        MATCH (n:{label})
        RETURN count(n) AS count
        """
    )

    return int(
        result[0]["count"]
    )


def _count_relationships(
    store: Neo4jOperationalGraph,
    relationship_type: str,
) -> int:
    result = store.execute_cypher(
        f"""
        MATCH ()-[r:{relationship_type}]->()
        RETURN count(r) AS count
        """
    )

    return int(
        result[0]["count"]
    )


def test_real_neo4j_operational_graph():
    """
    Real Neo4j integration test.

    This intentionally writes real SupportMind Help Desk data into
    the configured Neo4j database so the resulting graph can be
    inspected from Neo4j Browser.
    """

    assert ISSUES_CSV.exists(), (
        f"Missing file: {ISSUES_CSV}"
    )

    assert CHANGE_HISTORY_CSV.exists(), (
        f"Missing file: {CHANGE_HISTORY_CSV}"
    )

    # ---------------------------------------------------------
    # 1. Read real Help Desk data
    # ---------------------------------------------------------

    issues_df = pd.read_csv(
        ISSUES_CSV,
        low_memory=False,
    )

    change_history_df = pd.read_csv(
        CHANGE_HISTORY_CSV,
        low_memory=False,
    )

    assert len(issues_df) > 0
    assert len(change_history_df) > 0

    # ---------------------------------------------------------
    # 2. Build only tickets represented by workflow sample
    # ---------------------------------------------------------

    workflow_events = (
        _build_workflow_events(
            issues_df,
            change_history_df,
        )
    )

    assert len(
        workflow_events
    ) > 0

    workflow_ticket_ids = {
        event.ticket_id
        for event in workflow_events
    }

    tickets = [
        _build_ticket_from_row(
            row
        )
        for _, row in issues_df.iterrows()
        if (
            f"{str(row['issue_proj']).strip()}:"
            f"{int(float(row['issue_num']))}"
        )
        in workflow_ticket_ids
    ]

    tickets_by_id = {
        ticket.ticket_id: ticket
        for ticket in tickets
    }

    assert (
        len(tickets_by_id)
        > 0
    )

    # ---------------------------------------------------------
    # 3. Build agents represented in workflow sample
    # ---------------------------------------------------------

    agent_ids = sorted(
        {
            event.assignee
            for event in workflow_events
            if event.assignee
        }
    )

    agents = [
        Agent(
            agent_id=agent_id,
            name=agent_id,
            team="helpdesk",
            active=True,
            source="helpdesk",
            metadata={
                "graph_integration_test": True
            },
        )
        for agent_id in agent_ids
    ]

    # ---------------------------------------------------------
    # 4. Workflow intelligence
    # ---------------------------------------------------------

    workflow_intervals = (
        build_workflow_intervals(
            workflow_events
        )
    )

    assert len(
        workflow_intervals
    ) > 0

    workflow_metrics = (
        calculate_workflow_metrics(
            workflow_intervals
        )
    )

    if isinstance(
        workflow_metrics,
        dict,
    ):
        workflow_metrics_list = list(
            workflow_metrics.values()
        )
    else:
        workflow_metrics_list = list(
            workflow_metrics
        )

    handoffs = list(
        identify_handoffs(
            workflow_intervals
        )
    )

    # These are calculated to ensure the real workflow module
    # remains exercised by the graph integration test.
    bottlenecks = list(
        identify_bottlenecks(
            workflow_intervals
        )
    )

    workflow_variants = list(
        identify_workflow_variants(
            workflow_intervals
        )
    )

    assert len(
        workflow_metrics_list
    ) > 0

    # ---------------------------------------------------------
    # 5. Build operational graph
    # ---------------------------------------------------------

    graph = build_operational_graph(
        tickets=tickets,
        agents=agents,
        workflow_events=workflow_events,
        handoffs=handoffs,
    )

    assert len(
        graph.nodes
    ) > 0

    assert len(
        graph.relationships
    ) > 0

    # ---------------------------------------------------------
    # 6. Connect to existing Neo4j
    # ---------------------------------------------------------

    store = Neo4jOperationalGraph()

    try:
        health = store.health_check()

        assert health["healthy"] is True

        print(
            "\n--- NEO4J CONNECTION ---"
        )

        print(
            f"URI: {health['uri']}"
        )

        print(
            f"Database: {health['database']}"
        )

        # -----------------------------------------------------
        # 7. Persist graph
        # -----------------------------------------------------

        persist_result = (
            store.upsert_graph(
                graph
            )
        )

        assert (
            persist_result["nodes"]
            == len(graph.nodes)
        )

        assert (
            persist_result[
                "relationships"
            ]
            == len(graph.relationships)
        )

        # -----------------------------------------------------
        # 8. Verify real Neo4j nodes
        # -----------------------------------------------------

        node_counts = {}

        for label in [
            "Ticket",
            "Agent",
            "WorkflowEvent",
            "IssueType",
            "Priority",
            "Status",
            "Team",
        ]:
            node_counts[label] = (
                _count_nodes(
                    store,
                    label,
                )
            )

            assert (
                node_counts[label]
                >= 0
            )

        # -----------------------------------------------------
        # 9. Verify real relationships
        # -----------------------------------------------------

        relationship_counts = {}

        for relationship_type in [
            "HAS_TYPE",
            "HAS_PRIORITY",
            "HAS_STATUS",
            "ASSIGNED_TO",
            "HAS_WORKFLOW_EVENT",
            "NEXT",
            "MEMBER_OF",
            "HANDOFF_TO",
        ]:
            relationship_counts[
                relationship_type
            ] = _count_relationships(
                store,
                relationship_type,
            )

        assert (
            relationship_counts[
                "HAS_WORKFLOW_EVENT"
            ]
            > 0
        )

        assert (
            relationship_counts["NEXT"]
            > 0
        )

        # -----------------------------------------------------
        # 10. Verify one real ticket can be queried
        # -----------------------------------------------------

        first_ticket_id = (
            sorted(
                tickets_by_id
            )[0]
        )

        ticket_result = store.execute_cypher(
            """
            MATCH (t:Ticket {
                ticket_id: $ticket_id
            })
            RETURN
                t.ticket_id AS ticket_id,
                t.project AS project,
                t.issue_type AS issue_type,
                t.priority AS priority,
                t.status AS status
            """,
            {
                "ticket_id": first_ticket_id
            },
        )

        assert len(
            ticket_result
        ) == 1

        assert (
            ticket_result[0][
                "ticket_id"
            ]
            == first_ticket_id
        )

        # -----------------------------------------------------
        # 11. Verify graph neighborhood
        # -----------------------------------------------------

        neighborhood = (
            store.execute_cypher(
                """
                MATCH p=(
                    t:Ticket {
                        ticket_id: $ticket_id
                    }
                )-[*1..2]-(n)
                RETURN p
                LIMIT 20
                """,
                {
                    "ticket_id":
                        first_ticket_id,
                },
            )
        )

        assert len(
            neighborhood
        ) > 0

        # -----------------------------------------------------
        # 12. Print real graph summary
        # -----------------------------------------------------

        print(
            "\n--- REAL NEO4J GRAPH ---"
        )

        print(
            "Source issues rows: "
            f"{len(issues_df)}"
        )

        print(
            "Source change-history rows: "
            f"{len(change_history_df)}"
        )

        print(
            "Workflow events used: "
            f"{len(workflow_events)}"
        )

        print(
            "Workflow intervals: "
            f"{len(workflow_intervals)}"
        )

        print(
            "Workflow metrics: "
            f"{len(workflow_metrics_list)}"
        )

        print(
            "Bottlenecks: "
            f"{len(bottlenecks)}"
        )

        print(
            "Workflow variants: "
            f"{len(workflow_variants)}"
        )

        print(
            "Agents represented: "
            f"{len(agents)}"
        )

        print(
            "Tickets represented: "
            f"{len(tickets)}"
        )

        print(
            "Graph nodes submitted: "
            f"{persist_result['nodes']}"
        )

        print(
            "Graph relationships submitted: "
            f"{persist_result['relationships']}"
        )

        print(
            "Neo4j node counts: "
            f"{node_counts}"
        )

        print(
            "Neo4j relationship counts: "
            f"{relationship_counts}"
        )

        print(
            "Verified ticket: "
            f"{first_ticket_id}"
        )

    finally:
        store.close()