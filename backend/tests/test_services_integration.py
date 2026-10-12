from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from backend.app.intelligence.analytics.agent_analytics import (
    build_agent_analytics,
)
from backend.app.intelligence.analytics.organization import (
    build_organization_analytics,
)
from backend.app.intelligence.analytics.team_analytics import (
    build_team_analytics,
)
from backend.app.intelligence.analytics.ticket_analytics import (
    build_ticket_analytics,
)
from backend.app.intelligence.schemas.agent import Agent
from backend.app.intelligence.schemas.ticket import Ticket
from backend.app.intelligence.schemas.workflow import WorkflowEvent
from backend.app.intelligence.services.evidence_service import (
    build_evidence,
)
from backend.app.intelligence.services.organization_intelligence import (
    build_organization_intelligence,
)
from backend.app.intelligence.services.ticket_intelligence import (
    build_ticket_intelligence,
)
from backend.app.intelligence.services.workflow_intelligence import (
    build_workflow_intelligence,
)
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
) :
    value = _value_from_row(
        row,
        candidates,
    )

    parsed = _parse_optional_timestamp(
        value
    )

    assert parsed is not None, (
        f"Unable to parse required timestamp. "
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
            "services_integration_test": True,
        },
    )


def _build_workflow_events(
    issues_df: pd.DataFrame,
    change_history_df: pd.DataFrame,
) -> list[WorkflowEvent]:

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
        .dropna(
            subset=["created"]
        )
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
                    row.created.to_pydatetime()
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
                    "source_issue_id": row.issueid,
                    "field": field,
                    "change_group_id": (
                        row.change_group_id
                    ),
                },
            )
        )

    return events


def test_real_helpdesk_services_integration():
    """
    Final real-data integration test for the complete Services Layer.

    Uses:
        issues.csv
        issues_change_history.csv

    The test intentionally uses a deterministic workflow sample so
    execution remains practical while exercising the real Help Desk
    data flow.
    """

    # ---------------------------------------------------------
    # 1. Load real Help Desk data
    # ---------------------------------------------------------

    assert ISSUES_CSV.exists()
    assert CHANGE_HISTORY_CSV.exists()

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
    # 2. Build canonical tickets
    # ---------------------------------------------------------

    workflow_events = _build_workflow_events(
        issues_df,
        change_history_df,
    )

    assert len(
        workflow_events
    ) > 0

    workflow_ticket_ids = {
        event.ticket_id
        for event in workflow_events
    }

    tickets = [
        _build_ticket_from_row(row)
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

    assert len(
        tickets_by_id
    ) > 0

    # ---------------------------------------------------------
    # 3. Workflow intelligence
    # ---------------------------------------------------------

    workflow_intervals = (
        build_workflow_intervals(
            workflow_events
        )
    )

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

    workflow_metrics_by_ticket = {
        metrics.ticket_id: metrics
        for metrics in workflow_metrics_list
    }

    handoffs = list(
        identify_handoffs(
            workflow_intervals
        )
    )

    bottlenecks = list(
        identify_bottlenecks(
            workflow_intervals
        )
    )

    variants = list(
        identify_workflow_variants(
            workflow_intervals
        )
    )

    assert len(
        workflow_metrics_list
    ) > 0

    # ---------------------------------------------------------
    # 4. Ticket analytics
    # ---------------------------------------------------------

    handoffs_by_ticket: dict[
        str,
        list,
    ] = defaultdict(list)

    for handoff in handoffs:
        handoffs_by_ticket[
            handoff.ticket_id
        ].append(handoff)

    ticket_analytics = []

    for ticket_id, metrics in (
        workflow_metrics_by_ticket.items()
    ):
        ticket = tickets_by_id.get(
            ticket_id
        )

        if ticket is None:
            continue

        ticket_analytics.append(
            build_ticket_analytics(
                ticket=ticket,
                workflow_metrics=metrics,
                handoffs=(
                    handoffs_by_ticket.get(
                        ticket_id,
                        [],
                    )
                ),
            )
        )

    assert len(
        ticket_analytics
    ) > 0

    # ---------------------------------------------------------
    # 5. Agent analytics
    # ---------------------------------------------------------

    sampled_tickets = [
        tickets_by_id[
            analytics.ticket_id
        ]
        for analytics in ticket_analytics
        if analytics.ticket_id
        in tickets_by_id
        and tickets_by_id[
            analytics.ticket_id
        ].assignee
    ]

    agent_ids = sorted(
        {
            str(
                ticket.assignee
            ).strip()
            for ticket in sampled_tickets
            if ticket.assignee
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
                "services_integration_test": True
            },
        )
        for agent_id in agent_ids
    ]

    tickets_by_agent: dict[
        str,
        list[Ticket],
    ] = defaultdict(list)

    for ticket in sampled_tickets:
        tickets_by_agent[
            str(ticket.assignee).strip()
        ].append(ticket)

    agent_analytics = [
        build_agent_analytics(
            agent=agent,
            tickets=tickets_by_agent.get(
                agent.agent_id,
                [],
            ),
            workflow_metrics_by_ticket=(
                workflow_metrics_by_ticket
            ),
            handoffs=handoffs,
        )
        for agent in agents
    ]

    # ---------------------------------------------------------
    # 6. Team analytics
    # ---------------------------------------------------------

    team_result = build_team_analytics(
        team="helpdesk",
        agents=agents,
        agent_analytics=agent_analytics,
    )

    # ---------------------------------------------------------
    # 7. Organization analytics
    # ---------------------------------------------------------

    organization_result = (
        build_organization_analytics(
            agents=agents,
            ticket_analytics=ticket_analytics,
            agent_analytics=agent_analytics,
            team_analytics=[team_result],
            bottlenecks=bottlenecks,
            workflow_variants=variants,
            emerging_issues=[],
        )
    )

    # ---------------------------------------------------------
    # 8. Ticket Intelligence Service
    # ---------------------------------------------------------

    first_ticket_analytics = (
        ticket_analytics[0]
    )

    first_ticket = tickets_by_id[
        first_ticket_analytics.ticket_id
    ]

    first_ticket_metrics = (
        workflow_metrics_by_ticket.get(
            first_ticket.ticket_id
        )
    )

    first_ticket_events = [
        event
        for event in workflow_events
        if event.ticket_id
        == first_ticket.ticket_id
    ]

    first_ticket_handoffs = (
        handoffs_by_ticket.get(
            first_ticket.ticket_id,
            [],
        )
    )

    ticket_service_result = (
        build_ticket_intelligence(
            ticket=first_ticket,
            analytics=first_ticket_analytics,
            workflow_metrics=(
                first_ticket_metrics
            ),
            workflow_events=(
                first_ticket_events
            ),
            handoffs=(
                first_ticket_handoffs
            ),
        )
    )

    assert (
        ticket_service_result.ticket.ticket_id
        == first_ticket.ticket_id
    )

    assert (
        ticket_service_result.analytics.ticket_id
        == first_ticket.ticket_id
    )

    # ---------------------------------------------------------
    # 9. Workflow Intelligence Service
    # ---------------------------------------------------------

    workflow_service_result = (
        build_workflow_intelligence(
            ticket_id=first_ticket.ticket_id,
            events=first_ticket_events,
            metrics=first_ticket_metrics,
            handoffs=first_ticket_handoffs,
            bottlenecks=bottlenecks,
            variants=variants,
        )
    )

    assert (
        workflow_service_result.ticket_id
        == first_ticket.ticket_id
    )

    assert (
        workflow_service_result.metrics
        == first_ticket_metrics
    )

    # ---------------------------------------------------------
    # 10. Organization Intelligence Service
    # ---------------------------------------------------------

    organization_service_result = (
        build_organization_intelligence(
            organization=organization_result,
            agent_analytics=agent_analytics,
            team_analytics=[team_result],
            bottlenecks=bottlenecks,
            workflow_variants=variants,
            emerging_issues=[],
        )
    )

    assert (
        organization_service_result
        .organization
        == organization_result
    )

    assert (
        len(
            organization_service_result
            .agent_analytics
        )
        == len(agent_analytics)
    )

    assert (
        len(
            organization_service_result
            .team_analytics
        )
        == 1
    )

    # ---------------------------------------------------------
    # 11. Evidence Service
    # ---------------------------------------------------------

    evidence = build_evidence(
        ticket_analytics=[
            first_ticket_analytics
        ],
        organization_analytics=(
            organization_result
        ),
        source_dataset="helpdesk",
    )

    assert len(evidence) > 0

    assert any(
        item.scope == "ticket"
        for item in evidence
    )

    assert any(
        item.scope == "organization"
        for item in evidence
    )

    assert all(
        item.source_dataset
        == "helpdesk"
        for item in evidence
    )

    assert all(
        len(item.source_fields) > 0
        for item in evidence
    )

    # ---------------------------------------------------------
    # 12. Final consistency checks
    # ---------------------------------------------------------

    assert (
        organization_result.total_ticket_count
        == len(ticket_analytics)
    )

    assert (
        organization_result.total_transition_count
        == sum(
            item.transition_count
            for item in ticket_analytics
        )
    )

    assert (
        organization_result.total_handoff_count
        == sum(
            item.handoff_count
            for item in ticket_analytics
        )
    )

    assert (
        organization_result.total_lifecycle_seconds
        == sum(
            item.lifecycle_seconds
            for item in ticket_analytics
        )
    )

    assert (
        organization_result.total_waiting_seconds
        == sum(
            item.waiting_seconds
            for item in ticket_analytics
        )
    )

    # ---------------------------------------------------------
    # 13. Real-data summary
    # ---------------------------------------------------------

    print(
        "\n--- REAL SERVICES INTEGRATION ---"
    )

    print(
        f"issues.csv rows: "
        f"{len(issues_df)}"
    )

    print(
        "change-history rows: "
        f"{len(change_history_df)}"
    )

    print(
        "workflow events: "
        f"{len(workflow_events)}"
    )

    print(
        "workflow intervals: "
        f"{len(workflow_intervals)}"
    )

    print(
        "workflow metrics: "
        f"{len(workflow_metrics_list)}"
    )

    print(
        "ticket analytics: "
        f"{len(ticket_analytics)}"
    )

    print(
        "agents: "
        f"{len(agents)}"
    )

    print(
        "team tickets: "
        f"{team_result.assigned_ticket_count}"
    )

    print(
        "organization tickets: "
        f"{organization_result.total_ticket_count}"
    )

    print(
        "ticket intelligence events: "
        f"{len(ticket_service_result.workflow_events)}"
    )

    print(
        "workflow intelligence events: "
        f"{len(workflow_service_result.events)}"
    )

    print(
        "organization intelligence agents: "
        f"{len(organization_service_result.agent_analytics)}"
    )

    print(
        "evidence objects: "
        f"{len(evidence)}"
    )

    print(
        "Services integration: PASS"
    )