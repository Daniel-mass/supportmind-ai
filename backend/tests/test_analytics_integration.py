#analytics integrationn test
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


PROJECT_ROOT = Path(r"C:\Users\danie\supportmind-ai")

HELPDESK_ROOT = (
    PROJECT_ROOT
    / "rawsupportminddata"
    / "Help Desk Tickets"
    / "Help Desk Tickets"
)

ISSUES_CSV = HELPDESK_ROOT / "issues.csv"
CHANGE_HISTORY_CSV = HELPDESK_ROOT / "issues_change_history.csv"

# Keep the real-data integration deterministic and reasonably fast.
WORKFLOW_SAMPLE_SIZE = 5_000


def _value_from_row(
    row: pd.Series,
    candidates: list[str],
    default=None,
):
    """
    Return the first available non-null value from candidate columns.
    """
    for column in candidates:
        if column not in row.index:
            continue

        value = row[column]

        if pd.notna(value):
            return value

    return default


def _parse_optional_timestamp(value):
    """
    Parse a timestamp into a timezone-aware Python datetime.

    Returns None when the source value is missing or cannot be parsed.
    """
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
    *,
    field_name: str,
):
    """
    Parse a required timestamp from the real Help Desk row.

    Raises a clear assertion when none of the expected source columns
    contains a valid timestamp.
    """
    raw_value = _value_from_row(
        row,
        candidates,
        default=None,
    )

    parsed = _parse_optional_timestamp(raw_value)

    assert parsed is not None, (
        f"Unable to derive required '{field_name}' timestamp. "
        f"Checked columns: {candidates}. "
        f"Available columns: {list(row.index)}"
    )

    return parsed


def _build_ticket_from_row(row: pd.Series) -> Ticket:
    """
    Build the canonical Ticket model from one real issues.csv row.

    The Help Desk source does not expose a separate populated
    `created_at` field in the rows used by this dataset. The real
    `started` timestamp is therefore used as the lifecycle-start
    fallback for the canonical Ticket.created_at field.
    """

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

    raw_issue_num = _value_from_row(
        row,
        [
            "issue_num",
            "issue_number",
        ],
        default=0,
    )

    issue_num = int(float(raw_issue_num))

    ticket_id = f"{project}:{issue_num}"

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

    # ---------------------------------------------------------
    # Timestamp mapping
    # ---------------------------------------------------------
    #
    # Actual Help Desk data exposes `started`. When a separate
    # creation timestamp is absent, use `started` as the canonical
    # lifecycle start.
    #

    started_at = _parse_required_timestamp(
        row,
        [
            "issue_started_at",
            "started_at",
            "started",
        ],
        field_name="started_at",
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
                "finished",
                "finished_at",
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
                "resolved_at",
            ],
        )
    )

    # ---------------------------------------------------------
    # Non-temporal fields
    # ---------------------------------------------------------

    resolution = _value_from_row(
        row,
        [
            "issue_resolution",
            "resolution",
        ],
    )

    reporter = _value_from_row(
        row,
        [
            "issue_reporter",
            "reporter",
        ],
    )

    assignee = _value_from_row(
        row,
        [
            "issue_assignee",
            "assignee",
        ],
    )

    raw_contributor_count = _value_from_row(
        row,
        [
            "issue_contr_count",
            "contributor_count",
        ],
        default=0,
    )

    raw_comments_count = _value_from_row(
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
            float(raw_contributor_count)
        )
    except (TypeError, ValueError):
        contributor_count = 0

    try:
        comments_count = int(
            float(raw_comments_count)
        )
    except (TypeError, ValueError):
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
        resolution=resolution,
        reporter=reporter,
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
            "integration_test": True,
        },
    )


def _build_workflow_events(
    issues_df: pd.DataFrame,
    change_history_df: pd.DataFrame,
) -> list[WorkflowEvent]:
    """
    Convert real Help Desk change-history records into canonical
    WorkflowEvent objects.

    Only status and assignee changes are included because these are
    the workflow activities required by the Intelligence Layer.
    """

    required_issue_columns = {
        "id",
        "issue_proj",
        "issue_num",
        "issue_type",
        "issue_priority",
    }

    missing_issue_columns = (
        required_issue_columns
        - set(issues_df.columns)
    )

    assert not missing_issue_columns, (
        "issues.csv is missing required columns: "
        f"{sorted(missing_issue_columns)}"
    )

    required_history_columns = {
        "id",
        "issueid",
        "field",
        "value",
        "created",
    }

    missing_history_columns = (
        required_history_columns
        - set(change_history_df.columns)
    )

    assert not missing_history_columns, (
        "issues_change_history.csv is missing required columns: "
        f"{sorted(missing_history_columns)}"
    )

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
            "id": "source_issue_id",
        }
    )

    issue_context["ticket_id"] = (
        issue_context["issue_proj"]
        .astype(str)
        .str.strip()
        + ":"
        + issue_context["issue_num"]
        .astype(int)
        .astype(str)
    )

    workflow_raw = change_history_df[
        change_history_df["field"].isin(
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

    workflow_raw["created"] = pd.to_datetime(
        workflow_raw["created"],
        format="mixed",
        errors="coerce",
        utc=True,
    )

    workflow_raw = workflow_raw.dropna(
        subset=["created"]
    )

    workflow_raw = workflow_raw.sort_values(
        [
            "created",
            "issueid",
            "change_group_id",
            "id",
        ],
        kind="mergesort",
    ).head(
        WORKFLOW_SAMPLE_SIZE
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

    events: list[WorkflowEvent] = []

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

        state = state_by_ticket[
            ticket_id
        ]

        if field == "status":
            state["status"] = value
            activity = "status_change"

        else:
            state["assignee"] = value
            activity = "assignee_change"

        timestamp = (
            row.created.to_pydatetime()
        )

        events.append(
            WorkflowEvent(
                event_id=f"change-{row.id}",
                ticket_id=ticket_id,
                timestamp=timestamp,
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
                source="helpdesk_change_history",
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


def test_real_helpdesk_analytics_integration():
    """
    Real end-to-end integration test for the Intelligence Analytics layer.

    Real sources:
        issues.csv
        issues_change_history.csv

    Pipeline:
        Help Desk source
            -> canonical Ticket
            -> WorkflowEvent
            -> WorkflowInterval
            -> WorkflowMetrics
            -> Bottlenecks
            -> Handoffs
            -> Workflow Variants
            -> Ticket Analytics
            -> Agent Analytics
            -> Team Analytics
            -> Organization Analytics

    The workflow sample is bounded for runtime while remaining
    deterministic and sourced from the real dataset.
    """

    # ---------------------------------------------------------
    # 0. Source validation
    # ---------------------------------------------------------

    assert ISSUES_CSV.exists(), (
        f"Missing Help Desk file: {ISSUES_CSV}"
    )

    assert CHANGE_HISTORY_CSV.exists(), (
        f"Missing Help Desk file: {CHANGE_HISTORY_CSV}"
    )

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
    # 1. Canonical tickets from real issues.csv
    # ---------------------------------------------------------

    tickets = [
        _build_ticket_from_row(row)
        for _, row in issues_df.iterrows()
    ]

    assert len(tickets) == len(
        issues_df
    )

    tickets_by_id: dict[
        str,
        Ticket,
    ] = {}

    for ticket in tickets:
        tickets_by_id.setdefault(
            ticket.ticket_id,
            ticket,
        )

    assert len(tickets_by_id) > 0

    # ---------------------------------------------------------
    # 2. Real workflow events
    # ---------------------------------------------------------

    workflow_events = (
        _build_workflow_events(
            issues_df=issues_df,
            change_history_df=change_history_df,
        )
    )

    assert len(workflow_events) > 0

    # Every workflow event should map to a canonical ticket.
    missing_event_tickets = sorted(
        {
            event.ticket_id
            for event in workflow_events
            if event.ticket_id
            not in tickets_by_id
        }
    )

    assert not missing_event_tickets, (
        "Workflow events reference tickets "
        "that do not exist in the canonical ticket set: "
        f"{missing_event_tickets[:10]}"
    )

    # ---------------------------------------------------------
    # 3. Workflow intelligence
    # ---------------------------------------------------------

    workflow_intervals = (
        build_workflow_intervals(
            workflow_events
        )
    )

    assert len(workflow_intervals) > 0

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

    assert len(
        workflow_metrics_list
    ) > 0

    workflow_metrics_by_ticket = {
        metrics.ticket_id: metrics
        for metrics in workflow_metrics_list
    }

    assert len(
        workflow_metrics_by_ticket
    ) > 0

    bottlenecks = (
        identify_bottlenecks(
            workflow_intervals
        )
    )

    handoffs = (
        identify_handoffs(
            workflow_intervals
        )
    )

    workflow_variants = (
        identify_workflow_variants(
            workflow_intervals
        )
    )

    bottleneck_list = list(
        bottlenecks
    )

    handoff_list = list(
        handoffs
    )

    workflow_variant_list = list(
        workflow_variants
    )

    # ---------------------------------------------------------
    # 4. Ticket analytics
    # ---------------------------------------------------------

    handoffs_by_ticket: dict[
        str,
        list,
    ] = defaultdict(list)

    for handoff in handoff_list:
        handoffs_by_ticket[
            handoff.ticket_id
        ].append(handoff)

    ticket_analytics = []

    for (
        ticket_id,
        metrics,
    ) in workflow_metrics_by_ticket.items():
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

    sampled_ticket_ids = {
        analytics.ticket_id
        for analytics in ticket_analytics
    }

    sampled_tickets = [
        tickets_by_id[ticket_id]
        for ticket_id in sampled_ticket_ids
        if ticket_id in tickets_by_id
        and tickets_by_id[ticket_id].assignee
    ]

    agent_ids = sorted(
        {
            str(ticket.assignee).strip()
            for ticket in sampled_tickets
            if ticket.assignee
        }
    )

    # The raw Help Desk issues dataset does not expose a
    # canonical team dimension. A neutral "helpdesk" team is
    # therefore used only to exercise the team aggregation path.
    agents = [
        Agent(
            agent_id=agent_id,
            name=agent_id,
            team="helpdesk",
            active=True,
            source="helpdesk",
            metadata={
                "integration_test": True,
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

    agent_analytics = []

    for agent in agents:
        agent_result = (
            build_agent_analytics(
                agent=agent,
                tickets=tickets_by_agent.get(
                    agent.agent_id,
                    [],
                ),
                workflow_metrics_by_ticket=(
                    workflow_metrics_by_ticket
                ),
                handoffs=handoff_list,
            )
        )

        agent_analytics.append(
            agent_result
        )

    assert len(
        agent_analytics
    ) == len(agents)

    # ---------------------------------------------------------
    # 6. Team analytics
    # ---------------------------------------------------------

    team_result = build_team_analytics(
        team="helpdesk",
        agents=agents,
        agent_analytics=agent_analytics,
    )

    assert (
        team_result.team
        == "helpdesk"
    )

    # ---------------------------------------------------------
    # 7. Organization analytics
    # ---------------------------------------------------------

    organization_result = (
        build_organization_analytics(
            agents=agents,
            ticket_analytics=ticket_analytics,
            agent_analytics=agent_analytics,
            team_analytics=[
                team_result
            ],
            bottlenecks=bottleneck_list,
            workflow_variants=(
                workflow_variant_list
            ),
            emerging_issues=[],
        )
    )

    # ---------------------------------------------------------
    # 8. Cross-layer consistency checks
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

    assert (
        sum(
            item.count
            for item in (
                organization_result
                .priority_distribution
            )
        )
        == organization_result.total_ticket_count
    )

    assert (
        sum(
            item.count
            for item in (
                organization_result
                .status_distribution
            )
        )
        == organization_result.total_ticket_count
    )

    assert (
        team_result.assigned_ticket_count
        == sum(
            item.assigned_ticket_count
            for item in agent_analytics
        )
    )

    assert (
        team_result.total_lifecycle_seconds
        == sum(
            item.total_lifecycle_seconds
            for item in agent_analytics
        )
    )

    assert (
        organization_result.team_count
        == 1
    )

    assert (
        len(
            organization_result.team_workload
        )
        == 1
    )

    # ---------------------------------------------------------
    # 9. Real-data summary
    # ---------------------------------------------------------

    print(
        "\n--- REAL ANALYTICS INTEGRATION ---"
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
        "unique canonical tickets: "
        f"{len(tickets_by_id)}"
    )

    print(
        "workflow events used: "
        f"{len(workflow_events)}"
    )

    print(
        "workflow intervals: "
        f"{len(workflow_intervals)}"
    )

    print(
        "ticket workflow metrics: "
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
        "team assigned tickets: "
        f"{team_result.assigned_ticket_count}"
    )

    print(
        "organization tickets: "
        f"{organization_result.total_ticket_count}"
    )

    print(
        "bottlenecks: "
        f"{len(bottleneck_list)}"
    )

    print(
        "workflow variants: "
        f"{len(workflow_variant_list)}"
    )