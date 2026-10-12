from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
from backend.app.intelligence.api.agents import (
    get_agent_analytics_provider,
    router as agents_router,
)
from backend.app.intelligence.api.organization import (
    get_organization_intelligence_provider,
    router as organization_router,
)
from backend.app.intelligence.api.tickets import (
    get_ticket_intelligence_provider,
    router as tickets_router,
)
from backend.app.intelligence.api.workflow import (
    get_workflow_intelligence_provider,
    router as workflow_router,
)
from backend.app.intelligence.schemas.agent import Agent
from backend.app.intelligence.schemas.ticket import Ticket
from backend.app.intelligence.schemas.workflow import WorkflowEvent
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
):
    value = _value_from_row(
        row,
        candidates,
    )

    parsed = _parse_optional_timestamp(
        value
    )

    assert parsed is not None

    return parsed


def _build_ticket_from_row(
    row: pd.Series,
) -> Ticket:
    project = str(
        _value_from_row(
            row,
            ["issue_proj", "project"],
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
        ended_at=None,
        resolution_date=None,
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
            "api_integration_test": True,
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

    workflow_raw["created"] = pd.to_datetime(
        workflow_raw["created"],
        format="mixed",
        errors="coerce",
        utc=True,
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


def test_real_helpdesk_api_integration():
    """
    Final real-data FastAPI integration test.

    Builds real Help Desk intelligence and exposes it through
    all four completed Intelligence API endpoints.
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

    # ---------------------------------------------------------
    # 2. Build canonical tickets and workflow events
    # ---------------------------------------------------------

    workflow_events = _build_workflow_events(
        issues_df,
        change_history_df,
    )

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

    # ---------------------------------------------------------
    # 3. Workflow processing
    # ---------------------------------------------------------

    intervals = (
        build_workflow_intervals(
            workflow_events
        )
    )

    workflow_metrics = (
        calculate_workflow_metrics(
            intervals
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
        metric.ticket_id: metric
        for metric in workflow_metrics_list
    }

    handoffs = list(
        identify_handoffs(
            intervals
        )
    )

    bottlenecks = list(
        identify_bottlenecks(
            intervals
        )
    )

    variants = list(
        identify_workflow_variants(
            intervals
        )
    )

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

    for ticket_id, metric in (
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
                workflow_metrics=metric,
                handoffs=(
                    handoffs_by_ticket.get(
                        ticket_id,
                        [],
                    )
                ),
            )
        )

    # ---------------------------------------------------------
    # 5. Agent + team analytics
    # ---------------------------------------------------------

    sampled_tickets = [
        tickets_by_id[
            item.ticket_id
        ]
        for item in ticket_analytics
        if item.ticket_id in tickets_by_id
        and tickets_by_id[
            item.ticket_id
        ].assignee
    ]

    agent_ids = sorted(
        {
            str(ticket.assignee).strip()
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
            metadata={},
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

    team_result = build_team_analytics(
        team="helpdesk",
        agents=agents,
        agent_analytics=agent_analytics,
    )

    # ---------------------------------------------------------
    # 6. Organization analytics
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
    # 7. Build Services responses
    # ---------------------------------------------------------

    first_ticket_analytics = (
        ticket_analytics[0]
    )

    first_ticket = tickets_by_id[
        first_ticket_analytics.ticket_id
    ]

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

    ticket_intelligence = (
        build_ticket_intelligence(
            ticket=first_ticket,
            analytics=first_ticket_analytics,
            workflow_metrics=(
                workflow_metrics_by_ticket.get(
                    first_ticket.ticket_id
                )
            ),
            workflow_events=(
                first_ticket_events
            ),
            handoffs=(
                first_ticket_handoffs
            ),
        )
    )

    workflow_intelligence = (
        build_workflow_intelligence(
            ticket_id=first_ticket.ticket_id,
            events=first_ticket_events,
            metrics=(
                workflow_metrics_by_ticket.get(
                    first_ticket.ticket_id
                )
            ),
            handoffs=(
                first_ticket_handoffs
            ),
            bottlenecks=bottlenecks,
            variants=variants,
        )
    )

    organization_intelligence = (
        build_organization_intelligence(
            organization=organization_result,
            agent_analytics=agent_analytics,
            team_analytics=[team_result],
            bottlenecks=bottlenecks,
            workflow_variants=variants,
            emerging_issues=[],
        )
    )

    # ---------------------------------------------------------
    # 8. Build FastAPI app
    # ---------------------------------------------------------

    app = FastAPI()

    app.include_router(
        tickets_router,
        prefix="/api/intelligence",
    )

    app.include_router(
        workflow_router,
        prefix="/api/intelligence",
    )

    app.include_router(
        agents_router,
        prefix="/api/intelligence",
    )

    app.include_router(
        organization_router,
        prefix="/api/intelligence",
    )

    # ---------------------------------------------------------
    # 9. Wire real in-memory providers
    # ---------------------------------------------------------

    ticket_map = {
        first_ticket.ticket_id:
            ticket_intelligence
    }

    workflow_map = {
        first_ticket.ticket_id:
            workflow_intelligence
    }

    agent_map = {
        item.agent_id: item
        for item in agent_analytics
    }

    organization_provider = (
        lambda: organization_intelligence
    )

    app.dependency_overrides[
        get_ticket_intelligence_provider
    ] = lambda: (
        lambda ticket_id:
        ticket_map.get(ticket_id)
    )

    app.dependency_overrides[
        get_workflow_intelligence_provider
    ] = lambda: (
        lambda ticket_id:
        workflow_map.get(ticket_id)
    )

    app.dependency_overrides[
        get_agent_analytics_provider
    ] = lambda: (
        lambda agent_id:
        agent_map.get(agent_id)
    )

    app.dependency_overrides[
        get_organization_intelligence_provider
    ] = lambda: organization_provider

    client = TestClient(app)

    # ---------------------------------------------------------
    # 10. Test ticket endpoint
    # ---------------------------------------------------------

    response = client.get(
        "/api/intelligence/tickets/"
        f"{first_ticket.ticket_id}"
        "/intelligence"
    )

    assert response.status_code == 200

    ticket_payload = response.json()

    assert (
        ticket_payload["ticket"]["ticket_id"]
        == first_ticket.ticket_id
    )

    # ---------------------------------------------------------
    # 11. Test workflow endpoint
    # ---------------------------------------------------------

    response = client.get(
        "/api/intelligence/workflow/"
        f"{first_ticket.ticket_id}"
        "/intelligence"
    )

    assert response.status_code == 200

    workflow_payload = response.json()

    assert (
        workflow_payload["ticket_id"]
        == first_ticket.ticket_id
    )

    # ---------------------------------------------------------
    # 12. Test agent endpoint
    # ---------------------------------------------------------

    assert len(agent_analytics) > 0

    first_agent_id = (
        agent_analytics[0].agent_id
    )

    response = client.get(
        "/api/intelligence/agents/"
        f"{first_agent_id}"
        "/intelligence"
    )

    assert response.status_code == 200

    agent_payload = response.json()

    assert (
        agent_payload["agent_id"]
        == first_agent_id
    )

    # ---------------------------------------------------------
    # 13. Test organization endpoint
    # ---------------------------------------------------------

    response = client.get(
        "/api/intelligence/organization/"
        "intelligence"
    )

    assert response.status_code == 200

    organization_payload = (
        response.json()
    )

    assert (
        organization_payload[
            "organization"
        ][
            "total_ticket_count"
        ]
        == len(ticket_analytics)
    )

    assert (
        len(
            organization_payload[
                "agent_analytics"
            ]
        )
        == len(agent_analytics)
    )

    # ---------------------------------------------------------
    # 14. Final API summary
    # ---------------------------------------------------------

    print(
        "\n--- REAL API INTEGRATION ---"
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
        "workflow metrics: "
        f"{len(workflow_metrics_list)}"
    )

    print(
        "ticket analytics: "
        f"{len(ticket_analytics)}"
    )

    print(
        "agents: "
        f"{len(agent_analytics)}"
    )

    print(
        "organization tickets: "
        f"{organization_result.total_ticket_count}"
    )

    print(
        "Ticket API: PASS"
    )

    print(
        "Workflow API: PASS"
    )

    print(
        "Agent API: PASS"
    )

    print(
        "Organization API: PASS"
    )