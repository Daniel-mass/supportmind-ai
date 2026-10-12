#test_workflow_integration.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.intelligence.schemas.ticket import PriorityLevel
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


HELPDESK_DIR = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\Help Desk Tickets\Help Desk Tickets"
)

ISSUES_PATH = HELPDESK_DIR / "issues.csv"

CHANGE_HISTORY_PATH = (
    HELPDESK_DIR / "issues_change_history.csv"
)


def test_real_helpdesk_workflow_dataflow():
    """
    End-to-end workflow integration test using the real
    Help Desk issues.csv and issues_change_history.csv.

    Flow:

        raw Help Desk data
            ↓
        WorkflowEvent
            ↓
        WorkflowInterval
            ↓
        Metrics / Bottlenecks / Handoffs / Variants
    """

    # ---------------------------------------------------------
    # 1. Load real source datasets
    # ---------------------------------------------------------

    issues = pd.read_csv(ISSUES_PATH)
    history = pd.read_csv(CHANGE_HISTORY_PATH)

    assert issues.shape == (66691, 58)
    assert history.shape == (257508, 6)

    expected_history_columns = {
        "id",
        "issueid",
        "field",
        "value",
        "created",
        "change_group_id",
    }

    assert expected_history_columns.issubset(
        history.columns
    )

    # ---------------------------------------------------------
    # 2. Validate source relationship
    # ---------------------------------------------------------

    issue_ids = set(
        issues["id"]
        .dropna()
        .astype(int)
    )

    history_issue_ids = set(
        history["issueid"]
        .dropna()
        .astype(int)
    )

    assert history_issue_ids.issubset(issue_ids)

    # ---------------------------------------------------------
    # 3. Build issue lookup with explicit source column names
    # ---------------------------------------------------------

    issue_lookup = (
        issues[
            [
                "id",
                "issue_proj",
                "issue_num",
                "issue_type",
                "issue_priority",
            ]
        ]
        .drop_duplicates("id")
        .rename(
            columns={
                "id": "source_issue_id",
            }
        )
        .copy()
    )

    issue_lookup["source_issue_id"] = (
        issue_lookup["source_issue_id"]
        .astype(int)
    )

    # ---------------------------------------------------------
    # 4. Restrict change history to workflow fields
    # ---------------------------------------------------------

    workflow_history = history[
        history["field"].isin(
            ["status", "assignee"]
        )
    ].copy()

    assert len(workflow_history) > 0

    assert set(
        workflow_history["field"].unique()
    ).issubset(
        {"status", "assignee"}
    )

    # ---------------------------------------------------------
    # 5. Join real workflow events to real tickets
    # ---------------------------------------------------------

    workflow_history["issueid"] = (
        workflow_history["issueid"]
        .astype(int)
    )

    workflow_history = workflow_history.merge(
        issue_lookup,
        left_on="issueid",
        right_on="source_issue_id",
        how="left",
        validate="many_to_one",
    )

    assert workflow_history["issue_proj"].notna().all()
    assert workflow_history["issue_num"].notna().all()

    # ---------------------------------------------------------
    # 6. Normalize timestamps
    # ---------------------------------------------------------

    workflow_history["created"] = pd.to_datetime(
        workflow_history["created"],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    assert workflow_history["created"].notna().all()

    # ---------------------------------------------------------
    # 7. Deterministic integration subset
    # ---------------------------------------------------------

    integration_history = (
        workflow_history
        .sort_values(
            ["created", "id"]
        )
        .head(5000)
        .copy()
    )

    assert len(integration_history) == 5000

    # ---------------------------------------------------------
    # 8. Convert real rows → canonical WorkflowEvent
    # ---------------------------------------------------------

    events: list[WorkflowEvent] = []

    priority_map = {
        "lowest": PriorityLevel.LOWEST,
        "low": PriorityLevel.LOW,
        "medium": PriorityLevel.MEDIUM,
        "high": PriorityLevel.HIGH,
        "highest": PriorityLevel.HIGHEST,
        "blocker": PriorityLevel.BLOCKER,
        "unknown": PriorityLevel.UNKNOWN,
    }

    for row in integration_history.itertuples(
        index=False
    ):
        field = str(row.field)

        value = (
            None
            if pd.isna(row.value)
            else str(row.value).strip()
        )

        if not value:
            continue

        project = str(row.issue_proj)
        issue_num = int(row.issue_num)

        ticket_id = (
            f"{project}:{issue_num}"
        )

        raw_priority = (
            "unknown"
            if pd.isna(row.issue_priority)
            else str(row.issue_priority).strip()
        )

        priority = priority_map.get(
            raw_priority.lower(),
            PriorityLevel.UNKNOWN,
        )

        event_id = str(int(row.id))

        change_group_id = (
            int(row.change_group_id)
            if not pd.isna(row.change_group_id)
            else None
        )

        event = WorkflowEvent(
            event_id=event_id,
            ticket_id=ticket_id,
            timestamp=row.created.to_pydatetime(),
            activity=f"{field}_change",
            status=(
                value
                if field == "status"
                else None
            ),
            assignee=(
                value
                if field == "assignee"
                else None
            ),
            priority=priority,
            category=str(row.issue_type),
            source="helpdesk",
            metadata={
                "source_issue_id": int(row.issueid),
                "change_group_id": change_group_id,
                "source_field": field,
            },
        )

        events.append(event)

    assert len(events) > 0

    # Both workflow dimensions should be represented.
    activities = {
        event.activity
        for event in events
    }

    assert "status_change" in activities
    assert "assignee_change" in activities

    # ---------------------------------------------------------
    # 9. Build workflow intervals
    # ---------------------------------------------------------

    intervals = build_workflow_intervals(events)

    assert len(intervals) > 0

    assert all(
        interval.duration_seconds >= 0
        for interval in intervals
    )

    # ---------------------------------------------------------
    # 10. Workflow metrics
    # ---------------------------------------------------------

    metrics = calculate_workflow_metrics(
        intervals
    )

    assert len(metrics) > 0

    assert all(
        metric.lifecycle_seconds >= 0
        for metric in metrics
    )

    # ---------------------------------------------------------
    # 11. Bottleneck analysis
    # ---------------------------------------------------------

    bottlenecks = identify_bottlenecks(
        intervals
    )

    assert isinstance(
        bottlenecks,
        list,
    )

    status_intervals = [
        interval
        for interval in intervals
        if interval.status
    ]

    assert len(status_intervals) > 0
    assert len(bottlenecks) > 0

    assert all(
        0 <= item.bottleneck_score <= 1
        for item in bottlenecks
    )

    # ---------------------------------------------------------
    # 12. Handoff analysis
    # ---------------------------------------------------------

    handoffs = identify_handoffs(
        intervals
    )

    assert isinstance(
        handoffs,
        list,
    )

    assert all(
        handoff.from_agent
        != handoff.to_agent
        for handoff in handoffs
    )

    # ---------------------------------------------------------
    # 13. Workflow variants
    # ---------------------------------------------------------

    variants = identify_workflow_variants(
        intervals
    )

    assert len(variants) > 0

    total_variant_percentage = sum(
        variant.percentage
        for variant in variants
    )

    assert round(
        total_variant_percentage,
        6,
    ) == 100.0

    # ---------------------------------------------------------
    # 14. Integration evidence
    # ---------------------------------------------------------

    print(
        "\nReal Help Desk workflow integration"
    )
    print("=" * 70)

    print(
        "issues.csv rows             :",
        len(issues),
    )

    print(
        "change-history rows         :",
        len(history),
    )

    print(
        "workflow events used        :",
        len(events),
    )

    print(
        "workflow intervals          :",
        len(intervals),
    )

    print(
        "ticket workflow metrics     :",
        len(metrics),
    )

    print(
        "bottleneck candidates       :",
        len(bottlenecks),
    )

    print(
        "handoffs detected           :",
        len(handoffs),
    )

    print(
        "workflow variants           :",
        len(variants),
    )

    if bottlenecks:
        print(
            "top bottleneck              :",
            bottlenecks[0].status,
        )

    if variants:
        print(
            "top workflow variant        :",
            variants[0].path_string,
        )