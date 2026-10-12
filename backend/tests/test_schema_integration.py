from datetime import datetime, timezone

from backend.app.intelligence.schemas.agent import Agent
from backend.app.intelligence.schemas.evidence import (
    EvidenceScope,
    EvidenceType,
    IntelligenceEvidence,
)
from backend.app.intelligence.schemas.overview import (
    DistributionItem,
    MetricSummary,
    OverviewSummary,
)
from backend.app.intelligence.schemas.ticket import (
    PriorityLevel,
    Ticket,
)
from backend.app.intelligence.schemas.workflow import WorkflowEvent


def test_complete_intelligence_schema_flow():
    """
    Integration test for the complete Intelligence Layer schema module.

    Verifies that:
        Ticket
          ↓
        Agent
          ↓
        WorkflowEvent
          ↓
        IntelligenceEvidence
          ↓
        OverviewSummary

    can represent one coherent Help Desk intelligence flow.
    """

    created_at = datetime(
        2023,
        3,
        5,
        19,
        34,
        56,
        tzinfo=timezone.utc,
    )

    # ---------------------------------------------------------
    # 1. Ticket
    # ---------------------------------------------------------
    ticket = Ticket(
        ticket_id="C06ggn039:1923",
        project="C06ggn039",
        issue_num=1923,
        issue_type="Ticket",
        priority=PriorityLevel.HIGH,
        status="in_progress",
        created_at=created_at,
        source="helpdesk",
    )

    # ---------------------------------------------------------
    # 2. Agent
    # ---------------------------------------------------------
    agent = Agent(
        agent_id="agent-123",
        name="Support Agent 123",
        team="Service Desk",
        active=True,
        source="helpdesk",
    )

    # ---------------------------------------------------------
    # 3. Workflow event
    # ---------------------------------------------------------
    workflow_event = WorkflowEvent(
        event_id="EVT-1923-001",
        ticket_id=ticket.ticket_id,
        timestamp=created_at,
        activity="assignment",
        status=ticket.status,
        assignee=agent.agent_id,
        team=agent.team,
        priority=ticket.priority,
        category=ticket.issue_type,
        source="helpdesk",
    )

    # ---------------------------------------------------------
    # 4. Evidence generated from Help Desk analytics
    # ---------------------------------------------------------
    evidence = IntelligenceEvidence(
        evidence_id="EV-1923-WAIT",
        scope=EvidenceScope.TICKET,
        ticket_id=ticket.ticket_id,
        evidence_type=EvidenceType.WORKFLOW_METRIC,
        metric_name="waiting_time",
        value=18.4,
        unit="hours",
        source_dataset="issues_change_history.csv",
        source_fields=[
            "timestamp",
            "status",
        ],
        metadata={
            "status": "waiting",
        },
    )

    # ---------------------------------------------------------
    # 5. Dashboard overview
    # ---------------------------------------------------------
    overview = OverviewSummary(
        total_tickets=32661,
        open_tickets=412,
        closed_tickets=31890,
        priority_distribution=[
            DistributionItem(
                label="Medium",
                count=24786,
            ),
            DistributionItem(
                label="High",
                count=4493,
            ),
        ],
        status_distribution=[
            DistributionItem(
                label="closed",
                count=31890,
            ),
            DistributionItem(
                label="open",
                count=412,
            ),
        ],
        workflow_metrics=[
            MetricSummary(
                name="median_lifecycle_hours",
                value=14.2,
                unit="hours",
            ),
        ],
        handoff_metrics=[
            MetricSummary(
                name="average_handoffs",
                value=1.8,
                unit="handoffs",
            ),
        ],
        bottlenecks=[
            {
                "status": "pending_approval",
                "median_wait_hours": 18.7,
            },
        ],
        emerging_issues=[
            {
                "topic": "payment_failures",
                "change_percent": 23.0,
            },
        ],
    )

    # ---------------------------------------------------------
    # Integration assertions
    # ---------------------------------------------------------

    # Ticket ↔ Agent
    assert agent.agent_id == "agent-123"
    assert agent.team == "Service Desk"

    # Ticket ↔ WorkflowEvent
    assert workflow_event.ticket_id == ticket.ticket_id
    assert workflow_event.priority == ticket.priority
    assert workflow_event.status == ticket.status

    # Agent ↔ WorkflowEvent
    assert workflow_event.assignee == agent.agent_id
    assert workflow_event.team == agent.team

    # WorkflowEvent ↔ Evidence
    assert evidence.ticket_id == workflow_event.ticket_id
    assert evidence.source_dataset == "issues_change_history.csv"
    assert "timestamp" in evidence.source_fields
    assert "status" in evidence.source_fields

    # Ticket priority contract shared everywhere
    assert ticket.priority == PriorityLevel.HIGH
    assert workflow_event.priority == PriorityLevel.HIGH

    # Overview dashboard contract
    assert overview.total_tickets >= overview.open_tickets
    assert overview.total_tickets >= overview.closed_tickets
    assert len(overview.priority_distribution) > 0
    assert len(overview.status_distribution) > 0
    assert len(overview.workflow_metrics) > 0
    assert len(overview.handoff_metrics) > 0
    assert len(overview.bottlenecks) > 0
    assert len(overview.emerging_issues) > 0