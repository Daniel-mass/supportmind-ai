#evidence_service.py
from __future__ import annotations

from collections.abc import Iterable

from app.intelligence.analytics.organization import (
    OrganizationAnalytics,
)
from app.intelligence.analytics.ticket_analytics import (
    TicketAnalytics,
)
from app.intelligence.schemas.evidence import (
    IntelligenceEvidence,
)


def _evidence_id(
    scope: str,
    subject_id: str,
    metric_name: str,
) -> str:
    return (
        f"{scope}:"
        f"{subject_id}:"
        f"{metric_name}"
    )


def build_ticket_evidence(
    analytics: TicketAnalytics,
    *,
    source_dataset: str = "helpdesk",
) -> list[IntelligenceEvidence]:
    """
    Convert ticket-level analytics into traceable evidence objects.

    No new analytics or inference is performed here.
    """

    ticket_id = analytics.ticket_id

    evidence = [
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "ticket",
                ticket_id,
                "lifecycle_seconds",
            ),
            scope="ticket",
            ticket_id=ticket_id,
            evidence_type="ticket_metric",
            metric_name="lifecycle_seconds",
            value=analytics.lifecycle_seconds,
            unit="seconds",
            source_dataset=source_dataset,
            source_fields=[
                "started",
                "ended",
            ],
            metadata={
                "derived_from": "TicketAnalytics",
            },
        ),
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "ticket",
                ticket_id,
                "waiting_seconds",
            ),
            scope="ticket",
            ticket_id=ticket_id,
            evidence_type="ticket_metric",
            metric_name="waiting_seconds",
            value=analytics.waiting_seconds,
            unit="seconds",
            source_dataset=source_dataset,
            source_fields=[
                "status",
                "workflow_events",
            ],
            metadata={
                "derived_from": "TicketAnalytics",
            },
        ),
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "ticket",
                ticket_id,
                "transition_count",
            ),
            scope="ticket",
            ticket_id=ticket_id,
            evidence_type="workflow_metric",
            metric_name="transition_count",
            value=analytics.transition_count,
            unit="count",
            source_dataset=source_dataset,
            source_fields=[
                "issues_change_history.field",
                "issues_change_history.created",
            ],
            metadata={
                "derived_from": "TicketAnalytics",
            },
        ),
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "ticket",
                ticket_id,
                "handoff_count",
            ),
            scope="ticket",
            ticket_id=ticket_id,
            evidence_type="handoff",
            metric_name="handoff_count",
            value=analytics.handoff_count,
            unit="count",
            source_dataset=source_dataset,
            source_fields=[
                "assignee",
                "issues_change_history.field",
            ],
            metadata={
                "derived_from": "TicketAnalytics",
            },
        ),
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "ticket",
                ticket_id,
                "priority",
            ),
            scope="ticket",
            ticket_id=ticket_id,
            evidence_type="operational_signal",
            metric_name="priority",
            value=analytics.priority,
            unit=None,
            source_dataset=source_dataset,
            source_fields=[
                "issue_priority",
            ],
            metadata={
                "derived_from": "TicketAnalytics",
            },
        ),
    ]

    return evidence


def build_organization_evidence(
    analytics: OrganizationAnalytics,
    *,
    source_dataset: str = "helpdesk",
) -> list[IntelligenceEvidence]:
    """
    Convert organization-level analytics into traceable evidence.
    """

    evidence = [
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "organization",
                "support",
                "total_ticket_count",
            ),
            scope="organization",
            evidence_type="ticket_metric",
            metric_name="total_ticket_count",
            value=analytics.total_ticket_count,
            unit="count",
            source_dataset=source_dataset,
            source_fields=[
                "issues.csv",
            ],
            metadata={
                "derived_from": "OrganizationAnalytics",
            },
        ),
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "organization",
                "support",
                "open_ticket_count",
            ),
            scope="organization",
            evidence_type="ticket_metric",
            metric_name="open_ticket_count",
            value=analytics.open_ticket_count,
            unit="count",
            source_dataset=source_dataset,
            source_fields=[
                "issue_status",
            ],
            metadata={
                "derived_from": "OrganizationAnalytics",
            },
        ),
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "organization",
                "support",
                "high_priority_ticket_count",
            ),
            scope="organization",
            evidence_type="operational_signal",
            metric_name="high_priority_ticket_count",
            value=analytics.high_priority_ticket_count,
            unit="count",
            source_dataset=source_dataset,
            source_fields=[
                "issue_priority",
            ],
            metadata={
                "derived_from": "OrganizationAnalytics",
            },
        ),
        IntelligenceEvidence(
            evidence_id=_evidence_id(
                "organization",
                "support",
                "waiting_ticket_count",
            ),
            scope="organization",
            evidence_type="workflow_metric",
            metric_name="waiting_ticket_count",
            value=analytics.waiting_ticket_count,
            unit="count",
            source_dataset=source_dataset,
            source_fields=[
                "issue_status",
                "issues_change_history.field",
            ],
            metadata={
                "derived_from": "OrganizationAnalytics",
            },
        ),
    ]

    return evidence


def build_evidence(
    *,
    ticket_analytics: Iterable[
        TicketAnalytics
    ] | None = None,
    organization_analytics: OrganizationAnalytics | None = None,
    source_dataset: str = "helpdesk",
) -> list[IntelligenceEvidence]:
    """
    Build the complete evidence collection from existing analytics.
    """

    evidence: list[
        IntelligenceEvidence
    ] = []

    for analytics in (
        ticket_analytics or []
    ):
        evidence.extend(
            build_ticket_evidence(
                analytics,
                source_dataset=source_dataset,
            )
        )

    if organization_analytics is not None:
        evidence.extend(
            build_organization_evidence(
                organization_analytics,
                source_dataset=source_dataset,
            )
        )

    return evidence
