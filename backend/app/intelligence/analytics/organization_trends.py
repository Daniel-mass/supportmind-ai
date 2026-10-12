
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.intelligence.providers.helpdesk_provider import (
        HelpDeskIntelligenceSnapshot,
    )
from app.intelligence.schemas.trends import (
    MonthlyTrendPoint,
    OrganizationTrends,
    TrendCategory,
)


def _month_key(value: datetime) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _month_range(
    start: str,
    end: str,
) -> list[str]:
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))

    months: list[str] = []
    year, month = start_year, start_month

    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return months


def _category_trends(
    counts: Counter[tuple[str, str]],
) -> list[TrendCategory]:
    return [
        TrendCategory(
            month=month,
            category=category,
            count=count,
        )
        for (month, category), count in sorted(
            counts.items(),
            key=lambda item: (
                item[0][0],
                item[0][1].casefold(),
            ),
        )
    ]


def build_organization_trends(
    snapshot: HelpDeskIntelligenceSnapshot,
) -> OrganizationTrends:
    tickets = list(snapshot.tickets_by_id.values())

    all_dates = [
        ticket.created_at
        for ticket in tickets
    ]

    all_dates.extend(
        event.timestamp
        for events in snapshot.workflow_events_by_ticket.values()
        for event in events
    )

    if not all_dates:
        return OrganizationTrends(
            months=[],
            ticket_volume=[],
            issue_type_trends=[],
            priority_trends=[],
            status_trends=[],
            total_tickets=0,
            total_workflow_events=0,
        )

    first_month = min(
        _month_key(value) for value in all_dates
    )
    last_month = max(
        _month_key(value) for value in all_dates
    )
    months = _month_range(first_month, last_month)

    ticket_counts: Counter[str] = Counter()
    issue_counts: Counter[tuple[str, str]] = Counter()
    priority_counts: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[tuple[str, str]] = Counter()

    for ticket in tickets:
        month = _month_key(ticket.created_at)
        ticket_counts[month] += 1

        issue_type = ticket.issue_type.strip()
        if issue_type:
            issue_counts[(month, issue_type)] += 1

        priority = ticket.priority.value
        priority_counts[(month, priority)] += 1

    workflow_event_count = 0

    for events in snapshot.workflow_events_by_ticket.values():
        for event in events:
            workflow_event_count += 1

            # Status trends count actual events carrying a status,
            # not the current status repeated across historical months.
            if event.status:
                month = _month_key(event.timestamp)
                status_counts[(month, event.status.strip())] += 1

    return OrganizationTrends(
        months=months,
        ticket_volume=[
            MonthlyTrendPoint(
                month=month,
                count=ticket_counts[month],
            )
            for month in months
        ],
        issue_type_trends=_category_trends(issue_counts),
        priority_trends=_category_trends(priority_counts),
        status_trends=_category_trends(status_counts),
        total_tickets=len(tickets),
        total_workflow_events=workflow_event_count,
    )