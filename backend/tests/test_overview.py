import pytest
from pydantic import ValidationError

from backend.app.intelligence.schemas.overview import (
    DistributionItem,
    MetricSummary,
    OverviewSummary,
)


def make_valid_overview() -> OverviewSummary:
    return OverviewSummary(
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
            )
        ],
        handoff_metrics=[
            MetricSummary(
                name="average_handoffs",
                value=1.8,
                unit="handoffs",
            )
        ],
        bottlenecks=[
            {
                "status": "pending_approval",
                "median_wait_hours": 18.7,
            }
        ],
        emerging_issues=[
            {
                "topic": "payment_failures",
                "change_percent": 23.0,
            }
        ],
    )


def test_valid_overview_creation():
    overview = make_valid_overview()

    assert overview.total_tickets == 32661
    assert overview.open_tickets == 412
    assert overview.closed_tickets == 31890
    assert len(overview.priority_distribution) == 2
    assert len(overview.workflow_metrics) == 1


def test_distribution_item_requires_non_negative_count():
    with pytest.raises(ValidationError):
        DistributionItem(
            label="High",
            count=-1,
        )


def test_metric_summary_supports_numeric_values():
    metric = MetricSummary(
        name="median_wait",
        value=12.5,
        unit="hours",
    )

    assert metric.name == "median_wait"
    assert metric.value == 12.5
    assert metric.unit == "hours"


def test_empty_distributions_are_supported():
    overview = OverviewSummary(
        total_tickets=0,
        open_tickets=0,
        closed_tickets=0,
    )

    assert overview.priority_distribution == []
    assert overview.status_distribution == []
    assert overview.workflow_metrics == []
    assert overview.handoff_metrics == []
    assert overview.bottlenecks == []
    assert overview.emerging_issues == []


def test_negative_ticket_counts_are_rejected():
    with pytest.raises(ValidationError):
        OverviewSummary(
            total_tickets=-1,
            open_tickets=0,
            closed_tickets=0,
        )


def test_empty_distribution_label_is_rejected():
    with pytest.raises(ValidationError):
        DistributionItem(
            label="",
            count=5,
        )


def test_empty_metric_name_is_rejected():
    with pytest.raises(ValidationError):
        MetricSummary(
            name="",
            value=5,
        )


def test_assignment_validation_is_enabled():
    overview = make_valid_overview()

    with pytest.raises(ValidationError):
        overview.total_tickets = -10


def test_unexpected_fields_are_rejected():
    with pytest.raises(ValidationError):
        OverviewSummary(
            total_tickets=10,
            open_tickets=2,
            closed_tickets=8,
            unexpected_field="should_fail",
        )