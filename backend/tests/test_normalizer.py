#test_normalziaer.py
from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from backend.app.intelligence.ingestion.normalizer import (
    HelpDeskNormalizationError,
    HelpDeskNormalizer,
)
from backend.app.intelligence.schemas.ticket import (
    PriorityLevel,
)


def make_raw_row() -> pd.Series:
    return pd.Series(
        {
            "id": 12345,
            "issue_proj": "C06ggn039",
            "issue_num": 1923.0,
            "issue_type": "Ticket",
            "issue_priority": "High",
            "issue_status": "closed",
            "issue_created": "2023-03-05 19:34:56+00:00",
            "started": "2023-03-05 19:34:56+00:00",
            "ended": "2023-03-05 20:34:56+00:00",
            "issue_resolution_date": (
                "2023-03-06 10:00:00+00:00"
            ),
            "issue_resolution": "Done",
            "issue_reporter": "reporter-1",
            "issue_assignee": "agent-123",
            "issue_contr_count": 2,
            "issue_comments_count": 5,
        }
    )


def test_normalize_valid_ticket():
    normalizer = HelpDeskNormalizer()

    ticket = normalizer.normalize_ticket_row(
        make_raw_row()
    )

    assert ticket.ticket_id == "C06ggn039:1923"
    assert ticket.project == "C06ggn039"
    assert ticket.issue_num == 1923
    assert ticket.priority == PriorityLevel.HIGH
    assert ticket.status == "closed"
    assert ticket.reporter == "reporter-1"
    assert ticket.assignee == "agent-123"
    assert ticket.contributor_count == 2
    assert ticket.comments_count == 5


def test_priority_values_are_normalized():
    normalizer = HelpDeskNormalizer()

    row = make_raw_row()
    row["issue_priority"] = "medium"

    ticket = normalizer.normalize_ticket_row(row)

    assert ticket.priority == PriorityLevel.MEDIUM


def test_unknown_priority_is_supported():
    normalizer = HelpDeskNormalizer()

    row = make_raw_row()
    row["issue_priority"] = "unknown"

    ticket = normalizer.normalize_ticket_row(row)

    assert ticket.priority == PriorityLevel.UNKNOWN


def test_invalid_priority_is_rejected():
    normalizer = HelpDeskNormalizer()

    row = make_raw_row()
    row["issue_priority"] = "Critical"

    with pytest.raises(HelpDeskNormalizationError):
        normalizer.normalize_ticket_row(row)


def test_ticket_id_is_built_from_project_and_issue_num():
    normalizer = HelpDeskNormalizer()

    ticket = normalizer.normalize_ticket_row(
        make_raw_row()
    )

    assert ticket.ticket_id == "C06ggn039:1923"


def test_timestamps_are_timezone_aware():
    normalizer = HelpDeskNormalizer()

    ticket = normalizer.normalize_ticket_row(
        make_raw_row()
    )

    assert ticket.created_at.tzinfo is not None
    assert ticket.started_at is not None
    assert ticket.started_at.tzinfo is not None


def test_missing_optional_values_are_normalized():
    normalizer = HelpDeskNormalizer()

    row = make_raw_row()

    row["issue_assignee"] = None
    row["issue_reporter"] = None
    row["ended"] = None
    row["issue_resolution_date"] = None
    row["issue_resolution"] = None

    ticket = normalizer.normalize_ticket_row(row)

    assert ticket.assignee is None
    assert ticket.reporter is None
    assert ticket.ended_at is None
    assert ticket.resolution_date is None
    assert ticket.resolution is None


def test_missing_required_timestamp_is_rejected():
    normalizer = HelpDeskNormalizer()

    row = make_raw_row()
    row["issue_created"] = None

    with pytest.raises(HelpDeskNormalizationError):
        normalizer.normalize_ticket_row(row)


def test_required_columns_are_checked():
    normalizer = HelpDeskNormalizer()

    dataframe = pd.DataFrame(
        {
            "issue_proj": ["C01"],
            "issue_num": [1],
        }
    )

    with pytest.raises(HelpDeskNormalizationError):
        normalizer.normalize_tickets(dataframe)


def test_normalize_multiple_tickets():
    normalizer = HelpDeskNormalizer()

    row_1 = make_raw_row()

    row_2 = make_raw_row()
    row_2["issue_num"] = 1924
    row_2["issue_priority"] = "Medium"

    dataframe = pd.DataFrame(
        [row_1, row_2]
    )

    tickets = normalizer.normalize_tickets(
        dataframe
    )

    assert len(tickets) == 2
    assert tickets[0].ticket_id == "C06ggn039:1923"
    assert tickets[1].ticket_id == "C06ggn039:1924"


def test_empty_dataframe_returns_empty_list():
    normalizer = HelpDeskNormalizer()

    dataframe = pd.DataFrame(
        columns=[
            "issue_proj",
            "issue_num",
            "issue_type",
            "issue_priority",
            "issue_status",
            "issue_created",
        ]
    )

    tickets = normalizer.normalize_tickets(
        dataframe
    )

    assert tickets == []