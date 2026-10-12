import pandas as pd
import pytest

from backend.app.intelligence.ingestion.change_history_normalizer import (
    ChangeHistoryNormalizationError,
    ChangeHistoryNormalizer,
)
from backend.app.intelligence.schemas.ticket import (
    PriorityLevel,
    Ticket,
)


def make_ticket(
    ticket_id: str = "d1z0:186",
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        project="d1z0",
        issue_num=186,
        issue_type="Ticket",
        priority=PriorityLevel.HIGH,
        status="open",
        created_at=pd.Timestamp(
            "2016-03-24 15:00:00+00:00"
        ).to_pydatetime(),
        assignee="agent-1",
    )


def make_row() -> pd.Series:
    return pd.Series(
        {
            "id": 10810.0,
            "issueid": 11887.0,
            "field": "status",
            "value": "resolved",
            "created": "2016-03-24 15:35:53+00:00",
            "change_group_id": 10707.0,
        }
    )


def make_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 10810.0,
                "issueid": 11887.0,
                "field": "status",
                "value": "resolved",
                "created": "2016-03-24 15:35:53+00:00",
                "change_group_id": 10707.0,
            },
            {
                "id": 10823.0,
                "issueid": 11887.0,
                "field": "status",
                "value": "reopened",
                "created": "2016-03-24 16:12:19+00:00",
                "change_group_id": 10716.0,
            },
        ]
    )


def test_normalize_valid_status_change():
    normalizer = ChangeHistoryNormalizer()

    event = normalizer.normalize_event_row(
        row=make_row(),
        tickets_by_source_id={
            "11887": make_ticket(),
        },
    )

    assert event.event_id == "10810"
    assert event.ticket_id == "d1z0:186"
    assert event.activity == "status"
    assert event.status == "resolved"

    assert event.priority == PriorityLevel.HIGH
    assert event.assignee is None
    assert event.team is None

    assert event.source == "helpdesk_change_history"

    assert event.timestamp.tzinfo is not None

    assert event.metadata["source_issue_id"] == "11887"
    assert event.metadata["change_field"] == "status"
    assert event.metadata["change_value"] == "resolved"
    assert event.metadata["change_value_was_null"] is False
    assert event.metadata["change_group_id"] == "10707"


def test_non_status_change_is_preserved_as_activity():
    normalizer = ChangeHistoryNormalizer()

    row = make_row()

    row["field"] = "priority"
    row["value"] = "High"

    event = normalizer.normalize_event_row(
        row=row,
        tickets_by_source_id={
            "11887": make_ticket(),
        },
    )

    assert event.activity == "priority"
    assert event.status is None
    assert event.assignee is None

    assert event.metadata["change_field"] == "priority"
    assert event.metadata["change_value"] == "High"
    assert event.metadata["change_value_was_null"] is False


def test_assignee_change_preserves_historical_assignee():
    normalizer = ChangeHistoryNormalizer()

    row = make_row()

    row["field"] = "assignee"
    row["value"] = "4gsg"

    event = normalizer.normalize_event_row(
        row=row,
        tickets_by_source_id={
            "11887": make_ticket(),
        },
    )

    assert event.activity == "assignee"
    assert event.status is None
    assert event.assignee == "4gsg"
    assert event.team is None

    assert event.metadata["change_field"] == "assignee"
    assert event.metadata["change_value"] == "4gsg"
    assert event.metadata["change_value_was_null"] is False


def test_null_assignee_change_is_preserved_as_unassignment():
    normalizer = ChangeHistoryNormalizer()

    row = make_row()

    row["field"] = "assignee"
    row["value"] = None

    event = normalizer.normalize_event_row(
        row=row,
        tickets_by_source_id={
            "11887": make_ticket(),
        },
    )

    assert event.activity == "assignee"
    assert event.status is None
    assert event.assignee is None
    assert event.team is None

    assert event.metadata["change_field"] == "assignee"
    assert event.metadata["change_value"] is None
    assert event.metadata["change_value_was_null"] is True


def test_null_non_assignee_change_is_rejected():
    normalizer = ChangeHistoryNormalizer()

    row = make_row()

    row["field"] = "status"
    row["value"] = None

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="value cannot be null for change field 'status'",
    ):
        normalizer.normalize_event_row(
            row=row,
            tickets_by_source_id={
                "11887": make_ticket(),
            },
        )


def test_empty_non_assignee_change_is_rejected():
    normalizer = ChangeHistoryNormalizer()

    row = make_row()

    row["field"] = "status"
    row["value"] = "   "

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="value cannot be empty for change field 'status'",
    ):
        normalizer.normalize_event_row(
            row=row,
            tickets_by_source_id={
                "11887": make_ticket(),
            },
        )


def test_source_issue_id_maps_to_canonical_ticket():
    normalizer = ChangeHistoryNormalizer()

    ticket = make_ticket(
        ticket_id="d1z0:186",
    )

    event = normalizer.normalize_event_row(
        row=make_row(),
        tickets_by_source_id={
            "11887": ticket,
        },
    )

    assert event.ticket_id == "d1z0:186"


def test_missing_source_ticket_is_rejected():
    normalizer = ChangeHistoryNormalizer()

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="No canonical ticket found",
    ):
        normalizer.normalize_event_row(
            row=make_row(),
            tickets_by_source_id={},
        )


def test_missing_required_dataframe_columns_are_rejected():
    normalizer = ChangeHistoryNormalizer()

    dataframe = pd.DataFrame(
        {
            "id": [1],
            "issueid": [11887],
            "field": ["status"],
        }
    )

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="Missing required change-history columns",
    ):
        normalizer.normalize_events(
            dataframe=dataframe,
            tickets_by_source_id={
                "11887": make_ticket(),
            },
        )


def test_missing_required_row_column_is_rejected():
    normalizer = ChangeHistoryNormalizer()

    row = make_row().drop(
        labels=["created"],
    )

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="Missing required change-history columns",
    ):
        normalizer.normalize_event_row(
            row=row,
            tickets_by_source_id={
                "11887": make_ticket(),
            },
        )


def test_missing_event_id_is_rejected():
    normalizer = ChangeHistoryNormalizer()

    row = make_row()
    row["id"] = None

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="id cannot be null",
    ):
        normalizer.normalize_event_row(
            row=row,
            tickets_by_source_id={
                "11887": make_ticket(),
            },
        )


def test_invalid_timestamp_is_rejected():
    normalizer = ChangeHistoryNormalizer()

    row = make_row()
    row["created"] = "not-a-valid-timestamp"

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="Invalid change-history timestamp",
    ):
        normalizer.normalize_event_row(
            row=row,
            tickets_by_source_id={
                "11887": make_ticket(),
            },
        )


def test_normalize_multiple_events():
    normalizer = ChangeHistoryNormalizer()

    events = normalizer.normalize_events(
        dataframe=make_dataframe(),
        tickets_by_source_id={
            "11887": make_ticket(),
        },
    )

    assert len(events) == 2

    assert events[0].event_id == "10810"
    assert events[0].status == "resolved"

    assert events[1].event_id == "10823"
    assert events[1].status == "reopened"

    assert all(
        event.ticket_id == "d1z0:186"
        for event in events
    )


def test_empty_dataframe_returns_empty_list():
    normalizer = ChangeHistoryNormalizer()

    dataframe = pd.DataFrame(
        columns=[
            "id",
            "issueid",
            "field",
            "value",
            "created",
            "change_group_id",
        ]
    )

    events = normalizer.normalize_events(
        dataframe=dataframe,
        tickets_by_source_id={},
    )

    assert events == []


def test_build_ticket_source_id_map():
    normalizer = ChangeHistoryNormalizer()

    tickets = [
        make_ticket("d1z0:186"),
        make_ticket("d1z0:190"),
    ]

    mapping = normalizer.build_ticket_source_id_map(
        tickets=tickets,
        source_ids=[
            11887,
            11890,
        ],
    )

    assert set(mapping.keys()) == {
        "11887",
        "11890",
    }

    assert mapping["11887"].ticket_id == "d1z0:186"
    assert mapping["11890"].ticket_id == "d1z0:190"


def test_source_id_map_requires_equal_lengths():
    normalizer = ChangeHistoryNormalizer()

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="same number",
    ):
        normalizer.build_ticket_source_id_map(
            tickets=[
                make_ticket(),
            ],
            source_ids=[],
        )


def test_source_id_map_rejects_duplicate_ids():
    normalizer = ChangeHistoryNormalizer()

    tickets = [
        make_ticket("d1z0:186"),
        make_ticket("d1z0:190"),
    ]

    with pytest.raises(
        ChangeHistoryNormalizationError,
        match="Duplicate source issue ID",
    ):
        normalizer.build_ticket_source_id_map(
            tickets=tickets,
            source_ids=[
                11887,
                11887,
            ],
        )