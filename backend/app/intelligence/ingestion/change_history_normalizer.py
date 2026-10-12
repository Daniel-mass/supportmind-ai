#ingestion/change_history_normalizer.py
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from app.intelligence.schemas.ticket import Ticket
from app.intelligence.schemas.workflow import WorkflowEvent


class ChangeHistoryNormalizationError(ValueError):
    """Raised when Help Desk change-history data cannot be normalized."""


class ChangeHistoryNormalizer:
    """
    Normalize Help Desk change-history records into WorkflowEvent objects.

    Source relationship:

        issues.csv.id
            â†“
        issues_change_history.issueid
            â†“
        canonical Ticket

    This component performs normalization only.

    It does NOT calculate:
    - lifecycle duration
    - time in status
    - transitions
    - handoffs
    - bottlenecks
    - workflow variants
    - escalation decisions

    Source-specific null handling
    -----------------------------
    The Help Desk change-history dataset contains 1,840 records where:

        field = "assignee"
        value = NULL

    These records represent an assignee being cleared/unassigned.

    Therefore:
    - null assignee values are preserved as assignee=None
    - the event itself is preserved
    - metadata records that the source value was null
    - null values for other change fields remain invalid
    """

    REQUIRED_COLUMNS = frozenset(
        {
            "id",
            "issueid",
            "field",
            "value",
            "created",
            "change_group_id",
        }
    )

    SOURCE_NAME = "helpdesk_change_history"

    NULLABLE_VALUE_FIELDS = frozenset(
        {
            "assignee",
        }
    )

    def normalize_event_row(
        self,
        row: pd.Series,
        tickets_by_source_id: Mapping[str, Ticket],
    ) -> WorkflowEvent:
        """
        Normalize one change-history row into a canonical WorkflowEvent.

        Parameters
        ----------
        row:
            One row from issues_change_history.csv.

        tickets_by_source_id:
            Mapping from the original Help Desk `issues.csv.id`
            to the corresponding canonical Ticket.

        Returns
        -------
        WorkflowEvent
            Canonical workflow event.
        """

        self._validate_row_columns(row)

        event_id = self._normalize_identifier(
            row["id"],
            field_name="id",
        )

        source_ticket_id = self._normalize_identifier(
            row["issueid"],
            field_name="issueid",
        )

        ticket = tickets_by_source_id.get(source_ticket_id)

        if ticket is None:
            raise ChangeHistoryNormalizationError(
                "No canonical ticket found for change-history "
                f"issueid {source_ticket_id!r}."
            )

        field = self._normalize_text(
            row["field"],
            field_name="field",
        )

        raw_value = row["value"]

        value = self._normalize_change_value(
            raw_value,
            field=field,
        )

        timestamp = self._normalize_timestamp(
            row["created"],
        )

        change_group_id = self._normalize_optional_identifier(
            row["change_group_id"],
        )

        status = (
            value
            if field.casefold() == "status"
            else None
        )

        assignee = (
            value
            if field.casefold() == "assignee"
            else None
        )

        metadata = {
            "source_issue_id": source_ticket_id,
            "change_field": field,
            "change_value": value,
            "change_value_was_null": value is None,
        }

        if change_group_id is not None:
            metadata["change_group_id"] = change_group_id

        return WorkflowEvent(
            event_id=event_id,
            ticket_id=ticket.ticket_id,
            timestamp=timestamp,
            activity=field,
            status=status,
            # For assignee change-history records, the value represents
            # the historical assignee at that point in time.
            #
            # A null value means the assignee was cleared.
            #
            # For all other event types there is no historical assignee
            # information in this source, so assignee remains None.
            assignee=assignee,
            team=None,
            priority=ticket.priority,
            category=ticket.issue_type,
            source=self.SOURCE_NAME,
            metadata=metadata,
        )

    def normalize_events(
        self,
        dataframe: pd.DataFrame,
        tickets_by_source_id: Mapping[str, Ticket],
    ) -> list[WorkflowEvent]:
        """
        Normalize an entire change-history DataFrame.

        The source row order is preserved.

        Chronological sorting is intentionally delegated to the
        workflow layer, which already owns temporal ordering.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if not isinstance(tickets_by_source_id, Mapping):
            raise TypeError(
                "tickets_by_source_id must be a mapping of "
                "source issue IDs to Ticket objects."
            )

        self._validate_dataframe_columns(dataframe)

        if dataframe.empty:
            return []

        events: list[WorkflowEvent] = []

        for row_index, row in dataframe.iterrows():
            try:
                event = self.normalize_event_row(
                    row=row,
                    tickets_by_source_id=tickets_by_source_id,
                )
            except ChangeHistoryNormalizationError as exc:
                raise ChangeHistoryNormalizationError(
                    "Failed to normalize change-history row "
                    f"{row_index}: {exc}"
                ) from exc

            events.append(event)

        return events

    @staticmethod
    def build_ticket_source_id_map(
        tickets: list[Ticket],
        source_ids: list[object],
    ) -> dict[str, Ticket]:
        """
        Build a source-ID â†’ Ticket mapping.

        This helper exists for callers that already have the original
        `issues.csv.id` values aligned with their normalized tickets.

        The caller is responsible for preserving the row alignment
        between `tickets` and `source_ids`.
        """

        if not isinstance(tickets, list):
            raise TypeError(
                "tickets must be a list of Ticket objects."
            )

        if not isinstance(source_ids, list):
            raise TypeError(
                "source_ids must be a list."
            )

        if len(tickets) != len(source_ids):
            raise ChangeHistoryNormalizationError(
                "tickets and source_ids must contain the same "
                "number of records."
            )

        mapping: dict[str, Ticket] = {}

        for ticket, source_id in zip(
            tickets,
            source_ids,
            strict=True,
        ):
            if not isinstance(ticket, Ticket):
                raise TypeError(
                    "Every item in tickets must be a Ticket."
                )

            normalized_source_id = (
                ChangeHistoryNormalizer._normalize_identifier(
                    source_id,
                    field_name="source_id",
                )
            )

            if normalized_source_id in mapping:
                raise ChangeHistoryNormalizationError(
                    "Duplicate source issue ID encountered: "
                    f"{normalized_source_id!r}."
                )

            mapping[normalized_source_id] = ticket

        return mapping

    @staticmethod
    def _normalize_change_value(
        value: object,
        *,
        field: str,
    ) -> str | None:
        """
        Normalize a change-history value.

        Assignee changes are the only currently supported source field
        where NULL has an explicit meaning: the assignee was cleared.

        All other fields require a non-null, non-empty value.
        """

        normalized_field = field.casefold()

        if pd.isna(value):
            if normalized_field in (
                ChangeHistoryNormalizer.NULLABLE_VALUE_FIELDS
            ):
                return None

            raise ChangeHistoryNormalizationError(
                "value cannot be null for change field "
                f"{field!r}."
            )

        text = str(value).strip()

        if not text:
            if normalized_field in (
                ChangeHistoryNormalizer.NULLABLE_VALUE_FIELDS
            ):
                return None

            raise ChangeHistoryNormalizationError(
                "value cannot be empty for change field "
                f"{field!r}."
            )

        return text

    def _validate_row_columns(
        self,
        row: pd.Series,
    ) -> None:
        if not isinstance(row, pd.Series):
            raise TypeError(
                "row must be a pandas Series."
            )

        missing_columns = sorted(
            self.REQUIRED_COLUMNS - set(row.index)
        )

        if missing_columns:
            raise ChangeHistoryNormalizationError(
                "Missing required change-history columns: "
                + ", ".join(missing_columns)
            )

    def _validate_dataframe_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        missing_columns = sorted(
            self.REQUIRED_COLUMNS - set(dataframe.columns)
        )

        if missing_columns:
            raise ChangeHistoryNormalizationError(
                "Missing required change-history columns: "
                + ", ".join(missing_columns)
            )

    @staticmethod
    def _normalize_identifier(
        value: object,
        *,
        field_name: str,
    ) -> str:
        """
        Normalize numeric/string identifiers without losing integer IDs.

        Example:
            11887.0 â†’ "11887"
            "11887"  â†’ "11887"
        """

        if pd.isna(value):
            raise ChangeHistoryNormalizationError(
                f"{field_name} cannot be null."
            )

        if isinstance(value, str):
            text = value.strip()

            if not text:
                raise ChangeHistoryNormalizationError(
                    f"{field_name} cannot be empty."
                )

            return text

        try:
            numeric_value = float(value)

            if numeric_value.is_integer():
                return str(int(numeric_value))
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        if not text:
            raise ChangeHistoryNormalizationError(
                f"{field_name} cannot be empty."
            )

        return text

    @staticmethod
    def _normalize_optional_identifier(
        value: object,
    ) -> str | None:
        """Normalize an optional identifier."""

        if pd.isna(value):
            return None

        if isinstance(value, str):
            text = value.strip()
            return text or None

        try:
            numeric_value = float(value)

            if numeric_value.is_integer():
                return str(int(numeric_value))
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        return text or None

    @staticmethod
    def _normalize_text(
        value: object,
        *,
        field_name: str,
    ) -> str:
        """Normalize a required text field."""

        if pd.isna(value):
            raise ChangeHistoryNormalizationError(
                f"{field_name} cannot be null."
            )

        text = str(value).strip()

        if not text:
            raise ChangeHistoryNormalizationError(
                f"{field_name} cannot be empty."
            )

        return text

    @staticmethod
    def _normalize_timestamp(
        value: object,
    ):
        """Normalize a timestamp and guarantee timezone awareness."""

        if pd.isna(value):
            raise ChangeHistoryNormalizationError(
                "created cannot be null."
            )

        timestamp = pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )

        if pd.isna(timestamp):
            raise ChangeHistoryNormalizationError(
                f"Invalid change-history timestamp: {value!r}"
            )

        result = timestamp.to_pydatetime()

        if result.tzinfo is None or result.utcoffset() is None:
            raise ChangeHistoryNormalizationError(
                "created must be timezone-aware."
            )

        return result
