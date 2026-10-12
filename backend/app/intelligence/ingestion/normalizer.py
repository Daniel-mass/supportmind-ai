#ingestion/normalizer.py
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from app.intelligence.schemas.ticket import (
    PriorityLevel,
    Ticket,
)


class HelpDeskNormalizationError(Exception):
    """Raised when a Help Desk record cannot be normalized."""


class HelpDeskNormalizer:
    """
    Converts raw Help Desk records into canonical SupportMind schemas.

    Current scope:
        issues.csv â†’ Ticket

    Workflow-event normalization will be added separately once the
    change-history source schema is validated.
    """

    PRIORITY_MAP = {
        "lowest": PriorityLevel.LOWEST,
        "low": PriorityLevel.LOW,
        "medium": PriorityLevel.MEDIUM,
        "high": PriorityLevel.HIGH,
        "highest": PriorityLevel.HIGHEST,
        "blocker": PriorityLevel.BLOCKER,
        "unknown": PriorityLevel.UNKNOWN,
    }

    def _optional_string(
        self,
        value: Any,
    ) -> str | None:
        """Convert a raw value into a clean optional string."""

        if value is None or pd.isna(value):
            return None

        text = str(value).strip()

        return text or None

    def _optional_datetime(
        self,
        value: Any,
    ) -> datetime | None:
        """
        Convert a raw timestamp into a timezone-aware datetime.

        Help Desk timestamps are treated as UTC.
        """

        if value is None or pd.isna(value):
            return None

        parsed = pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
            format="mixed",
        )

        if pd.isna(parsed):
            raise HelpDeskNormalizationError(
                f"Unable to parse timestamp: {value!r}"
            )

        return parsed.to_pydatetime()

    def _required_datetime(
        self,
        value: Any,
        field_name: str,
    ) -> datetime:
        """Parse a required timestamp."""

        parsed = self._optional_datetime(value)

        if parsed is None:
            raise HelpDeskNormalizationError(
                f"Required timestamp '{field_name}' is missing"
            )

        return parsed

    def _priority(
        self,
        value: Any,
    ) -> PriorityLevel:
        """Normalize the raw Help Desk priority."""

        if value is None or pd.isna(value):
            return PriorityLevel.UNKNOWN

        normalized = str(value).strip().lower()

        if normalized not in self.PRIORITY_MAP:
            raise HelpDeskNormalizationError(
                f"Unsupported priority value: {value!r}"
            )

        return self.PRIORITY_MAP[normalized]

    def _non_negative_int(
        self,
        value: Any,
        field_name: str,
    ) -> int:
        """Convert a numeric count into a non-negative integer."""

        if value is None or pd.isna(value):
            return 0

        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise HelpDeskNormalizationError(
                f"Invalid numeric value for '{field_name}': {value!r}"
            ) from exc

        if result < 0:
            raise HelpDeskNormalizationError(
                f"'{field_name}' cannot be negative: {result}"
            )

        return result

    def normalize_ticket_row(
        self,
        row: pd.Series,
    ) -> Ticket:
        """
        Convert one raw issues.csv row into a canonical Ticket.
        """

        project = self._optional_string(
            row.get("issue_proj")
        )

        issue_num_raw = row.get("issue_num")

        if project is None:
            raise HelpDeskNormalizationError(
                "issue_proj is required"
            )

        if issue_num_raw is None or pd.isna(issue_num_raw):
            raise HelpDeskNormalizationError(
                "issue_num is required"
            )

        try:
            issue_num = int(issue_num_raw)
        except (TypeError, ValueError) as exc:
            raise HelpDeskNormalizationError(
                f"Invalid issue_num: {issue_num_raw!r}"
            ) from exc

        if issue_num < 0:
            raise HelpDeskNormalizationError(
                "issue_num cannot be negative"
            )

        ticket_id = f"{project}:{issue_num}"

        metadata = {
            "raw_id": self._optional_string(
                row.get("id")
            ),
        }

        return Ticket(
            ticket_id=ticket_id,
            project=project,
            issue_num=issue_num,
            issue_type=(
                self._optional_string(
                    row.get("issue_type")
                )
                or "Unknown"
            ),
            priority=self._priority(
                row.get("issue_priority")
            ),
            status=(
                self._optional_string(
                    row.get("issue_status")
                )
                or "unknown"
            ),
            created_at=self._required_datetime(
                row.get("issue_created"),
                "issue_created",
            ),
            started_at=self._optional_datetime(
                row.get("started")
            ),
            ended_at=self._optional_datetime(
                row.get("ended")
            ),
            resolution_date=self._optional_datetime(
                row.get("issue_resolution_date")
            ),
            resolution=self._optional_string(
                row.get("issue_resolution")
            ),
            reporter=self._optional_string(
                row.get("issue_reporter")
            ),
            assignee=self._optional_string(
                row.get("issue_assignee")
            ),
            contributor_count=self._non_negative_int(
                row.get("issue_contr_count"),
                "issue_contr_count",
            ),
            comments_count=self._non_negative_int(
                row.get("issue_comments_count"),
                "issue_comments_count",
            ),
            source="helpdesk",
            metadata=metadata,
        )

    def normalize_tickets(
        self,
        dataframe: pd.DataFrame,
    ) -> list[Ticket]:
        """
        Normalize all rows in an issues.csv DataFrame.
        """

        if dataframe.empty:
            return []

        required_columns = {
            "issue_proj",
            "issue_num",
            "issue_type",
            "issue_priority",
            "issue_status",
            "issue_created",
        }

        missing_columns = required_columns - set(
            dataframe.columns
        )

        if missing_columns:
            raise HelpDeskNormalizationError(
                "Missing required issues.csv columns: "
                f"{sorted(missing_columns)}"
            )

        tickets: list[Ticket] = []

        for index, row in dataframe.iterrows():
            try:
                tickets.append(
                    self.normalize_ticket_row(row)
                )
            except HelpDeskNormalizationError as exc:
                raise HelpDeskNormalizationError(
                    f"Failed to normalize issues.csv row "
                    f"{index}: {exc}"
                ) from exc

        return tickets
