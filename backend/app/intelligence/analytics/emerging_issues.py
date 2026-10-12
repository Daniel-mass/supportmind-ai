from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Iterable

from app.intelligence.schemas.overview import (
    DistributionItem,
)
from app.intelligence.schemas.ticket import Ticket


DEFAULT_RECENT_DAYS = 30
DEFAULT_MIN_RECENT_COUNT = 5
DEFAULT_MAX_RESULTS = 10


def identify_emerging_issues(
    tickets: Iterable[Ticket],
    recent_days: int = DEFAULT_RECENT_DAYS,
    min_recent_count: int = DEFAULT_MIN_RECENT_COUNT,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[DistributionItem]:
    """
    Identify issue types whose recent ticket volume is increasing.

    The dataset's latest ticket creation timestamp is used as the
    reference point. This keeps the calculation deterministic for
    historical datasets and avoids depending on the current wall clock.

    Two equal windows are compared:

        recent window
        previous window

    An issue type is considered emerging when:

    - it has at least ``min_recent_count`` tickets in the recent window
    - its recent volume is greater than its previous-window volume

    The returned DistributionItem count represents the number of
    tickets created in the recent window.

    Notes
    -----
    The existing API contract exposes DistributionItem only, so the
    growth percentage is intentionally used for ranking internally
    rather than added to the public schema.
    """

    ticket_list = list(tickets)

    if not ticket_list:
        return []

    if recent_days <= 0:
        raise ValueError("recent_days must be greater than zero")

    if min_recent_count <= 0:
        raise ValueError(
            "min_recent_count must be greater than zero"
        )

    if max_results <= 0:
        raise ValueError(
            "max_results must be greater than zero"
        )

    latest_created_at = max(
        ticket.created_at
        for ticket in ticket_list
    )

    recent_start = (
        latest_created_at
        - timedelta(days=recent_days)
    )

    previous_start = (
        recent_start
        - timedelta(days=recent_days)
    )

    recent_counts: Counter[str] = Counter()
    previous_counts: Counter[str] = Counter()

    labels: dict[str, str] = {}

    for ticket in ticket_list:
        issue_type = ticket.issue_type.strip()

        if not issue_type:
            continue

        normalized_issue_type = issue_type.casefold()

        labels.setdefault(
            normalized_issue_type,
            issue_type,
        )

        created_at = ticket.created_at

        if (
            recent_start
            < created_at
            <= latest_created_at
        ):
            recent_counts[
                normalized_issue_type
            ] += 1

        elif (
            previous_start
            < created_at
            <= recent_start
        ):
            previous_counts[
                normalized_issue_type
            ] += 1

    candidates: list[
        tuple[
            str,
            int,
            int,
            float,
            int,
        ]
    ] = []

    for normalized_issue_type, recent_count in (
        recent_counts.items()
    ):
        if recent_count < min_recent_count:
            continue

        previous_count = previous_counts.get(
            normalized_issue_type,
            0,
        )

        if recent_count <= previous_count:
            continue

        if previous_count == 0:
            growth_score = float("inf")
        else:
            growth_score = (
                (recent_count - previous_count)
                / previous_count
            )

        absolute_growth = (
            recent_count - previous_count
        )

        candidates.append(
            (
                normalized_issue_type,
                recent_count,
                previous_count,
                growth_score,
                absolute_growth,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[3],
            item[4],
            item[1],
            labels[item[0]].casefold(),
        ),
        reverse=True,
    )

    return [
        DistributionItem(
            label=labels[normalized_issue_type],
            count=recent_count,
        )
        for (
            normalized_issue_type,
            recent_count,
            _previous_count,
            _growth_score,
            _absolute_growth,
        ) in candidates[:max_results]
    ]