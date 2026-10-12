#helpdeskingestionintgegrationn test
from pathlib import Path

from backend.app.intelligence.ingestion.helpdesk_loader import (
    HelpDeskLoader,
    HelpDeskFileSet,
)
from backend.app.intelligence.ingestion.normalizer import (
    HelpDeskNormalizer,
)
from backend.app.intelligence.schemas.ticket import (
    PriorityLevel,
)


HELPDESK_DIR = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\Help Desk Tickets\Help Desk Tickets"
)

ISSUES_PATH = HELPDESK_DIR / "issues.csv"


def test_real_helpdesk_ingestion_pipeline():
    """
    End-to-end integration test for:

        real issues.csv
            ↓
        HelpDeskLoader
            ↓
        HelpDeskNormalizer
            ↓
        canonical Ticket objects
    """

    # ---------------------------------------------------------
    # 1. Configure real source
    # ---------------------------------------------------------
    files = HelpDeskFileSet(
        issues_csv=ISSUES_PATH,
    )

    loader = HelpDeskLoader()
    normalizer = HelpDeskNormalizer()

    # ---------------------------------------------------------
    # 2. Load real CSV
    # ---------------------------------------------------------
    sources = loader.load_all(files)

    assert "issues" in sources

    issues_df = sources["issues"]

    # Known real dataset size
    assert len(issues_df) == 66691
    assert issues_df.shape[1] == 58

    # ---------------------------------------------------------
    # 3. Normalize real Help Desk records
    # ---------------------------------------------------------
    tickets = normalizer.normalize_tickets(
        issues_df
    )

    # One canonical Ticket per source row at this stage.
    assert len(tickets) == 66691

    # ---------------------------------------------------------
    # 4. Validate representative ticket values
    # ---------------------------------------------------------
    first_ticket = tickets[0]

    assert first_ticket.ticket_id
    assert first_ticket.project
    assert first_ticket.issue_num >= 0
    assert first_ticket.issue_type
    assert first_ticket.status
    assert first_ticket.created_at.tzinfo is not None

    # ---------------------------------------------------------
    # 5. Validate canonical priority contract
    # ---------------------------------------------------------
    valid_priorities = {
        PriorityLevel.LOWEST,
        PriorityLevel.LOW,
        PriorityLevel.MEDIUM,
        PriorityLevel.HIGH,
        PriorityLevel.HIGHEST,
        PriorityLevel.BLOCKER,
        PriorityLevel.UNKNOWN,
    }

    for ticket in tickets[:1000]:
        assert ticket.priority in valid_priorities

    # ---------------------------------------------------------
    # 6. Validate issue identity construction
    # ---------------------------------------------------------
    raw_first = issues_df.iloc[0]

    expected_ticket_id = (
        f"{raw_first['issue_proj']}:"
        f"{int(raw_first['issue_num'])}"
    )

    assert first_ticket.ticket_id == expected_ticket_id

    # ---------------------------------------------------------
    # 7. Validate timestamps across a sample
    # ---------------------------------------------------------
    for ticket in tickets[:1000]:
        assert ticket.created_at.tzinfo is not None

        if ticket.started_at is not None:
            assert ticket.started_at.tzinfo is not None

        if ticket.ended_at is not None:
            assert ticket.ended_at.tzinfo is not None

        if ticket.resolution_date is not None:
            assert ticket.resolution_date.tzinfo is not None

    # ---------------------------------------------------------
    # 8. Basic sanity checks
    # ---------------------------------------------------------
    ticket_ids = [
        ticket.ticket_id
        for ticket in tickets
    ]

    assert all(ticket_ids)
    assert all(ticket.project for ticket in tickets)
    assert all(ticket.issue_type for ticket in tickets)

    print("\nReal Help Desk ingestion integration test")
    print("=" * 60)
    print("Source rows       :", len(issues_df))
    print("Source columns    :", len(issues_df.columns))
    print("Canonical tickets :", len(tickets))
    print("First ticket      :", first_ticket.ticket_id)
    print("First priority    :", first_ticket.priority.value)