
from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path

import pandas as pd
import pytest
from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.escalation_agent.agent import (
    EscalationAgent,
)
from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationAgentStatus,
)
from app.agents.escalation_agent.operational_intelligence import (
    EscalationOperationalIntelligence,
)
from app.intelligence.analytics.agent_analytics import (
    build_agent_analytics,
)
from app.intelligence.analytics.organization import (
    build_organization_analytics,
)
from app.intelligence.analytics.team_analytics import (
    build_team_analytics,
)
from app.intelligence.analytics.ticket_analytics import (
    build_ticket_analytics,
)
from app.intelligence.ingestion.change_history_normalizer import (
    ChangeHistoryNormalizer,
)
from app.intelligence.ingestion.helpdesk_loader import (
    HelpDeskFileSet,
    HelpDeskLoader,
)
from app.intelligence.ingestion.normalizer import (
    HelpDeskNormalizer,
)
from app.intelligence.schemas.agent import (
    Agent,
)
from app.intelligence.workflow.bottlenecks import (
    identify_bottlenecks,
)
from app.intelligence.workflow.event_builder import (
    build_workflow_intervals,
)
from app.intelligence.workflow.handoffs import (
    identify_handoffs,
)
from app.intelligence.workflow.metrics import (
    calculate_workflow_metrics,
)
from app.intelligence.workflow.variants import (
    identify_workflow_variants,
)


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_ESCALATION") != "1",
    reason="Set RUN_REAL_ESCALATION=1 to run the real Escalation Agent integration test.",
)


HELPDESK_DIR = Path(
    r"C:\Users\danie\supportmind-ai"
    r"\rawsupportminddata"
    r"\Help Desk Tickets"
    r"\Help Desk Tickets"
)

ISSUES_PATH = HELPDESK_DIR / "issues.csv"

CHANGE_HISTORY_PATH = (
    HELPDESK_DIR / "issues_change_history.csv"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "SUPPORTMIND_LLM_MODEL",
    "qwen3:8b",
)


def build_model_client() -> OllamaChatCompletionClient:
    """
    Build the real Ollama model client used by AutoGen.
    """

    return OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
    )


async def close_model_client(
    model_client: OllamaChatCompletionClient,
) -> None:
    """
    Close the model client inside the same event loop in which it
    was used.
    """

    close_method = getattr(
        model_client,
        "close",
        None,
    )

    if close_method is None:
        return

    result = close_method()

    if inspect.isawaitable(result):
        await result


def build_agents(
    tickets,
) -> list[Agent]:
    """
    Build canonical Agent entities from real ticket assignees.

    No operational metric is calculated here.

    The identities come directly from the real issues.csv data.
    """

    agent_ids = sorted(
        {
            ticket.assignee
            for ticket in tickets
            if ticket.assignee
        }
    )

    return [
        Agent(
            agent_id=agent_id,
            name=agent_id,
            team=None,
            active=True,
            source="helpdesk",
        )
        for agent_id in agent_ids
    ]


def build_agent_ticket_groups(
    tickets,
) -> dict[str, list]:
    """
    Group real tickets by their canonical assignee.
    """

    groups: dict[str, list] = {}

    for ticket in tickets:
        if not ticket.assignee:
            continue

        groups.setdefault(
            ticket.assignee,
            [],
        ).append(ticket)

    return groups


def build_team_analytics_from_real_data(
    agents: list[Agent],
    agent_analytics: list,
):
    """
    Build team analytics from available real agent analytics.

    The supplied Help Desk schema does not provide a canonical team
    assignment in issues.csv, so no artificial team assignment is
    created.
    """

    teams = sorted(
        {
            agent.team
            for agent in agents
            if agent.team
        }
    )

    return [
        build_team_analytics(
            team=team,
            agents=[
                agent
                for agent in agents
                if agent.team == team
            ],
            agent_analytics=[
                analytics
                for analytics in agent_analytics
                if analytics.team == team
            ],
        )
        for team in teams
    ]


def run_real_operational_pipeline():
    """
    Execute the real Help Desk â†’ Intelligence pipeline.

    Returns
    -------
    dict
        Real artifacts required by the Escalation Agent test.
    """

    # ---------------------------------------------------------
    # 1. Load real source files
    # ---------------------------------------------------------

    files = HelpDeskFileSet(
        issues_csv=ISSUES_PATH,
        change_history_csv=CHANGE_HISTORY_PATH,
    )

    loader = HelpDeskLoader()

    sources = loader.load_all(files)

    assert "issues" in sources
    assert "change_history" in sources

    issues_df = sources["issues"]
    change_history_df = sources["change_history"]

    # ---------------------------------------------------------
    # 2. Validate real dataset sizes
    # ---------------------------------------------------------

    assert len(issues_df) == 66691
    assert len(change_history_df) == 257508

    # ---------------------------------------------------------
    # 3. Normalize real tickets
    # ---------------------------------------------------------

    ticket_normalizer = HelpDeskNormalizer()

    tickets = ticket_normalizer.normalize_tickets(
        issues_df,
    )

    assert len(tickets) == 66691

    # ---------------------------------------------------------
    # 4. Build the REAL source-ID â†’ Ticket mapping
    #
    # issues.csv.id
    #       â†•
    # issues_change_history.issueid
    # ---------------------------------------------------------

    change_history_normalizer = ChangeHistoryNormalizer()

    source_ids = (
        issues_df["id"]
        .tolist()
    )

    tickets_by_source_id = (
        change_history_normalizer.build_ticket_source_id_map(
            tickets=tickets,
            source_ids=source_ids,
        )
    )

    assert len(tickets_by_source_id) == 66691

    # ---------------------------------------------------------
    # 5. Normalize REAL change history
    # ---------------------------------------------------------

    workflow_events = (
        change_history_normalizer.normalize_events(
            dataframe=change_history_df,
            tickets_by_source_id=tickets_by_source_id,
        )
    )

    assert len(workflow_events) == 257508

    assert all(
        event.timestamp.tzinfo is not None
        for event in workflow_events
    )

    assert all(
        event.ticket_id
        for event in workflow_events
    )

    # ---------------------------------------------------------
    # 6. Build workflow intervals
    # ---------------------------------------------------------

    workflow_intervals = build_workflow_intervals(
        workflow_events,
    )

    assert workflow_intervals

    # ---------------------------------------------------------
    # 7. Calculate real workflow metrics
    # ---------------------------------------------------------

    workflow_metrics = calculate_workflow_metrics(
        workflow_intervals,
    )

    assert workflow_metrics

    workflow_metrics_by_ticket = {
        metrics.ticket_id: metrics
        for metrics in workflow_metrics
    }

    # ---------------------------------------------------------
    # 8. Calculate real handoffs
    # ---------------------------------------------------------

    handoffs = identify_handoffs(
        workflow_intervals,
    )

    # ---------------------------------------------------------
    # 9. Calculate real bottlenecks
    # ---------------------------------------------------------

    bottlenecks = identify_bottlenecks(
        workflow_intervals,
    )

    # ---------------------------------------------------------
    # 10. Calculate real workflow variants
    # ---------------------------------------------------------

    workflow_variants = identify_workflow_variants(
        workflow_intervals,
    )

    # ---------------------------------------------------------
    # 11. Build real Ticket Analytics
    # ---------------------------------------------------------

    ticket_analytics = [
        build_ticket_analytics(
            ticket=ticket,
            workflow_metrics=(
                workflow_metrics_by_ticket.get(
                    ticket.ticket_id,
                )
            ),
            handoffs=[
                handoff
                for handoff in handoffs
                if handoff.ticket_id == ticket.ticket_id
            ],
        )
        for ticket in tickets
        if ticket.ticket_id in workflow_metrics_by_ticket
    ]

    assert ticket_analytics

    # ---------------------------------------------------------
    # 12. Build real Agent Analytics
    # ---------------------------------------------------------

    agents = build_agents(tickets)

    tickets_by_agent = build_agent_ticket_groups(
        tickets,
    )

    handoffs_by_agent: dict[str, list] = {}

    for handoff in handoffs:
        if handoff.from_agent:
            handoffs_by_agent.setdefault(
                handoff.from_agent,
                [],
            ).append(handoff)

        if handoff.to_agent:
            handoffs_by_agent.setdefault(
                handoff.to_agent,
                [],
            ).append(handoff)

    agent_analytics = []

    for agent in agents:
        agent_analytics.append(
            build_agent_analytics(
                agent=agent,
                tickets=tickets_by_agent.get(
                    agent.agent_id,
                    [],
                ),
                workflow_metrics_by_ticket=(
                    workflow_metrics_by_ticket
                ),
                handoffs=handoffs_by_agent.get(
                    agent.agent_id,
                    [],
                ),
            )
        )

    assert agent_analytics

    # ---------------------------------------------------------
    # 13. Build real Team Analytics
    #
    # No artificial team values are introduced.
    # ---------------------------------------------------------

    team_analytics = (
        build_team_analytics_from_real_data(
            agents=agents,
            agent_analytics=agent_analytics,
        )
    )

    # ---------------------------------------------------------
    # 14. Build real Organization Analytics
    # ---------------------------------------------------------

    organization_analytics = (
        build_organization_analytics(
            agents=agents,
            ticket_analytics=ticket_analytics,
            agent_analytics=agent_analytics,
            team_analytics=team_analytics,
            bottlenecks=bottlenecks,
            workflow_variants=workflow_variants,
        )
    )

    # ---------------------------------------------------------
    # 15. Select a real ticket with workflow history
    # ---------------------------------------------------------

    representative_ticket = next(
        ticket
        for ticket in tickets
        if ticket.ticket_id in workflow_metrics_by_ticket
    )

    representative_ticket_analytics = next(
        analytics
        for analytics in ticket_analytics
        if analytics.ticket_id
        == representative_ticket.ticket_id
    )

    # ---------------------------------------------------------
    # 16. Convert existing Intelligence Layer outputs into
    #     Escalation operational signals.
    #
    #     The adapter performs mapping only.
    # ---------------------------------------------------------

    operational_signals = (
        EscalationOperationalIntelligence.build_operational_signals(
            ticket_analytics=representative_ticket_analytics,
            organization_analytics=organization_analytics,
            bottlenecks=bottlenecks,
            workflow_variants=workflow_variants,
        )
    )

    assert operational_signals

    return {
        "issues_df": issues_df,
        "change_history_df": change_history_df,
        "tickets": tickets,
        "workflow_events": workflow_events,
        "workflow_intervals": workflow_intervals,
        "workflow_metrics": workflow_metrics,
        "handoffs": handoffs,
        "bottlenecks": bottlenecks,
        "workflow_variants": workflow_variants,
        "ticket_analytics": ticket_analytics,
        "agent_analytics": agent_analytics,
        "team_analytics": team_analytics,
        "organization_analytics": organization_analytics,
        "representative_ticket": representative_ticket,
        "representative_ticket_analytics": (
            representative_ticket_analytics
        ),
        "operational_signals": operational_signals,
    }


async def run_real_escalation_agent(
    operational_data: dict,
):
    """
    Run the real Escalation Agent using real operational signals.
    """

    representative_ticket = operational_data[
        "representative_ticket"
    ]

    operational_signals = operational_data[
        "operational_signals"
    ]

    state = EscalationAgentState(
        request=(
            "Analyze escalation risk for the real support ticket "
            f"{representative_ticket.ticket_id} using the supplied "
            "operational intelligence. Determine whether the case "
            "requires escalation. Do not invent missing information."
        ),
        ticket_id=representative_ticket.ticket_id,
        operational_signals=operational_signals,
    )

    model_client = build_model_client()

    try:
        agent = EscalationAgent(
            model_client=model_client,
        )

        result = await agent.run(
            state,
        )

        return result

    finally:
        await close_model_client(
            model_client,
        )


def test_real_escalation_agent_end_to_end():
    """
    Real integration test:

        issues.csv
            â†“
        HelpDeskNormalizer
            â†“
        issues_change_history.csv
            â†“
        ChangeHistoryNormalizer
            â†“
        Workflow Intelligence
            â†“
        Intelligence Analytics
            â†“
        EscalationOperationalIntelligence
            â†“
        EscalationAgent
            â†“
        real Ollama / Qwen3:8b
    """

    operational_data = (
        run_real_operational_pipeline()
    )

    result = asyncio.run(
        run_real_escalation_agent(
            operational_data,
        )
    )

    # ---------------------------------------------------------
    # Validate real operational pipeline
    # ---------------------------------------------------------

    assert len(
        operational_data["issues_df"]
    ) == 66691

    assert len(
        operational_data["change_history_df"]
    ) == 257508

    assert len(
        operational_data["tickets"]
    ) == 66691

    assert len(
        operational_data["workflow_events"]
    ) == 257508

    assert operational_data[
        "workflow_intervals"
    ]

    assert operational_data[
        "workflow_metrics"
    ]

    assert operational_data[
        "ticket_analytics"
    ]

    assert operational_data[
        "agent_analytics"
    ]

    assert operational_data[
        "operational_signals"
    ]

    # ---------------------------------------------------------
    # Validate Escalation Agent
    # ---------------------------------------------------------

    assert (
        result.status
        is EscalationAgentStatus.COMPLETED
    )

    assert result.iteration_count == 1

    assert not result.errors

    response = result.execution_metadata.get(
        "agent_response",
    )

    assert isinstance(
        response,
        str,
    )

    assert response.strip()

    print(
        "\nReal Escalation Agent Integration"
    )
    print("=" * 70)
    print(
        "Issues rows              :",
        len(operational_data["issues_df"]),
    )
    print(
        "Change-history rows      :",
        len(operational_data["change_history_df"]),
    )
    print(
        "Canonical tickets        :",
        len(operational_data["tickets"]),
    )
    print(
        "Workflow events          :",
        len(operational_data["workflow_events"]),
    )
    print(
        "Workflow intervals       :",
        len(operational_data["workflow_intervals"]),
    )
    print(
        "Workflow metrics         :",
        len(operational_data["workflow_metrics"]),
    )
    print(
        "Handoffs                 :",
        len(operational_data["handoffs"]),
    )
    print(
        "Bottlenecks              :",
        len(operational_data["bottlenecks"]),
    )
    print(
        "Workflow variants        :",
        len(operational_data["workflow_variants"]),
    )
    print(
        "Ticket analytics         :",
        len(operational_data["ticket_analytics"]),
    )
    print(
        "Agent analytics          :",
        len(operational_data["agent_analytics"]),
    )
    print(
        "Operational signals      :",
        len(operational_data["operational_signals"]),
    )
    print(
        "Representative ticket   :",
        operational_data[
            "representative_ticket"
        ].ticket_id,
    )
    print(
        "Agent status             :",
        result.status.value,
    )
    print(
        "Agent iterations         :",
        result.iteration_count,
    )
    print(
        "Model                    :",
        OLLAMA_MODEL,
    )
    print(
        "\nEscalation Agent response:"
    )
    print("-" * 70)
    print(response)

