from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.intelligence.analytics.agent_analytics import (
    AgentAnalytics,
    build_agent_analytics,
)
from app.intelligence.analytics.emerging_issues import (
    identify_emerging_issues,
)
from app.intelligence.analytics.organization import (
    OrganizationAnalytics,
    build_organization_analytics,
)
from app.intelligence.analytics.team_analytics import (
    TeamAnalytics,
    build_team_analytics,
)
from app.intelligence.analytics.ticket_analytics import (
    TicketAnalytics,
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
from app.intelligence.schemas.agent import Agent
from app.intelligence.schemas.ticket import Ticket
from app.intelligence.schemas.workflow import WorkflowEvent
from app.intelligence.services.organization_intelligence import (
    OrganizationIntelligence,
    build_organization_intelligence,
)
from app.intelligence.services.ticket_intelligence import (
    TicketIntelligence,
    build_ticket_intelligence,
)
from app.intelligence.services.workflow_intelligence import (
    WorkflowIntelligence,
    build_workflow_intelligence,
)
from app.intelligence.analytics.organization_trends import (
    build_organization_trends,
)
from app.intelligence.schemas.trends import OrganizationTrends
from app.intelligence.workflow.bottlenecks import (
    BottleneckCandidate,
    identify_bottlenecks,
)
from app.intelligence.workflow.event_builder import (
    WorkflowInterval,
    build_workflow_intervals,
)
from app.intelligence.workflow.handoffs import (
    HandoffRecord,
    identify_handoffs,
)
from app.intelligence.workflow.metrics import (
    WorkflowMetrics,
    calculate_workflow_metrics,
)
from app.intelligence.workflow.variants import (
    WorkflowVariant,
    identify_workflow_variants,
)


class HelpDeskProviderError(RuntimeError):
    """Raised when Help Desk intelligence cannot be constructed."""


@dataclass(frozen=True)
class HelpDeskIntelligenceSnapshot:
    tickets_by_id: dict[str, Ticket]

    ticket_analytics_by_id: dict[
        str,
        TicketAnalytics,
    ]

    workflow_metrics_by_id: dict[
        str,
        WorkflowMetrics,
    ]

    workflow_events_by_ticket: dict[
        str,
        list[WorkflowEvent],
    ]

    workflow_intervals_by_ticket: dict[
        str,
        list[WorkflowInterval],
    ]

    handoffs_by_ticket: dict[
        str,
        list[HandoffRecord],
    ]

    bottlenecks_by_ticket: dict[
        str,
        list[BottleneckCandidate],
    ]

    variants_by_ticket: dict[
        str,
        list[WorkflowVariant],
    ]

    agents_by_id: dict[
        str,
        Agent,
    ]

    agent_analytics_by_id: dict[
        str,
        AgentAnalytics,
    ]

    team_analytics_by_team: dict[
        str,
        TeamAnalytics,
    ]

    organization_analytics: OrganizationAnalytics


class HelpDeskIntelligenceProvider:
    def __init__(
        self,
        files: HelpDeskFileSet,
    ) -> None:
        self.files = files

        self.loader = HelpDeskLoader()

        self.ticket_normalizer = HelpDeskNormalizer()

        self.change_history_normalizer = (
            ChangeHistoryNormalizer()
        )

        self._snapshot: (
            HelpDeskIntelligenceSnapshot | None
        ) = None

    def build_snapshot(
        self,
    ) -> HelpDeskIntelligenceSnapshot:
        if self.files.change_history_csv is None:
            raise HelpDeskProviderError(
                "change_history_csv is required "
                "for ticket intelligence."
            )

        sources = self.loader.load_all(self.files)

        issues_dataframe = sources.get("issues")

        change_history_dataframe = (
            sources.get("change_history")
        )

        if issues_dataframe is None:
            raise HelpDeskProviderError(
                "The configured issues.csv source "
                "was not loaded."
            )

        if change_history_dataframe is None:
            raise HelpDeskProviderError(
                "The configured change-history source "
                "was not loaded."
            )

        # =========================================================
        # 1. Normalize tickets
        # =========================================================

        tickets = (
            self.ticket_normalizer.normalize_tickets(
                issues_dataframe
            )
        )

        if not tickets:
            raise HelpDeskProviderError(
                "No tickets were found after normalization."
            )

        # =========================================================
        # 2. Build source-ID → canonical Ticket mapping
        # =========================================================

        source_ids = (
            issues_dataframe["id"].tolist()
            if "id" in issues_dataframe.columns
            else []
        )

        if len(source_ids) != len(tickets):
            raise HelpDeskProviderError(
                "The number of normalized tickets does "
                "not match the number of source issue IDs."
            )

        tickets_by_source_id = (
            self.change_history_normalizer
            .build_ticket_source_id_map(
                tickets=tickets,
                source_ids=source_ids,
            )
        )

        # =========================================================
        # 3. Normalize workflow events
        # =========================================================

        workflow_events = (
            self.change_history_normalizer
            .normalize_events(
                dataframe=change_history_dataframe,
                tickets_by_source_id=tickets_by_source_id,
            )
        )

        # =========================================================
        # 4. Build workflow intervals
        # =========================================================

        intervals = build_workflow_intervals(
            workflow_events
        )

        # =========================================================
        # 5. Calculate workflow metrics
        # =========================================================

        workflow_metrics = calculate_workflow_metrics(
            intervals
        )

        # =========================================================
        # 6. Identify handoffs
        # =========================================================

        handoffs = identify_handoffs(
            intervals
        )

        # =========================================================
        # 7. Canonical ticket mappings
        # =========================================================

        tickets_by_id = {
            ticket.ticket_id: ticket
            for ticket in tickets
        }

        workflow_metrics_by_id = {
            metric.ticket_id: metric
            for metric in workflow_metrics
        }

        # =========================================================
        # 8. Group workflow events by ticket
        # =========================================================

        events_by_ticket: dict[
            str,
            list[WorkflowEvent],
        ] = defaultdict(list)

        for event in workflow_events:
            events_by_ticket[
                event.ticket_id
            ].append(event)

        # =========================================================
        # 9. Group workflow intervals by ticket
        # =========================================================

        intervals_by_ticket: dict[
            str,
            list[WorkflowInterval],
        ] = defaultdict(list)

        for interval in intervals:
            intervals_by_ticket[
                interval.ticket_id
            ].append(interval)

        # =========================================================
        # 10. Group handoffs by ticket
        # =========================================================

        handoffs_by_ticket: dict[
            str,
            list[HandoffRecord],
        ] = defaultdict(list)

        for handoff in handoffs:
            handoffs_by_ticket[
                handoff.ticket_id
            ].append(handoff)

        # =========================================================
        # 11. Ticket-level bottlenecks and variants
        # =========================================================

        bottlenecks_by_ticket: dict[
            str,
            list[BottleneckCandidate],
        ] = defaultdict(list)

        variants_by_ticket: dict[
            str,
            list[WorkflowVariant],
        ] = defaultdict(list)

        for (
            ticket_id,
            ticket_intervals,
        ) in intervals_by_ticket.items():

            bottlenecks_by_ticket[
                ticket_id
            ] = identify_bottlenecks(
                ticket_intervals
            )

            variants_by_ticket[
                ticket_id
            ] = identify_workflow_variants(
                ticket_intervals
            )

        # =========================================================
        # 12. Build ticket analytics
        # =========================================================

        ticket_analytics_by_id: dict[
            str,
            TicketAnalytics,
        ] = {}

        for ticket in tickets:
            metric = workflow_metrics_by_id.get(
                ticket.ticket_id
            )

            if metric is None:
                continue

            ticket_handoffs = (
                handoffs_by_ticket.get(
                    ticket.ticket_id,
                    [],
                )
            )

            analytics = build_ticket_analytics(
                ticket=ticket,
                workflow_metrics=metric,
                handoffs=ticket_handoffs,
            )

            ticket_analytics_by_id[
                ticket.ticket_id
            ] = analytics

        # =========================================================
        # 13. Build canonical Agent objects
        # =========================================================

        agent_ids: set[str] = set()

        for ticket in tickets:
            if ticket.assignee is None:
                continue

            agent_id = str(
                ticket.assignee
            ).strip()

            if agent_id:
                agent_ids.add(agent_id)

        for handoff in handoffs:
            if handoff.from_agent is not None:
                from_agent = str(
                    handoff.from_agent
                ).strip()

                if from_agent:
                    agent_ids.add(from_agent)

            if handoff.to_agent is not None:
                to_agent = str(
                    handoff.to_agent
                ).strip()

                if to_agent:
                    agent_ids.add(to_agent)

        agents_by_id: dict[
            str,
            Agent,
        ] = {}

        for agent_id in sorted(agent_ids):
            agents_by_id[agent_id] = Agent(
                agent_id=agent_id,
                source="helpdesk",
                metadata={
                    "derived_from": "ticket_assignee",
                },
            )

        # =========================================================
        # 14. Build Agent analytics
        # =========================================================

        all_handoffs = list(handoffs)

        agent_analytics_by_id: dict[
            str,
            AgentAnalytics,
        ] = {}

        for agent_id, agent in agents_by_id.items():
            analytics = build_agent_analytics(
                agent=agent,
                tickets=tickets,
                workflow_metrics_by_ticket=(
                    workflow_metrics_by_id
                ),
                handoffs=all_handoffs,
            )

            agent_analytics_by_id[
                agent_id
            ] = analytics

        agent_analytics_list = list(
            agent_analytics_by_id.values()
        )

        # =========================================================
        # 15. Build Team analytics
        # =========================================================
        #
        # Current Agent objects have team=None because the
        # current Help Desk source contract does not provide a
        # reliable agent-team mapping.
        #
        # Therefore there are currently no team analytics to
        # construct. We do not invent team names.
        # =========================================================

        team_analytics_by_team: dict[
            str,
            TeamAnalytics,
        ] = {}

        team_names = sorted(
            {
                agent.team.strip()
                for agent in agents_by_id.values()
                if agent.team is not None
                and agent.team.strip()
            }
        )

        for team in team_names:
            team_analytics_by_team[
                team
            ] = build_team_analytics(
                team=team,
                agents=agents_by_id.values(),
                agent_analytics=agent_analytics_list,
            )

        team_analytics_list = list(
            team_analytics_by_team.values()
        )

        # =========================================================
        # 16. Build organization-wide bottlenecks
        # =========================================================

        organization_bottlenecks = (
            identify_bottlenecks(
                intervals
            )
            if intervals
            else []
        )

        # =========================================================
        # 17. Build organization-wide workflow variants
        # =========================================================

        organization_workflow_variants = (
            identify_workflow_variants(
                intervals
            )
            if intervals
            else []
        )

        # =========================================================
        # 18. Identify emerging issues
        # =========================================================
        #
        # Emerging issues are calculated from canonical Ticket
        # objects using their normalized created_at timestamps
        # and issue_type values.
        #
        # The dedicated analytics module compares the latest
        # 30-day window against the preceding 30-day window.
        #
        # This intentionally differs from issue-type frequency:
        # an issue must demonstrate recent growth to be considered
        # emerging.
        # =========================================================

        emerging_issues = identify_emerging_issues(
            tickets=tickets,
        )

        # =========================================================
        # 19. Build organization analytics
        # =========================================================

        organization_analytics = (
            build_organization_analytics(
                agents=agents_by_id.values(),
                ticket_analytics=(
                    ticket_analytics_by_id.values()
                ),
                agent_analytics=agent_analytics_list,
                team_analytics=team_analytics_list,
                bottlenecks=organization_bottlenecks,
                workflow_variants=(
                    organization_workflow_variants
                ),
                emerging_issues=emerging_issues,
            )
        )

        # =========================================================
        # 20. Build immutable intelligence snapshot
        # =========================================================

        snapshot = HelpDeskIntelligenceSnapshot(
            tickets_by_id=tickets_by_id,
            ticket_analytics_by_id=(
                ticket_analytics_by_id
            ),
            workflow_metrics_by_id=(
                workflow_metrics_by_id
            ),
            workflow_events_by_ticket=dict(
                events_by_ticket
            ),
            workflow_intervals_by_ticket=dict(
                intervals_by_ticket
            ),
            handoffs_by_ticket=dict(
                handoffs_by_ticket
            ),
            bottlenecks_by_ticket=dict(
                bottlenecks_by_ticket
            ),
            variants_by_ticket=dict(
                variants_by_ticket
            ),
            agents_by_id=agents_by_id,
            agent_analytics_by_id=(
                agent_analytics_by_id
            ),
            team_analytics_by_team=(
                team_analytics_by_team
            ),
            organization_analytics=(
                organization_analytics
            ),
        )

        self._snapshot = snapshot

        return snapshot

    def get_snapshot(
        self,
    ) -> HelpDeskIntelligenceSnapshot:
        if self._snapshot is None:
            return self.build_snapshot()

        return self._snapshot

    # =============================================================
    # Ticket Intelligence
    # =============================================================

    def get_ticket_intelligence(
        self,
        ticket_id: str,
    ) -> TicketIntelligence | None:
        snapshot = self.get_snapshot()

        ticket = snapshot.tickets_by_id.get(
            ticket_id
        )

        if ticket is None:
            return None

        analytics = (
            snapshot.ticket_analytics_by_id.get(
                ticket_id
            )
        )

        if analytics is None:
            raise HelpDeskProviderError(
                "Ticket exists but ticket analytics "
                "could not be constructed for "
                f"'{ticket_id}'."
            )

        workflow_metrics = (
            snapshot.workflow_metrics_by_id.get(
                ticket_id
            )
        )

        workflow_events = (
            snapshot.workflow_events_by_ticket.get(
                ticket_id,
                [],
            )
        )

        handoffs = (
            snapshot.handoffs_by_ticket.get(
                ticket_id,
                [],
            )
        )

        return build_ticket_intelligence(
            ticket=ticket,
            analytics=analytics,
            workflow_metrics=workflow_metrics,
            workflow_events=workflow_events,
            handoffs=handoffs,
        )

    # =============================================================
    # Workflow Intelligence
    # =============================================================

    def get_workflow_intelligence(
        self,
        ticket_id: str,
    ) -> WorkflowIntelligence | None:
        snapshot = self.get_snapshot()

        if ticket_id not in snapshot.tickets_by_id:
            return None

        return build_workflow_intelligence(
            ticket_id=ticket_id,
            events=(
                snapshot.workflow_events_by_ticket.get(
                    ticket_id,
                    [],
                )
            ),
            metrics=(
                snapshot.workflow_metrics_by_id.get(
                    ticket_id
                )
            ),
            handoffs=(
                snapshot.handoffs_by_ticket.get(
                    ticket_id,
                    [],
                )
            ),
            bottlenecks=(
                snapshot.bottlenecks_by_ticket.get(
                    ticket_id,
                    [],
                )
            ),
            variants=(
                snapshot.variants_by_ticket.get(
                    ticket_id,
                    [],
                )
            ),
        )

    # =============================================================
    # Agent Intelligence
    # =============================================================

    def get_agent_intelligence(
        self,
        agent_id: str,
    ) -> AgentAnalytics | None:
        snapshot = self.get_snapshot()

        normalized_agent_id = agent_id.strip()

        if not normalized_agent_id:
            return None

        return snapshot.agent_analytics_by_id.get(
            normalized_agent_id
        )

    # =============================================================
    # Organization Intelligence
    # =============================================================

    def get_organization_intelligence(
        self,
    ) -> OrganizationIntelligence | None:
        snapshot = self.get_snapshot()

        return build_organization_intelligence(
            organization=(
                snapshot.organization_analytics
            ),
            agent_analytics=list(
                snapshot.agent_analytics_by_id.values()
            ),
            team_analytics=list(
                snapshot.team_analytics_by_team.values()
            ),
            bottlenecks=(
                snapshot.organization_analytics.bottlenecks
            ),
            workflow_variants=(
                snapshot.organization_analytics.workflow_variants
            ),
            emerging_issues=(
                snapshot.organization_analytics.emerging_issues
            ),
        )

    def get_organization_trends(
        self,
    ) -> OrganizationTrends:
        snapshot = self.get_snapshot()

        return build_organization_trends(snapshot)


def _required_environment_variable(
    name: str,
) -> Path:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise HelpDeskProviderError(
            f"Required environment variable "
            f"'{name}' is not configured."
        )

    return Path(value.strip())


@lru_cache(maxsize=1)
def get_helpdesk_provider(
) -> HelpDeskIntelligenceProvider:
    files = HelpDeskFileSet(
        issues_csv=_required_environment_variable(
            "SUPPORTMIND_ISSUES_CSV"
        ),
        change_history_csv=(
            _required_environment_variable(
                "SUPPORTMIND_CHANGE_HISTORY_CSV"
            )
        ),
    )

    return HelpDeskIntelligenceProvider(
        files=files
    )