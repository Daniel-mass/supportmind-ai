
from __future__ import annotations
#operational_intelleginece.py
from collections.abc import Iterable

from app.intelligence.analytics.agent_analytics import AgentAnalytics
from app.intelligence.analytics.organization import OrganizationAnalytics
from app.intelligence.analytics.team_analytics import TeamAnalytics
from app.intelligence.analytics.ticket_analytics import TicketAnalytics
from app.intelligence.workflow.bottlenecks import BottleneckCandidate
from app.intelligence.workflow.variants import WorkflowVariant

from app.agents.escalation_agent.contracts.state import EscalationSignal


class EscalationOperationalIntelligence:
    """
    Adapter between the existing Intelligence Layer and the Escalation Agent.

    This component does not calculate operational analytics.

    It only converts already-computed Intelligence Layer analytics into
    EscalationSignal contracts consumed by the Escalation Agent.
    """

    @staticmethod
    def from_ticket_analytics(
        analytics: TicketAnalytics,
    ) -> list[EscalationSignal]:
        """
        Convert ticket-level operational analytics into escalation signals.
        """

        if not isinstance(analytics, TicketAnalytics):
            raise TypeError(
                "analytics must be an instance of TicketAnalytics."
            )

        values = {
            "ticket.project": analytics.project,
            "ticket.issue_type": analytics.issue_type,
            "ticket.priority": analytics.priority,
            "ticket.status": analytics.status,
            "ticket.lifecycle_seconds": analytics.lifecycle_seconds,
            "ticket.active_seconds": analytics.active_seconds,
            "ticket.waiting_seconds": analytics.waiting_seconds,
            "ticket.active_ratio": analytics.active_ratio,
            "ticket.waiting_ratio": analytics.waiting_ratio,
            "ticket.transition_count": analytics.transition_count,
            "ticket.unique_status_count": analytics.unique_status_count,
            "ticket.unique_assignee_count": analytics.unique_assignee_count,
            "ticket.handoff_count": analytics.handoff_count,
            "ticket.is_high_priority": analytics.is_high_priority,
            "ticket.is_waiting": analytics.is_waiting,
        }

        return [
            EscalationSignal(
                name=name,
                value=str(value),
                source="ticket_analytics",
                metadata={
                    "ticket_id": analytics.ticket_id,
                },
            )
            for name, value in values.items()
        ]

    @staticmethod
    def from_agent_analytics(
        analytics: AgentAnalytics,
    ) -> list[EscalationSignal]:
        """
        Convert agent-level operational analytics into escalation signals.
        """

        if not isinstance(analytics, AgentAnalytics):
            raise TypeError(
                "analytics must be an instance of AgentAnalytics."
            )

        values = {
            "agent.assigned_ticket_count": analytics.assigned_ticket_count,
            "agent.open_ticket_count": analytics.open_ticket_count,
            "agent.closed_ticket_count": analytics.closed_ticket_count,
            "agent.high_priority_ticket_count": (
                analytics.high_priority_ticket_count
            ),
            "agent.waiting_ticket_count": analytics.waiting_ticket_count,
            "agent.total_lifecycle_seconds": (
                analytics.total_lifecycle_seconds
            ),
            "agent.average_lifecycle_seconds": (
                analytics.average_lifecycle_seconds
            ),
            "agent.total_waiting_seconds": (
                analytics.total_waiting_seconds
            ),
            "agent.average_waiting_seconds": (
                analytics.average_waiting_seconds
            ),
            "agent.total_transition_count": (
                analytics.total_transition_count
            ),
            "agent.incoming_handoff_count": (
                analytics.incoming_handoff_count
            ),
            "agent.outgoing_handoff_count": (
                analytics.outgoing_handoff_count
            ),
            "agent.unique_project_count": (
                analytics.unique_project_count
            ),
            "agent.unique_issue_type_count": (
                analytics.unique_issue_type_count
            ),
        }

        return [
            EscalationSignal(
                name=name,
                value=str(value),
                source="agent_analytics",
                metadata={
                    "agent_id": analytics.agent_id,
                    "agent_name": analytics.name,
                    "team": analytics.team,
                    "active": analytics.active,
                },
            )
            for name, value in values.items()
        ]

    @staticmethod
    def from_team_analytics(
        analytics: TeamAnalytics,
    ) -> list[EscalationSignal]:
        """
        Convert team-level operational analytics into escalation signals.
        """

        if not isinstance(analytics, TeamAnalytics):
            raise TypeError(
                "analytics must be an instance of TeamAnalytics."
            )

        values = {
            "team.assigned_ticket_count": analytics.assigned_ticket_count,
            "team.open_ticket_count": analytics.open_ticket_count,
            "team.closed_ticket_count": analytics.closed_ticket_count,
            "team.high_priority_ticket_count": (
                analytics.high_priority_ticket_count
            ),
            "team.waiting_ticket_count": analytics.waiting_ticket_count,
            "team.total_lifecycle_seconds": (
                analytics.total_lifecycle_seconds
            ),
            "team.average_lifecycle_seconds": (
                analytics.average_lifecycle_seconds
            ),
            "team.total_waiting_seconds": (
                analytics.total_waiting_seconds
            ),
            "team.average_waiting_seconds": (
                analytics.average_waiting_seconds
            ),
            "team.total_transition_count": (
                analytics.total_transition_count
            ),
            "team.incoming_handoff_count": (
                analytics.incoming_handoff_count
            ),
            "team.outgoing_handoff_count": (
                analytics.outgoing_handoff_count
            ),
        }

        return [
            EscalationSignal(
                name=name,
                value=str(value),
                source="team_analytics",
                metadata={
                    "team": analytics.team,
                },
            )
            for name, value in values.items()
        ]

    @staticmethod
    def from_organization_analytics(
        analytics: OrganizationAnalytics,
    ) -> list[EscalationSignal]:
        """
        Convert organization-level operational analytics into escalation
        signals.

        Only scalar organization metrics are exposed here. Distribution
        objects and nested workflow objects are handled separately.
        """

        if not isinstance(analytics, OrganizationAnalytics):
            raise TypeError(
                "analytics must be an instance of OrganizationAnalytics."
            )

        values = {
            "organization.agent_count": analytics.agent_count,
            "organization.active_agent_count": (
                analytics.active_agent_count
            ),
            "organization.team_count": analytics.team_count,
            "organization.total_ticket_count": (
                analytics.total_ticket_count
            ),
            "organization.open_ticket_count": (
                analytics.open_ticket_count
            ),
            "organization.closed_ticket_count": (
                analytics.closed_ticket_count
            ),
            "organization.high_priority_ticket_count": (
                analytics.high_priority_ticket_count
            ),
            "organization.waiting_ticket_count": (
                analytics.waiting_ticket_count
            ),
            "organization.total_lifecycle_seconds": (
                analytics.total_lifecycle_seconds
            ),
            "organization.average_lifecycle_seconds": (
                analytics.average_lifecycle_seconds
            ),
            "organization.total_waiting_seconds": (
                analytics.total_waiting_seconds
            ),
            "organization.average_waiting_seconds": (
                analytics.average_waiting_seconds
            ),
            "organization.total_transition_count": (
                analytics.total_transition_count
            ),
            "organization.total_handoff_count": (
                analytics.total_handoff_count
            ),
        }

        return [
            EscalationSignal(
                name=name,
                value=str(value),
                source="organization_analytics",
                metadata={},
            )
            for name, value in values.items()
        ]

    @staticmethod
    def from_bottlenecks(
        bottlenecks: Iterable[BottleneckCandidate],
    ) -> list[EscalationSignal]:
        """
        Convert existing bottleneck candidates into operational signals.

        Bottleneck scores are consumed exactly as produced by the
        Intelligence Layer. No new thresholds or calculations are applied.
        """

        signals: list[EscalationSignal] = []

        for index, bottleneck in enumerate(bottlenecks):
            if not isinstance(bottleneck, BottleneckCandidate):
                raise TypeError(
                    "Every bottleneck must be an instance of "
                    "BottleneckCandidate."
                )

            metadata = {
                "status": bottleneck.status,
                "index": index,
            }

            values = {
                "workflow.bottleneck.interval_count": (
                    bottleneck.interval_count
                ),
                "workflow.bottleneck.ticket_count": (
                    bottleneck.ticket_count
                ),
                "workflow.bottleneck.total_seconds": (
                    bottleneck.total_seconds
                ),
                "workflow.bottleneck.average_seconds": (
                    bottleneck.average_seconds
                ),
                "workflow.bottleneck.volume_share": (
                    bottleneck.volume_share
                ),
                "workflow.bottleneck.duration_share": (
                    bottleneck.duration_share
                ),
                "workflow.bottleneck.recurrence_rate": (
                    bottleneck.recurrence_rate
                ),
                "workflow.bottleneck.score": (
                    bottleneck.bottleneck_score
                ),
            }

            for name, value in values.items():
                signals.append(
                    EscalationSignal(
                        name=name,
                        value=str(value),
                        source="bottleneck_analysis",
                        metadata=metadata,
                    )
                )

        return signals

    @staticmethod
    def from_workflow_variants(
        variants: Iterable[WorkflowVariant],
    ) -> list[EscalationSignal]:
        """
        Convert existing workflow variants into contextual operational
        signals.

        Workflow variants are descriptive context. This adapter does not
        classify any variant as good, bad, risky, or escalatory.
        """

        signals: list[EscalationSignal] = []

        for index, variant in enumerate(variants):
            if not isinstance(variant, WorkflowVariant):
                raise TypeError(
                    "Every workflow variant must be an instance of "
                    "WorkflowVariant."
                )

            metadata = {
                "variant_id": variant.variant_id,
                "index": index,
            }

            values = {
                "workflow.variant.path": variant.path_string,
                "workflow.variant.ticket_count": variant.ticket_count,
                "workflow.variant.percentage": variant.percentage,
            }

            for name, value in values.items():
                signals.append(
                    EscalationSignal(
                        name=name,
                        value=str(value),
                        source="workflow_variant_analysis",
                        metadata=metadata,
                    )
                )

        return signals

    @staticmethod
    def build_operational_signals(
        *,
        ticket_analytics: TicketAnalytics | None = None,
        agent_analytics: AgentAnalytics | None = None,
        team_analytics: TeamAnalytics | None = None,
        organization_analytics: OrganizationAnalytics | None = None,
        bottlenecks: Iterable[BottleneckCandidate] | None = None,
        workflow_variants: Iterable[WorkflowVariant] | None = None,
    ) -> list[EscalationSignal]:
        """
        Build the complete operational-signal collection for the
        Escalation Agent.

        Every input is already-computed Intelligence Layer output.
        This method only adapts those objects into EscalationSignal
        contracts.
        """

        signals: list[EscalationSignal] = []

        if ticket_analytics is not None:
            signals.extend(
                EscalationOperationalIntelligence.from_ticket_analytics(
                    ticket_analytics
                )
            )

        if agent_analytics is not None:
            signals.extend(
                EscalationOperationalIntelligence.from_agent_analytics(
                    agent_analytics
                )
            )

        if team_analytics is not None:
            signals.extend(
                EscalationOperationalIntelligence.from_team_analytics(
                    team_analytics
                )
            )

        if organization_analytics is not None:
            signals.extend(
                EscalationOperationalIntelligence.from_organization_analytics(
                    organization_analytics
                )
            )

        if bottlenecks is not None:
            signals.extend(
                EscalationOperationalIntelligence.from_bottlenecks(
                    bottlenecks
                )
            )

        if workflow_variants is not None:
            signals.extend(
                EscalationOperationalIntelligence.from_workflow_variants(
                    workflow_variants
                )
            )

        return signals
