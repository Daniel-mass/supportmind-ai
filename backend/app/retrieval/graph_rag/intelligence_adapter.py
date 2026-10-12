"""
intelligence_adapter.py

Adapter between the SupportMind Intelligence Layer operational graph
and the Graph-RAG generation layer.

This module does not build or modify the operational graph.

It consumes the existing:
    - GraphQuery registry
    - Neo4jOperationalGraph
    - Intelligence Layer graph schema

and converts operational graph results into GraphRAGEvidence objects.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from app.intelligence.graph.graph_queries import (
    GET_AGENT_HANDOFFS,
    GET_AGENT_TICKETS,
    GET_PRIORITY_DISTRIBUTION,
    GET_STATUS_DISTRIBUTION,
    GET_STATUS_TRANSITIONS,
    GET_TEAM_MEMBERS,
    GET_TEAM_TICKET_WORKLOAD,
    GET_TICKET,
    GET_TICKET_NEIGHBORHOOD,
    GET_TICKET_WORKFLOW,
    GET_TICKET_WORKFLOW_TRANSITIONS,
    GET_WORKFLOW_EVENT_VOLUME,
    GraphQuery,
)

from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
)


# ============================================================================
# Operational graph protocol
# ============================================================================


class OperationalGraphProtocol(Protocol):
    """
    Minimal interface required from the existing operational graph.

    Keeping this as a Protocol makes the adapter easy to unit-test without
    starting a real Neo4j instance.
    """

    def execute_query(
        self,
        query: GraphQuery,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a registered, parameterized graph query."""
        ...


# ============================================================================
# Intelligence Adapter
# ============================================================================


class IntelligenceAdapter:
    """
    Translate operational Help Desk graph data into Graph-RAG evidence.

    The adapter is intentionally thin. Cypher remains owned by the
    Intelligence Layer query registry, while Graph-RAG owns only the
    evidence representation.
    """

    def __init__(
        self,
        operational_graph: OperationalGraphProtocol,
    ) -> None:
        if operational_graph is None:
            raise ValueError(
                "operational_graph cannot be None."
            )

        if not hasattr(
            operational_graph,
            "execute_query",
        ):
            raise TypeError(
                "operational_graph must expose execute_query()."
            )

        self.operational_graph = operational_graph

    # ========================================================================
    # Ticket evidence
    # ========================================================================

    def get_ticket_evidence(
        self,
        ticket_id: str,
        *,
        include_neighborhood: bool = True,
        include_workflow: bool = True,
        include_transitions: bool = True,
    ) -> list[GraphRAGEvidence]:
        """
        Collect Help Desk graph evidence for one ticket.
        """

        ticket_id = self._validate_identifier(
            ticket_id,
            field_name="ticket_id",
        )

        evidence: list[GraphRAGEvidence] = []

        evidence.extend(
            self._query_ticket(ticket_id)
        )

        if include_neighborhood:
            evidence.extend(
                self._query_ticket_neighborhood(
                    ticket_id
                )
            )

        if include_workflow:
            evidence.extend(
                self._query_ticket_workflow(
                    ticket_id
                )
            )

        if include_transitions:
            evidence.extend(
                self._query_ticket_transitions(
                    ticket_id
                )
            )

        return self._assign_evidence_ids(
            evidence
        )

    # ========================================================================
    # Agent evidence
    # ========================================================================

    def get_agent_evidence(
        self,
        agent_id: str,
        *,
        include_handoffs: bool = True,
        include_tickets: bool = True,
    ) -> list[GraphRAGEvidence]:
        """
        Collect operational evidence for one support agent.
        """

        agent_id = self._validate_identifier(
            agent_id,
            field_name="agent_id",
        )

        evidence: list[GraphRAGEvidence] = []

        if include_handoffs:
            rows = self.operational_graph.execute_query(
                GET_AGENT_HANDOFFS,
                {"agent_id": agent_id},
            )

            evidence.extend(
                self._convert_rows(
                    rows=rows,
                    source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
                    evidence_type=GraphRAGEvidenceType.HANDOFF,
                    source_prefix=(
                        f"agent:{agent_id}:handoff"
                    ),
                    formatter=self._format_agent_handoff,
                    query_name=GET_AGENT_HANDOFFS.name,
                )
            )

        if include_tickets:
            rows = self.operational_graph.execute_query(
                GET_AGENT_TICKETS,
                {"agent_id": agent_id},
            )

            evidence.extend(
                self._convert_rows(
                    rows=rows,
                    source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
                    evidence_type=GraphRAGEvidenceType.TICKET_CONTEXT,
                    source_prefix=(
                        f"agent:{agent_id}:ticket"
                    ),
                    formatter=self._format_agent_ticket,
                    query_name=GET_AGENT_TICKETS.name,
                )
            )

        return self._assign_evidence_ids(
            evidence
        )

    # ========================================================================
    # Team evidence
    # ========================================================================

    def get_team_evidence(
        self,
        team: str,
        *,
        include_members: bool = True,
        include_workload: bool = True,
    ) -> list[GraphRAGEvidence]:
        """
        Collect operational graph evidence for one team.
        """

        team = self._validate_identifier(
            team,
            field_name="team",
        )

        evidence: list[GraphRAGEvidence] = []

        if include_members:
            rows = self.operational_graph.execute_query(
                GET_TEAM_MEMBERS,
                {"team": team},
            )

            evidence.extend(
                self._convert_rows(
                    rows=rows,
                    source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
                    evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
                    source_prefix=f"team:{team}:member",
                    formatter=self._format_team_member,
                    query_name=GET_TEAM_MEMBERS.name,
                )
            )

        if include_workload:
            rows = self.operational_graph.execute_query(
                GET_TEAM_TICKET_WORKLOAD,
                {"team": team},
            )

            evidence.extend(
                self._convert_rows(
                    rows=rows,
                    source=GraphRAGEvidenceSource.ORGANIZATION_INTELLIGENCE,
                    evidence_type=GraphRAGEvidenceType.METRIC,
                    source_prefix=f"team:{team}:workload",
                    formatter=self._format_team_workload,
                    query_name=GET_TEAM_TICKET_WORKLOAD.name,
                )
            )

        return self._assign_evidence_ids(
            evidence
        )

    # ========================================================================
    # Organization evidence
    # ========================================================================

    def get_organization_evidence(
        self,
    ) -> list[GraphRAGEvidence]:
        """
        Collect organization-level operational evidence.

        These queries intentionally use the existing Intelligence Layer
        query registry rather than introducing new Cypher here.
        """

        evidence: list[GraphRAGEvidence] = []

        organization_queries = (
            (
                GET_STATUS_TRANSITIONS,
                GraphRAGEvidenceType.TREND,
                "organization:status-transition",
                self._format_status_transition,
            ),
            (
                GET_WORKFLOW_EVENT_VOLUME,
                GraphRAGEvidenceType.OPERATIONAL_SIGNAL,
                "organization:event-volume",
                self._format_workflow_event_volume,
            ),
            (
                GET_PRIORITY_DISTRIBUTION,
                GraphRAGEvidenceType.METRIC,
                "organization:priority",
                self._format_priority_distribution,
            ),
            (
                GET_STATUS_DISTRIBUTION,
                GraphRAGEvidenceType.METRIC,
                "organization:status",
                self._format_status_distribution,
            ),
        )

        for (
            query,
            evidence_type,
            source_prefix,
            formatter,
        ) in organization_queries:
            rows = self.operational_graph.execute_query(
                query
            )

            evidence.extend(
                self._convert_rows(
                    rows=rows,
                    source=(
                        GraphRAGEvidenceSource
                        .ORGANIZATION_INTELLIGENCE
                    ),
                    evidence_type=evidence_type,
                    source_prefix=source_prefix,
                    formatter=formatter,
                    query_name=query.name,
                )
            )

        return self._assign_evidence_ids(
            evidence
        )

    # ========================================================================
    # Query-aware organization evidence
    # ========================================================================

    def get_relevant_organization_evidence(
        self,
        query: str,
    ) -> list[GraphRAGEvidence]:
        """
        Collect only organization-level evidence relevant to a query.

        This method is specifically intended for Graph-RAG.

        The existing get_organization_evidence() method intentionally
        returns the complete organization evidence and remains unchanged.

        Selection is deterministic and bounded so that organization
        intelligence
        does not overwhelm the retrieval context.
        """

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string."
            )

        normalized_query = query.strip().lower()

        if not normalized_query:
            raise ValueError(
                "query must not be empty."
            )

        selected_queries = (
            self._select_organization_queries(
                normalized_query
            )
        )

        if not selected_queries:
            return []

        # Maximum number of organization rows allowed per category.
        #
        # This prevents one organization-level query from flooding
        # the final Graph-RAG evidence context.
        limits = {
            GET_PRIORITY_DISTRIBUTION.name: 4,
            GET_STATUS_DISTRIBUTION.name: 4,
            GET_WORKFLOW_EVENT_VOLUME.name: 3,
            GET_STATUS_TRANSITIONS.name: 2,
        }

        evidence: list[GraphRAGEvidence] = []

        for query_definition in selected_queries:
            rows = self.operational_graph.execute_query(
                query_definition
            )

            limit = limits.get(
                query_definition.name,
                0,
            )

            bounded_rows = rows[:limit]

            evidence.extend(
                self._convert_organization_rows(
                    query_definition,
                    bounded_rows,
                )
            )

        return self._assign_evidence_ids(
            evidence
        )

    def _select_organization_queries(
        self,
        normalized_query: str,
    ) -> list[GraphQuery]:
        """
        Select canonical organization graph queries relevant to a query.

        The ordering is deterministic so the resulting evidence remains
        predictable and easy to test.
        """

        selected_queries: list[GraphQuery] = []

        if self._query_mentions_priority(
            normalized_query
        ):
            selected_queries.append(
                GET_PRIORITY_DISTRIBUTION
            )

        if self._query_mentions_status(
            normalized_query
        ):
            selected_queries.append(
                GET_STATUS_DISTRIBUTION
            )

        if self._query_mentions_workflow(
            normalized_query
        ):
            selected_queries.append(
                GET_WORKFLOW_EVENT_VOLUME
            )

        if self._query_mentions_transition_or_delay(
            normalized_query
        ):
            selected_queries.append(
                GET_STATUS_TRANSITIONS
            )

        return selected_queries

    @staticmethod
    def _query_mentions_priority(
        query: str,
    ) -> bool:
        """
        Return whether the query asks about ticket priorities.
        """

        keywords = (
            "priority",
            "priorities",
        )

        return any(
            keyword in query
            for keyword in keywords
        )

    @staticmethod
    def _query_mentions_status(
        query: str,
    ) -> bool:
        """
        Return whether the query asks about ticket status distribution.

        A query specifically about status transitions should not also
        trigger status-distribution evidence unless it explicitly asks
        about statuses separately.
        """

        status_distribution_phrases = (
            "status distribution",
            "status distributions",
            "ticket status",
            "ticket statuses",
            "ticket state",
            "ticket states",
            "how many tickets are open",
            "how many tickets are closed",
            "how many tickets are resolved",
            "status breakdown",
            "status breakdowns",
        )

        if any(
            phrase in query
            for phrase in status_distribution_phrases
        ):
            return True

        standalone_status_words = (
            "statuses",
        )

        return any(
            word in query
            for word in standalone_status_words
        )

    @staticmethod
    def _query_mentions_workflow(
        query: str,
    ) -> bool:
        """
        Return whether the query asks about workflow activity.
        """

        keywords = (
            "workflow",
            "activity",
            "activities",
            "event",
            "events",
        )

        return any(
            keyword in query
            for keyword in keywords
        )

    @staticmethod
    def _query_mentions_transition_or_delay(
        query: str,
    ) -> bool:
        """
        Return whether the query asks about transitions or delays.
        """

        keywords = (
            "delay",
            "delays",
            "transition",
            "transitions",
            "bottleneck",
            "bottlenecks",
        )

        return any(
            keyword in query
            for keyword in keywords
        )

    def _convert_organization_rows(
        self,
        query: GraphQuery,
        rows: Iterable[dict[str, Any]],
    ) -> list[GraphRAGEvidence]:
        """
        Convert rows from one organization graph query into evidence.

        The existing organization-specific formatters are reused so that
        evidence produced by this query-aware method has exactly the same
        representation as evidence produced by get_organization_evidence().
        """

        formatter_map = {
            GET_STATUS_TRANSITIONS.name: (
                GraphRAGEvidenceType.TREND,
                "organization:status-transition",
                self._format_status_transition,
            ),
            GET_WORKFLOW_EVENT_VOLUME.name: (
                GraphRAGEvidenceType.OPERATIONAL_SIGNAL,
                "organization:event-volume",
                self._format_workflow_event_volume,
            ),
            GET_PRIORITY_DISTRIBUTION.name: (
                GraphRAGEvidenceType.METRIC,
                "organization:priority",
                self._format_priority_distribution,
            ),
            GET_STATUS_DISTRIBUTION.name: (
                GraphRAGEvidenceType.METRIC,
                "organization:status",
                self._format_status_distribution,
            ),
        }

        try:
            evidence_type, source_prefix, formatter = (
                formatter_map[query.name]
            )
        except KeyError as exc:
            raise ValueError(
                "Unsupported organization graph query: "
                f"{query.name}"
            ) from exc

        return self._convert_rows(
            rows=rows,
            source=(
                GraphRAGEvidenceSource
                .ORGANIZATION_INTELLIGENCE
            ),
            evidence_type=evidence_type,
            source_prefix=source_prefix,
            formatter=formatter,
            query_name=query.name,
        )

    # ========================================================================
    # Ticket query helpers
    # ========================================================================

    def _query_ticket(
        self,
        ticket_id: str,
    ) -> list[GraphRAGEvidence]:
        rows = self.operational_graph.execute_query(
            GET_TICKET,
            {"ticket_id": ticket_id},
        )

        return self._convert_rows(
            rows=rows,
            source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
            evidence_type=GraphRAGEvidenceType.TICKET_CONTEXT,
            source_prefix=f"ticket:{ticket_id}",
            formatter=self._format_ticket,
            query_name=GET_TICKET.name,
        )

    def _query_ticket_neighborhood(
        self,
        ticket_id: str,
    ) -> list[GraphRAGEvidence]:
        rows = self.operational_graph.execute_query(
            GET_TICKET_NEIGHBORHOOD,
            {"ticket_id": ticket_id},
        )

        return self._convert_rows(
            rows=rows,
            source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
            evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
            source_prefix=f"ticket:{ticket_id}:neighborhood",
            formatter=self._format_ticket_neighborhood,
            query_name=GET_TICKET_NEIGHBORHOOD.name,
        )

    def _query_ticket_workflow(
        self,
        ticket_id: str,
    ) -> list[GraphRAGEvidence]:
        rows = self.operational_graph.execute_query(
            GET_TICKET_WORKFLOW,
            {"ticket_id": ticket_id},
        )

        return self._convert_rows(
            rows=rows,
            source=GraphRAGEvidenceSource.WORKFLOW,
            evidence_type=GraphRAGEvidenceType.WORKFLOW_EVENT,
            source_prefix=f"ticket:{ticket_id}:workflow",
            formatter=self._format_workflow_event,
            query_name=GET_TICKET_WORKFLOW.name,
        )

    def _query_ticket_transitions(
        self,
        ticket_id: str,
    ) -> list[GraphRAGEvidence]:
        rows = self.operational_graph.execute_query(
            GET_TICKET_WORKFLOW_TRANSITIONS,
            {"ticket_id": ticket_id},
        )

        return self._convert_rows(
            rows=rows,
            source=GraphRAGEvidenceSource.WORKFLOW,
            evidence_type=GraphRAGEvidenceType.WORKFLOW_EVENT,
            source_prefix=(
                f"ticket:{ticket_id}:transition"
            ),
            formatter=self._format_workflow_transition,
            query_name=GET_TICKET_WORKFLOW_TRANSITIONS.name,
        )

    # ========================================================================
    # Generic conversion
    # ========================================================================

    def _convert_rows(
        self,
        *,
        rows: Iterable[dict[str, Any]],
        source: GraphRAGEvidenceSource,
        evidence_type: GraphRAGEvidenceType,
        source_prefix: str,
        formatter: Any,
        query_name: str,
    ) -> list[GraphRAGEvidence]:
        evidence: list[GraphRAGEvidence] = []

        for index, row in enumerate(
            rows,
            start=1,
        ):
            if not isinstance(
                row,
                dict,
            ):
                raise TypeError(
                    "Operational graph query results "
                    "must contain dictionaries."
                )

            text = formatter(row)

            if not text:
                continue

            evidence.append(
                GraphRAGEvidence(
                    evidence_id=f"E{index}",
                    source=source,
                    evidence_type=evidence_type,
                    text=text,
                    source_reference=(
                        f"{source_prefix}:{index}"
                    ),
                    metadata={
                        "query_name": query_name,
                        "query_row": index,
                    },
                )
            )

        return evidence

    # ========================================================================
    # Formatting helpers
    # ========================================================================

    @classmethod
    def _format_ticket(
        cls,
        row: dict[str, Any],
    ) -> str:
        node = cls._serialize_value(
            row.get("t")
        )

        if not node:
            return ""

        return cls._format_mapping(
            "Ticket context",
            node,
        )

    @classmethod
    def _format_ticket_neighborhood(
        cls,
        row: dict[str, Any],
    ) -> str:
        ticket = cls._serialize_value(
            row.get("t")
        )

        paths = row.get(
            "paths",
            [],
        )

        ticket_text = cls._format_mapping(
            "Ticket neighborhood",
            ticket,
        )

        if not paths:
            return ticket_text

        path_summaries = [
            cls._format_path(path)
            for path in paths
        ]

        path_summaries = [
            value
            for value in path_summaries
            if value
        ]

        if not path_summaries:
            return ticket_text

        return (
            f"{ticket_text}. "
            "Connected graph evidence: "
            + " | ".join(path_summaries)
        )

    @classmethod
    def _format_workflow_event(
        cls,
        row: dict[str, Any],
    ) -> str:
        event = cls._serialize_value(
            row.get("e")
        )

        if not event:
            return ""

        return cls._format_mapping(
            "Workflow event",
            event,
        )

    @classmethod
    def _format_workflow_transition(
        cls,
        row: dict[str, Any],
    ) -> str:
        first = cls._serialize_value(
            row.get("e1")
        )

        second = cls._serialize_value(
            row.get("e2")
        )

        if not first and not second:
            return ""

        first_status = (
            first.get("status")
            if isinstance(first, dict)
            else None
        )

        second_status = (
            second.get("status")
            if isinstance(second, dict)
            else None
        )

        return (
            "Workflow transition: "
            f"{first_status or 'unknown'} -> "
            f"{second_status or 'unknown'}."
        )

    @classmethod
    def _format_agent_handoff(
        cls,
        row: dict[str, Any],
    ) -> str:
        source_agent = cls._serialize_value(
            row.get("a")
        )
        relationship = cls._serialize_value(
            row.get("r")
        )
        target_agent = cls._serialize_value(
            row.get("b")
        )

        source_id = (
            source_agent.get("agent_id")
            if isinstance(source_agent, dict)
            else "unknown"
        )

        target_id = (
            target_agent.get("agent_id")
            if isinstance(target_agent, dict)
            else "unknown"
        )

        handoff_id = (
            relationship.get("handoff_id")
            if isinstance(relationship, dict)
            else "unknown"
        )

        return (
            "Agent handoff: "
            f"{source_id} -> {target_id} "
            f"(handoff_id={handoff_id})."
        )

    @classmethod
    def _format_agent_ticket(
        cls,
        row: dict[str, Any],
    ) -> str:
        ticket = cls._serialize_value(
            row.get("t")
        )

        if not ticket:
            return ""

        return cls._format_mapping(
            "Agent ticket",
            ticket,
        )

    @classmethod
    def _format_team_member(
        cls,
        row: dict[str, Any],
    ) -> str:
        agent = cls._serialize_value(
            row.get("a")
        )

        if not agent:
            return ""

        return cls._format_mapping(
            "Team member",
            agent,
        )

    @classmethod
    def _format_team_workload(
        cls,
        row: dict[str, Any],
    ) -> str:
        status = row.get(
            "status",
            "unknown",
        )

        count = row.get(
            "ticket_count",
            0,
        )

        return (
            "Team workload: "
            f"{status} = {count} tickets."
        )

    @classmethod
    def _format_status_transition(
        cls,
        row: dict[str, Any],
    ) -> str:
        return (
            "Organization workflow transition: "
            f"{row.get('from_status', 'unknown')} -> "
            f"{row.get('to_status', 'unknown')} "
            f"({row.get('transition_count', 0)} occurrences)."
        )

    @classmethod
    def _format_workflow_event_volume(
        cls,
        row: dict[str, Any],
    ) -> str:
        return (
            "Workflow event volume: "
            f"{row.get('activity', 'unknown')} = "
            f"{row.get('event_count', 0)} events."
        )

    @classmethod
    def _format_priority_distribution(
        cls,
        row: dict[str, Any],
    ) -> str:
        return (
            "Priority distribution: "
            f"{row.get('priority', 'unknown')} = "
            f"{row.get('ticket_count', 0)} tickets."
        )

    @classmethod
    def _format_status_distribution(
        cls,
        row: dict[str, Any],
    ) -> str:
        return (
            "Status distribution: "
            f"{row.get('status', 'unknown')} = "
            f"{row.get('ticket_count', 0)} tickets."
        )

    # ========================================================================
    # Serialization helpers
    # ========================================================================

    @classmethod
    def _serialize_value(
        cls,
        value: Any,
    ) -> Any:
        """
        Convert Neo4j graph values into JSON-friendly structures.

        The adapter deliberately uses duck-typing so unit tests do not
        require a live Neo4j driver.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): cls._serialize_value(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):
            return [
                cls._serialize_value(
                    item
                )
                for item in value
            ]

        items_method = getattr(
            value,
            "items",
            None,
        )

        if callable(items_method):
            try:
                return {
                    str(key): cls._serialize_value(
                        item
                    )
                    for key, item in items_method()
                }
            except Exception:
                pass

        properties = getattr(
            value,
            "_properties",
            None,
        )

        if isinstance(
            properties,
            dict,
        ):
            return {
                str(key): cls._serialize_value(
                    item
                )
                for key, item in properties.items()
            }

        return str(value)

    @classmethod
    def _format_mapping(
        cls,
        title: str,
        mapping: Any,
    ) -> str:
        if not isinstance(
            mapping,
            dict,
        ):
            return (
                f"{title}: {mapping}"
            )

        parts = [
            f"{key}={value}"
            for key, value in mapping.items()
            if value is not None
        ]

        if not parts:
            return title

        return (
            f"{title}: "
            + ", ".join(parts)
            + "."
        )

    # ========================================================================
    # Validation / IDs
    # ========================================================================

    @staticmethod
    def _validate_identifier(
        value: str,
        *,
        field_name: str,
    ) -> str:
        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return value

    @staticmethod
    def _assign_evidence_ids(
        evidence: list[GraphRAGEvidence],
    ) -> list[GraphRAGEvidence]:
        """
        Reassign deterministic local evidence IDs.

        Final global evidence numbering can be performed by
        evidence_fusion.py when multiple sources are combined.
        """

        return [
            item.model_copy(
                update={
                    "evidence_id": f"E{index}"
                }
            )
            for index, item in enumerate(
                evidence,
                start=1,
            )
        ]