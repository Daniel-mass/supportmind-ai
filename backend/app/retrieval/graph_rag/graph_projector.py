"""
CSV -> Neo4j graph projection for SupportMind AI.

This module converts the cleaned Customer Support Ticket Dataset
into a relationship-oriented Neo4j graph.

Graph
-----

(:Customer)-[:CREATED]->(:Ticket)
(:Ticket)-[:ABOUT]->(:Product)
(:Ticket)-[:HAS_TYPE]->(:TicketType)
(:Ticket)-[:HAS_PRIORITY]->(:Priority)
(:Ticket)-[:HAS_STATUS]->(:Status)
(:Ticket)-[:VIA_CHANNEL]->(:Channel)

This module does NOT:
- perform retrieval
- call an LLM
- generate embeddings
- write to Qdrant
- perform graph retrieval

SupportMind AI
"""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.retrieval.graph_rag.graph_schema import (
    ChannelNode,
    CustomerNode,
    GraphNode,
    GraphRelationship,
    NodeLabel,
    PriorityNode,
    ProductNode,
    RelationshipType,
    StatusNode,
    TicketGraphProjection,
    TicketNode,
    TicketTypeNode,
)

from app.retrieval.graph_rag.neo4j_store import (
    GraphWriteResult,
    Neo4jConfig,
    Neo4jStore,
)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BATCH_SIZE = 500

CSV_COLUMNS = {
    "Ticket ID",
    "Customer Name",
    "Customer Email",
    "Customer Age",
    "Customer Gender",
    "Product Purchased",
    "Date of Purchase",
    "Ticket Type",
    "Ticket Subject",
    "Ticket Description",
    "Ticket Status",
    "Resolution",
    "Ticket Priority",
    "Ticket Channel",
    "First Response Time",
    "Time to Resolution",
    "Customer Satisfaction Rating",
}


# ============================================================================
# Row contract
# ============================================================================


class SupportTicketRow(BaseModel):
    """
    Validated representation of one customer-support row.

    Resolution is intentionally excluded because the cleaned dataset
    contains no usable resolution values.
    """

    model_config = ConfigDict(
        extra="ignore"
    )

    ticket_id: str = Field(
        min_length=1
    )

    customer_name: str = Field(
        min_length=1
    )

    customer_email: str | None = None

    customer_age: int | None = Field(
        default=None,
        ge=0,
        le=150,
    )

    customer_gender: str | None = None

    product_purchased: str | None = None

    date_of_purchase: date | None = None

    ticket_type: str | None = None
    ticket_subject: str | None = None
    ticket_description: str | None = None

    ticket_status: str | None = None
    ticket_priority: str | None = None
    ticket_channel: str | None = None

    first_response_time: datetime | None = None
    time_to_resolution: datetime | None = None

    customer_satisfaction_rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
    )


# ============================================================================
# Projection statistics
# ============================================================================


class ProjectionStats(BaseModel):
    """Statistics generated during graph projection."""

    model_config = ConfigDict(
        extra="forbid"
    )

    rows_read: int = 0
    rows_projected: int = 0
    rows_skipped: int = 0

    validation_errors: int = 0

    nodes_created: int = 0
    nodes_matched: int = 0

    relationships_created: int = 0
    relationships_matched: int = 0


# Backwards-compatible alias if anything still imports the old name.
GraphProjectionStats = ProjectionStats


# ============================================================================
# ID utilities
# ============================================================================


def _normalize_identity(
    value: str,
) -> str:

    value = value.strip().lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def _stable_id(
    prefix: str,
    value: str,
) -> str:

    normalized = _normalize_identity(
        value
    )

    digest = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]

    return f"{prefix}:{digest}"


def _customer_id(
    *,
    email: str | None,
    name: str,
) -> str:

    if email:
        return _stable_id(
            "customer",
            email,
        )

    return _stable_id(
        "customer",
        name,
    )


def _ticket_id(
    ticket_id: str,
) -> str:

    return f"ticket:{ticket_id.strip()}"


def _product_id(
    product: str,
) -> str:

    return _stable_id(
        "product",
        product,
    )


def _ticket_type_id(
    ticket_type: str,
) -> str:

    return _stable_id(
        "ticket-type",
        ticket_type,
    )


def _priority_id(
    priority: str,
) -> str:

    return _stable_id(
        "priority",
        priority,
    )


def _status_id(
    status: str,
) -> str:

    return _stable_id(
        "status",
        status,
    )


def _channel_id(
    channel: str,
) -> str:

    return _stable_id(
        "channel",
        channel,
    )


# ============================================================================
# Parsing
# ============================================================================


def _clean_string(
    value: Any,
) -> str | None:

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() in {
        "null",
        "none",
        "nan",
        "na",
        "n/a",
    }:
        return None

    return value


def _parse_int(
    value: Any,
) -> int | None:

    value = _clean_string(value)

    if value is None:
        return None

    try:
        return int(
            float(value)
        )
    except ValueError:
        return None


def _parse_float(
    value: Any,
) -> float | None:

    value = _clean_string(value)

    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _parse_date(
    value: Any,
) -> date | None:

    value = _clean_string(value)

    if value is None:
        return None

    formats = (
        "%d-%m-%Y",
        "%Y-%m-%d",
    )

    for fmt in formats:

        try:
            return datetime.strptime(
                value,
                fmt,
            ).date()

        except ValueError:
            continue

    return None


def _parse_datetime(
    value: Any,
) -> datetime | None:

    value = _clean_string(value)

    if value is None:
        return None

    formats = (
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    )

    for fmt in formats:

        try:
            return datetime.strptime(
                value,
                fmt,
            )

        except ValueError:
            continue

    return None


# ============================================================================
# Row mapping
# ============================================================================


def _row_to_contract(
    row: dict[str, Any],
) -> SupportTicketRow:

    return SupportTicketRow(
        ticket_id=(
            _clean_string(
                row.get("Ticket ID")
            )
            or ""
        ),

        customer_name=(
            _clean_string(
                row.get("Customer Name")
            )
            or ""
        ),

        customer_email=_clean_string(
            row.get("Customer Email")
        ),

        customer_age=_parse_int(
            row.get("Customer Age")
        ),

        customer_gender=_clean_string(
            row.get("Customer Gender")
        ),

        product_purchased=_clean_string(
            row.get("Product Purchased")
        ),

        date_of_purchase=_parse_date(
            row.get("Date of Purchase")
        ),

        ticket_type=_clean_string(
            row.get("Ticket Type")
        ),

        ticket_subject=_clean_string(
            row.get("Ticket Subject")
        ),

        ticket_description=_clean_string(
            row.get("Ticket Description")
        ),

        ticket_status=_clean_string(
            row.get("Ticket Status")
        ),

        ticket_priority=_clean_string(
            row.get("Ticket Priority")
        ),

        ticket_channel=_clean_string(
            row.get("Ticket Channel")
        ),

        first_response_time=_parse_datetime(
            row.get("First Response Time")
        ),

        time_to_resolution=_parse_datetime(
            row.get("Time to Resolution")
        ),

        customer_satisfaction_rating=_parse_float(
            row.get(
                "Customer Satisfaction Rating"
            )
        ),
    )


# ============================================================================
# Graph projector
# ============================================================================


class GraphProjector:
    """Projects structured support-ticket rows into Neo4j."""

    def __init__(
        self,
        store: Neo4jStore,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.store = store
        self.batch_size = batch_size

    # =========================================================================
    # CSV API
    # =========================================================================

    def project_csv(
        self,
        csv_path: str | Path,
    ) -> ProjectionStats:
        """
        Read a CSV structurally and project it.

        This is useful for direct CLI/testing.
        """

        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(
                f"CSV file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"CSV path is not a file: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            self._validate_columns(
                reader.fieldnames
            )

            rows = list(reader)

        return self.project_rows(
            rows=rows,
            source_file=path.name,
        )

    # =========================================================================
    # Structured row API
    # =========================================================================

    def project_rows(
        self,
        rows: Iterable[
            dict[str, Any]
        ],
        *,
        source_file: str | None = None,
        document_id: str | None = None,
    ) -> ProjectionStats:
        """
        Project already-structured CSV rows.

        This is the API used by GraphIngestionPipeline.
        """

        self.store.ensure_schema()

        stats = ProjectionStats()

        node_batch: list[GraphNode] = []
        relationship_batch: list[
            GraphRelationship
        ] = []

        for raw_row in rows:

            stats.rows_read += 1

            try:

                ticket = _row_to_contract(
                    raw_row
                )

                projection = (
                    self._build_projection(
                        ticket
                    )
                )

                nodes, relationships = (
                    self._projection_to_graph(
                        projection,
                        source_file=source_file,
                        document_id=document_id,
                    )
                )

            except (
                ValidationError,
                ValueError,
            ):

                stats.rows_skipped += 1
                stats.validation_errors += 1
                continue

            node_batch.extend(nodes)
            relationship_batch.extend(
                relationships
            )

            stats.rows_projected += 1

            if len(node_batch) >= self.batch_size:

                result = self._flush(
                    node_batch,
                    relationship_batch,
                )

                self._merge_write_stats(
                    stats,
                    result,
                )

                node_batch.clear()
                relationship_batch.clear()

        if node_batch:

            result = self._flush(
                node_batch,
                relationship_batch,
            )

            self._merge_write_stats(
                stats,
                result,
            )

        return stats

    # =========================================================================
    # Projection construction
    # =========================================================================

    def _build_projection(
        self,
        ticket: SupportTicketRow,
    ) -> TicketGraphProjection:

        customer = CustomerNode(
            customer_id=_customer_id(
                email=ticket.customer_email,
                name=ticket.customer_name,
            ),
            name=ticket.customer_name,
            email=ticket.customer_email,
            age=ticket.customer_age,
            gender=ticket.customer_gender,
        )

        ticket_node = TicketNode(
            ticket_id=ticket.ticket_id,
            subject=ticket.ticket_subject,
            description=ticket.ticket_description,
            ticket_status=ticket.ticket_status,
            ticket_type=ticket.ticket_type,
            ticket_priority=ticket.ticket_priority,
            ticket_channel=ticket.ticket_channel,
            product_purchased=ticket.product_purchased,
            date_of_purchase=ticket.date_of_purchase,
            first_response_time=ticket.first_response_time,
            time_to_resolution=ticket.time_to_resolution,
            customer_satisfaction_rating=(
                ticket.customer_satisfaction_rating
            ),
        )

        product = None

        if ticket.product_purchased:

            product = ProductNode(
                product_id=_product_id(
                    ticket.product_purchased
                ),
                name=ticket.product_purchased,
            )

        ticket_type = None

        if ticket.ticket_type:

            ticket_type = TicketTypeNode(
                type_id=_ticket_type_id(
                    ticket.ticket_type
                ),
                name=ticket.ticket_type,
            )

        priority = None

        if ticket.ticket_priority:

            priority = PriorityNode(
                priority_id=_priority_id(
                    ticket.ticket_priority
                ),
                name=ticket.ticket_priority,
            )

        status = None

        if ticket.ticket_status:

            status = StatusNode(
                status_id=_status_id(
                    ticket.ticket_status
                ),
                name=ticket.ticket_status,
            )

        channel = None

        if ticket.ticket_channel:

            channel = ChannelNode(
                channel_id=_channel_id(
                    ticket.ticket_channel
                ),
                name=ticket.ticket_channel,
            )

        return TicketGraphProjection(
            customer=customer,
            ticket=ticket_node,
            product=product,
            ticket_type=ticket_type,
            priority=priority,
            status=status,
            channel=channel,
        )

    # =========================================================================
    # Generic graph contracts
    # =========================================================================

    def _projection_to_graph(
        self,
        projection: TicketGraphProjection,
        *,
        source_file: str | None = None,
        document_id: str | None = None,
    ) -> tuple[
        list[GraphNode],
        list[GraphRelationship],
    ]:

        nodes: list[GraphNode] = []
        relationships: list[
            GraphRelationship
        ] = []

        customer_id = (
            projection.customer.customer_id
        )

        ticket_id = _ticket_id(
            projection.ticket.ticket_id
        )

        # ------------------------------------------------------------------
        # Customer
        # ------------------------------------------------------------------

        nodes.append(
            GraphNode(
                node_id=customer_id,
                label=NodeLabel.CUSTOMER,
                properties={
                    "name": projection.customer.name,
                    "email": projection.customer.email,
                    "age": projection.customer.age,
                    "gender": projection.customer.gender,
                },
            )
        )

        # ------------------------------------------------------------------
        # Ticket
        # ------------------------------------------------------------------

        ticket_properties = {
            "ticket_id": projection.ticket.ticket_id,
            "description": projection.ticket.description,
            "subject": projection.ticket.subject,
            "status": projection.ticket.ticket_status,
            "ticket_type": projection.ticket.ticket_type,
            "priority": projection.ticket.ticket_priority,
            "channel": projection.ticket.ticket_channel,
            "product_purchased": (
                projection.ticket.product_purchased
            ),
            "date_of_purchase": (
                projection.ticket.date_of_purchase.isoformat()
                if projection.ticket.date_of_purchase
                else None
            ),
            "first_response_time": (
                projection.ticket.first_response_time.isoformat()
                if projection.ticket.first_response_time
                else None
            ),
            "time_to_resolution": (
                projection.ticket.time_to_resolution.isoformat()
                if projection.ticket.time_to_resolution
                else None
            ),
            "customer_satisfaction_rating": (
                projection.ticket.customer_satisfaction_rating
            ),
        }

        if source_file:
            ticket_properties[
                "source_file"
            ] = source_file

        if document_id:
            ticket_properties[
                "document_id"
            ] = document_id

        nodes.append(
            GraphNode(
                node_id=ticket_id,
                label=NodeLabel.TICKET,
                properties=ticket_properties,
            )
        )

        relationships.append(
            GraphRelationship(
                source_id=customer_id,
                source_label=NodeLabel.CUSTOMER,
                relationship_type=(
                    RelationshipType.CREATED
                ),
                target_id=ticket_id,
                target_label=NodeLabel.TICKET,
            )
        )

        # ------------------------------------------------------------------
        # Product
        # ------------------------------------------------------------------

        if projection.product:

            product_id = (
                projection.product.product_id
            )

            nodes.append(
                GraphNode(
                    node_id=product_id,
                    label=NodeLabel.PRODUCT,
                    properties={
                        "name": (
                            projection.product.name
                        )
                    },
                )
            )

            relationships.append(
                GraphRelationship(
                    source_id=ticket_id,
                    source_label=NodeLabel.TICKET,
                    relationship_type=(
                        RelationshipType.ABOUT
                    ),
                    target_id=product_id,
                    target_label=NodeLabel.PRODUCT,
                )
            )

        # ------------------------------------------------------------------
        # Ticket Type
        # ------------------------------------------------------------------

        if projection.ticket_type:

            type_id = (
                projection.ticket_type.type_id
            )

            nodes.append(
                GraphNode(
                    node_id=type_id,
                    label=NodeLabel.TICKET_TYPE,
                    properties={
                        "name": (
                            projection.ticket_type.name
                        )
                    },
                )
            )

            relationships.append(
                GraphRelationship(
                    source_id=ticket_id,
                    source_label=NodeLabel.TICKET,
                    relationship_type=(
                        RelationshipType.HAS_TYPE
                    ),
                    target_id=type_id,
                    target_label=(
                        NodeLabel.TICKET_TYPE
                    ),
                )
            )

        # ------------------------------------------------------------------
        # Priority
        # ------------------------------------------------------------------

        if projection.priority:

            priority_id = (
                projection.priority.priority_id
            )

            nodes.append(
                GraphNode(
                    node_id=priority_id,
                    label=NodeLabel.PRIORITY,
                    properties={
                        "name": (
                            projection.priority.name
                        )
                    },
                )
            )

            relationships.append(
                GraphRelationship(
                    source_id=ticket_id,
                    source_label=NodeLabel.TICKET,
                    relationship_type=(
                        RelationshipType.HAS_PRIORITY
                    ),
                    target_id=priority_id,
                    target_label=NodeLabel.PRIORITY,
                )
            )

        # ------------------------------------------------------------------
        # Status
        # ------------------------------------------------------------------

        if projection.status:

            status_id = (
                projection.status.status_id
            )

            nodes.append(
                GraphNode(
                    node_id=status_id,
                    label=NodeLabel.STATUS,
                    properties={
                        "name": (
                            projection.status.name
                        )
                    },
                )
            )

            relationships.append(
                GraphRelationship(
                    source_id=ticket_id,
                    source_label=NodeLabel.TICKET,
                    relationship_type=(
                        RelationshipType.HAS_STATUS
                    ),
                    target_id=status_id,
                    target_label=NodeLabel.STATUS,
                )
            )

        # ------------------------------------------------------------------
        # Channel
        # ------------------------------------------------------------------

        if projection.channel:

            channel_id = (
                projection.channel.channel_id
            )

            nodes.append(
                GraphNode(
                    node_id=channel_id,
                    label=NodeLabel.CHANNEL,
                    properties={
                        "name": (
                            projection.channel.name
                        )
                    },
                )
            )

            relationships.append(
                GraphRelationship(
                    source_id=ticket_id,
                    source_label=NodeLabel.TICKET,
                    relationship_type=(
                        RelationshipType.VIA_CHANNEL
                    ),
                    target_id=channel_id,
                    target_label=NodeLabel.CHANNEL,
                )
            )

        return nodes, relationships

    # =========================================================================
    # Batch
    # =========================================================================

    def _flush(
        self,
        nodes: list[GraphNode],
        relationships: list[
            GraphRelationship
        ],
    ) -> GraphWriteResult:

        if not nodes and not relationships:
            return GraphWriteResult()

        return self.store.write_projection(
            nodes=nodes,
            relationships=relationships,
        )

    @staticmethod
    def _merge_write_stats(
        stats: ProjectionStats,
        result: GraphWriteResult,
    ) -> None:

        stats.nodes_created += (
            result.nodes_created
        )

        stats.nodes_matched += (
            result.nodes_matched
        )

        stats.relationships_created += (
            result.relationships_created
        )

        stats.relationships_matched += (
            result.relationships_matched
        )

    # =========================================================================
    # Validation
    # =========================================================================

    @staticmethod
    def _validate_columns(
        fieldnames: list[str] | None,
    ) -> None:

        if not fieldnames:
            raise ValueError(
                "CSV does not contain a header row."
            )

        actual = {
            field.strip()
            for field in fieldnames
            if field
        }

        missing = (
            CSV_COLUMNS - actual
        )

        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )


# ============================================================================
# Environment
# ============================================================================


def create_store_from_environment() -> Neo4jStore:
    """Create Neo4jStore using environment configuration."""

    import os

    config = Neo4jConfig(
        uri=os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        ),
        username=os.getenv(
            "NEO4J_USERNAME",
            "neo4j",
        ),
        password=os.getenv(
            "NEO4J_PASSWORD",
            "",
        ),
        database=os.getenv(
            "NEO4J_DATABASE",
            "neo4j",
        ),
    )

    return Neo4jStore(config)