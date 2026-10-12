"""
Graph schema contracts for SupportMind AI.

This module defines the canonical Neo4j graph model used by the
SupportMind customer-support graph.

The schema is intentionally small and relationship-focused.

CSV source:
    Customer Support Ticket Dataset

Graph model:

    (:Customer)-[:CREATED]->(:Ticket)
    (:Ticket)-[:ABOUT]->(:Product)
    (:Ticket)-[:HAS_TYPE]->(:TicketType)
    (:Ticket)-[:HAS_PRIORITY]->(:Priority)
    (:Ticket)-[:HAS_STATUS]->(:Status)
    (:Ticket)-[:VIA_CHANNEL]->(:Channel)

Design principles:
    - Stable business identifiers
    - Explicit relationships
    - No duplicate entity nodes for scalar attributes
    - Pydantic validation before persistence
    - No LLM dependency
    - No retrieval logic
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Graph labels
# ---------------------------------------------------------------------------


class NodeLabel(StrEnum):
    """Canonical Neo4j node labels."""

    CUSTOMER = "Customer"
    TICKET = "Ticket"
    PRODUCT = "Product"
    TICKET_TYPE = "TicketType"
    PRIORITY = "Priority"
    STATUS = "Status"
    CHANNEL = "Channel"


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------


class RelationshipType(StrEnum):
    """Canonical Neo4j relationship types."""

    CREATED = "CREATED"
    ABOUT = "ABOUT"
    HAS_TYPE = "HAS_TYPE"
    HAS_PRIORITY = "HAS_PRIORITY"
    HAS_STATUS = "HAS_STATUS"
    VIA_CHANNEL = "VIA_CHANNEL"


# ---------------------------------------------------------------------------
# Base graph contracts
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """Base contract for a Neo4j node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    label: NodeLabel
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    """Base contract for a Neo4j relationship."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    source_label: NodeLabel

    relationship_type: RelationshipType

    target_id: str = Field(min_length=1)
    target_label: NodeLabel

    properties: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------


class CustomerNode(BaseModel):
    """
    Customer entity.

    Email is used as the preferred stable identity because customer names
    are not guaranteed to be unique.
    """

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    email: str | None = None
    age: int | None = Field(default=None, ge=0, le=150)
    gender: str | None = None

    @field_validator("name", "email", "gender", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> Any:
        if value is None:
            return None

        value = str(value).strip()

        return value or None


# ---------------------------------------------------------------------------
# Ticket
# ---------------------------------------------------------------------------


class TicketNode(BaseModel):
    """Customer support ticket entity."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1)

    subject: str | None = None
    description: str | None = None

    ticket_status: str | None = None
    ticket_type: str | None = None
    ticket_priority: str | None = None
    ticket_channel: str | None = None

    product_purchased: str | None = None

    date_of_purchase: date | None = None
    first_response_time: datetime | None = None
    time_to_resolution: datetime | None = None

    customer_satisfaction_rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
    )

    @field_validator(
        "subject",
        "description",
        "ticket_status",
        "ticket_type",
        "ticket_priority",
        "ticket_channel",
        "product_purchased",
        mode="before",
    )
    @classmethod
    def normalize_strings(cls, value: Any) -> Any:
        if value is None:
            return None

        value = str(value).strip()

        return value or None


# ---------------------------------------------------------------------------
# Canonical entity nodes
# ---------------------------------------------------------------------------


class ProductNode(BaseModel):
    """Product entity."""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class TicketTypeNode(BaseModel):
    """Ticket type entity."""

    model_config = ConfigDict(extra="forbid")

    type_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class PriorityNode(BaseModel):
    """Ticket priority entity."""

    model_config = ConfigDict(extra="forbid")

    priority_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class StatusNode(BaseModel):
    """Ticket status entity."""

    model_config = ConfigDict(extra="forbid")

    status_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class ChannelNode(BaseModel):
    """Ticket channel entity."""

    model_config = ConfigDict(extra="forbid")

    channel_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Graph projection contract
# ---------------------------------------------------------------------------


class TicketGraphProjection(BaseModel):
    """
    Complete graph projection for one support ticket row.

    This is the contract passed from graph_projector.py to neo4j_store.py.
    """

    model_config = ConfigDict(extra="forbid")

    customer: CustomerNode
    ticket: TicketNode

    product: ProductNode | None = None
    ticket_type: TicketTypeNode | None = None
    priority: PriorityNode | None = None
    status: StatusNode | None = None
    channel: ChannelNode | None = None