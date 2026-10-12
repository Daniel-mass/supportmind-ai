#vector_filters.py
"""
Provider-agnostic vector search filters.

These schemas define how SupportMind requests
metadata filtering during vector search.

Concrete vector databases (Qdrant, Pinecone,
Milvus, etc.) are responsible for translating
these filters into their native query language.

SupportMind AI
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FilterOperator(str, Enum):
    """
    Supported comparison operators.
    """

    EQUAL = "eq"
    NOT_EQUAL = "ne"

    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"

    LESS_THAN = "lt"
    LESS_EQUAL = "lte"

    IN = "in"
    NOT_IN = "not_in"

    CONTAINS = "contains"

    EXISTS = "exists"


class FilterCondition(BaseModel):
    """
    Single metadata filter.

    Example

    metadata.source == "pdf"

    metadata.page >= 5

    metadata.category in [...]
    """

    field: str = Field(
        ...,
        description="Metadata field."
    )

    operator: FilterOperator = Field(
        ...,
        description="Comparison operator."
    )

    value: Any = Field(
        ...,
        description="Comparison value."
    )


class VectorFilter(BaseModel):
    """
    Collection of metadata filters.

    Logical operation is AND.

    OR logic can be added later if required.
    """

    conditions: list[FilterCondition] = Field(
        default_factory=list,
        description="Metadata filter conditions."
    )

    def is_empty(self) -> bool:
        """
        Returns True when no filters exist.
        """
        return len(self.conditions) == 0

    def add(
        self,
        field: str,
        operator: FilterOperator,
        value: Any,
    ) -> None:
        """
        Append a filter condition.
        """

        self.conditions.append(

            FilterCondition(
                field=field,
                operator=operator,
                value=value,
            )

        )