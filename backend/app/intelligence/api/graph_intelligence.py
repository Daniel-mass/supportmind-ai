
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from neo4j import READ_ACCESS
from neo4j.graph import Node, Path, Relationship
from pydantic import BaseModel, Field

from app.intelligence.graph.operational_graph import (
    Neo4jOperationalGraph,
    create_neo4j_operational_graph,
)


router = APIRouter(
    prefix="/graph",
    tags=["intelligence-graph"],
)

NODE_LABELS = {
    "Ticket",
    "Agent",
    "WorkflowEvent",
    "IssueType",
    "Priority",
    "Status",
    "Team",
}


# ---------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------

class GraphNodeResponse(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]


class GraphEdgeResponse(BaseModel):
    id: str
    source: str
    target: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    returned_nodes: int
    returned_edges: int
    limit: int


class CypherWorkbenchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    limit: int = Field(default=100, ge=1, le=500)


# ---------------------------------------------------------------------
# Neo4j service and serialization
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_graph_service() -> Neo4jOperationalGraph:
    """Reuse one Neo4j driver for Graph Intelligence requests."""
    return create_neo4j_operational_graph()


def _json_safe(value: Any) -> Any:
    """Convert Neo4j values into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "iso_format"):
        return value.iso_format()

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def _workbench_json(value: Any) -> Any:
    """
    Serialize Neo4j nodes, relationships and paths while preserving
    their structure for graph visualization.
    """
    if isinstance(value, Node):
        return {
            "_type": "node",
            "id": value.element_id,
            "labels": sorted(value.labels),
            "properties": {
                key: _workbench_json(item)
                for key, item in dict(value).items()
            },
        }

    if isinstance(value, Relationship):
        return {
            "_type": "relationship",
            "id": value.element_id,
            "type": value.type,
            "source": value.start_node.element_id,
            "target": value.end_node.element_id,
            "properties": {
                key: _workbench_json(item)
                for key, item in dict(value).items()
            },
        }

    if isinstance(value, Path):
        return {
            "_type": "path",
            "nodes": [
                _workbench_json(node)
                for node in value.nodes
            ],
            "relationships": [
                _workbench_json(relationship)
                for relationship in value.relationships
            ],
        }

    if isinstance(value, dict):
        return {
            str(key): _workbench_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_workbench_json(item) for item in value]

    return _json_safe(value)


def _run_query(
    graph: Neo4jOperationalGraph,
    cypher: str,
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run existing application queries using the graph service."""
    try:
        return graph.execute_cypher(cypher, parameters)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to query the Neo4j operational graph.",
        ) from exc


def _execute_workbench_read(
    graph: Neo4jOperationalGraph,
    cypher: str,
    parameters: dict[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Execute workbench queries inside a Neo4j read transaction.

    The existing graph service and its driver are reused. The callback
    consumes the result before the transaction closes.
    """
    try:
        with graph._driver.session(
            database=graph.config.database,
            default_access_mode=READ_ACCESS,
        ) as session:

            def run_query(
                tx: Any,
            ) -> tuple[list[str], list[dict[str, Any]]]:
                result = tx.run(cypher, parameters or {})
                columns = list(result.keys())
                rows = result.data()
                return columns, rows

            return session.execute_read(run_query)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Neo4j read query failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------
# Existing endpoint: graph overview
# ---------------------------------------------------------------------

@router.get("/overview")
def get_graph_overview() -> dict[str, Any]:
    """Return real graph counts, label distribution, and connectivity."""
    graph = get_graph_service()

    try:
        health = graph.health_check()
    except Exception:
        return {
            "healthy": False,
            "database": None,
            "node_count": 0,
            "relationship_count": 0,
            "node_labels": [],
            "relationship_types": [],
        }

    if not health.get("healthy"):
        return {
            "healthy": False,
            "database": health.get("database"),
            "node_count": 0,
            "relationship_count": 0,
            "node_labels": [],
            "relationship_types": [],
        }

    counts = _run_query(
        graph,
        """
        MATCH (n)
        WITH count(n) AS nodes
        OPTIONAL MATCH ()-[r]->()
        RETURN nodes, count(r) AS relationships
        """,
    )

    label_rows = _run_query(
        graph,
        """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC, label
        """,
    )

    relationship_rows = _run_query(
        graph,
        """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC, type
        """,
    )

    totals = counts[0] if counts else {}

    return {
        "healthy": True,
        "database": health.get("database"),
        "node_count": totals.get("nodes", 0),
        "relationship_count": totals.get("relationships", 0),
        "node_labels": _json_safe(label_rows),
        "relationship_types": _json_safe(relationship_rows),
    }


# ---------------------------------------------------------------------
# Existing endpoint: graph exploration
# ---------------------------------------------------------------------

@router.get("/explore", response_model=GraphResponse)
def explore_graph(
    label: str | None = Query(default=None),
    search: str = Query(default="", max_length=150),
    limit: int = Query(default=100, ge=1, le=250),
) -> GraphResponse:
    """Return a bounded graph slice for visualization."""
    if label is not None and label not in NODE_LABELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported node label.",
        )

    graph = get_graph_service()

    _run_query(
        graph,
        "RETURN 1 AS connected",
    )

    node_rows = _run_query(
        graph,
        """
        MATCH (n)
        WHERE ($label IS NULL OR $label IN labels(n))
          AND (
            $search = ''
            OR any(
                key IN keys(n)
                WHERE toLower(toString(n[key]))
                    CONTAINS toLower($search)
            )
          )
        WITH n
        ORDER BY labels(n), elementId(n)
        LIMIT $limit
        RETURN
            elementId(n) AS id,
            labels(n) AS labels,
            properties(n) AS properties
        """,
        {
            "label": label,
            "search": search.strip(),
            "limit": limit,
        },
    )

    node_ids = [row["id"] for row in node_rows]
    edge_rows: list[dict[str, Any]] = []

    if node_ids:
        edge_rows = _run_query(
            graph,
            """
            MATCH (source)-[r]->(target)
            WHERE elementId(source) IN $node_ids
              AND elementId(target) IN $node_ids
            RETURN
                elementId(r) AS id,
                elementId(source) AS source,
                elementId(target) AS target,
                type(r) AS type,
                properties(r) AS properties
            LIMIT $edge_limit
            """,
            {
                "node_ids": node_ids,
                "edge_limit": min(limit * 3, 750),
            },
        )

    nodes = [
        GraphNodeResponse(
            id=str(row["id"]),
            labels=list(row.get("labels") or []),
            properties=_json_safe(row.get("properties") or {}),
        )
        for row in node_rows
    ]

    edges = [
        GraphEdgeResponse(
            id=str(row["id"]),
            source=str(row["source"]),
            target=str(row["target"]),
            type=str(row["type"]),
            properties=_json_safe(row.get("properties") or {}),
        )
        for row in edge_rows
    ]

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        returned_nodes=len(nodes),
        returned_edges=len(edges),
        limit=limit,
    )


# ---------------------------------------------------------------------
# Existing endpoint: selected node and its immediate relationships
# ---------------------------------------------------------------------

@router.get("/nodes/{node_id}")
def get_graph_node_details(node_id: str) -> dict[str, Any]:
    """Return a node's properties and its immediate relationships."""
    if not node_id.strip() or len(node_id) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid node ID.",
        )

    graph = get_graph_service()

    rows = _run_query(
        graph,
        """
        MATCH (n)
        WHERE elementId(n) = $node_id
           OR n.node_id = $node_id
        RETURN
            elementId(n) AS id,
            labels(n) AS labels,
            properties(n) AS properties
        LIMIT 1
        """,
        {"node_id": node_id},
    )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph node not found.",
        )

    relationship_rows = _run_query(
        graph,
        """
        MATCH (n)-[r]-(other)
        WHERE elementId(n) = $node_id
        RETURN
            elementId(r) AS id,
            type(r) AS type,
            CASE
                WHEN startNode(r) = n THEN 'outgoing'
                ELSE 'incoming'
            END AS direction,
            elementId(other) AS connected_node_id,
            labels(other) AS connected_node_labels,
            properties(other) AS connected_node_properties,
            properties(r) AS properties
        ORDER BY type(r)
        LIMIT 100
        """,
        {"node_id": rows[0]["id"]},
    )

    return {
        "node": {
            "id": str(rows[0]["id"]),
            "labels": list(rows[0].get("labels") or []),
            "properties": _json_safe(
                rows[0].get("properties") or {}
            ),
        },
        "relationships": _json_safe(relationship_rows),
    }


# ---------------------------------------------------------------------
# New endpoint: real graph schema and relationship patterns
# ---------------------------------------------------------------------

@router.get("/schema")
def get_graph_workbench_schema() -> dict[str, Any]:
    """
    Return actual node-label counts and observed relationship patterns.

    A pattern includes source labels, relationship type, target labels,
    and the number of matching relationships.
    """
    graph = get_graph_service()

    labels_query = """
    MATCH (n)
    UNWIND labels(n) AS label
    RETURN label, count(*) AS node_count
    ORDER BY node_count DESC, label
    """

    relationships_query = """
    MATCH (source)-[r]->(target)
    WITH
        labels(source) AS source_labels,
        type(r) AS relationship_type,
        labels(target) AS target_labels,
        count(*) AS relationship_count
    RETURN
        source_labels,
        relationship_type,
        target_labels,
        relationship_count
    ORDER BY relationship_count DESC
    LIMIT 250
    """

    _, label_rows = _execute_workbench_read(
        graph,
        labels_query,
    )

    _, relationship_rows = _execute_workbench_read(
        graph,
        relationships_query,
    )

    return {
        "node_labels": _workbench_json(label_rows),
        "relationships": _workbench_json(relationship_rows),
    }


# ---------------------------------------------------------------------
# New endpoint: bounded, read-only Cypher workbench
# ---------------------------------------------------------------------

def _validate_workbench_query(query: str) -> str:
    """
    Accept a single read query and reject write/administrative clauses.

    Validation is an additional safeguard, not a replacement for
    Neo4j's read-transaction enforcement.
    """
    query = query.strip()

    if query.endswith(";"):
        query = query[:-1].strip()

    if not query or ";" in query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Enter one Cypher query at a time.",
        )

    # Ignore quoted strings and comments while checking clauses.
    inspected = re.sub(
        r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/""",
        " ",
        query,
        flags=re.DOTALL,
    )

    allowed_start = re.match(
        r"^\s*(MATCH|OPTIONAL\s+MATCH|UNWIND|WITH|RETURN)\b",
        inspected,
        flags=re.IGNORECASE,
    )

    if not allowed_start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Start with MATCH, OPTIONAL MATCH, UNWIND, WITH, "
                "or RETURN. CALL and administrative queries are disabled."
            ),
        )

    if not re.search(
        r"\bRETURN\b",
        inspected,
        flags=re.IGNORECASE,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The query must return results.",
        )

    forbidden = re.compile(
        r"\b("
        r"CREATE|MERGE|DELETE|SET|REMOVE|DROP|"
        r"LOAD\s+CSV|FOREACH|CALL|USE|"
        r"GRANT|DENY|REVOKE|ALTER|RENAME|"
        r"TERMINATE|START|STOP|SHOW|EXPLAIN|PROFILE"
        r")\b",
        flags=re.IGNORECASE,
    )

    if forbidden.search(inspected):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Write, procedure, and administrative clauses "
                "are not allowed in this workbench."
            ),
        )

    return query


@router.post("/cypher")
def run_graph_workbench_query(
    request: CypherWorkbenchRequest,
) -> dict[str, Any]:
    """
    Execute a bounded read query and return table-ready JSON.

    Nodes, relationships, and paths retain their structure so the
    frontend can render them as a graph when appropriate.
    """
    graph = get_graph_service()
    query = _validate_workbench_query(request.query)
    started = time.perf_counter()

    bounded_query = (
        "CALL {\n"
        f"{query}\n"
        "} RETURN * LIMIT $workbench_limit"
    )

    columns, raw_rows = _execute_workbench_read(
        graph,
        bounded_query,
        {"workbench_limit": request.limit + 1},
    )

    truncated = len(raw_rows) > request.limit
    raw_rows = raw_rows[:request.limit]

    return {
        "columns": columns,
        "rows": _workbench_json(raw_rows),
        "row_count": len(raw_rows),
        "truncated": truncated,
        "limit": request.limit,
        "execution_ms": round(
            (time.perf_counter() - started) * 1000,
            2,
        ),
    }
