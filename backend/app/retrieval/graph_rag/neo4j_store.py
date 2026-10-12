"""
Neo4j persistence layer for SupportMind AI.

Responsibilities
----------------
- Manage Neo4j driver lifecycle.
- Verify connectivity.
- Configure graph constraints.
- Persist graph nodes and relationships.
- Support transactional batch writes.
- Provide graph statistics and controlled Cypher reads.

This module intentionally contains NO:
- CSV parsing
- entity extraction
- LLM calls
- embeddings
- Qdrant logic
- retrieval/ranking logic

Graph projection logic belongs in graph_projector.py.
Graph retrieval logic belongs in graph_retriever.py.

SupportMind AI
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import Neo4jError

from app.retrieval.graph_rag.graph_schema import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelationshipType,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class Neo4jConfig:
    """Immutable Neo4j connection configuration."""

    uri: str
    username: str
    password: str
    database: str = "neo4j"

    max_connection_pool_size: int = 50
    connection_timeout: float = 10.0
    max_transaction_retry_time: float = 30.0

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("Neo4j URI cannot be empty.")

        if not self.username.strip():
            raise ValueError("Neo4j username cannot be empty.")

        if not self.password:
            raise ValueError("Neo4j password cannot be empty.")

        if not self.database.strip():
            raise ValueError("Neo4j database cannot be empty.")

        if self.max_connection_pool_size <= 0:
            raise ValueError(
                "max_connection_pool_size must be greater than zero."
            )

        if self.connection_timeout <= 0:
            raise ValueError(
                "connection_timeout must be greater than zero."
            )

        if self.max_transaction_retry_time <= 0:
            raise ValueError(
                "max_transaction_retry_time must be greater than zero."
            )


# ============================================================================
# Result contracts
# ============================================================================


@dataclass(frozen=True, slots=True)
class GraphWriteResult:
    """Summary of one graph write operation."""

    nodes_created: int = 0
    nodes_matched: int = 0

    relationships_created: int = 0
    relationships_matched: int = 0


@dataclass(frozen=True, slots=True)
class GraphStats:
    """Basic graph statistics."""

    node_count: int
    relationship_count: int


# ============================================================================
# Neo4j Store
# ============================================================================


class Neo4jStore:
    """
    Infrastructure layer for Neo4j persistence.

    The store does not know where graph data came from.
    """

    def __init__(
        self,
        config: Neo4jConfig,
        *,
        driver: Driver | None = None,
    ) -> None:

        self.config = config

        self._driver: Driver = driver or GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
            max_connection_pool_size=config.max_connection_pool_size,
            connection_timeout=config.connection_timeout,
            max_transaction_retry_time=config.max_transaction_retry_time,
        )

        self._closed = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @property
    def driver(self) -> Driver:
        """Return the underlying Neo4j driver."""

        self._ensure_open()

        return self._driver

    def verify_connection(self) -> bool:
        """Verify Neo4j connectivity and authentication."""

        self._ensure_open()

        try:
            self._driver.verify_connectivity()
            return True

        except Neo4jError:
            return False

    def close(self) -> None:
        """Close the Neo4j driver."""

        if not self._closed:
            self._driver.close()
            self._closed = True

    def __enter__(self) -> "Neo4jStore":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Neo4jStore has already been closed."
            )

    # =========================================================================
    # Session
    # =========================================================================

    def _session(self) -> Session:
        """Create a configured Neo4j session."""

        self._ensure_open()

        return self._driver.session(
            database=self.config.database
        )

    # =========================================================================
    # Schema
    # =========================================================================

    def ensure_schema(self) -> None:
        """
        Create uniqueness constraints for graph node IDs.

        Idempotent and safe to call before ingestion.
        """

        self._ensure_open()

        statements = [
            """
            CREATE CONSTRAINT customer_node_id_unique
            IF NOT EXISTS
            FOR (n:Customer)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT ticket_node_id_unique
            IF NOT EXISTS
            FOR (n:Ticket)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT product_node_id_unique
            IF NOT EXISTS
            FOR (n:Product)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT ticket_type_node_id_unique
            IF NOT EXISTS
            FOR (n:TicketType)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT priority_node_id_unique
            IF NOT EXISTS
            FOR (n:Priority)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT status_node_id_unique
            IF NOT EXISTS
            FOR (n:Status)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT channel_node_id_unique
            IF NOT EXISTS
            FOR (n:Channel)
            REQUIRE n.node_id IS UNIQUE
            """,
        ]

        with self._session() as session:
            for statement in statements:
                session.run(statement).consume()

    # =========================================================================
    # Node persistence
    # =========================================================================

    def upsert_node(
        self,
        node: GraphNode,
    ) -> GraphWriteResult:
        """Create or update one graph node."""

        self._ensure_open()

        statement = self._node_merge_statement(
            node.label
        )

        with self._session() as session:

            result = session.run(
                statement,
                node_id=node.node_id,
                properties=dict(node.properties),
            ).single()

            if result is None:
                raise RuntimeError(
                    f"No write result returned for node "
                    f"{node.node_id!r}."
                )

            return GraphWriteResult(
                nodes_created=int(
                    result["created"]
                ),
                nodes_matched=int(
                    result["matched"]
                ),
            )

    def _node_merge_statement(
        self,
        label: NodeLabel,
    ) -> str:
        """Build a trusted node MERGE statement."""

        allowed_labels = {
            NodeLabel.CUSTOMER,
            NodeLabel.TICKET,
            NodeLabel.PRODUCT,
            NodeLabel.TICKET_TYPE,
            NodeLabel.PRIORITY,
            NodeLabel.STATUS,
            NodeLabel.CHANNEL,
        }

        if label not in allowed_labels:
            raise ValueError(
                f"Unsupported graph node label: {label}"
            )

        return f"""
        MERGE (n:{label.value} {{node_id: $node_id}})

        ON CREATE SET
            n += $properties,
            n.created_at = datetime()

        ON MATCH SET
            n += $properties,
            n.updated_at = datetime()

        RETURN
            CASE
                WHEN n.created_at IS NOT NULL
                AND n.updated_at IS NULL
                THEN 1
                ELSE 0
            END AS created,

            CASE
                WHEN n.updated_at IS NOT NULL
                THEN 1
                ELSE 0
            END AS matched
        """

    # =========================================================================
    # Relationship persistence
    # =========================================================================

    def upsert_relationship(
        self,
        relationship: GraphRelationship,
    ) -> GraphWriteResult:
        """Create or update one relationship."""

        self._ensure_open()

        statement = self._relationship_merge_statement(
            source_label=relationship.source_label,
            relationship_type=relationship.relationship_type,
            target_label=relationship.target_label,
        )

        with self._session() as session:

            result = session.run(
                statement,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                properties=dict(relationship.properties),
            ).single()

            if result is None:
                raise RuntimeError(
                    "No relationship write result returned."
                )

            if result["source_exists"] == 0:
                raise ValueError(
                    f"Source node "
                    f"{relationship.source_id!r} "
                    f"does not exist."
                )

            if result["target_exists"] == 0:
                raise ValueError(
                    f"Target node "
                    f"{relationship.target_id!r} "
                    f"does not exist."
                )

            return GraphWriteResult(
                relationships_created=int(
                    result["created"]
                ),
                relationships_matched=int(
                    result["matched"]
                ),
            )

    def _relationship_merge_statement(
        self,
        *,
        source_label: NodeLabel,
        relationship_type: RelationshipType,
        target_label: NodeLabel,
    ) -> str:
        """Build a trusted relationship MERGE statement."""

        allowed_labels = {
            NodeLabel.CUSTOMER,
            NodeLabel.TICKET,
            NodeLabel.PRODUCT,
            NodeLabel.TICKET_TYPE,
            NodeLabel.PRIORITY,
            NodeLabel.STATUS,
            NodeLabel.CHANNEL,
        }

        allowed_relationships = {
            RelationshipType.CREATED,
            RelationshipType.ABOUT,
            RelationshipType.HAS_TYPE,
            RelationshipType.HAS_PRIORITY,
            RelationshipType.HAS_STATUS,
            RelationshipType.VIA_CHANNEL,
        }

        if source_label not in allowed_labels:
            raise ValueError(
                f"Unsupported source label: {source_label}"
            )

        if target_label not in allowed_labels:
            raise ValueError(
                f"Unsupported target label: {target_label}"
            )

        if relationship_type not in allowed_relationships:
            raise ValueError(
                f"Unsupported relationship type: "
                f"{relationship_type}"
            )

        return f"""
        MATCH (
            source:{source_label.value}
            {{node_id: $source_id}}
        )

        MATCH (
            target:{target_label.value}
            {{node_id: $target_id}}
        )

        MERGE (
            source
        )-[r:{relationship_type.value}]->(
            target
        )

        ON CREATE SET
            r += $properties,
            r.created_at = datetime()

        ON MATCH SET
            r += $properties,
            r.updated_at = datetime()

        RETURN
            1 AS source_exists,
            1 AS target_exists,

            CASE
                WHEN r.created_at IS NOT NULL
                AND r.updated_at IS NULL
                THEN 1
                ELSE 0
            END AS created,

            CASE
                WHEN r.updated_at IS NOT NULL
                THEN 1
                ELSE 0
            END AS matched
        """

    # =========================================================================
    # Transactional projection write
    # =========================================================================

    def write_projection(
        self,
        nodes: Iterable[GraphNode],
        relationships: Iterable[GraphRelationship],
    ) -> GraphWriteResult:
        """
        Write nodes and relationships atomically.
        """

        self._ensure_open()

        node_list = list(nodes)
        relationship_list = list(
            relationships
        )

        if not node_list and not relationship_list:
            return GraphWriteResult()

        with self._session() as session:

            return session.execute_write(
                self._write_projection_transaction,
                node_list,
                relationship_list,
            )

    def _write_projection_transaction(
        self,
        tx: Any,
        nodes: list[GraphNode],
        relationships: list[GraphRelationship],
    ) -> GraphWriteResult:

        total_nodes_created = 0
        total_nodes_matched = 0

        total_relationships_created = 0
        total_relationships_matched = 0

        # --------------------------------------------------------------
        # Nodes
        # --------------------------------------------------------------

        for node in nodes:

            statement = self._node_merge_statement(
                node.label
            )

            result = tx.run(
                statement,
                node_id=node.node_id,
                properties=dict(node.properties),
            ).single()

            if result is None:
                raise RuntimeError(
                    f"No result returned while writing node "
                    f"{node.node_id!r}."
                )

            total_nodes_created += int(
                result["created"]
            )

            total_nodes_matched += int(
                result["matched"]
            )

        # --------------------------------------------------------------
        # Relationships
        # --------------------------------------------------------------

        for relationship in relationships:

            statement = (
                self._relationship_merge_statement(
                    source_label=relationship.source_label,
                    relationship_type=relationship.relationship_type,
                    target_label=relationship.target_label,
                )
            )

            result = tx.run(
                statement,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                properties=dict(
                    relationship.properties
                ),
            ).single()

            if result is None:
                raise RuntimeError(
                    "No result returned while "
                    "writing relationship."
                )

            if result["source_exists"] == 0:
                raise ValueError(
                    f"Source node "
                    f"{relationship.source_id!r} "
                    f"does not exist."
                )

            if result["target_exists"] == 0:
                raise ValueError(
                    f"Target node "
                    f"{relationship.target_id!r} "
                    f"does not exist."
                )

            total_relationships_created += int(
                result["created"]
            )

            total_relationships_matched += int(
                result["matched"]
            )

        return GraphWriteResult(
            nodes_created=total_nodes_created,
            nodes_matched=total_nodes_matched,
            relationships_created=(
                total_relationships_created
            ),
            relationships_matched=(
                total_relationships_matched
            ),
        )

    # =========================================================================
    # Multiple projections
    # =========================================================================

    def write_projections(
        self,
        projections: Iterable[
            tuple[
                Iterable[GraphNode],
                Iterable[GraphRelationship],
            ]
        ],
    ) -> GraphWriteResult:

        total = GraphWriteResult()

        for nodes, relationships in projections:

            result = self.write_projection(
                nodes,
                relationships,
            )

            total = GraphWriteResult(
                nodes_created=(
                    total.nodes_created
                    + result.nodes_created
                ),
                nodes_matched=(
                    total.nodes_matched
                    + result.nodes_matched
                ),
                relationships_created=(
                    total.relationships_created
                    + result.relationships_created
                ),
                relationships_matched=(
                    total.relationships_matched
                    + result.relationships_matched
                ),
            )

        return total

    # =========================================================================
    # Read operations
    # =========================================================================

    def run_read(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:

        self._ensure_open()

        with self._session() as session:

            result = session.run(
                query,
                parameters or {},
            )

            return [
                record.data()
                for record in result
            ]

    def get_stats(self) -> GraphStats:

        query = """
        MATCH (n)
        WITH count(n) AS node_count

        OPTIONAL MATCH ()-[r]->()

        RETURN
            node_count,
            count(r) AS relationship_count
        """

        rows = self.run_read(query)

        if not rows:
            return GraphStats(
                node_count=0,
                relationship_count=0,
            )

        return GraphStats(
            node_count=int(
                rows[0]["node_count"]
            ),
            relationship_count=int(
                rows[0]["relationship_count"]
            ),
        )

    def get_label_counts(self) -> dict[str, int]:

        query = """
        MATCH (n)
        UNWIND labels(n) AS label

        RETURN
            label,
            count(*) AS count

        ORDER BY label
        """

        rows = self.run_read(query)

        return {
            str(row["label"]): int(
                row["count"]
            )
            for row in rows
        }

    def get_relationship_counts(
        self,
    ) -> dict[str, int]:

        query = """
        MATCH ()-[r]->()

        RETURN
            type(r) AS relationship_type,
            count(*) AS count

        ORDER BY relationship_type
        """

        rows = self.run_read(query)

        return {
            str(row["relationship_type"]): int(
                row["count"]
            )
            for row in rows
        }

    # =========================================================================
    # Visualization
    # =========================================================================

    def get_visualization_subgraph(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        query = """
        MATCH (source)-[relationship]->(target)

        RETURN
            source.node_id AS source_id,
            labels(source) AS source_labels,
            type(relationship) AS relationship_type,
            target.node_id AS target_id,
            labels(target) AS target_labels

        LIMIT $limit
        """

        return self.run_read(
            query,
            {"limit": limit},
        )

    # =========================================================================
    # Explicit maintenance
    # =========================================================================

    def clear_database(self) -> None:
        """
        Explicitly delete all graph data.

        NEVER called automatically.
        """

        self._ensure_open()

        query = """
        MATCH (n)
        DETACH DELETE n
        """

        with self._session() as session:
            session.run(query).consume()