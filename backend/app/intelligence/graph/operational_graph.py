from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from app.intelligence.graph.graph_builder import (
    GraphNode,
    GraphRelationship,
    OperationalGraph,
)
from app.intelligence.graph.graph_queries import (
    GraphQuery,
)


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Neo4jConfig:
    """
    Neo4j connection configuration.

    Credentials are loaded from environment variables and are never
    hard-coded in the source code.
    """

    uri: str
    username: str
    password: str
    database: str

    @classmethod
    def from_environment(
        cls,
    ) -> "Neo4jConfig":
        uri = os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        )

        username = os.getenv(
            "NEO4J_USERNAME",
            "neo4j",
        )

        password = os.getenv(
            "NEO4J_PASSWORD",
        )

        database = os.getenv(
            "NEO4J_DATABASE",
            "neo4j",
        )

        if not password:
            raise RuntimeError(
                "NEO4J_PASSWORD is not configured. "
                "Set it in the project .env file."
            )

        return cls(
            uri=uri,
            username=username,
            password=password,
            database=database,
        )


# ---------------------------------------------------------------------
# Neo4j graph store
# ---------------------------------------------------------------------

class Neo4jOperationalGraph:
    """
    Neo4j persistence and query service for the SupportMind
    operational graph.

    Responsibilities:
    - connect to Neo4j
    - verify connectivity
    - create node constraints
    - persist graph nodes
    - persist graph relationships
    - execute registered Cypher queries

    Does not:
    - build graph semantics
    - perform ML/NLP
    - perform visualization
    - contain dashboard logic
    """

    NODE_LABELS = {
        "Ticket",
        "Agent",
        "WorkflowEvent",
        "IssueType",
        "Priority",
        "Status",
        "Team",
    }

    RELATIONSHIP_TYPES = {
        "HAS_TYPE",
        "HAS_PRIORITY",
        "HAS_STATUS",
        "ASSIGNED_TO",
        "HAS_WORKFLOW_EVENT",
        "NEXT",
        "MEMBER_OF",
        "HANDOFF_TO",
    }

    IDENTIFIER_PATTERN = re.compile(
        r"^[A-Za-z_][A-Za-z0-9_]*$"
    )

    NODE_BATCH_SIZE = 1000
    RELATIONSHIP_BATCH_SIZE = 1000

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        driver: Driver | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else Neo4jConfig.from_environment()
        )

        self._driver = (
            driver
            if driver is not None
            else GraphDatabase.driver(
                self.config.uri,
                auth=(
                    self.config.username,
                    self.config.password,
                ),
            )
        )

    # -----------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------

    def verify_connectivity(self) -> bool:
        """
        Verify that Neo4j is reachable.
        """
        self._driver.verify_connectivity()
        return True

    def close(self) -> None:
        """
        Close the Neo4j driver.
        """
        self._driver.close()

    # -----------------------------------------------------------------
    # Schema / constraints
    # -----------------------------------------------------------------

    def ensure_constraints(self) -> None:
        """
        Create node-id uniqueness constraints if they do not exist.
        """

        statements = [
            """
            CREATE CONSTRAINT ticket_node_id_unique
            IF NOT EXISTS
            FOR (n:Ticket)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT agent_node_id_unique
            IF NOT EXISTS
            FOR (n:Agent)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT workflow_event_node_id_unique
            IF NOT EXISTS
            FOR (n:WorkflowEvent)
            REQUIRE n.node_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT issue_type_node_id_unique
            IF NOT EXISTS
            FOR (n:IssueType)
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
            CREATE CONSTRAINT team_node_id_unique
            IF NOT EXISTS
            FOR (n:Team)
            REQUIRE n.node_id IS UNIQUE
            """,
        ]

        with self._driver.session(
            database=self.config.database
        ) as session:
            for statement in statements:
                session.run(statement)

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    @classmethod
    def _validate_node_label(
        cls,
        label: str,
    ) -> str:
        if label not in cls.NODE_LABELS:
            raise ValueError(
                f"Unsupported Neo4j node label: {label}"
            )

        if not cls.IDENTIFIER_PATTERN.fullmatch(
            label
        ):
            raise ValueError(
                f"Invalid Neo4j node label: {label}"
            )

        return label

    @classmethod
    def _validate_relationship_type(
        cls,
        relationship_type: str,
    ) -> str:
        if (
            relationship_type
            not in cls.RELATIONSHIP_TYPES
        ):
            raise ValueError(
                "Unsupported Neo4j relationship type: "
                f"{relationship_type}"
            )

        if not cls.IDENTIFIER_PATTERN.fullmatch(
            relationship_type
        ):
            raise ValueError(
                "Invalid Neo4j relationship type: "
                f"{relationship_type}"
            )

        return relationship_type

    @staticmethod
    def _clean_properties(
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Remove None-valued properties because Neo4j properties
        should represent actual known values only.
        """
        return {
            key: value
            for key, value in properties.items()
            if value is not None
        }

    @classmethod
    def _node_row(
        cls,
        node: GraphNode,
    ) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "properties": cls._clean_properties(
                node.properties
            ),
        }

    @classmethod
    def _relationship_row(
        cls,
        relationship: GraphRelationship,
    ) -> dict[str, Any]:
        return {
            "relationship_id": (
                relationship.relationship_id
            ),
            "source_id": (
                relationship.source_id
            ),
            "target_id": (
                relationship.target_id
            ),
            "properties": cls._clean_properties(
                relationship.properties
            ),
        }

    # -----------------------------------------------------------------
    # Node persistence
    # -----------------------------------------------------------------

    @classmethod
    def _merge_nodes_transaction(
        cls,
        tx,
        label: str,
        rows: list[dict[str, Any]],
    ) -> None:
        cls._validate_node_label(label)

        query = f"""
        UNWIND $rows AS row
        MERGE (n:{label} {{node_id: row.node_id}})
        SET n += row.properties
        """

        tx.run(
            query,
            rows=rows,
        ).consume()

    def _persist_nodes(
        self,
        nodes: list[GraphNode],
    ) -> None:
        nodes_by_label: dict[
            str,
            list[GraphNode],
        ] = defaultdict(list)

        for node in nodes:
            nodes_by_label[
                node.label
            ].append(node)

        with self._driver.session(
            database=self.config.database
        ) as session:

            for label, label_nodes in (
                nodes_by_label.items()
            ):
                for start in range(
                    0,
                    len(label_nodes),
                    self.NODE_BATCH_SIZE,
                ):
                    batch = label_nodes[
                        start:
                        start + self.NODE_BATCH_SIZE
                    ]

                    rows = [
                        self._node_row(
                            node
                        )
                        for node in batch
                    ]

                    session.execute_write(
                        self._merge_nodes_transaction,
                        label,
                        rows,
                    )

    # -----------------------------------------------------------------
    # Relationship persistence
    # -----------------------------------------------------------------

    @classmethod
    def _merge_relationships_transaction(
        cls,
        tx,
        relationship_type: str,
        rows: list[dict[str, Any]],
    ) -> None:
        cls._validate_relationship_type(
            relationship_type
        )

        query = f"""
        UNWIND $rows AS row

        MATCH (source {{
            node_id: row.source_id
        }})

        MATCH (target {{
            node_id: row.target_id
        }})

        MERGE (
            source
        )-[r:{relationship_type} {{
            relationship_id: row.relationship_id
        }}]->(
            target
        )

        SET r += row.properties
        """

        tx.run(
            query,
            rows=rows,
        ).consume()

    def _persist_relationships(
        self,
        relationships: list[
            GraphRelationship
        ],
    ) -> None:
        relationships_by_type: dict[
            str,
            list[GraphRelationship],
        ] = defaultdict(list)

        for relationship in relationships:
            relationships_by_type[
                relationship.relationship_type
            ].append(relationship)

        with self._driver.session(
            database=self.config.database
        ) as session:

            for (
                relationship_type,
                typed_relationships,
            ) in relationships_by_type.items():

                for start in range(
                    0,
                    len(typed_relationships),
                    self.RELATIONSHIP_BATCH_SIZE,
                ):
                    batch = typed_relationships[
                        start:
                        start
                        + self.RELATIONSHIP_BATCH_SIZE
                    ]

                    rows = [
                        self._relationship_row(
                            relationship
                        )
                        for relationship in batch
                    ]

                    session.execute_write(
                        self._merge_relationships_transaction,
                        relationship_type,
                        rows,
                    )

    # -----------------------------------------------------------------
    # Public persistence API
    # -----------------------------------------------------------------

    def upsert_graph(
        self,
        graph: OperationalGraph,
        *,
        ensure_constraints: bool = True,
    ) -> dict[str, int]:
        """
        Upsert an entire operational graph into Neo4j.

        Returns the number of graph nodes and relationships submitted.
        MERGE keeps repeated runs idempotent.
        """

        if ensure_constraints:
            self.ensure_constraints()

        self._persist_nodes(
            graph.nodes
        )

        self._persist_relationships(
            graph.relationships
        )

        return {
            "nodes": len(
                graph.nodes
            ),
            "relationships": len(
                graph.relationships
            ),
        }

    # -----------------------------------------------------------------
    # Query execution
    # -----------------------------------------------------------------

    def execute_query(
        self,
        query: GraphQuery,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a parameterized GraphQuery and return records as dicts.
        """

        with self._driver.session(
            database=self.config.database
        ) as session:
            result = session.run(
                query.cypher,
                parameters or {},
            )

            return result.data()
    def execute_cypher(
       self,
       cypher: str,
       parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
      """
    Execute raw parameterized Cypher.

    Intended for internal operational graph integration
    and future service-layer wrappers.
    """

      with self._driver.session(
           database=self.config.database
    ) as session:
         result = session.run(
            cypher,
            parameters or {},
        )

         return result.data()
    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """
        Return a lightweight operational health result.
        """

        self.verify_connectivity()

        with self._driver.session(
            database=self.config.database
        ) as session:
            record = session.run(
                """
                RETURN
                    "neo4j" AS database,
                    1 AS healthy
                """
            ).single()

        return {
            "healthy": bool(
                record["healthy"]
            )
            if record
            else False,
            "database": (
                record["database"]
                if record
                else self.config.database
            ),
            "uri": self.config.uri,
        }


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def create_neo4j_operational_graph() -> Neo4jOperationalGraph:
    """
    Create the production Neo4j operational graph service
    from the existing project environment variables.
    """
    return Neo4jOperationalGraph()
