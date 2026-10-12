"""
graph_retriever.py

Production Neo4j Graph-RAG retriever for SupportMind AI.

Responsibilities
----------------
1. Connect to Neo4j through the official Python driver.
2. Execute read-only graph retrieval.
3. Match query terms against the real SupportMind graph schema.
4. Traverse bounded SupportMind relationships.
5. Convert graph evidence into RetrievalResult objects.
6. Preserve graph/document metadata.
7. Assign deterministic graph scores and ranks.
8. Remain independent of LLMs, embeddings, Qdrant and reranking.

Non-responsibilities
--------------------
- No LLM generation.
- No embedding generation.
- No RRF fusion.
- No cross-encoder reranking.
- No CSV ingestion.
- No graph mutation.
- No database deletion.

SupportMind Graph
-----------------

Customer
    └── CREATED ──> Ticket
                         ├── ABOUT ────────> Product
                         ├── HAS_TYPE ─────> TicketType
                         ├── HAS_PRIORITY ─> Priority
                         ├── HAS_STATUS ───> Status
                         └── VIA_CHANNEL ──> Channel
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from neo4j import Driver, GraphDatabase, Session

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalSource,
)


# ==============================================================
# Configuration
# ==============================================================


@dataclass(frozen=True)
class Neo4jConfig:
    """
    Neo4j connection configuration.
    """

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    @classmethod
    def from_environment(cls) -> "Neo4jConfig":
        """
        Build configuration from environment variables.
        """

        return cls(
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

    def validate(self) -> None:
        """
        Validate Neo4j configuration.
        """

        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ValueError(
                "Neo4j URI must be a non-empty string."
            )

        if not isinstance(
            self.username,
            str,
        ) or not self.username.strip():
            raise ValueError(
                "Neo4j username must be a non-empty string."
            )

        if not isinstance(
            self.password,
            str,
        ) or not self.password:
            raise ValueError(
                "Neo4j password must be a non-empty string."
            )

        if not isinstance(
            self.database,
            str,
        ) or not self.database.strip():
            raise ValueError(
                "Neo4j database must be a non-empty string."
            )


# ==============================================================
# Graph Retriever
# ==============================================================


class GraphRetriever:
    """
    Read-only Neo4j Graph-RAG retriever.

    The implementation is deliberately schema-aware.

    It does NOT perform unrestricted graph traversal.

    Retrieval process:

        Query
          ↓
        Tokenization
          ↓
        Search actual SupportMind properties
          ↓
        Identify matching nodes
          ↓
        Retrieve bounded graph evidence
          ↓
        Score
          ↓
        RetrievalResult[]
    """

    SOURCE = RetrievalSource.GRAPH

    # ----------------------------------------------------------
    # Actual SupportMind searchable properties
    # ----------------------------------------------------------

    DEFAULT_SEARCH_PROPERTIES: tuple[str, ...] = (
        "name",
        "email",
        "subject",
        "description",
        "ticket_id",
        "ticket_status",
        "ticket_type",
        "ticket_priority",
        "ticket_channel",
        "product_purchased",
        "source_file",
        "document_id",
        "node_id",
    )

    # ----------------------------------------------------------
    # Actual SupportMind labels
    # ----------------------------------------------------------

    SUPPORTED_LABELS: tuple[str, ...] = (
        "Customer",
        "Ticket",
        "Product",
        "TicketType",
        "Priority",
        "Status",
        "Channel",
    )

    # ----------------------------------------------------------
    # Actual SupportMind relationships
    # ----------------------------------------------------------

    SUPPORTED_RELATIONSHIPS: tuple[str, ...] = (
        "CREATED",
        "ABOUT",
        "HAS_TYPE",
        "HAS_PRIORITY",
        "HAS_STATUS",
        "VIA_CHANNEL",
    )

    _WORD_PATTERN = re.compile(
        r"\b\w+\b",
        re.UNICODE,
    )

    def __init__(
        self,
        driver: Driver | None = None,
        config: Neo4jConfig | None = None,
        search_properties: Sequence[str] | None = None,
        max_hops: int = 2,
    ) -> None:
        """
        Initialize GraphRetriever.
        """

        if not isinstance(
            max_hops,
            int,
        ) or isinstance(
            max_hops,
            bool,
        ):
            raise ValueError(
                "max_hops must be an integer."
            )

        if max_hops not in (
            1,
            2,
        ):
            raise ValueError(
                "max_hops must be either 1 or 2."
            )

        self._config = (
            config
            if config is not None
            else Neo4jConfig.from_environment()
        )

        self._config.validate()

        self._search_properties = (
            self._validate_search_properties(
                search_properties
            )
        )

        self._max_hops = max_hops

        self._driver = (
            driver
            if driver is not None
            else GraphDatabase.driver(
                self._config.uri,
                auth=(
                    self._config.username,
                    self._config.password,
                ),
            )
        )

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def database(self) -> str:
        """
        Return configured Neo4j database.
        """

        return self._config.database

    @property
    def max_hops(self) -> int:
        """
        Return configured graph traversal depth.
        """

        return self._max_hops

    @property
    def search_properties(self) -> tuple[str, ...]:
        """
        Return searchable graph properties.
        """

        return self._search_properties

    # ==========================================================
    # Connectivity
    # ==========================================================

    def verify_connection(self) -> bool:
        """
        Verify Neo4j connectivity.
        """

        try:
            with self._driver.session(
                database=self._config.database
            ) as session:

                session.run(
                    "RETURN 1 AS ok"
                ).consume()

            return True

        except Exception as exc:
          print(
          f"[GraphRetriever] verify_connection failed: "
          f"{type(exc).__name__}: {exc}",
          flush=True,
    )
          return False

    def close(self) -> None:
        """
        Close Neo4j driver.
        """

        self._driver.close()

    # ==========================================================
    # Retrieval
    # ==========================================================

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalResult]:
        """
        Execute graph retrieval.
        """

        self._validate_request(
            request
        )

        query_tokens = self._tokenize(
            request.query
        )

        if not query_tokens:
            return []

        with self._driver.session(
            database=self._config.database
        ) as session:

            records = self._execute_retrieval(
                session=session,
                query_tokens=query_tokens,
                limit=request.top_k,
            )

        results: list[RetrievalResult] = []

        for record in records:

            result = self._convert_record(
                record
            )

            if result is not None:
                results.append(
                    result
                )

        results.sort(
            key=lambda result: (
                -float(result.score),
                result.id,
            )
        )

        final_results: list[RetrievalResult] = []

        for rank, result in enumerate(
            results[: request.top_k],
            start=1,
        ):

            final_results.append(
                result.model_copy(
                    update={
                        "rank": rank,
                    }
                )
            )

        return final_results

    # ==========================================================
    # Cypher Retrieval
    # ==========================================================

    def _execute_retrieval(
        self,
        session: Session,
        query_tokens: Sequence[str],
        limit: int,
    ) -> list[Any]:
        """
        Execute bounded schema-aware graph retrieval.

        Important:
        We deliberately avoid:

            MATCH (n)
            OPTIONAL MATCH (n)-[*1..2]-(related)

        because unrestricted variable-length traversal can generate
        very large path sets on a connected graph.

        Instead we:
            1. search indexed/known properties
            2. identify matching nodes
            3. traverse only known SupportMind relationships
            4. return a bounded number of graph paths
        """

        property_expressions = ", ".join(
            f"n.`{property_name}`"
            for property_name in self._search_properties
        )

        # ------------------------------------------------------
        # One-hop relationships
        # ------------------------------------------------------

        one_hop_query = f"""
        UNWIND $query_tokens AS token

        MATCH (n)

        WITH
            n,
            token,
            [
                value IN [{property_expressions}]
                WHERE value IS NOT NULL
            ] AS property_values

        WITH
            n,
            token,
            any(
                value IN property_values
                WHERE toLower(
                    toString(value)
                ) CONTAINS token
            ) AS token_match

        WHERE token_match

        WITH
            n,
            count(DISTINCT token) AS matched_tokens

        OPTIONAL MATCH path =
            (n)-[
                r:CREATED
                |ABOUT
                |HAS_TYPE
                |HAS_PRIORITY
                |HAS_STATUS
                |VIA_CHANNEL
            ]-(related)

        WITH
            n,
            matched_tokens,
            collect(
                DISTINCT path
            )[0..20] AS paths

        RETURN
            elementId(n) AS neo4j_element_id,
            labels(n) AS labels,
            properties(n) AS node_properties,
            matched_tokens,
            size(paths) AS graph_path_count,
            [
                path IN paths
                WHERE path IS NOT NULL
                |
                {{
                    length: length(path),
                    relationships: [
                        relationship IN relationships(path)
                        |
                        {{
                            type: type(relationship),
                            properties: properties(relationship)
                        }}
                    ],
                    nodes: [
                        node IN nodes(path)
                        |
                        {{
                            element_id: elementId(node),
                            labels: labels(node),
                            properties: properties(node)
                        }}
                    ]
                }}
            ] AS graph_paths

        ORDER BY
            matched_tokens DESC,
            graph_path_count DESC,
            neo4j_element_id ASC

        LIMIT $limit
        """

        records = session.run(
            one_hop_query,
            query_tokens=[
                token.lower()
                for token in query_tokens
            ],
            limit=int(limit),
        )

        return list(records)

    # ==========================================================
    # Result Conversion
    # ==========================================================

    @classmethod
    def _convert_record(
        cls,
        record: Any,
    ) -> RetrievalResult | None:
        """
        Convert Neo4j record into RetrievalResult.
        """

        properties = dict(
            record.get(
                "node_properties",
                {},
            )
            or {}
        )

        neo4j_element_id = record.get(
            "neo4j_element_id"
        )

        if not neo4j_element_id:
            return None

        node_id = cls._first_non_empty(
            properties,
            (
                "node_id",
            ),
            default=str(
                neo4j_element_id
            ),
        )

        document_id = cls._first_non_empty(
            properties,
            (
                "document_id",
                "source_file",
            ),
            default=str(
                node_id
            ),
        )

        text = cls._first_non_empty(
            properties,
            (
                "description",
                "subject",
                "name",
                "product_purchased",
                "ticket_id",
            ),
            default="",
        )

        matched_tokens = int(
            record.get(
                "matched_tokens",
                0,
            )
            or 0
        )

        graph_path_count = int(
            record.get(
                "graph_path_count",
                0,
            )
            or 0
        )

        relationship_count = cls._count_relationships(
            record.get(
                "graph_paths",
                [],
            )
            or []
        )

        score = cls._calculate_graph_score(
            matched_tokens=matched_tokens,
            relationship_count=relationship_count,
            graph_path_count=graph_path_count,
        )

        metadata = dict(
            properties
        )

        metadata["neo4j_element_id"] = str(
            neo4j_element_id
        )

        metadata["graph_labels"] = list(
            record.get(
                "labels",
                [],
            )
            or []
        )

        metadata["matched_tokens"] = (
            matched_tokens
        )

        metadata["relationship_count"] = (
            relationship_count
        )

        metadata["graph_path_count"] = (
            graph_path_count
        )

        metadata["graph_paths"] = (
            record.get(
                "graph_paths",
                [],
            )
            or []
        )

        return RetrievalResult(
            id=str(
                node_id
            ),
            node_id=str(
                node_id
            ),
            document_id=str(
                document_id
            ),
            text=str(
                text
            ),
            score=score,
            rank=1,
            source=cls.SOURCE,
            metadata=metadata,
            retrieval_scores={
                "graph_score": score,
                "matched_tokens": float(
                    matched_tokens
                ),
                "relationship_count": float(
                    relationship_count
                ),
                "graph_path_count": float(
                    graph_path_count
                ),
            },
        )

    # ==========================================================
    # Graph Scoring
    # ==========================================================

    @staticmethod
    def _calculate_graph_score(
        matched_tokens: int,
        relationship_count: int,
        graph_path_count: int,
    ) -> float:
        """
        Calculate deterministic graph relevance.

        Formula:

            score =
                matched_tokens
                +
                0.05 * relationship_count
                +
                0.25 * graph_path_count

        Token matching remains the strongest signal.
        Graph connectivity provides supporting evidence.
        """

        if matched_tokens < 0:
            raise ValueError(
                "matched_tokens cannot be negative."
            )

        if relationship_count < 0:
            raise ValueError(
                "relationship_count cannot be negative."
            )

        if graph_path_count < 0:
            raise ValueError(
                "graph_path_count cannot be negative."
            )

        return float(
            matched_tokens
            + (
                0.05
                * relationship_count
            )
            + (
                0.25
                * graph_path_count
            )
        )

    # ==========================================================
    # Graph Evidence Helpers
    # ==========================================================

    @staticmethod
    def _count_relationships(
        graph_paths: Sequence[Any],
    ) -> int:
        """
        Count relationships represented in graph paths.
        """

        count = 0

        for path in graph_paths:

            if not isinstance(
                path,
                dict,
            ):
                continue

            relationships = path.get(
                "relationships",
                [],
            )

            if isinstance(
                relationships,
                list,
            ):
                count += len(
                    relationships
                )

        return count

    # ==========================================================
    # Tokenization
    # ==========================================================

    @classmethod
    def _tokenize(
        cls,
        text: str,
    ) -> list[str]:
        """
        Deterministically tokenize query.
        """

        if not isinstance(
            text,
            str,
        ):
            raise ValueError(
                "Text must be a string."
            )

        return [
            token.lower()
            for token in cls._WORD_PATTERN.findall(
                text
            )
        ]

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_request(
        request: RetrievalRequest,
    ) -> None:
        """
        Validate RetrievalRequest.
        """

        if request is None:
            raise ValueError(
                "Retrieval request cannot be None."
            )

        if not isinstance(
            request,
            RetrievalRequest,
        ):
            raise ValueError(
                "request must be a RetrievalRequest instance."
            )

    @classmethod
    def _validate_search_properties(
        cls,
        properties: Sequence[str] | None,
    ) -> tuple[str, ...]:
        """
        Validate searchable property names.
        """

        if properties is None:
            return cls.DEFAULT_SEARCH_PROPERTIES

        if isinstance(
            properties,
            (str, bytes),
        ):
            raise ValueError(
                "search_properties must be a sequence of strings."
            )

        validated: list[str] = []

        for property_name in properties:

            if (
                not isinstance(
                    property_name,
                    str,
                )
                or not property_name.strip()
            ):
                raise ValueError(
                    "Every search property must be "
                    "a non-empty string."
                )

            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*",
                property_name,
            ):
                raise ValueError(
                    f"Invalid Neo4j property name: "
                    f"{property_name!r}"
                )

            validated.append(
                property_name
            )

        if not validated:
            raise ValueError(
                "At least one search property is required."
            )

        return tuple(
            dict.fromkeys(
                validated
            )
        )

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def _first_non_empty(
        properties: dict[str, Any],
        keys: Sequence[str],
        default: str,
    ) -> str:
        """
        Return first non-empty property value.
        """

        for key in keys:

            value = properties.get(
                key
            )

            if (
                value is not None
                and str(value).strip()
            ):
                return str(
                    value
                )

        return default


# ==============================================================
# Application Factory
# ==============================================================


def create_graph_retriever(
    config: Neo4jConfig | None = None,
) -> GraphRetriever:
    """
    Create application-level GraphRetriever.
    """

    return GraphRetriever(
        config=config
    )