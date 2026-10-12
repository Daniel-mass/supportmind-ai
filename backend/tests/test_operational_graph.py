import pytest

from backend.app.intelligence.graph.graph_builder import (
    GraphNode,
    GraphRelationship,
    OperationalGraph,
)
from backend.app.intelligence.graph.graph_queries import (
    GET_TICKET,
)
from backend.app.intelligence.graph.operational_graph import (
    Neo4jConfig,
    Neo4jOperationalGraph,
)


class FakeResult:
    def __init__(
        self,
        records=None,
        single_record=None,
    ):
        self._records = records or []

        # If an explicit single record was not supplied,
        # emulate Neo4j's .single() by returning the first record.
        if single_record is not None:
            self._single_record = single_record
        elif self._records:
            self._single_record = self._records[0]
        else:
            self._single_record = None

    def data(self):
        return self._records

    def single(self):
        return self._single_record

    def consume(self):
        return None


class FakeTransaction:
    def __init__(self):
        self.queries = []

    def run(
        self,
        query,
        **parameters,
    ):
        self.queries.append(
            (
                query,
                parameters,
            )
        )

        return FakeResult()


class FakeSession:
    def __init__(self):
        self.transaction = (
            FakeTransaction()
        )

        self.queries = []

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def run(
        self,
        query,
        parameters=None,
    ):
        self.queries.append(
            (
                query,
                parameters or {},
            )
        )

        # Emulate the specific health-check response
        # expected from operational_graph.py.
        if (
            "AS database" in query
            and "AS healthy" in query
        ):
            return FakeResult(
                records=[
                    {
                        "database": "neo4j",
                        "healthy": 1,
                    }
                ],
            )

        # Generic query response used by test_execute_query.
        return FakeResult(
            records=[
                {
                    "value": 1,
                }
            ],
        )

    def execute_write(
        self,
        callback,
        *args,
    ):
        return callback(
            self.transaction,
            *args,
        )


class FakeDriver:
    def __init__(self):
        self.session_instance = (
            FakeSession()
        )

        self.verified = False
        self.closed = False

    def session(
        self,
        database=None,
    ):
        return self.session_instance

    def verify_connectivity(self):
        self.verified = True

    def close(self):
        self.closed = True


def make_store():
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
        database="neo4j",
    )

    driver = FakeDriver()

    store = Neo4jOperationalGraph(
        config=config,
        driver=driver,
    )

    return store, driver


def test_neo4j_config_from_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "NEO4J_URI",
        "bolt://localhost:7687",
    )

    monkeypatch.setenv(
        "NEO4J_USERNAME",
        "neo4j",
    )

    monkeypatch.setenv(
        "NEO4J_PASSWORD",
        "secret",
    )

    monkeypatch.setenv(
        "NEO4J_DATABASE",
        "neo4j",
    )

    config = (
        Neo4jConfig
        .from_environment()
    )

    assert (
        config.uri
        == "bolt://localhost:7687"
    )

    assert (
        config.username
        == "neo4j"
    )

    assert (
        config.password
        == "secret"
    )

    assert (
        config.database
        == "neo4j"
    )


def test_missing_password_raises_error(
    monkeypatch,
):
    monkeypatch.delenv(
        "NEO4J_PASSWORD",
        raising=False,
    )

    monkeypatch.setenv(
        "NEO4J_URI",
        "bolt://localhost:7687",
    )

    monkeypatch.setenv(
        "NEO4J_USERNAME",
        "neo4j",
    )

    with pytest.raises(
        RuntimeError,
        match="NEO4J_PASSWORD",
    ):
        Neo4jConfig.from_environment()


def test_verify_connectivity():
    store, driver = make_store()

    assert (
        store.verify_connectivity()
        is True
    )

    assert driver.verified is True


def test_close_closes_driver():
    store, driver = make_store()

    store.close()

    assert driver.closed is True


def test_node_label_validation():
    store, _ = make_store()

    assert (
        store._validate_node_label(
            "Ticket"
        )
        == "Ticket"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Neo4j node label",
    ):
        store._validate_node_label(
            "UnknownLabel"
        )


def test_relationship_type_validation():
    store, _ = make_store()

    assert (
        store._validate_relationship_type(
            "ASSIGNED_TO"
        )
        == "ASSIGNED_TO"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Neo4j relationship type",
    ):
        store._validate_relationship_type(
            "UNKNOWN_RELATIONSHIP"
        )


def test_none_properties_are_removed():
    properties = {
        "ticket_id": "PROJ:1",
        "status": None,
        "priority": "High",
    }

    cleaned = (
        Neo4jOperationalGraph
        ._clean_properties(
            properties
        )
    )

    assert cleaned == {
        "ticket_id": "PROJ:1",
        "priority": "High",
    }


def test_upsert_graph_persists_nodes_and_relationships():
    store, driver = make_store()

    graph = OperationalGraph(
        nodes=[
            GraphNode(
                node_id="ticket:PROJ:1",
                label="Ticket",
                properties={
                    "ticket_id": "PROJ:1",
                },
            ),
            GraphNode(
                node_id="agent:agent-001",
                label="Agent",
                properties={
                    "agent_id": "agent-001",
                },
            ),
        ],
        relationships=[
            GraphRelationship(
                relationship_id="assigned:1",
                relationship_type="ASSIGNED_TO",
                source_id="ticket:PROJ:1",
                target_id="agent:agent-001",
                properties={},
            )
        ],
    )

    result = store.upsert_graph(
        graph,
        ensure_constraints=False,
    )

    assert result == {
        "nodes": 2,
        "relationships": 1,
    }

    # Two node persistence queries + one relationship persistence query.
    assert len(
        driver
        .session_instance
        .transaction
        .queries
    ) == 3


def test_execute_query():
    store, driver = make_store()

    records = store.execute_query(
        GET_TICKET,
        {
            "ticket_id": "PROJ:1",
        },
    )

    assert records == [
        {
            "value": 1,
        }
    ]

    assert len(
        driver
        .session_instance
        .queries
    ) == 1


def test_health_check():
    store, driver = make_store()

    result = store.health_check()

    assert result["healthy"] is True

    assert (
        result["database"]
        == "neo4j"
    )

    assert (
        result["uri"]
        == "bolt://localhost:7687"
    )

    assert driver.verified is True