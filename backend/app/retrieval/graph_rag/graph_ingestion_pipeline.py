"""
Graph-specific ingestion pipeline for SupportMind AI.

Purpose
-------
Build the Neo4j graph from structured CSV source data without modifying
the shared CSVLoader or the common vector ingestion pipeline.

Flow
----

FilesystemConnector
        ↓
LoaderRegistry
        ↓
CSVLoader
        ↓
IngestedDocument
        ↓
ParserPipeline
        ↓
Graph-specific original CSV reader
        ↓
GraphProjector
        ↓
Neo4jStore

Important
---------
The shared CSVLoader remains unchanged.

CSVLoader converts CSV data into text because that representation is useful
for the common ingestion/RAG pipeline.

Neo4j requires the original structured columns, so this graph-specific
branch reads the original CSV directly with csv.DictReader.

SupportMind AI
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from app.ingestion.connectors.filesystem_connector import (
    FilesystemConnector,
)

from app.ingestion.parsers.parser_pipeline import (
    ParserPipeline,
)

from app.retrieval.graph_rag.graph_projector import (
    GraphProjector,
    ProjectionStats,
)

from app.retrieval.graph_rag.neo4j_store import (
    Neo4jStore,
)


logger = logging.getLogger(
    "supportmind.graph_ingestion"
)


# ============================================================================
# Result
# ============================================================================


class GraphIngestionResult:
    """Result of one graph ingestion run."""

    def __init__(
        self,
        source_directory: str,
    ) -> None:

        self.source_directory = (
            source_directory
        )

        self.files_discovered = 0

        self.csv_files_processed = 0
        self.csv_files_failed = 0

        self.rows_read = 0
        self.rows_projected = 0
        self.rows_skipped = 0

        self.nodes_created = 0
        self.nodes_matched = 0

        self.relationships_created = 0
        self.relationships_matched = 0

        self.validation_errors = 0

        self.errors: list[str] = []

        self.metadata: dict[
            str,
            Any,
        ] = {}

        self.success = False


# ============================================================================
# Pipeline
# ============================================================================


class GraphIngestionPipeline:
    """
    Graph-specific ingestion pipeline.

    The common ingestion pipeline remains untouched.

    This pipeline exists specifically for projecting structured
    source data into Neo4j.
    """

    def __init__(
        self,
        *,
        neo4j_store: Neo4jStore,
        filesystem_connector: (
            FilesystemConnector | None
        ) = None,
        parser_pipeline: (
            ParserPipeline | None
        ) = None,
        batch_size: int = 500,
    ) -> None:

        self.neo4j_store = neo4j_store

        self.filesystem_connector = (
            filesystem_connector
            if filesystem_connector is not None
            else FilesystemConnector()
        )

        self.parser_pipeline = (
            parser_pipeline
            if parser_pipeline is not None
            else ParserPipeline()
        )

        self.graph_projector = GraphProjector(
            store=neo4j_store,
            batch_size=batch_size,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def run(
        self,
        source_directory: str,
    ) -> GraphIngestionResult:
        """
        Execute graph-specific ingestion.
        """

        result = GraphIngestionResult(
            source_directory=source_directory
        )

        source_path = Path(
            source_directory
        )

        if not source_path.exists():

            result.errors.append(
                f"Source directory not found: "
                f"{source_path}"
            )

            return result

        if not source_path.is_dir():

            result.errors.append(
                f"Source path is not a directory: "
                f"{source_path}"
            )

            return result

        try:

            logger.info(
                "Starting graph ingestion: %s",
                source_path,
            )

            documents = (
                self.filesystem_connector
                .ingest_directory(
                    str(source_path)
                )
            )

            result.files_discovered = (
                len(documents)
            )

            if not documents:

                raise RuntimeError(
                    "No supported documents discovered."
                )

            for document in documents:

                if (
                    document.source.lower()
                    != "csv"
                ):
                    logger.debug(
                        "Skipping non-CSV source: %s",
                        document.source_file,
                    )
                    continue

                try:

                    self._process_csv_document(
                        document=document,
                        source_directory=source_path,
                        result=result,
                    )

                    result.csv_files_processed += 1

                except Exception as exc:

                    result.csv_files_failed += 1

                    error = (
                        f"{document.source_file}: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    result.errors.append(
                        error
                    )

                    logger.exception(
                        "Graph ingestion failed for %s",
                        document.source_file,
                    )

            result.success = (
                result.csv_files_processed > 0
                and result.csv_files_failed == 0
                and result.rows_projected > 0
            )

        except Exception as exc:

            result.success = False

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            result.errors.append(
                error
            )

            logger.exception(
                "Graph ingestion pipeline failed."
            )

        return result

    # =========================================================================
    # CSV processing
    # =========================================================================

    def _process_csv_document(
        self,
        *,
        document: Any,
        source_directory: Path,
        result: GraphIngestionResult,
    ) -> None:
        """
        Process one CSV document.
        """

        csv_path = (
            source_directory
            / document.source_file
        )

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Original CSV file not found: "
                f"{csv_path}"
            )

        logger.info(
            "Processing graph CSV: %s",
            csv_path,
        )

        # ------------------------------------------------------------------
        # Execute common parser pipeline.
        #
        # This proves the document successfully passed through the existing
        # ingestion architecture.
        # ------------------------------------------------------------------

        parsed_document = (
            self.parser_pipeline.run(
                document
            )
        )

        table_rows = (
            parsed_document.metadata.get(
                "table",
                [],
            )
        )

        result.metadata[
            f"{document.source_file}:parser_table_rows"
        ] = len(table_rows)

        result.metadata[
            f"{document.source_file}:document_id"
        ] = document.document_id

        # ------------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT reconstruct rows from document.content.
        #
        # CSVLoader intentionally converts the CSV into text.
        # The graph branch needs the original structured columns.
        # ------------------------------------------------------------------

        rows = self._read_structured_csv(
            csv_path
        )

        if not rows:

            raise ValueError(
                f"CSV contains no rows: "
                f"{csv_path}"
            )

        logger.info(
            "Structured CSV loaded: %s | rows=%d",
            csv_path.name,
            len(rows),
        )

        # ------------------------------------------------------------------
        # Graph projection
        # ------------------------------------------------------------------

        projection_stats = (
            self.graph_projector.project_rows(
                rows=rows,
                source_file=document.source_file,
                document_id=document.document_id,
            )
        )

        self._merge_projection_stats(
            result=result,
            stats=projection_stats,
        )

        result.metadata[
            f"{document.source_file}:columns"
        ] = list(rows[0].keys())

        result.metadata[
            f"{document.source_file}:rows"
        ] = len(rows)

    # =========================================================================
    # Structured CSV reader
    # =========================================================================

    @staticmethod
    def _read_structured_csv(
        csv_path: Path,
    ) -> list[dict[str, str]]:
        """
        Read original CSV structure without altering CSVLoader.

        This is intentionally graph-pipeline-specific.
        """

        with csv_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            if not reader.fieldnames:

                raise ValueError(
                    f"CSV has no header: "
                    f"{csv_path}"
                )

            rows: list[
                dict[str, str]
            ] = []

            for row in reader:

                cleaned_row = {
                    str(key).strip(): (
                        value.strip()
                        if isinstance(
                            value,
                            str,
                        )
                        else value
                    )
                    for key, value in row.items()
                    if key is not None
                }

                rows.append(
                    cleaned_row
                )

            return rows

    # =========================================================================
    # Statistics
    # =========================================================================

    @staticmethod
    def _merge_projection_stats(
        *,
        result: GraphIngestionResult,
        stats: ProjectionStats,
    ) -> None:

        result.rows_read += (
            stats.rows_read
        )

        result.rows_projected += (
            stats.rows_projected
        )

        result.rows_skipped += (
            stats.rows_skipped
        )

        result.validation_errors += (
            stats.validation_errors
        )

        result.nodes_created += (
            stats.nodes_created
        )

        result.nodes_matched += (
            stats.nodes_matched
        )

        result.relationships_created += (
            stats.relationships_created
        )

        result.relationships_matched += (
            stats.relationships_matched
        )