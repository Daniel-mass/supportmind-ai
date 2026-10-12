"""
ingestion_orchestrator.py

End-to-end orchestration layer for SupportMind AI.

Ingestion pipeline:

Filesystem
    ↓
Loader Registry
    ↓
Parser Pipeline
    ↓
Preprocessing Pipeline
    ↓
Chunking Pipeline
    ↓
Embedding Pipeline
    ↓
Vector Pipeline
    ↓
Qdrant

Retrieval pipeline:

User Query
    ↓
Query Embedding
    ↓
RetrievalService
    ↓
Dense + Sparse
    ↓
RRF
    ↓
Optional Graph-RAG
    ↓
Optional Reranking
    ↓
ContextBuilder
    ↓
BuiltContext

The orchestrator connects existing components.
It does not implement provider-specific algorithms.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.ingestion.connectors.filesystem_connector import (
    FilesystemConnector,
)

from app.ingestion.parsers.parser_pipeline import (
    ParserPipeline,
)

from app.ingestion.preprocessors.preprocessing_pipeline import (
    PreprocessingPipeline,
)

from app.ingestion.chunking.chunking_pipeline import (
    ChunkingPipeline,
)

from app.embeddings.embedding_pipeline import (
    EmbeddingPipeline,
)

from app.vector_store.vector_pipeline import (
    VectorPipeline,
)

from app.vector_store.vector_registry import (
    vector_registry,
)

from app.vector_store.qdrant_store import (
    QdrantStore,
)

from app.orchestrator.orchestration_config import (
    OrchestrationConfig,
)

from app.orchestrator.orchestration_schema import (
    OrchestrationResult,
    StageMetric,
)

from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)


logger = logging.getLogger(
    "supportmind.orchestrator"
)


class IngestionOrchestrator:
    """
    Coordinates the complete SupportMind ingestion workflow
    and exposes retrieval through the RetrievalService.

    Responsibilities:
        - document ingestion
        - preprocessing
        - chunking
        - embedding
        - vector persistence
        - query embedding
        - retrieval orchestration

    Non-responsibilities:
        - retrieval algorithms
        - BM25 implementation
        - RRF implementation
        - graph traversal
        - reranking implementation
        - LLM generation
        - query routing
    """

    def __init__(
        self,
        config: OrchestrationConfig | None = None,
        filesystem_connector: FilesystemConnector | None = None,
        parser_pipeline: ParserPipeline | None = None,
        preprocessing_pipeline: PreprocessingPipeline | None = None,
        chunking_pipeline: ChunkingPipeline | None = None,
        embedding_pipeline: EmbeddingPipeline | None = None,
        vector_pipeline: VectorPipeline | None = None,
        retrieval_service: RetrievalService | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else OrchestrationConfig()
        )

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

        self.preprocessing_pipeline = (
            preprocessing_pipeline
            if preprocessing_pipeline is not None
            else PreprocessingPipeline()
        )

        self.chunking_pipeline = (
            chunking_pipeline
            if chunking_pipeline is not None
            else ChunkingPipeline(
                default_strategy=self.config.chunking_strategy
            )
        )

        self.embedding_pipeline = (
            embedding_pipeline
            if embedding_pipeline is not None
            else EmbeddingPipeline(
                default_provider=self.config.embedding_provider
            )
        )

        self.vector_pipeline = (
            vector_pipeline
            if vector_pipeline is not None
            else VectorPipeline(
                registry=vector_registry
            )
        )

        # ------------------------------------------------------
        # Retrieval Service
        # ------------------------------------------------------

        self.retrieval_service = retrieval_service

    # ==========================================================
    # Public API
    # ==========================================================

    def run(
        self,
        source_directory: str,
    ) -> OrchestrationResult:

        result = OrchestrationResult(
            source_directory=source_directory
        )

        started = time.perf_counter()

        logger.info(
            "Starting SupportMind E2E ingestion: %s",
            source_directory,
        )

        try:

            # --------------------------------------------------
            # Stage 1
            # Filesystem → Loader → IngestedDocument
            # --------------------------------------------------

            documents = self._run_stage(
                result=result,
                stage_name="filesystem_ingestion",
                function=lambda: (
                    self.filesystem_connector
                    .ingest_directory(
                        source_directory
                    )
                ),
            )

            result.documents_discovered = len(
                documents
            )

            if not documents:

                raise RuntimeError(
                    "No supported documents were discovered."
                )

            # --------------------------------------------------
            # Stage 2
            # Qdrant provider registration
            # --------------------------------------------------

            self._register_vector_provider()

            # --------------------------------------------------
            # Stage 3
            # Diagnose collection state
            # --------------------------------------------------

            self._diagnose_collection_before_ingestion(
                result=result
            )

            # --------------------------------------------------
            # Stage 4
            # Process documents
            # --------------------------------------------------

            all_embedded_nodes = []

            for document in documents:

                try:

                    logger.info(
                        "Processing document: %s",
                        document.source_file,
                    )

                    embedded_nodes = (
                        self._process_document(
                            document
                        )
                    )

                    all_embedded_nodes.extend(
                        embedded_nodes
                    )

                    result.documents_processed += 1

                    logger.info(
                        "Document processed successfully: %s | "
                        "embedded nodes=%d",
                        document.source_file,
                        len(embedded_nodes),
                    )

                except Exception as exc:

                    result.documents_failed += 1

                    error = (
                        f"{document.source_file}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    result.errors.append(
                        error
                    )

                    logger.exception(
                        "Document processing failed: %s",
                        document.source_file,
                    )

            if not all_embedded_nodes:

                raise RuntimeError(
                    "No embeddings were produced."
                )

            # --------------------------------------------------
            # Stage 5
            # Current ingestion counts
            # --------------------------------------------------

            result.chunks_created = len(
                all_embedded_nodes
            )

            result.embedding_provider = (
                self.config.embedding_provider
            )

            result.vector_provider = (
                self.config.vector_provider
            )

            result.collection_name = (
                self.config.collection_name
            )

            # --------------------------------------------------
            # Diagnose generated node IDs
            # --------------------------------------------------

            self._diagnose_embedded_nodes(
                result=result,
                embedded_nodes=all_embedded_nodes,
            )

            # --------------------------------------------------
            # Stage 6
            # Embedding dimension
            # --------------------------------------------------

            first_node = all_embedded_nodes[0]

            dimension = (
                first_node.embedding_dimension
            )

            result.embedding_dimension = dimension

            # --------------------------------------------------
            # Stage 7
            # Ensure Qdrant collection
            # --------------------------------------------------

            collection_existed_before = (
                self.vector_pipeline.collection_exists(
                    collection_name=(
                        self.config.collection_name
                    ),
                    provider_name=(
                        self.config.vector_provider
                    ),
                )
            )

            result.metadata[
                "collection_existed_before_upsert"
            ] = collection_existed_before

            self._ensure_collection(
                dimension=dimension
            )

            # --------------------------------------------------
            # Stage 8
            # Upsert vectors
            # --------------------------------------------------

            logger.info(
                "Upserting %d embedded nodes into Qdrant collection '%s'.",
                len(all_embedded_nodes),
                self.config.collection_name,
            )

            self._upsert_vectors(
                all_embedded_nodes
            )

            result.vectors_created = len(
                all_embedded_nodes
            )

            # --------------------------------------------------
            # Stage 9
            # Verify persistence
            # --------------------------------------------------

            stored_count = (
                self.vector_pipeline.count(
                    collection_name=(
                        self.config.collection_name
                    ),
                    provider_name=(
                        self.config.vector_provider
                    ),
                )
            )

            result.metadata[
                "qdrant_vector_count"
            ] = stored_count

            self._diagnose_persistence(
                result=result,
                expected_vectors=len(
                    all_embedded_nodes
                ),
                stored_count=stored_count,
            )

            vectors_before = result.metadata.get(
                "qdrant_vector_count_before"
            )

            if (
                vectors_before is not None
                and stored_count >= vectors_before
            ):

                actual_delta = (
                    stored_count
                    - vectors_before
                )

                result.metadata[
                    "qdrant_vector_delta"
                ] = actual_delta

                if actual_delta == len(
                    all_embedded_nodes
                ):

                    logger.info(
                        "Qdrant persistence diagnosis: "
                        "PASS — current ingestion added exactly "
                        "%d vectors.",
                        actual_delta,
                    )

                elif actual_delta == 0:

                    logger.warning(
                        "Qdrant persistence diagnosis: "
                        "NO NET INCREASE — vectors may have "
                        "been overwritten because node IDs "
                        "already existed."
                    )

                else:

                    logger.warning(
                        "Qdrant persistence diagnosis: "
                        "MISMATCH — expected delta=%d, "
                        "actual delta=%d.",
                        len(all_embedded_nodes),
                        actual_delta,
                    )

            # --------------------------------------------------
            # Stage 10
            # Health verification
            # --------------------------------------------------

            healthy = (
                self.vector_pipeline.health_check(
                    provider_name=(
                        self.config.vector_provider
                    )
                )
            )

            result.metadata[
                "vector_store_healthy"
            ] = healthy

            if not healthy:

                raise RuntimeError(
                    "Vector store health check failed."
                )

            result.success = True

        except Exception as exc:

            result.success = False

            error = (
                f"{type(exc).__name__}: {exc}"
            )

            result.errors.append(
                error
            )

            logger.exception(
                "E2E orchestration failed."
            )

        finally:

            result.completed_at = time.time()

            elapsed = (
                time.perf_counter() - started
            )

            result.metadata[
                "total_duration_seconds"
            ] = round(
                elapsed,
                4
            )

        return result

    # ==========================================================
    # Retrieval
    # ==========================================================

    def configure_retrieval(
        self,
        *,
        retrieval_service: RetrievalService | None = None,
        config: RetrievalServiceConfig | None = None,
    ) -> None:
        """
        Configure the RetrievalService used by this orchestrator.

        RetrievalService remains responsible for the actual retrieval
        pipeline. The orchestrator only connects it to the ingestion
        components.
        """

        if retrieval_service is not None:

            self.retrieval_service = (
                retrieval_service
            )

            return

        self.retrieval_service = (
            self._create_retrieval_service(
                config=config
            )
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_graph: bool = False,
        rerank: bool = False,
        rerank_top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Execute retrieval against the orchestrator's configured
        Qdrant collection.

        Flow:

            query
                ↓
            EmbeddingPipeline
                ↓
            query vector
                ↓
            RetrievalRequest
                ↓
            RetrievalService
                ↓
            BuiltContext
        """

        if self.retrieval_service is None:

            self.configure_retrieval()

        if not query or not query.strip():

            raise ValueError(
                "query cannot be empty."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        # ------------------------------------------------------
        # Query embedding
        # ------------------------------------------------------

        query_vector = (
            self._embed_query(
                query
            )
        )

        # ------------------------------------------------------
        # Retrieval request
        # ------------------------------------------------------

        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            collection_name=(
                self.config.collection_name
            ),
            provider=(
                self.config.vector_provider
            ),
            query_vector=query_vector,
            filters=(
                filters
                if filters is not None
                else {}
            ),
            include_graph=include_graph,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )

        # ------------------------------------------------------
        # RetrievalService
        # ------------------------------------------------------

        start = time.perf_counter()

        built_context = (
            self.retrieval_service.retrieve(
                request
            )
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        logger.info(
            "Retrieval completed | "
            "query='%s' | collection='%s' | "
            "results=%d | latency=%.2f ms",
            query,
            self.config.collection_name,
            len(built_context.items),
            latency_ms,
        )

        return built_context

    def _create_retrieval_service(
        self,
        *,
        config: RetrievalServiceConfig | None = None,
    ) -> RetrievalService:
        """
        Create the RetrievalService connected to this orchestrator's
        existing retrieval components.

        This method intentionally performs no retrieval algorithm itself.
        """

        from app.retrieval.dense.dense_retriever import (
            DenseRetriever,
        )

        from app.retrieval.sparse.sparse_retriever import (
            SparseRetriever,
        )

        from app.retrieval.graph_rag.graph_retriever import (
            GraphRetriever,
        )

        from app.retrieval.reranker.cross_encoder_baseline import (
            CrossEncoderBaseline,
        )

        # ------------------------------------------------------
        # Dense retriever
        # ------------------------------------------------------

        dense_retriever = DenseRetriever(
            pipeline=self.vector_pipeline,
        )

        # ------------------------------------------------------
        # Sparse retriever
        # ------------------------------------------------------
        
        qdrant_store = QdrantStore()
        sparse_retriever = (
            SparseRetriever.from_qdrant(
                qdrant_store=qdrant_store,
                collection_name=(
                    self.config.collection_name
                ),
                 batch_size=100,
            )
        )

        # ------------------------------------------------------
        # Graph retriever
        # ------------------------------------------------------

        graph_retriever = None

        if (
            config is None
            or config.use_graph
        ):

            try:

                graph_retriever = (
                    GraphRetriever()
                )

            except Exception as exc:

                logger.warning(
                    "GraphRetriever could not be initialized: %s",
                    exc,
                )

        # ------------------------------------------------------
        # Reranker
        # ------------------------------------------------------

        reranker = None

        if (
            config is None
            or config.use_reranker
        ):

            try:

                reranker = (
                    CrossEncoderBaseline()
                )

            except Exception as exc:

                logger.warning(
                    "CrossEncoderBaseline could not be initialized: %s",
                    exc,
                )

        # ------------------------------------------------------
        # RetrievalService
        # ------------------------------------------------------

        return RetrievalService(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            graph_retriever=graph_retriever,
            reranker=reranker,
            config=(
                config
                if config is not None
                else RetrievalServiceConfig()
            ),
        )

    def _embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Embed a query using the same embedding provider used
        during document ingestion.

        Query embeddings are never persisted.
        """

        from llama_index.core.schema import (
            TextNode,
        )

        query_node = TextNode(
            text=query
        )

        embedded = (
            self.embedding_pipeline.embed(
                query_node,
                provider=(
                    self.config.embedding_provider
                ),
            )
        )

        return list(
            embedded.embedding
        )

    # ==========================================================
    # Document Processing
    # ==========================================================

    def _process_document(
        self,
        document,
    ):

        logger.info(
            "Processing document: %s",
            document.source_file,
        )

        parsed_document = (
            self.parser_pipeline.run(
                document
            )
        )

        preprocessing_result = (
            self.preprocessing_pipeline.run(
                parsed_document.content,
                lowercase=self.config.lowercase,
                remove_stopwords=(
                    self.config.remove_stopwords
                ),
                spell_correction=(
                    self.config.spell_correction
                ),
                lemmatize=self.config.lemmatize,
                stem=self.config.stem,
            )
        )

        parsed_document.metadata.update(
            {
                "language": (
                    preprocessing_result[
                        "language"
                    ]
                ),
                "preprocessing": {
                    "lowercase": (
                        self.config.lowercase
                    ),
                    "remove_stopwords": (
                        self.config.remove_stopwords
                    ),
                    "spell_correction": (
                        self.config.spell_correction
                    ),
                    "lemmatize": (
                        self.config.lemmatize
                    ),
                    "stem": (
                        self.config.stem
                    ),
                },
            }
        )

        parsed_document.content = (
            preprocessing_result[
                "normalized_text"
            ]
        )

        nodes = (
            self.chunking_pipeline.chunk(
                parsed_document,
                strategy=(
                    self.config.chunking_strategy
                ),
            )
        )

        if not nodes:

            raise RuntimeError(
                f"No chunks generated for "
                f"{document.source_file}"
            )

        logger.info(
            "Chunking completed: %s | chunks=%d",
            document.source_file,
            len(nodes),
        )

        embedded_nodes = (
            self.embedding_pipeline.embed(
                nodes,
                provider=(
                    self.config.embedding_provider
                ),
            )
        )

        if not isinstance(
            embedded_nodes,
            list,
        ):

            embedded_nodes = [
                embedded_nodes
            ]

        logger.info(
            "Embedding completed: %s | vectors=%d",
            document.source_file,
            len(embedded_nodes),
        )

        return embedded_nodes

    # ==========================================================
    # Diagnostics
    # ==========================================================

    def _diagnose_collection_before_ingestion(
        self,
        result: OrchestrationResult,
    ) -> None:

        collection_name = (
            self.config.collection_name
        )

        provider_name = (
            self.config.vector_provider
        )

        try:

            exists = (
                self.vector_pipeline.collection_exists(
                    collection_name=collection_name,
                    provider_name=provider_name,
                )
            )

            result.metadata[
                "collection_exists_before_ingestion"
            ] = exists

            if not exists:

                result.metadata[
                    "qdrant_vector_count_before"
                ] = 0

                logger.info(
                    "DIAGNOSTIC — Collection '%s' "
                    "does not exist before ingestion.",
                    collection_name,
                )

                return

            existing_count = (
                self.vector_pipeline.count(
                    collection_name=collection_name,
                    provider_name=provider_name,
                )
            )

            result.metadata[
                "qdrant_vector_count_before"
            ] = existing_count

            logger.info(
                "DIAGNOSTIC — Qdrant collection '%s' "
                "already exists with %d vectors.",
                collection_name,
                existing_count,
            )

            if existing_count > 0:

                logger.warning(
                    "DIAGNOSTIC — Existing vectors detected. "
                    "This ingestion will NOT automatically clear "
                    "the collection."
                )

        except Exception as exc:

            logger.warning(
                "DIAGNOSTIC — Unable to inspect Qdrant "
                "collection before ingestion: %s",
                exc,
            )

            result.metadata[
                "qdrant_vector_count_before"
            ] = None

    def _diagnose_embedded_nodes(
        self,
        result: OrchestrationResult,
        embedded_nodes,
    ) -> None:

        try:

            node_ids = [
                str(node.node_id)
                for node in embedded_nodes
            ]

            unique_node_ids = set(
                node_ids
            )

            duplicate_count = (
                len(node_ids)
                - len(unique_node_ids)
            )

            result.metadata[
                "generated_node_count"
            ] = len(node_ids)

            result.metadata[
                "unique_generated_node_count"
            ] = len(unique_node_ids)

            result.metadata[
                "duplicate_generated_node_ids"
            ] = duplicate_count

            result.metadata[
                "sample_node_ids"
            ] = node_ids[:5]

            logger.info(
                "DIAGNOSTIC — Generated nodes=%d | "
                "unique node IDs=%d | duplicate IDs=%d",
                len(node_ids),
                len(unique_node_ids),
                duplicate_count,
            )

        except Exception as exc:

            logger.warning(
                "DIAGNOSTIC — Unable to inspect "
                "EmbeddedNode IDs: %s",
                exc,
            )

    def _diagnose_persistence(
        self,
        result: OrchestrationResult,
        expected_vectors: int,
        stored_count: int,
    ) -> None:

        before_count = result.metadata.get(
            "qdrant_vector_count_before"
        )

        result.metadata[
            "expected_vectors_current_run"
        ] = expected_vectors

        result.metadata[
            "qdrant_vector_count_after"
        ] = stored_count

        if before_count is None:

            logger.warning(
                "DIAGNOSTIC — Cannot calculate Qdrant "
                "vector delta."
            )

            return

        delta = (
            stored_count
            - before_count
        )

        result.metadata[
            "qdrant_vector_delta"
        ] = delta

        logger.info(
            "=================================================="
        )

        logger.info(
            "VECTOR COUNT DIAGNOSTIC"
        )

        logger.info(
            "Qdrant count BEFORE ingestion : %d",
            before_count,
        )

        logger.info(
            "Vectors generated THIS run   : %d",
            expected_vectors,
        )

        logger.info(
            "Qdrant count AFTER ingestion  : %d",
            stored_count,
        )

        logger.info(
            "Qdrant count DELTA             : %d",
            delta,
        )

        logger.info(
            "=================================================="
        )

        if delta == expected_vectors:

            logger.info(
                "DIAGNOSIS — PASS: Qdrant added "
                "exactly the number of vectors generated."
            )

        elif delta == 0:

            logger.warning(
                "DIAGNOSIS — Qdrant count did not increase."
            )

        elif delta < expected_vectors:

            logger.warning(
                "DIAGNOSIS — Qdrant increased by fewer "
                "vectors than were generated."
            )

        else:

            logger.error(
                "DIAGNOSIS — Qdrant increased by MORE "
                "vectors than were generated."
            )

    # ==========================================================
    # Vector Store
    # ==========================================================

    def _register_vector_provider(
        self,
    ) -> None:

        if self.config.vector_provider != "qdrant":

            raise ValueError(
                "Initial E2E validation currently "
                "supports Qdrant only."
            )

        if self.vector_pipeline.list_providers():

            if self.vector_pipeline.registry.has_provider(
                "qdrant"
            ):

                self.vector_pipeline.set_default_provider(
                    "qdrant"
                )

                return

        qdrant_store = QdrantStore()

        self.vector_pipeline.register_provider(
            qdrant_store,
            make_default=True,
        )

    def _ensure_collection(
        self,
        dimension: int,
    ) -> None:

        exists = (
            self.vector_pipeline.collection_exists(
                collection_name=(
                    self.config.collection_name
                ),
                provider_name=(
                    self.config.vector_provider
                ),
            )
        )

        if exists:

            logger.info(
                "Qdrant collection '%s' already exists. "
                "Reusing existing collection.",
                self.config.collection_name,
            )

            return

        logger.info(
            "Creating Qdrant collection '%s' "
            "with dimension=%d.",
            self.config.collection_name,
            dimension,
        )

        self.vector_pipeline.create_collection(
            collection_name=(
                self.config.collection_name
            ),
            dimension=dimension,
            provider_name=(
                self.config.vector_provider
            ),
        )

    def _upsert_vectors(
        self,
        embedded_nodes,
    ) -> None:

        self.vector_pipeline.upsert_batch(
            collection_name=(
                self.config.collection_name
            ),
            nodes=embedded_nodes,
            provider_name=(
                self.config.vector_provider
            ),
        )

    # ==========================================================
    # Stage Utility
    # ==========================================================

    def _run_stage(
        self,
        result: OrchestrationResult,
        stage_name: str,
        function,
    ):

        started = time.perf_counter()

        try:

            output = function()

            elapsed = (
                time.perf_counter()
                - started
            )

            items_out = (
                len(output)
                if hasattr(output, "__len__")
                else 1
            )

            result.stages.append(
                StageMetric(
                    stage=stage_name,
                    status="success",
                    duration_seconds=round(
                        elapsed,
                        4
                    ),
                    items_out=items_out,
                )
            )

            return output

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - started
            )

            result.stages.append(
                StageMetric(
                    stage=stage_name,
                    status="failed",
                    duration_seconds=round(
                        elapsed,
                        4
                    ),
                    error=str(exc),
                )
            )

            raise

    # ==========================================================
    # Lifecycle
    # ==========================================================

    def close(self) -> None:
        """
        Close retrieval and vector components when supported.
        """

        if self.retrieval_service is not None:

            close_method = getattr(
                self.retrieval_service,
                "close",
                None,
            )

            if callable(close_method):

                close_method()