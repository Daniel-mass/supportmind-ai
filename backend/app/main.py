# backend/app/main.py

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# ==========================================================
# Environment Configuration
# ==========================================================
# Project structure:
# supportmind-ai/
# ├── .env
# └── backend/
#     └── app/
#         └── main.py
#
# parents[2] resolves to the SupportMind AI project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Load configuration before importing application modules.
# The root .env is the source of truth for this application.
load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

if ENV_FILE.is_file():
    LOGGER.info("Loaded environment configuration from %s", ENV_FILE)
else:
    LOGGER.warning(
        "Project .env file was not found at %s",
        ENV_FILE,
    )


# ==========================================================
# FastAPI and Middleware
# ==========================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ==========================================================
# Intelligence API Routers
# ==========================================================

from app.intelligence.api.agents import (
    router as agent_intelligence_router,
)
from app.intelligence.api.organization import (
    router as organization_intelligence_router,
)
from app.intelligence.api.tickets import (
    router as ticket_intelligence_router,
)
from app.intelligence.api.workflow import (
    router as workflow_intelligence_router,
)
from app.intelligence.providers.helpdesk_provider import (
    get_helpdesk_provider,
)

from app.intelligence.api.graph_intelligence import (
    router as graph_intelligence_router,
)
# ==========================================================
# Support API Routers
# ==========================================================

from app.support.api.assist import (
    router as support_assist_router,
)


# ==========================================================
# Graph-RAG API
# ==========================================================

from app.retrieval.graph_rag.api import (
    router as graph_rag_router,
)


# ==========================================================
# Evaluation API
# ==========================================================

from app.evaluation.api.evaluation import (
    router as evaluation_router,
)


# ==========================================================
# Observability APIs
# ==========================================================

from app.observability.api.mlflow import (
    router as mlflow_router,
)
from app.observability.api.phoenix import (
    router as phoenix_router,
)


# ==========================================================
# Application Lifecycle
# ==========================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Initialize application-wide intelligence before serving
    incoming requests.

    The Help Desk intelligence snapshot is built during startup
    and reused through the provider's existing cache.
    """

    LOGGER.info("Starting SupportMind AI...")

    provider = get_helpdesk_provider()

    # Warm the Help Desk intelligence snapshot.
    provider.build_snapshot()

    LOGGER.info("Help Desk intelligence snapshot initialized.")

    yield

    LOGGER.info("Shutting down SupportMind AI...")


# ==========================================================
# Application Factory
# ==========================================================

def create_app() -> FastAPI:
    """Create and configure the SupportMind AI application."""

    application = FastAPI(
        title="SupportMind AI",
        version="1.0.0",
        description=(
            "Enterprise Customer and Issue "
            "Intelligence Platform."
        ),
        lifespan=lifespan,
    )

    # ------------------------------------------------------
    # CORS
    # ------------------------------------------------------

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------
    # Intelligence endpoints
    # ------------------------------------------------------

    application.include_router(
        ticket_intelligence_router,
        prefix="/api/v1",
    )

    application.include_router(
        workflow_intelligence_router,
        prefix="/api/v1",
    )

    application.include_router(
        agent_intelligence_router,
        prefix="/api/v1",
    )

    application.include_router(
        organization_intelligence_router,
        prefix="/api/v1",
    )
    application.include_router(
       graph_intelligence_router,
       prefix="/api/v1",
)
    # ------------------------------------------------------
    # Support endpoints
    # ------------------------------------------------------

    application.include_router(
        support_assist_router,
        prefix="/api/v1",
    )

    # ------------------------------------------------------
    # Graph-RAG endpoints
    # ------------------------------------------------------

    application.include_router(
        graph_rag_router,
    )

    # ------------------------------------------------------
    # Evaluation endpoints
    # ------------------------------------------------------

    application.include_router(
        evaluation_router,
        prefix="/api/v1",
    )

    # ------------------------------------------------------
    # Observability endpoints
    # ------------------------------------------------------

    application.include_router(
        mlflow_router,
        prefix="/api/v1",
    )

    application.include_router(
        phoenix_router,
        prefix="/api/v1",
    )

    return application


# ==========================================================
# Application Entry Point
# ==========================================================

app = create_app()
