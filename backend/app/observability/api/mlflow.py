from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from mlflow import MlflowClient
from mlflow.entities import ViewType


router = APIRouter(
    prefix="/mlflow",
    tags=["MLflow"],
)


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


def _get_client() -> MlflowClient:
    return MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )


def _serialize_experiment(experiment: Any) -> dict[str, Any]:
    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "artifact_location": experiment.artifact_location,
        "lifecycle_stage": experiment.lifecycle_stage,
        "tags": dict(experiment.tags or {}),
    }


def _serialize_run(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "run_uuid": run.info.run_id,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": {
            key: value
            for key, value in run.data.tags.items()
            if not key.startswith("mlflow.")
        },
    }


@router.get("/health")
def mlflow_health() -> dict[str, Any]:
    try:
        client = _get_client()

        experiments = client.search_experiments(
            max_results=1
        )

        return {
            "status": "healthy",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_count_check": len(experiments),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "tracking_uri": MLFLOW_TRACKING_URI,
                "error": str(exc),
            },
        )


@router.get("/experiments")
def list_experiments(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
) -> dict[str, Any]:

    try:
        client = _get_client()

        experiments = client.search_experiments(
            max_results=limit,
        )

        return {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "count": len(experiments),
            "experiments": [
                _serialize_experiment(experiment)
                for experiment in experiments
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@router.get("/runs")
def list_runs(
    experiment_id: str | None = None,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> dict[str, Any]:

    try:
        client = _get_client()

        if experiment_id:
            experiment_ids = [experiment_id]
        else:
            experiments = client.search_experiments(
                max_results=1000
            )
            experiment_ids = [
                experiment.experiment_id
                for experiment in experiments
            ]

        if not experiment_ids:
            return {
                "tracking_uri": MLFLOW_TRACKING_URI,
                "count": 0,
                "runs": [],
            }

        runs = client.search_runs(
            experiment_ids=experiment_ids,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=limit,
            order_by=[
                "start_time DESC",
            ],
        )

        return {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "count": len(runs),
            "runs": [
                _serialize_run(run)
                for run in runs
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:

    try:
        client = _get_client()

        run = client.get_run(run_id)

        return _serialize_run(run)

    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"MLflow run not found: {run_id}. {exc}",
        )