from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.evaluation.deepeval.result_store import (
    load_latest_deepeval_result,
)


router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "deepeval"
    / "results"
)

RESULT_TEXT_PATH = RESULTS_DIR / "result.txt"


def _load_terminal_result() -> dict[str, Any]:
    if not RESULT_TEXT_PATH.exists():
        return {
            "available": False,
            "content": None,
        }

    content = RESULT_TEXT_PATH.read_text(
        encoding="utf-8"
    )

    faithfulness_match = re.search(
        r"Faithfulness:\s*([0-9.]+)",
        content,
    )

    answer_relevancy_match = re.search(
        r"Answer Relevancy:\s*([0-9.]+)",
        content,
    )

    return {
        "available": True,
        "content": content,
        "parsed_metrics": {
            "faithfulness": (
                float(faithfulness_match.group(1))
                if faithfulness_match
                else None
            ),
            "answer_relevancy": (
                float(answer_relevancy_match.group(1))
                if answer_relevancy_match
                else None
            ),
        },
    }


@router.get("/summary")
def get_evaluation_summary() -> dict[str, Any]:
    result = load_latest_deepeval_result()

    return {
        "status": "available" if result else "not_available",
        "framework": (
            result.get("evaluation", {}).get("framework")
            if result
            else None
        ),
        "evaluation_type": (
            result.get("evaluation", {}).get("evaluation_type")
            if result
            else None
        ),
        "model": (
            result.get("model")
            if result
            else None
        ),
        "metrics": (
            result.get("metrics")
            if result
            else None
        ),
    }


@router.get("/deepeval")
def get_deepeval_result() -> dict[str, Any]:
    result = load_latest_deepeval_result()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No saved DeepEval result found.",
        )

    return result


@router.get("/deepeval/raw")
def get_deepeval_terminal_result() -> dict[str, Any]:
    result = _load_terminal_result()

    if not result["available"]:
        raise HTTPException(
            status_code=404,
            detail="DeepEval terminal result file not found.",
        )

    return result