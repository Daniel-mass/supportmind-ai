from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results"
)

LATEST_RESULT_PATH = RESULTS_DIR / "latest.json"


def _json_safe(value: Any) -> Any:
    """
    Convert common Python / DeepEval values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return str(value)


def save_deepeval_result(
    *,
    query: str,
    answer: str,
    retrieval_context: list[str],
    metrics: dict[str, Any],
    grounding: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Save the completed DeepEval evaluation result.

    This function only stores evaluation output.
    It does not execute DeepEval.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "evaluation": {
            "framework": "DeepEval",
            "evaluation_type": "graph_rag",
        },
        "model": {
            "provider": "ollama",
            "name": "qwen3:8b",
        },
        "input": {
            "query": query,
        },
        "output": {
            "answer": answer,
        },
        "retrieval": {
            "context_count": len(retrieval_context),
            "context": retrieval_context,
        },
        "grounding": grounding,
        "metrics": metrics,
        "metadata": metadata or {},
    }

    result = _json_safe(result)

    LATEST_RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return result


def load_latest_deepeval_result() -> dict[str, Any] | None:
    """
    Load the most recently saved DeepEval result.
    """

    if not LATEST_RESULT_PATH.exists():
        return None

    return json.loads(
        LATEST_RESULT_PATH.read_text(
            encoding="utf-8"
        )
    )