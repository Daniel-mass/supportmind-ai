"""
SupportMind Phoenix observability.

Provides:
- Phoenix/OpenTelemetry setup.
- A reusable OpenTelemetry tracer for application-level
  SupportMind orchestration spans.
- Helpers for recording OpenInference-compatible
  input/output payloads.
"""

from __future__ import annotations

import json
import os
from typing import Any

from opentelemetry import trace
from phoenix.otel import register
from openinference.instrumentation.autogen_agentchat import (
    AutogenAgentChatInstrumentor,
)


# ============================================================================
# Configuration
# ============================================================================

PHOENIX_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT",
    "http://localhost:6006/v1/traces",
)

PHOENIX_PROJECT_NAME = os.getenv(
    "PHOENIX_PROJECT_NAME",
    "supportmind-agents",
)

SUPPORTMIND_TRACER_NAME = (
    "supportmind.agent-orchestrator"
)


# ============================================================================
# Phoenix Setup
# ============================================================================


def setup_phoenix() -> object:
    """
    Configure Phoenix OpenTelemetry tracing.

    This function should be called once by the application or
    an end-to-end test before executing the agent workflow.

    Returns:
        The Phoenix tracer provider.
    """

    tracer_provider = register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=PHOENIX_ENDPOINT,
    )

    AutogenAgentChatInstrumentor().instrument(
        tracer_provider=tracer_provider,
    )

    return tracer_provider


# ============================================================================
# SupportMind Tracer
# ============================================================================


def get_supportmind_tracer() -> trace.Tracer:
    """
    Return the application-level SupportMind tracer.

    The tracer is intentionally obtained through OpenTelemetry's
    global tracer provider so that:

        setup_phoenix()
            ↓
        OpenTelemetry provider
            ↓
        SupportMind tracer
            ↓
        Orchestrator spans
            ↓
        AutoGen agent spans

    This allows the entire multi-agent workflow to appear under
    the same Phoenix trace.

    Returns:
        OpenTelemetry tracer used by SupportMind application code.
    """

    return trace.get_tracer(
        SUPPORTMIND_TRACER_NAME
    )


# ============================================================================
# Payload Serialization
# ============================================================================


def _json_safe(value: Any) -> Any:
    """
    Convert common SupportMind/Pydantic values into
    JSON-serializable structures.

    This is intentionally defensive because agent results
    may contain enums, Pydantic models, lists, dictionaries,
    or other application objects.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if hasattr(value, "model_dump"):
        return _json_safe(
            value.model_dump(
                mode="json"
            )
        )

    if hasattr(value, "value"):
        return _json_safe(
            value.value
        )

    return str(value)


def _serialize_payload(value: Any) -> str:
    """
    Serialize a payload into JSON for Phoenix.
    """

    safe_value = _json_safe(value)

    return json.dumps(
        safe_value,
        ensure_ascii=False,
        default=str,
    )


# ============================================================================
# OpenInference Payload Helpers
# ============================================================================


def set_span_input(
    span: trace.Span,
    value: Any,
) -> None:
    """
    Record a span input using OpenInference conventions.
    """

    span.set_attribute(
        "input.value",
        _serialize_payload(value),
    )

    span.set_attribute(
        "input.mime_type",
        "application/json",
    )


def set_span_output(
    span: trace.Span,
    value: Any,
) -> None:
    """
    Record a span output using OpenInference conventions.
    """

    span.set_attribute(
        "output.value",
        _serialize_payload(value),
    )

    span.set_attribute(
        "output.mime_type",
        "application/json",
    )


def set_span_text_input(
    span: trace.Span,
    value: str,
) -> None:
    """
    Record a plain-text span input.
    """

    span.set_attribute(
        "input.value",
        value,
    )

    span.set_attribute(
        "input.mime_type",
        "text/plain",
    )


def set_span_text_output(
    span: trace.Span,
    value: str,
) -> None:
    """
    Record a plain-text span output.
    """

    span.set_attribute(
        "output.value",
        value,
    )

    span.set_attribute(
        "output.mime_type",
        "text/plain",
    )


# ============================================================================
# Span Kind Helpers
# ============================================================================


def set_span_kind(
    span: trace.Span,
    kind: str,
) -> None:
    """
    Set the OpenInference span kind.

    Examples:
        AGENT
        CHAIN
        RETRIEVER
        TOOL
        LLM
    """

    span.set_attribute(
        "openinference.span.kind",
        kind,
    )