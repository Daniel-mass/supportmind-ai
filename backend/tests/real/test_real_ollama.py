"""
Real Ollama integration test for SupportMind AI.

This test connects to a real local Ollama server and performs
real inference through the SupportMind Ollama provider.

It is intentionally separate from unit tests because it requires:
- Ollama server
- configured model
- actual model inference
"""

from __future__ import annotations

import asyncio
import os

import pytest
from autogen_core import CancellationToken
from autogen_core.models import UserMessage

from app.llm.config import LLMSettings
from app.llm.providers.ollama import OllamaProvider


RUN_REAL_OLLAMA = os.getenv("RUN_REAL_OLLAMA", "0") == "1"


pytestmark = pytest.mark.skipif(
    not RUN_REAL_OLLAMA,
    reason=(
        "Real Ollama integration disabled. "
        "Set RUN_REAL_OLLAMA=1 to run."
    ),
)


def test_real_ollama_provider_inference() -> None:
    """
    Verify that the SupportMind Ollama provider can perform
    real inference through Ollama.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_host=os.getenv(
            "OLLAMA_HOST",
            "http://127.0.0.1:11434",
        ),
        ollama_model=os.getenv(
            "SUPPORTMIND_LLM_MODEL",
            "qwen3:8b",
        ),
        temperature=0.0,
        max_tokens=128,
        timeout=120.0,
    )

    provider = OllamaProvider(settings)

    client = provider.create_model_client()

    async def run_inference() -> str:
        try:
            result = await client.create(
                messages=[
                    UserMessage(
                        content=(
                            "/no_think\n"
                            "Reply with exactly "
                            "SUPPORTMIND_OLLAMA_OK "
                            "and nothing else."
                        ),
                        source="user",
                    )
                ],
                cancellation_token=CancellationToken(),
            )

            content = getattr(result, "content", None)

            if not isinstance(content, str):
                raise AssertionError(
                    "Ollama returned a response without textual content."
                )

            return content.strip()

        finally:
            await client.close()

    response = asyncio.run(run_inference())

    assert response, (
        "Ollama returned an empty textual response. "
        "The model may have consumed the generation budget "
        "with reasoning output."
    )

    assert "SUPPORTMIND_OLLAMA_OK" in response