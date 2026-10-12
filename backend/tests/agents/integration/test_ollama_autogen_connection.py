import pytest

from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient


@pytest.mark.asyncio
async def test_real_ollama_autogen_connection() -> None:
    client = OllamaChatCompletionClient(
        model="qwen3:8b",
        host="http://127.0.0.1:11434",
    )

    try:
        result = await client.create(
            [
                UserMessage(
                    content="Say hello in one sentence.",
                    source="user",
                )
            ]
        )

        print()
        print("=" * 80)
        print("OLLAMA + AUTOGEN CONNECTION TEST")
        print("=" * 80)
        print(result)

        assert result is not None

    finally:
        await client.close()