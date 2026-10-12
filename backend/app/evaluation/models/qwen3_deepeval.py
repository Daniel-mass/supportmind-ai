from __future__ import annotations

import asyncio
from typing import Any

import ollama
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel


class Qwen3DeepEvalModel(DeepEvalBaseLLM):
    """DeepEval adapter for a local Qwen3 model served by Ollama."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_predict: int | None = None,
    ) -> None:
        self.base_url = base_url
        self.temperature = temperature
        self.num_predict = num_predict

        super().__init__(model=model)

    def load_model(self) -> ollama.Client:
        """Create the local Ollama client used by the evaluator."""

        return ollama.Client(host=self.base_url)

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> str | BaseModel:
        """Generate text or structured Pydantic output using Ollama."""

        options: dict[str, Any] = {
            "temperature": self.temperature,
        }

        if self.num_predict is not None:
            options["num_predict"] = self.num_predict

        options.update(kwargs)

        response = self.model.chat(
            model=self.name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=schema.model_json_schema() if schema else None,
            options=options,
        )

        content = response.message.content

        if schema is not None:
            return schema.model_validate_json(content)

        return content

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> str | BaseModel:
        """Async wrapper around the synchronous Ollama generation."""

        return await asyncio.to_thread(
            self.generate,
            prompt,
            schema,
            **kwargs,
        )

    def get_model_name(self) -> str:
        """Return the model name shown by DeepEval."""

        return self.name