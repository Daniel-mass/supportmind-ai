from __future__ import annotations

from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentError,
    RetrievalAgentState,
    RetrievalAgentStatus,
    RetrievalStrategy,
)
from app.agents.retrieval_agent.prompts import (
    RETRIEVAL_AGENT_SYSTEM_PROMPT,
    build_retrieval_prompt,
)
from app.agents.retrieval_agent.tools.retrieval_tools import RetrievalTools
from app.retrieval.graph_rag.graph_rag_models import GraphRAGRequest
from app.retrieval.models.retrieval_models import RetrievalRequest


RETRIEVAL_AGENT_NAME = "retrieval_agent"


class RetrievalAgent:
    """
    AutoGen-based Retrieval Agent.

    The agent reasons about retrieval strategy and delegates actual
    retrieval execution to deterministic SupportMind services through
    RetrievalTools.
    """

    def __init__(
        self,
        model_client: Any,
        retrieval_tools: RetrievalTools,
        *,
        name: str = RETRIEVAL_AGENT_NAME,
    ) -> None:
        if model_client is None:
            raise ValueError("model_client must not be None.")

        if retrieval_tools is None:
            raise ValueError("retrieval_tools must not be None.")

        if not isinstance(retrieval_tools, RetrievalTools):
            raise TypeError(
                "retrieval_tools must be an instance of RetrievalTools."
            )

        self.name = name
        self._tools = retrieval_tools

        self._agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            system_message=RETRIEVAL_AGENT_SYSTEM_PROMPT,
        )

    @property
    def agent(self) -> AssistantAgent:
        """Return the underlying AutoGen AssistantAgent."""
        return self._agent

    @property
    def tools(self) -> RetrievalTools:
        """Return the deterministic retrieval tool adapter."""
        return self._tools

    def build_task(self, state: RetrievalAgentState) -> str:
        """
        Build the AutoGen task prompt from the current retrieval state.
        """
        if not isinstance(state, RetrievalAgentState):
            raise TypeError(
                "state must be an instance of RetrievalAgentState."
            )

        return build_retrieval_prompt(state)

    def select_strategy(
        self,
        state: RetrievalAgentState,
    ) -> RetrievalStrategy:
        """
        Select the retrieval execution path.

        The current explicit state flag is authoritative:

        use_graph_rag=True  -> GRAPH_RAG
        use_graph_rag=False -> STANDARD

        This keeps execution deterministic while the AutoGen agent
        provides reasoning around the selected capability.
        """
        if not isinstance(state, RetrievalAgentState):
            raise TypeError(
                "state must be an instance of RetrievalAgentState."
            )

        if state.use_graph_rag:
            return RetrievalStrategy.GRAPH_RAG

        return RetrievalStrategy.STANDARD

    def _build_graph_rag_request(
        self,
        state: RetrievalAgentState,
    ) -> GraphRAGRequest:
        """
        Convert the existing RetrievalRequest into the Graph-RAG contract.
        """
        request = state.request

        return GraphRAGRequest(
            query=request.query,
            retrieval_request=request,
            include_intelligence=True,
            include_workflow=True,
            max_evidence_items=20,
            max_evidence_characters=16000,
            metadata={
                "source": self.name,
                "retrieval_agent": True,
            },
        )

    def _update_standard_result(
        self,
        state: RetrievalAgentState,
    ) -> None:
        """
        Execute standard retrieval and update the state.
        """
        context = self._tools.retrieve(state.request)

        state.context = context
        state.graph_rag_response = None
        state.execution_metadata = self._tools.execution_metadata()

        state.evidence_ids = [
            item.id
            for item in context.selected_items
            if item.id
        ]

    def _update_graph_rag_result(
        self,
        state: RetrievalAgentState,
    ) -> None:
        """
        Execute Graph-RAG and update the state.
        """
        request = self._build_graph_rag_request(state)

        response = self._tools.graph_rag(request)

        state.graph_rag_response = response
        state.context = None

        state.evidence_ids = [
            evidence.evidence_id
            for evidence in response.evidence
            if evidence.evidence_id
        ]

        state.execution_metadata = {
            "provider": response.provider,
            "model_name": response.model_name,
            "latency_ms": response.latency_ms,
            "evidence_count": len(response.evidence),
            "grounding_status": response.grounding.status.value,
            "grounding_valid": response.grounding.valid,
        }

    def _execute_strategy(
        self,
        state: RetrievalAgentState,
    ) -> None:
        """
        Execute the deterministic retrieval path selected for the state.
        """
        if state.strategy == RetrievalStrategy.STANDARD:
            self._update_standard_result(state)
            return

        if state.strategy == RetrievalStrategy.GRAPH_RAG:
            self._update_graph_rag_result(state)
            return

        raise ValueError(
            f"Unsupported retrieval strategy: {state.strategy}"
        )

    def _record_error(
        self,
        state: RetrievalAgentState,
        exc: Exception,
    ) -> None:
        """
        Record a retrieval-agent failure without fabricating a result.
        """
        state.status = RetrievalAgentStatus.FAILED
        state.errors.append(
            RetrievalAgentError(
                source=self.name,
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=False,
            )
        )

    async def run(
        self,
        state: RetrievalAgentState,
    ) -> RetrievalAgentState:
        """
        Execute the Retrieval Agent workflow.

        The AutoGen model is used to reason about the current retrieval
        task. Deterministic retrieval execution is performed through
        RetrievalTools.
        """
        if not isinstance(state, RetrievalAgentState):
            raise TypeError(
                "state must be an instance of RetrievalAgentState."
            )

        state.status = RetrievalAgentStatus.RUNNING
        state.iteration_count += 1

        try:
            # Build the reasoning task so the AutoGen agent receives
            # the complete retrieval state.
            task = self.build_task(state)

            # Run the AutoGen reasoning layer.
            result = await self._agent.run(task=task)

            # Ensure the AutoGen agent actually produced a response.
            response = self._extract_final_response(result)

            if response is None:
                raise RuntimeError(
                    "Retrieval Agent completed without producing a textual response."
                )

            # Select the deterministic execution strategy.
            state.strategy = self.select_strategy(state)

            # Execute the existing retrieval capability.
            self._execute_strategy(state)

            state.status = RetrievalAgentStatus.COMPLETED

            return state

        except Exception as exc:
            self._record_error(state, exc)
            return state

    @staticmethod
    def _extract_final_response(result: Any) -> str | None:
        """
        Extract the final textual response from an AutoGen result.
        """
        messages = getattr(result, "messages", None)

        if not messages:
            return None

        for message in reversed(messages):
            if isinstance(message, TextMessage):
                content = message.content

                if isinstance(content, str) and content.strip():
                    return content.strip()

        return None