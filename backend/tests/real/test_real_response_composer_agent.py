from __future__ import annotations

import asyncio
import os

from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.response_composer_agent.agent import (
    ResponseComposerAgent,
)
from app.agents.response_composer_agent.contracts.state import (
    ResponseAudience,
    ResponseComposerAgentState,
    ResponseEvidence,
    ResponseEscalationSignal,
    ResponseResolutionSignal,
    ResponseTone,
)
from app.llm.config import LLMSettings


def build_real_model_client() -> OllamaChatCompletionClient:
    """
    Build the real Ollama model client used for integration testing.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_host=os.getenv(
            "OLLAMA_HOST",
            "http://127.0.0.1:11434",
        ),
        ollama_model=os.getenv(
            "SUPPORTMIND_GRAPH_RAG_MODEL",
            "qwen3:8b",
        ),
        temperature=0.1,
        max_tokens=1024,
        timeout=120.0,
    )

    return OllamaChatCompletionClient(
        model=settings.ollama_model,
        host=settings.ollama_host,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def build_real_composer_state() -> ResponseComposerAgentState:
    """
    Build a realistic upstream state.

    These signals represent outputs that would normally come from:

    Classification → Retrieval → Resolution → Escalation
    """

    return ResponseComposerAgentState(
        request=(
            "I cannot log in to my NovaCare account. "
            "How can I recover access?"
        ),
        ticket_id="SUPPORT-REAL-001",
        audience=ResponseAudience.CUSTOMER,
        tone=ResponseTone.PROFESSIONAL,
        classification_signals=[
            {
                "name": "category",
                "value": "Account",
                "confidence": 0.91,
                "source": "classification_agent",
            },
            {
                "name": "priority",
                "value": "Medium",
                "confidence": 0.82,
                "source": "classification_agent",
            },
            {
                "name": "sentiment",
                "value": "negative",
                "confidence": 0.88,
                "source": "classification_agent",
            },
        ],
        evidence=[
            ResponseEvidence(
                evidence_id="E1",
                source="knowledge_base",
                content=(
                    "Customers who cannot access their NovaCare "
                    "account should use the password recovery option "
                    "on the account login page."
                ),
                source_reference="account_management_guide.pdf",
                score=0.94,
            ),
            ResponseEvidence(
                evidence_id="E2",
                source="knowledge_base",
                content=(
                    "If password recovery does not restore account "
                    "access, the customer should contact support "
                    "for further account assistance."
                ),
                source_reference="account_management_guide.pdf",
                score=0.89,
            ),
        ],
        resolution=ResponseResolutionSignal(
            resolution_type="account_action",
            recommendation=(
                "Use the password recovery option on the "
                "NovaCare account login page."
            ),
            rationale=(
                "The supplied account-management evidence "
                "supports the password recovery procedure."
            ),
            confidence=0.93,
            requires_human_review=False,
            proposed_actions=[
                "Use the password recovery option.",
                "Contact support if recovery does not restore access.",
            ],
        ),
        escalation=ResponseEscalationSignal(
            escalation_type="not_required",
            should_escalate=False,
            reason="No escalation requirement was supplied.",
            rationale=(
                "The supplied upstream escalation decision "
                "does not require escalation."
            ),
            confidence=0.91,
            requires_human_review=False,
        ),
    )


async def run_real_response_composer_test() -> ResponseComposerAgentState:
    """
    Execute the real Response Composer Agent against Ollama.

    Model creation, inference, and cleanup all occur inside
    the same asyncio event loop.
    """

    model_client = build_real_model_client()

    try:
        agent = ResponseComposerAgent(
            model_client=model_client,
        )

        state = build_real_composer_state()

        result = await agent.run(state)

        return result

    finally:
        close_method = getattr(
            model_client,
            "close",
            None,
        )

        if close_method is not None:
            close_result = close_method()

            if asyncio.iscoroutine(close_result):
                await close_result


def test_real_response_composer_agent():
    """
    Real integration test.

    Requires:
    - Ollama running
    - Qwen3:8b available locally
    """

    result = asyncio.run(
        run_real_response_composer_test()
    )

    composed_response = result.composed_response

    print("\n" + "=" * 70)
    print("REAL RESPONSE COMPOSER AGENT RESULT")
    print("=" * 70)

    print(
        f"Status              : {result.status.value}"
    )

    print(
        f"Iterations          : {result.iteration_count}"
    )

    print(
        f"Evidence count      : {len(result.evidence)}"
    )

    if composed_response is not None:
        print(
            f"Evidence IDs        : "
            f"{composed_response.evidence_ids}"
        )

        print(
            f"Escalation required : "
            f"{composed_response.escalation_required}"
        )

        print(
            f"Human review        : "
            f"{composed_response.human_review_required}"
        )

        print(
            f"Response            : "
            f"{composed_response.response}"
        )
    else:
        print("Evidence IDs        : []")
        print("Escalation required : None")
        print("Human review        : None")
        print("Response            : None")

    print("=" * 70)

    assert result.successful is True

    assert composed_response is not None

    assert composed_response.response.strip()

    assert (
        composed_response.audience
        is ResponseAudience.CUSTOMER
    )

    assert (
        composed_response.tone
        is ResponseTone.PROFESSIONAL
    )

    assert (
        composed_response.escalation_required
        is False
    )

    assert (
        composed_response.human_review_required
        is False
    )

    available_evidence_ids = {
        evidence.evidence_id
        for evidence in result.evidence
    }

    assert all(
        evidence_id in available_evidence_ids
        for evidence_id in composed_response.evidence_ids
    )

    assert "password" in composed_response.response.lower()

    assert not result.errors