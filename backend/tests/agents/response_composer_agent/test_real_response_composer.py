from __future__ import annotations

import os

import pytest
from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.response_composer_agent.agent import (
    ResponseComposerAgent,
)
from app.agents.response_composer_agent.contracts.state import (
    ResponseAudience,
    ResponseClassificationSignal,
    ResponseComposerAgentState,
    ResponseComposerAgentStatus,
    ResponseEvidence,
    ResponseEscalationSignal,
    ResponseResolutionSignal,
    ResponseTone,
)


@pytest.mark.asyncio
async def test_real_response_composer_with_ollama() -> None:
    """
    Real local integration test for the Response Composer Agent.

    This test intentionally bypasses:
    - Coordinator
    - AgentOrchestrator
    - Phoenix
    - Qdrant
    - Neo4j
    - Retrieval Agent
    - Resolution Agent
    - Escalation Agent

    Those upstream outputs are represented by validated state objects.

    The test verifies only the real boundary:

        ResponseComposerAgentState
                    ↓
        ResponseComposerAgent
                    ↓
              Ollama / Qwen3
                    ↓
             composed response
                    ↓
             structural validation
                    ↓
                 COMPLETED
    """

    ollama_host = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    )

    model_name = os.getenv(
        "OLLAMA_MODEL",
        "qwen3:8b",
    )

    print()
    print("=" * 70)
    print("REAL RESPONSE COMPOSER AGENT TEST")
    print("=" * 70)
    print(f"Ollama URL : {ollama_host}")
    print(f"Model      : {model_name}")

    # ------------------------------------------------------------------
    # Build realistic upstream classification signals
    # ------------------------------------------------------------------

    classification_signals = [
        ResponseClassificationSignal(
            name="category",
            value="billing",
            confidence=0.94,
            source="classification_agent",
        ),
        ResponseClassificationSignal(
            name="priority",
            value="high",
            confidence=0.91,
            source="classification_agent",
        ),
        ResponseClassificationSignal(
            name="sentiment",
            value="frustrated",
            confidence=0.96,
            source="classification_agent",
        ),
        ResponseClassificationSignal(
            name="complexity",
            value="moderate",
            confidence=0.88,
            source="classification_agent",
        ),
    ]

    # ------------------------------------------------------------------
    # Build realistic retrieval / Graph-RAG evidence
    # ------------------------------------------------------------------

    evidence = [
        ResponseEvidence(
            evidence_id="E1",
            source="refund_policy",
            content=(
                "Customers may request a refund within 14 days "
                "of the original purchase, subject to the applicable "
                "subscription plan terms."
            ),
            source_reference="refund-policy-node-001",
            score=0.94,
            metadata={
                "context_source": "bm25",
                "evidence_type": "policy",
            },
        ),
        ResponseEvidence(
            evidence_id="E2",
            source="subscription_policy",
            content=(
                "Refund eligibility depends on the subscription plan "
                "and the applicable refund period."
            ),
            source_reference="subscription-policy-node-002",
            score=0.91,
            metadata={
                "context_source": "vector",
                "evidence_type": "policy",
            },
        ),
        ResponseEvidence(
            evidence_id="E3",
            source="support_ticket_history",
            content=(
                "The customer previously contacted support about "
                "the unresolved subscription refund request."
            ),
            source_reference="ticket-history-node-003",
            score=0.87,
            metadata={
                "context_source": "graph",
                "evidence_type": "ticket_history",
            },
        ),
        ResponseEvidence(
            evidence_id="E4",
            source="escalation_policy",
            content=(
                "Escalation is not required when the available "
                "information is sufficient to continue the normal "
                "support process."
            ),
            source_reference="escalation-policy-node-004",
            score=0.82,
            metadata={
                "context_source": "graph",
                "evidence_type": "policy",
            },
        ),
    ]

    # ------------------------------------------------------------------
    # Existing Resolution Agent output
    # ------------------------------------------------------------------

    resolution = ResponseResolutionSignal(
        resolution_type="policy_based",
        recommendation=(
            "The refund eligibility depends on the subscription "
            "plan and the applicable refund period. The customer "
            "must provide the purchase date and subscription plan "
            "details to determine eligibility."
        ),
        rationale=(
            "The refund policy states that refunds are available "
            "within 14 days of purchase, while the subscription "
            "policy indicates that eligibility depends on the "
            "plan and refund period."
        ),
        confidence=0.75,
        requires_human_review=True,
        proposed_actions=[
            "Request the customer's purchase date.",
            "Request the customer's subscription plan.",
            "Determine refund eligibility using the applicable policy.",
        ],
    )

    # ------------------------------------------------------------------
    # Existing Escalation Agent output
    # ------------------------------------------------------------------

    escalation = ResponseEscalationSignal(
        escalation_type="no_escalation",
        should_escalate=False,
        reason=(
            "The available policy evidence provides a normal "
            "support path, but additional customer information "
            "is required before eligibility can be determined."
        ),
        rationale=(
            "The issue can continue through the normal support "
            "process once the purchase date and subscription plan "
            "are obtained."
        ),
        recommended_team=None,
        recommended_path=None,
        evidence_ids=["E4"],
        confidence=0.95,
        requires_human_review=False,
        proposed_actions=[],
    )

    # ------------------------------------------------------------------
    # Complete Response Composer state
    # ------------------------------------------------------------------

    state = ResponseComposerAgentState(
        request=(
            "A customer is asking about a refund for a subscription "
            "and is frustrated because the issue has not been resolved. "
            "Provide a clear response explaining what information is "
            "needed to determine the refund eligibility."
        ),
        ticket_id="TEST-RESPONSE-COMPOSER-001",
        audience=ResponseAudience.CUSTOMER,
        tone=ResponseTone.PROFESSIONAL,
        classification_signals=classification_signals,
        evidence=evidence,
        resolution=resolution,
        escalation=escalation,
    )

    # ------------------------------------------------------------------
    # Create the real Ollama model client
    # ------------------------------------------------------------------

    model_client = OllamaChatCompletionClient(
        model=model_name,
        host=ollama_host,
    )

    agent = ResponseComposerAgent(
        model_client=model_client,
    )

    try:
        # --------------------------------------------------------------
        # Execute the REAL Response Composer Agent
        # --------------------------------------------------------------

        result = await agent.run(state)

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------

        print()
        print("-" * 70)
        print("RESPONSE COMPOSER RESULT")
        print("-" * 70)

        print(f"Status             : {result.status}")
        print(f"Iterations         : {result.iteration_count}")
        print(f"Successful         : {result.successful}")
        print(f"Has response       : {result.has_response}")
        print(f"Evidence count     : {len(result.evidence)}")
        print(f"Execution metadata : {result.execution_metadata}")

        if result.errors:
            print()
            print("Errors:")
            for error in result.errors:
                print(
                    f"  source={error.source}"
                    f" | type={error.error_type}"
                    f" | message={error.message}"
                )

        # --------------------------------------------------------------
        # Assertions
        # --------------------------------------------------------------

        assert result.status is ResponseComposerAgentStatus.COMPLETED
        assert result.successful is True
        assert result.has_response is True
        assert result.composed_response is not None

        composed = result.composed_response

        assert composed.response.strip()
        assert composed.audience is ResponseAudience.CUSTOMER
        assert composed.tone is ResponseTone.PROFESSIONAL

        # Every evidence ID attached by the composer must exist
        # in the supplied evidence.
        available_evidence_ids = {
            item.evidence_id
            for item in result.evidence
        }

        assert set(composed.evidence_ids).issubset(
            available_evidence_ids
        )

        # The composer must preserve the upstream escalation decision.
        assert (
            composed.escalation_required
            is escalation.should_escalate
        )

        # Human review must reflect the upstream signals.
        assert composed.human_review_required is True

        print()
        print("Structured Response:")
        print()
        print(f"Response:")
        print(composed.response)
        print()
        print(f"Evidence IDs        : {composed.evidence_ids}")
        print(
            "Escalation required : "
            f"{composed.escalation_required}"
        )
        print(
            "Human review        : "
            f"{composed.human_review_required}"
        )

        print()
        print("=" * 70)
        print("REAL RESPONSE COMPOSER TEST PASSED")
        print("=" * 70)

    finally:
        # OllamaChatCompletionClient owns resources that should be closed
        # after the real integration test.
        await model_client.close()