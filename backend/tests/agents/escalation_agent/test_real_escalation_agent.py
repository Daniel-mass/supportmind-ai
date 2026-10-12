from __future__ import annotations

import os

import pytest
from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.escalation_agent.agent import (
    EscalationAgent,
)
from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationAgentStatus,
    EscalationClassificationSignal,
    EscalationEvidence,
    EscalationResolutionSignal,
)


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b",
)


@pytest.mark.asyncio
async def test_real_escalation_agent_end_to_end() -> None:
    """
    Real local Escalation Agent contract test.

    This test uses:
        Ollama
        Qwen3:8b
        AutoGen AssistantAgent
        EscalationAgent
        EscalationOutputParser
        Pydantic validation

    It intentionally does NOT execute the complete SupportMind
    multi-agent workflow.

    Success criteria:
        1. Ollama responds.
        2. Escalation Agent returns structured JSON.
        3. JSON passes Pydantic validation.
        4. Evidence IDs are valid.
        5. state.recommendation is populated.
        6. Agent status is COMPLETED.
    """

    print("\n")
    print("=" * 80)
    print("REAL LOCAL ESCALATION AGENT TEST")
    print("=" * 80)
    print(f"Ollama URL : {OLLAMA_BASE_URL}")
    print(f"Model      : {OLLAMA_MODEL}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Build realistic local escalation state
    # ------------------------------------------------------------------

    state = EscalationAgentState(
        request=(
            "A customer is asking about a refund for a subscription "
            "and is frustrated because the issue has not been resolved."
        ),
        ticket_id="LOCAL-ESC-001",

        classification_signals=[
            EscalationClassificationSignal(
                name="sentiment",
                value="negative",
                confidence=0.94,
                source="classification_agent",
            ),
            EscalationClassificationSignal(
                name="complexity",
                value="MULTI_TURN",
                confidence=0.87,
                source="classification_agent",
            ),
            EscalationClassificationSignal(
                name="priority",
                value="medium",
                confidence=0.82,
                source="classification_agent",
            ),
            EscalationClassificationSignal(
                name="category",
                value="billing",
                confidence=0.91,
                source="classification_agent",
            ),
        ],

        resolution=EscalationResolutionSignal(
            resolution_type="policy_based",
            recommendation=(
                "The refund eligibility depends on the subscription "
                "plan and the applicable refund period. The customer "
                "must provide the purchase date and subscription plan "
                "details to determine eligibility."
            ),
            rationale=(
                "The supplied refund and subscription policies indicate "
                "that eligibility depends on the purchase date, plan, "
                "and applicable refund period."
            ),
            confidence=0.75,
            requires_human_review=True,
            proposed_actions=[
                "Inform the customer about the applicable refund policy.",
                "Request the purchase date.",
                "Request the subscription plan details.",
            ],
            metadata={
                "resolution_agent_status": "completed",
            },
        ),

        operational_signals=[],

        evidence=[
            EscalationEvidence(
                evidence_id="E1",
                source="refund_policy",
                content=(
                    "Refunds are allowed within 14 days of purchase "
                    "subject to the applicable subscription terms."
                ),
                source_reference="refund-policy-document",
                score=0.96,
            ),
            EscalationEvidence(
                evidence_id="E2",
                source="subscription_policy",
                content=(
                    "Refund eligibility depends on the subscription "
                    "plan and the applicable refund period."
                ),
                source_reference="subscription-policy-document",
                score=0.93,
            ),
            EscalationEvidence(
                evidence_id="E3",
                source="payment_policy",
                content=(
                    "Pending payment issues should follow the standard "
                    "payment resolution workflow unless an operational "
                    "escalation condition is present."
                ),
                source_reference="payment-policy-document",
                score=0.91,
            ),
            EscalationEvidence(
                evidence_id="E4",
                source="support_policy",
                content=(
                    "Customer frustration alone does not constitute an "
                    "automatic escalation condition."
                ),
                source_reference="support-policy-document",
                score=0.89,
            ),
            EscalationEvidence(
                evidence_id="E5",
                source="gateway_policy",
                content=(
                    "Payment gateway timeout conditions may require "
                    "specialist escalation when an active operational "
                    "condition is confirmed."
                ),
                source_reference="gateway-policy-document",
                score=0.87,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Create real Ollama model client
    # ------------------------------------------------------------------

    model_client = OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_BASE_URL,
    )

    agent = EscalationAgent(
        model_client=model_client,
    )

    try:
        # --------------------------------------------------------------
        # Run the REAL local Escalation Agent
        # --------------------------------------------------------------

        result = await agent.run(state)

        # --------------------------------------------------------------
        # Print diagnostic information
        # --------------------------------------------------------------

        print("\n")
        print("=" * 80)
        print("ESCALATION AGENT RESULT")
        print("=" * 80)

        print(f"Status           : {result.status}")
        print(f"Iterations       : {result.iteration_count}")
        print(f"Successful       : {result.successful}")
        print(f"Has recommendation: {result.has_recommendation}")
        print(f"Evidence count   : {len(result.evidence)}")

        print("\nErrors:")
        if result.errors:
            for error in result.errors:
                print(
                    f"  [{error.error_type}] "
                    f"{error.message}"
                )
        else:
            print("  None")

        print("\nStructured Recommendation:")

        if result.recommendation is not None:
            print(
                result.recommendation.model_dump_json(
                    indent=2
                )
            )
        else:
            print("  None")

        print("\nRaw Agent Response:")
        raw_response = result.execution_metadata.get(
            "agent_response"
        )

        if raw_response:
            print("-" * 80)
            print(raw_response)
            print("-" * 80)
        else:
            print("  None")

        print("\nExecution Metadata:")
        for key, value in result.execution_metadata.items():
            if key in {
                "agent_response",
                "raw_agent_response",
            }:
                continue

            print(f"  {key}: {value}")

        print("=" * 80)

        # --------------------------------------------------------------
        # Assertions
        # --------------------------------------------------------------

        assert result.status is EscalationAgentStatus.COMPLETED

        assert result.successful is True

        assert result.recommendation is not None

        assert result.has_recommendation is True

        assert len(result.errors) == 0

        recommendation = result.recommendation

        # Pydantic already guarantees this type, but keeping the
        # assertion makes the contract explicit for this test.
        assert recommendation.escalation_type is not None

        assert isinstance(
            recommendation.should_escalate,
            bool,
        )

        assert recommendation.reason.strip()

        assert recommendation.rationale.strip()

        assert 0 <= recommendation.confidence <= 1

        assert isinstance(
            recommendation.requires_human_review,
            bool,
        )

        assert isinstance(
            recommendation.evidence_ids,
            list,
        )

        # --------------------------------------------------------------
        # Evidence grounding assertion
        # --------------------------------------------------------------

        valid_evidence_ids = {
            evidence.evidence_id
            for evidence in state.evidence
        }

        assert set(
            recommendation.evidence_ids
        ).issubset(valid_evidence_ids)

        # --------------------------------------------------------------
        # Raw response must have been preserved
        # --------------------------------------------------------------

        assert result.execution_metadata.get(
            "agent_response"
        )

        print("\n")
        print("✅ REAL LOCAL ESCALATION AGENT TEST PASSED")
        print(
            "Structured recommendation was successfully "
            "produced and validated."
        )
        print("=" * 80)

    finally:
        await agent.close()