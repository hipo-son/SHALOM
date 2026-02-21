import os
import sys

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from shalom.core.llm_provider import LLMProvider
from shalom.agents.design_layer import CoarseSelector, FineSelector


@pytest.mark.integration
def test_design_layer_api_integration():
    """End-to-end integration test for the Design Layer using a real LLM API."""
    print("=== SHALOM Design Layer API Integration Test ===")

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("No OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable set.")

    # Provider selection (priority: OpenAI -> Anthropic)
    provider_type = "openai" if os.environ.get("OPENAI_API_KEY") else "anthropic"
    model_name = "gpt-4o" if provider_type == "openai" else "claude-sonnet-4-6"

    print(f"\n[1] Initializing LLM Provider: {provider_type} (model: {model_name})")
    llm = LLMProvider(provider_type=provider_type, model_name=model_name)

    target_objective = (
        "A highly efficient, low-cost transition metal alloy surface "
        "for Hydrogen Evolution Reaction (HER)."
    )
    print(f"\n[2] Target Objective: {target_objective}")

    # Stage 1: Coarse Selector
    print("\n[3] Running Coarse Selector...")
    coarse_selector = CoarseSelector(llm_provider=llm)
    try:
        candidates = coarse_selector.select(target_objective)
        print(f"-> {len(candidates)} candidates selected:")
        for i, c in enumerate(candidates):
            print(f"   [{i + 1}] {c.material_name} (elements: {c.elements})")
            print(f"       reasoning: {c.reasoning}")
    except Exception as e:
        pytest.fail(f"Coarse Selector execution error: {e}")

    if not candidates:
        pytest.fail("No candidates were selected.")

    # Stage 2: Fine Selector
    print("\n[4] Running Fine Selector...")
    fine_selector = FineSelector(llm_provider=llm)
    try:
        winner = fine_selector.rank_and_select(target_objective, candidates)
        print("\n=== [Winner] ===")
        print(f"Material: {winner.candidate.material_name}")
        print(f"Reasoning: {winner.candidate.reasoning}")
        print(f"Score: {winner.score}")
        print(f"Justification:\n{winner.ranking_justification}")
    except Exception as e:
        pytest.fail(f"Fine Selector execution error: {e}")


if __name__ == "__main__":
    test_design_layer_api_integration()
