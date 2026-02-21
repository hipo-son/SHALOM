import os
from shalom.agents.review_layer import ReviewAgent
from shalom.core.schemas import ReviewResult


def test_review_agent_mock(mock_llm):
    agent = ReviewAgent(llm_provider=mock_llm)

    # Verify OUTCAR parsing with the dummy fixture
    dummy_outcar_path = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_outcar.txt")

    parsed = agent.parse_outcar(dummy_outcar_path)
    assert parsed["is_converged"] is True
    assert parsed["energy"] == -34.567890

    # Mock LLM review response
    mock_result = ReviewResult(
        is_successful=True,
        energy=-34.567890,
        forces_max=0.01,
        feedback_for_design="Calculation converged successfully.",
    )
    mock_llm.generate_structured_output.return_value = mock_result

    result = agent.review("Find a stable material", dummy_outcar_path)
    assert result.is_successful is True
    assert result.energy == -34.567890
    assert mock_llm.generate_structured_output.called
