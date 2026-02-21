from unittest.mock import MagicMock
from shalom.core.llm_provider import LLMProvider
from shalom.agents.design_layer import CoarseSelector
from shalom.core.schemas import MaterialCandidate


def test_llm_seed_reproducibility():
    """Verifies that repeated calls with the same input produce identical arguments.

    Ensures the seed parameter is consistently forwarded to generate_structured_output,
    supporting deterministic reproducibility without making real LLM calls.
    """
    mock_llm = MagicMock(spec=LLMProvider)

    c1 = MaterialCandidate(
        material_name="Mat1", elements=["X"], reasoning="", expected_properties={}
    )

    # Mock return value
    mock_response = MagicMock()
    mock_response.candidates = [c1]
    mock_llm.generate_structured_output.return_value = mock_response

    selector = CoarseSelector(llm_provider=mock_llm)

    # 1st call
    selector.select("Test Objective 1")
    # 2nd call
    selector.select("Test Objective 1")

    assert mock_llm.generate_structured_output.call_count == 2

    # verify that both calls passed the same arguments exactly.
    call1_args = mock_llm.generate_structured_output.call_args_list[0]
    call2_args = mock_llm.generate_structured_output.call_args_list[1]

    assert call1_args == call2_args
