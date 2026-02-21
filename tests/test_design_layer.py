from unittest.mock import MagicMock

from shalom.core.schemas import MaterialCandidate, RankedMaterial
from shalom.core.llm_provider import LLMProvider
from shalom.agents.design_layer import CoarseSelector, FineSelector


def test_coarse_selector_mock():
    """Tests CoarseSelector data flow with a mocked LLM."""
    # 1. Create mock LLM provider
    mock_llm = MagicMock(spec=LLMProvider)

    # 2. Define mock response data
    mock_response = MagicMock()
    mock_response.candidates = [
        MaterialCandidate(
            material_name="Copper (111)",
            elements=["Cu"],
            reasoning="d-band center is appropriate for intermediate binding.",
            expected_properties={"surface_energy": "low"},
        ),
        MaterialCandidate(
            material_name="Platinum",
            elements=["Pt"],
            reasoning="High catalytic activity.",
            expected_properties={"cost": "high"},
        ),
    ]

    # Configure mock to return fake response on generate_structured_output
    mock_llm.generate_structured_output.return_value = mock_response

    # 3. Run agent
    selector = CoarseSelector(llm_provider=mock_llm)
    candidates = selector.select("Find a good catalyst for hydrogen evolution")

    # 4. Verify
    assert len(candidates) == 2
    assert candidates[0].elements == ["Cu"]
    assert mock_llm.generate_structured_output.called


def test_fine_selector_mock():
    """Tests FineSelector pipeline with a mocked LLM."""
    mock_llm = MagicMock(spec=LLMProvider)

    candidates = [
        MaterialCandidate(
            material_name="Copper (111)", elements=["Cu"], reasoning="Good", expected_properties={}
        ),
        MaterialCandidate(
            material_name="Platinum", elements=["Pt"], reasoning="Better", expected_properties={}
        ),
    ]

    # Mock top-ranked candidate response
    mock_llm.generate_structured_output.return_value = RankedMaterial(
        candidate=candidates[1],
        score=0.95,
        ranking_justification="Pt has the best theoretical performance.",
    )

    selector = FineSelector(llm_provider=mock_llm)
    winner = selector.rank_and_select("Best catalyst", candidates)

    assert winner.score == 0.95
    assert winner.candidate.material_name == "Platinum"
