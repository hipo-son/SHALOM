import pytest
from unittest.mock import MagicMock

from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import MaterialCandidate


@pytest.fixture
def mock_llm() -> MagicMock:
    """Provides a mocked LLMProvider for isolated testing."""
    return MagicMock(spec=LLMProvider)


@pytest.fixture
def sample_candidates():
    """Provides a default list of MaterialCandidates for testing selectors."""
    return [
        MaterialCandidate(
            material_name="Copper (111)",
            elements=["Cu"],
            reasoning="d-band center is appropriate.",
            expected_properties={"surface_energy": "low"},
        ),
        MaterialCandidate(
            material_name="Platinum",
            elements=["Pt"],
            reasoning="High catalytic activity.",
            expected_properties={"cost": "high"},
        ),
    ]
