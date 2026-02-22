"""Canned LLM provider for --dry-run mode (no API calls).

Provides deterministic, pre-defined responses that exercise the full pipeline
without requiring an API key or incurring cost.
"""

from collections import deque
from typing import Any, Deque, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from shalom.agents.design_layer import CandidateListResponse
from shalom.agents.simulation_layer import GeneratorResponse
from shalom.core.schemas import (
    CandidateScore,
    EvaluationResponse,
    MaterialCandidate,
    RankedMaterial,
)

T = TypeVar("T", bound=BaseModel)


class CannedLLMProvider:
    """LLM provider returning pre-defined responses for --dry-run mode.

    Responses can be:
    - Single instance: reused for every call with that response_model.
    - List: consumed FIFO; last element reused once queue is exhausted.

    Attributes:
        provider_type: Always ``"canned"``.
        model_name: Always ``"dry-run"``.
        usage_callback: Unused, present for interface compatibility.
    """

    def __init__(self, responses: Dict[Type[BaseModel], Any]) -> None:
        self.provider_type = "canned"
        self.model_name = "dry-run"
        self.usage_callback = None

        self._responses: Dict[Type, Any] = {}
        self._queues: Dict[Type, Deque] = {}
        for model, value in responses.items():
            if isinstance(value, list):
                self._queues[model] = deque(value)
                self._responses[model] = value[-1]  # fallback = last element
            else:
                self._responses[model] = value

    def generate_structured_output(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        seed: Optional[int] = 42,
    ) -> T:
        """Return the pre-defined canned response for the given model type."""
        if response_model not in self._responses and response_model not in self._queues:
            raise ValueError(f"No canned response for {response_model.__name__}")
        queue = self._queues.get(response_model)
        if queue:
            return queue.popleft()
        return self._responses[response_model]


# ---------------------------------------------------------------------------
# Pre-built canned responses for the smoke_test scenario
# ---------------------------------------------------------------------------

_CANNED_CANDIDATES = [
    MaterialCandidate(
        material_name="TiO2 (Rutile)",
        elements=["Ti", "O"],
        reasoning=(
            "TiO2 rutile is the most studied photocatalyst. Bandgap ~3.0 eV "
            "(UV-active), but serves as an excellent baseline."
        ),
        expected_properties={"bandgap": "3.0 eV", "crystal_structure": "rutile"},
    ),
    MaterialCandidate(
        material_name="BiVO4",
        elements=["Bi", "V", "O"],
        reasoning=(
            "Monoclinic BiVO4 has a bandgap of ~2.4 eV (visible light), "
            "favorable valence band position for water oxidation."
        ),
        expected_properties={"bandgap": "2.4 eV", "crystal_structure": "monoclinic scheelite"},
    ),
    MaterialCandidate(
        material_name="Fe2O3 (Hematite)",
        elements=["Fe", "O"],
        reasoning=(
            "Hematite has a ~2.1 eV bandgap, is earth-abundant and stable "
            "in aqueous environments. Major challenge is poor charge transport."
        ),
        expected_properties={"bandgap": "2.1 eV", "crystal_structure": "corundum"},
    ),
]

_CANNED_RANKED = RankedMaterial(
    candidate=_CANNED_CANDIDATES[1],  # BiVO4 wins
    score=0.85,
    ranking_justification=(
        "BiVO4 has optimal bandgap (2.4 eV) for visible-light water splitting, "
        "favorable band edge positions, and demonstrated experimental performance."
    ),
)

_CANNED_GEOMETRY_CODE = GeneratorResponse(
    python_code=(
        "# Dry-run: simple Cu bulk as placeholder (real LLM generates BiVO4)\n"
        "atoms = bulk('Cu', 'fcc', a=3.61)\n"
    ),
    explanation="Simplified bulk structure for dry-run mode (real LLM generates target material).",
)

# For multi-agent mode: 6 evaluator responses
_CANNED_EVALUATIONS = [
    EvaluationResponse(
        perspective=perspective,
        scores=[
            CandidateScore(
                material_name=c.material_name,
                score=score,
                confidence=0.6,
                justification=f"Evaluated from {perspective} perspective.",
            )
            for c, score in zip(_CANNED_CANDIDATES, scores)
        ],
    )
    for perspective, scores in [
        ("stability", [0.9, 0.7, 0.8]),
        ("target_property", [0.4, 0.9, 0.6]),
        ("dft_feasibility", [0.9, 0.7, 0.6]),
        ("synthesizability", [0.8, 0.7, 0.7]),
        ("novelty", [0.2, 0.6, 0.3]),
        ("environmental_cost", [0.7, 0.5, 0.9]),
    ]
]

CANNED_RESPONSES: Dict[Type[BaseModel], Any] = {
    CandidateListResponse: CandidateListResponse(candidates=_CANNED_CANDIDATES),
    RankedMaterial: _CANNED_RANKED,
    GeneratorResponse: _CANNED_GEOMETRY_CODE,
    EvaluationResponse: _CANNED_EVALUATIONS,  # list → queue-based consumption
}
