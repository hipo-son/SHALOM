import json
from typing import List

from pydantic import BaseModel

from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import MaterialCandidate, RankedMaterial


class CandidateListResponse(BaseModel):
    """Wrapper schema for a list of MaterialCandidate objects."""

    candidates: List[MaterialCandidate]


class CoarseSelector:
    """Step 1 of the Design Layer: Coarse Selector.

    Initially screens 3 to 5 promising candidates from a vast chemical space
    that meet the target objective.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.system_prompt = """[v1.0.0]
        You are the "Coarse Selector", working as a world-class computational materials scientist.
        Select the 3 to 5 most promising material candidates to achieve the given
        natural language Target Objective.

        [Selection Guidelines]
        1. Use established physical/chemical intuition such as periodic table trends,
           electronegativity, and d-band center theory.
        2. Each selection must have clear scientific reasoning — no random picks.
        3. Candidates should include some diversity (alloys, doping, etc.)
           rather than being limited to a single obvious material.
        4. You MUST respond in JSON format as a Candidates list.
        """

    def select(self, target_objective: str, context: str = "") -> List[MaterialCandidate]:
        """Generate and return a list of material candidates for a given objective.

        Args:
            target_objective: The target natural language objective.
            context: Additional contextual information.

        Returns:
            A list of 3-5 selected candidate materials.
        """
        user_prompt = (
            f"Target Objective: {target_objective}\n\n"
            f"Additional Context:\n{context}\n\n"
            "Please provide 3 to 5 candidate materials."
        )

        response = self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=CandidateListResponse,
        )
        return response.candidates


class FineSelector:
    """Step 2 of the Design Layer: Fine Selector.

    Ranks the candidates provided by the Coarse Selector based on expected
    properties and selects the optimal material.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.system_prompt = """[v1.0.0]
        You are the "Fine Selector".
        You are given a small pool of material candidates that passed the coarse screening.
        Your task is to precisely evaluate each candidate against the Target Objective
        on a scale of 0.0 to 1.0, and select exactly ONE "Winner".

        [Evaluation Guidelines]
        1. Deeply analyze each candidate's 'reasoning' and 'expected_properties'.
        2. Consider the simulation (DFT) cost vs. success probability
           (overly complex or large cells are penalized).
        3. The candidate with the highest score advances to the Simulation Layer.
        """

    def rank_and_select(
        self, target_objective: str, candidates: List[MaterialCandidate]
    ) -> RankedMaterial:
        """Rank the candidates and return the top-scoring material.

        Args:
            target_objective: The target natural language objective.
            candidates: A list of candidate materials.

        Returns:
            The final selected material with score and justification.
        """
        candidates_json = json.dumps([c.model_dump() for c in candidates], indent=2)

        user_prompt = (
            f"Target Objective: {target_objective}\n\n"
            f"Candidates Pool:\n{candidates_json}\n\n"
            "Please assign a score (0.0 to 1.0) and provide ranking justification. "
            "Return ONLY the best matching material as the RankedMaterial."
        )

        return self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=RankedMaterial,
        )
