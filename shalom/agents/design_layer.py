import json
import logging
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from shalom.agents.evaluators import (
    aggregate_scores,
    create_default_evaluators,
    evaluate_parallel,
    SpecialistEvaluator,
)
from shalom._config_loader import load_prompt
from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import EvaluationDetails, MaterialCandidate, RankedMaterial

logger = logging.getLogger(__name__)


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
        self.system_prompt = load_prompt("coarse_selector")

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
        self.system_prompt = load_prompt("fine_selector")

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


class MultiAgentFineSelector:
    """Multi-perspective candidate evaluation with veto logic and parallel execution.

    Uses 6 specialist evaluators to score candidates from different scientific
    perspectives, applies veto filtering, and selects the top candidate via
    confidence-weighted averaging.

    Includes a Design Micro-loop: when ALL candidates are vetoed, the veto
    reasons are fed back to a CoarseSelector (if provided) to generate a
    fresh candidate pool, up to ``max_design_retries`` times.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        evaluators: Optional[List[SpecialistEvaluator]] = None,
        weights: Optional[Dict[str, float]] = None,
        veto_thresholds: Optional[Dict[str, float]] = None,
        strict_veto: bool = True,
        parallel: bool = True,
        api_max_retries: int = 3,
    ):
        self.llm = llm_provider
        self.evaluators = evaluators or create_default_evaluators(llm_provider)
        self.weights = weights
        self.veto_thresholds = veto_thresholds
        self.strict_veto = strict_veto
        self.parallel = parallel
        self.api_max_retries = api_max_retries

    def _run_evaluations(
        self, target_objective: str, candidates: List[MaterialCandidate]
    ) -> list:
        """Run all evaluators (parallel or sequential)."""
        if self.parallel:
            return evaluate_parallel(
                self.evaluators,
                target_objective,
                candidates,
                api_max_retries=self.api_max_retries,
            )
        return [e.evaluate(target_objective, candidates) for e in self.evaluators]

    def rank_and_select(
        self,
        target_objective: str,
        candidates: List[MaterialCandidate],
        coarse_selector: Optional["CoarseSelector"] = None,
        max_design_retries: int = 2,
    ) -> RankedMaterial:
        """Evaluate candidates with 6 specialists, apply veto, and select winner.

        If all candidates are vetoed and a ``coarse_selector`` is provided,
        veto reasons are fed back as context to generate fresh candidates
        (Design Micro-loop), up to ``max_design_retries`` times.

        Args:
            target_objective: Natural language simulation objective.
            candidates: List of material candidates to evaluate.
            coarse_selector: Optional CoarseSelector for the micro-loop.
            max_design_retries: Max micro-loop iterations.

        Returns:
            The winning RankedMaterial with confidence-weighted score.
        """
        current_candidates = candidates

        for retry in range(max_design_retries + 1):
            evaluations = self._run_evaluations(target_objective, current_candidates)
            result, veto_reasons = aggregate_scores(
                evaluations,
                current_candidates,
                weights=self.weights,
                veto_thresholds=self.veto_thresholds,
                strict_veto=self.strict_veto,
            )

            if result is not None:
                return result.model_copy(update={
                    "evaluation_details": EvaluationDetails(
                        evaluations=evaluations,
                        veto_reasons=veto_reasons,
                        micro_loop_retries=retry,
                    ),
                })

            # All candidates vetoed
            if coarse_selector is None or retry == max_design_retries:
                # Fallback: pick the original first candidate with a warning
                logger.warning(
                    "All candidates vetoed after %d retries. Returning fallback.",
                    retry + 1,
                )
                fallback = RankedMaterial(
                    candidate=current_candidates[0],
                    score=0.0,
                    ranking_justification=(
                        "WARNING: All candidates were vetoed by specialist evaluators. "
                        "This is a fallback selection. Veto reasons: "
                        + "; ".join(veto_reasons[:3])
                    ),
                )
                return fallback.model_copy(update={
                    "evaluation_details": EvaluationDetails(
                        evaluations=evaluations,
                        veto_reasons=veto_reasons,
                        micro_loop_retries=max_design_retries,
                    ),
                })

            # Micro-loop: feed veto reasons back to CoarseSelector
            veto_feedback = "\n".join(veto_reasons)
            logger.warning(
                "All %d candidates vetoed. Re-running CoarseSelector (retry %d/%d).",
                len(current_candidates),
                retry + 1,
                max_design_retries,
            )
            current_candidates = coarse_selector.select(
                target_objective,
                context=(
                    f"[VETO FEEDBACK FROM SPECIALIST EVALUATORS]\n{veto_feedback}\n\n"
                    "The previous candidates were ALL rejected. "
                    "Please suggest new candidates that avoid these issues."
                ),
            )

        # Should not reach here, but safety fallback
        return RankedMaterial(  # pragma: no cover
            candidate=candidates[0],
            score=0.0,
            ranking_justification="Fallback: micro-loop exhausted.",
        )


def get_fine_selector(
    llm_provider: LLMProvider,
    mode: str = "simple",
    **kwargs,
) -> Union[FineSelector, MultiAgentFineSelector]:
    """Factory function to create a FineSelector by mode.

    Args:
        llm_provider: The LLM provider instance.
        mode: ``"simple"`` for single-LLM FineSelector,
              ``"multi_agent"`` for multi-perspective evaluation.
        **kwargs: Additional keyword arguments passed to MultiAgentFineSelector.

    Returns:
        A FineSelector or MultiAgentFineSelector instance.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode == "simple":
        return FineSelector(llm_provider)
    if mode == "multi_agent":
        return MultiAgentFineSelector(llm_provider, **kwargs)
    raise ValueError(f"Unknown selector mode: '{mode}'. Use 'simple' or 'multi_agent'.")
