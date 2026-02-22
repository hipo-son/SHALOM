"""Multi-agent specialist evaluators for candidate material assessment.

Provides parallel, multi-perspective evaluation with veto filtering,
confidence-weighted scoring, and rate-limit-safe API calls.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from shalom._config_loader import load_config, load_prompt
from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import (
    CandidateScore,
    EvaluationResponse,
    MaterialCandidate,
    RankedMaterial,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights and veto thresholds (loaded from config)
# ---------------------------------------------------------------------------

_eval_config = load_config("evaluator_weights")
DEFAULT_WEIGHTS: Dict[str, float] = _eval_config["weights"]
DEFAULT_VETO_THRESHOLDS: Dict[str, float] = _eval_config["veto_thresholds"]

# ---------------------------------------------------------------------------
# Confidence prompting rule (appended to every specialist prompt)
# ---------------------------------------------------------------------------

_CONFIDENCE_RULE = "\n\n" + load_prompt("eval_confidence_rule")

# ---------------------------------------------------------------------------
# Specialist system prompts (loaded from .md files)
# ---------------------------------------------------------------------------

_STABILITY_PROMPT = load_prompt("eval_stability") + _CONFIDENCE_RULE
_TARGET_PROPERTY_PROMPT = load_prompt("eval_target_property") + _CONFIDENCE_RULE
_DFT_FEASIBILITY_PROMPT = load_prompt("eval_dft_feasibility") + _CONFIDENCE_RULE
_SYNTHESIZABILITY_PROMPT = load_prompt("eval_synthesizability") + _CONFIDENCE_RULE
_NOVELTY_PROMPT = load_prompt("eval_novelty") + _CONFIDENCE_RULE
_ENVIRONMENTAL_COST_PROMPT = load_prompt("eval_environmental_cost") + _CONFIDENCE_RULE

_DEFAULT_PROMPTS: Dict[str, str] = {
    "stability": _STABILITY_PROMPT,
    "target_property": _TARGET_PROPERTY_PROMPT,
    "dft_feasibility": _DFT_FEASIBILITY_PROMPT,
    "synthesizability": _SYNTHESIZABILITY_PROMPT,
    "novelty": _NOVELTY_PROMPT,
    "environmental_cost": _ENVIRONMENTAL_COST_PROMPT,
}


# ---------------------------------------------------------------------------
# SpecialistEvaluator
# ---------------------------------------------------------------------------


class SpecialistEvaluator:
    """Evaluates candidates from a specific scientific perspective.

    Each evaluator scores ALL candidates in a single LLM call, returning
    an EvaluationResponse with per-candidate CandidateScore entries.
    """

    def __init__(self, llm_provider: LLMProvider, perspective: str, system_prompt: str):
        self.llm = llm_provider
        self.perspective = perspective
        self.system_prompt = system_prompt

    def evaluate(
        self, target_objective: str, candidates: List[MaterialCandidate]
    ) -> EvaluationResponse:
        """Score ALL candidates from this specialist's perspective.

        Args:
            target_objective: Natural language simulation objective.
            candidates: List of material candidates to evaluate.

        Returns:
            EvaluationResponse with scores for each candidate.
        """
        candidates_json = json.dumps(
            [c.model_dump() for c in candidates], indent=2
        )
        user_prompt = (
            f"Target Objective: {target_objective}\n\n"
            f"Candidates:\n{candidates_json}\n\n"
            f"Score each candidate from the '{self.perspective}' perspective (0.0 to 1.0). "
            "Return scores for ALL candidates."
        )

        return self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=EvaluationResponse,
        )


# ---------------------------------------------------------------------------
# Parallel evaluation with rate-limit protection
# ---------------------------------------------------------------------------


def _evaluate_with_retry(
    evaluator: SpecialistEvaluator,
    target_objective: str,
    candidates: List[MaterialCandidate],
    semaphore: threading.Semaphore,
    max_retries: int = 3,
) -> EvaluationResponse:
    """Single evaluator call with semaphore-gated rate limiting and exponential backoff."""
    for attempt in range(max_retries):
        with semaphore:
            try:
                return evaluator.evaluate(target_objective, candidates)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) + (hash(evaluator.perspective) % 1000) / 1000
                logger.warning(
                    "Evaluator '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                    evaluator.perspective,
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
    # unreachable, but satisfies type checker
    raise RuntimeError("Unreachable")  # pragma: no cover


def evaluate_parallel(
    evaluators: List[SpecialistEvaluator],
    target_objective: str,
    candidates: List[MaterialCandidate],
    max_workers: Optional[int] = None,
    max_concurrent_api_calls: int = 4,
    api_max_retries: int = 3,
) -> List[EvaluationResponse]:
    """Run all specialist evaluations in parallel with rate limit protection.

    Args:
        evaluators: List of specialist evaluators to run.
        target_objective: Natural language simulation objective.
        candidates: List of material candidates.
        max_workers: ThreadPoolExecutor worker count. Defaults to evaluator count.
        max_concurrent_api_calls: Semaphore limit to prevent RPM/TPM exhaustion.
        api_max_retries: Max retries per evaluator on API failure.

    Returns:
        List of EvaluationResponse, one per evaluator.
    """
    semaphore = threading.Semaphore(max_concurrent_api_calls)
    max_workers = max_workers or len(evaluators)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_with_retry,
                e,
                target_objective,
                candidates,
                semaphore,
                api_max_retries,
            ): e.perspective
            for e in evaluators
        }
        results: List[EvaluationResponse] = []
        for future in as_completed(futures):
            perspective = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(
                    "Evaluator '%s' failed after retries: %s", perspective, e,
                )
        if len(results) < len(futures):
            logger.warning(
                "%d/%d evaluators succeeded.", len(results), len(futures),
            )
    return results


# ---------------------------------------------------------------------------
# Score aggregation with veto filtering and confidence weighting
# ---------------------------------------------------------------------------


def aggregate_scores(
    evaluations: List[EvaluationResponse],
    candidates: List[MaterialCandidate],
    weights: Optional[Dict[str, float]] = None,
    veto_thresholds: Optional[Dict[str, float]] = None,
    strict_veto: bool = True,
) -> Tuple[Optional[RankedMaterial], List[str]]:
    """Confidence-weighted average with veto filtering.

    Steps:
        1. Collect all perspective scores for each candidate.
        2. Apply veto: if any perspective score < threshold, candidate is excluded.
           (strict_veto=False halves all thresholds for metastable material exploration.)
        3. Compute weighted average using score * confidence as effective score.
        4. Return the top-scoring candidate as RankedMaterial.
        5. If all candidates are vetoed, return (None, veto_reasons).

    Args:
        evaluations: List of EvaluationResponse from specialist evaluators.
        candidates: Original candidate list (for building RankedMaterial).
        weights: Per-perspective weight overrides.
        veto_thresholds: Per-perspective veto threshold overrides.
        strict_veto: If False, all thresholds are halved (metastable search).

    Returns:
        Tuple of (RankedMaterial or None, list of veto reason strings).
    """
    w = weights or DEFAULT_WEIGHTS
    thresholds = dict(veto_thresholds or DEFAULT_VETO_THRESHOLDS)

    if not strict_veto:
        thresholds = {k: v * 0.5 for k, v in thresholds.items()}

    # Build a lookup: material_name -> {perspective -> CandidateScore}
    score_map: Dict[str, Dict[str, CandidateScore]] = {}
    for ev in evaluations:
        for cs in ev.scores:
            if cs.material_name not in score_map:
                score_map[cs.material_name] = {}
            score_map[cs.material_name][ev.perspective] = cs

    veto_reasons: List[str] = []
    candidate_totals: List[Tuple[str, float, str]] = []  # (name, weighted_avg, justifications)

    for cand in candidates:
        name = cand.material_name
        perspectives = score_map.get(name, {})

        # Check veto
        vetoed = False
        for perspective, threshold in thresholds.items():
            if perspective in perspectives:
                cs = perspectives[perspective]
                if cs.score < threshold:
                    reason = (
                        f"{name}: vetoed by '{perspective}' "
                        f"(score={cs.score:.2f} < threshold={threshold:.2f}). "
                        f"Reason: {cs.justification}"
                    )
                    veto_reasons.append(reason)
                    vetoed = True
                    break

        if vetoed:
            continue

        # Compute confidence-weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        justification_parts: List[str] = []

        for perspective, weight in w.items():
            if perspective in perspectives:
                cs = perspectives[perspective]
                effective = cs.score * cs.confidence
                weighted_sum += weight * effective
                total_weight += weight
                justification_parts.append(
                    f"{perspective}: {cs.score:.2f} (conf={cs.confidence:.2f})"
                )

        avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        justification = "; ".join(justification_parts)
        candidate_totals.append((name, avg, justification))

    if not candidate_totals:
        return None, veto_reasons

    # Pick the best
    candidate_totals.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score, best_justification = candidate_totals[0]

    # Find the original MaterialCandidate (guard against LLM hallucinating a name)
    best_candidate = next(
        (c for c in candidates if c.material_name == best_name), None
    )
    if best_candidate is None:
        logger.warning(
            "LLM returned unknown material '%s'; falling back to top candidate.",
            best_name,
        )
        best_candidate = candidates[0]

    return (
        RankedMaterial(
            candidate=best_candidate,
            score=min(best_score, 1.0),
            ranking_justification=best_justification,
        ),
        veto_reasons,
    )


# ---------------------------------------------------------------------------
# Factory: create default evaluators
# ---------------------------------------------------------------------------


def create_default_evaluators(
    llm_provider: LLMProvider,
) -> List[SpecialistEvaluator]:
    """Create the default set of 6 specialist evaluators with physics-grounded prompts."""
    return [
        SpecialistEvaluator(llm_provider, perspective, prompt)
        for perspective, prompt in _DEFAULT_PROMPTS.items()
    ]
