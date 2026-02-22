"""Multi-agent specialist evaluators for candidate material assessment.

Provides parallel, multi-perspective evaluation with veto filtering,
confidence-weighted scoring, and rate-limit-safe API calls.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import (
    CandidateScore,
    EvaluationResponse,
    MaterialCandidate,
    RankedMaterial,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights and veto thresholds
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "stability": 0.20,
    "target_property": 0.25,
    "dft_feasibility": 0.15,
    "synthesizability": 0.15,
    "novelty": 0.15,
    "environmental_cost": 0.10,
}

DEFAULT_VETO_THRESHOLDS: Dict[str, float] = {
    "stability": 0.3,
    "target_property": 0.2,
    "dft_feasibility": 0.2,
    "synthesizability": 0.2,
    "novelty": 0.0,  # no veto for novelty
    "environmental_cost": 0.2,
}

# ---------------------------------------------------------------------------
# Confidence prompting rule (appended to every specialist prompt)
# ---------------------------------------------------------------------------

_CONFIDENCE_RULE = (
    "\n\n[CONFIDENCE RULE] Set confidence > 0.5 ONLY if you can cite specific "
    "evidence from crystallographic databases (ICSD, Materials Project, AFLOW) "
    "or published literature. If your assessment is based on chemical intuition "
    "without concrete references, confidence MUST be <= 0.5."
)

# ---------------------------------------------------------------------------
# Specialist system prompts
# ---------------------------------------------------------------------------

_STABILITY_PROMPT = """[v1.0.0] You are the "Stability Evaluator".
Score each candidate material on thermodynamic and kinetic stability (0.0 to 1.0).

[Evaluation Criteria]
1. Evaluate using formation energy trends, convex hull distance, and known phase diagrams.
2. Reference Hume-Rothery rules for alloys (atomic size, electronegativity, valence electron count).
3. If evaluating as metastable (when explicitly noted), identify the required stabilization
   mechanism (e.g., epitaxial substrate matching, high-pressure synthesis phase, kinetic
   trapping via rapid quenching).
4. Consider decomposition pathways and competing phases.""" + _CONFIDENCE_RULE

_TARGET_PROPERTY_PROMPT = """[v1.0.0] You are the "Target Property Evaluator".
Score each candidate on alignment with the target objective (0.0 to 1.0).

[Evaluation Criteria]
1. First, identify the dominant physical descriptors for the given target objective:
   - Catalysis: d-band center, adsorption energy, surface reactivity
   - Quantum materials: band topology, flat bands, spin-orbit coupling
   - Battery cathodes: ion radius, diffusion activation barrier, voltage
   - Thermoelectrics: Seebeck coefficient, lattice thermal conductivity, power factor
2. Then score each candidate based on alignment with these descriptors.
3. Use known structure-property relationships and electronegativity differences.""" + _CONFIDENCE_RULE

_DFT_FEASIBILITY_PROMPT = """[v1.0.0] You are the "DFT Feasibility Evaluator".
Score each candidate on computational cost and convergence difficulty (0.0 to 1.0).

[Evaluation Criteria]
1. Number of atoms in the primitive cell (fewer = cheaper).
2. Magnetic ordering complexity (AFM, spin-frustrated systems are harder).
3. Electron correlation: does it need DFT+U or hybrid functionals (HSE06)?
4. Expected SCF convergence difficulty (metallic surfaces, charge sloshing).
5. A simple bulk crystal scores high; a large supercell with defects scores low.""" + _CONFIDENCE_RULE

_SYNTHESIZABILITY_PROMPT = """[v1.0.0] You are the "Synthesizability Evaluator".
Score each candidate on experimental synthesis feasibility (0.0 to 1.0).

[Evaluation Criteria]
1. Competing phase analysis and synthesis energy barriers.
2. Availability of known precursor reaction pathways.
3. Goldschmidt tolerance factor for perovskites, Hume-Rothery rules for alloys.
4. Do NOT penalize solely for lack of prior experimental reports — assess the
   thermodynamic/kinetic pathway feasibility instead.
5. Consider whether similar compositions or structural motifs have been synthesized.""" + _CONFIDENCE_RULE

_NOVELTY_PROMPT = """[v1.0.0] You are the "Novelty Evaluator".
Score each candidate on scientific novelty and originality (0.0 to 1.0).

[Evaluation Criteria]
1. Is this a trivial elemental substitution of a known material (low novelty)?
2. Does it introduce genuinely new structural motifs, compositions, or design principles?
3. Would this material generate interest in top-tier journals (Nature, Science)?
4. Consider the gap between known materials and this candidate.""" + _CONFIDENCE_RULE

_ENVIRONMENTAL_COST_PROMPT = """[v1.0.0] You are the "Environmental & Cost Evaluator".
Score each candidate on element availability and environmental impact (0.0 to 1.0).

[Evaluation Criteria]
1. Penalize use of rare/expensive elements: Ir, Ru, Re, Os, Rh, Pd.
2. Penalize toxic elements: Pb, Tl, Cd, Hg, As.
3. Penalize conflict minerals where applicable.
4. Prefer earth-abundant alternatives (Fe, Cu, Mn, Ni, Ti, Zn, Al).
5. A material using only abundant, non-toxic elements scores ~1.0.""" + _CONFIDENCE_RULE

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
        import json

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
            results.append(future.result())
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

    # Find the original MaterialCandidate
    best_candidate = next(c for c in candidates if c.material_name == best_name)

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
