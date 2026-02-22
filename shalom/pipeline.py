"""End-to-end Pipeline Orchestrator connecting Design -> Simulation -> Review.

Usage::

    from shalom import Pipeline

    result = Pipeline(objective="Find 2D HER catalyst").run()
    print(result.status)          # "awaiting_dft"
    print(result.ranked_material) # winner
    print(result.structure_path)  # generated POSCAR path
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

from pydantic import BaseModel, Field

from shalom.agents.design_layer import CoarseSelector, MultiAgentFineSelector, get_fine_selector
from shalom.agents.review_layer import ReviewAgent
from shalom.agents.simulation_layer import GeometryGenerator, GeometryReviewer
from shalom.backends import get_backend
from shalom.backends.vasp_config import (
    AccuracyLevel,
    CalculationType,
    get_preset,
)
from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import (
    PipelineResult,
    PipelineStatus,
)

logger = logging.getLogger(__name__)

# Type alias
StepCallback = Callable[[str, dict], None]


class PipelineConfig(BaseModel):
    """Configuration for a Pipeline run."""

    backend_name: str = Field(
        default="vasp", description="DFT backend name ('vasp' or 'qe')."
    )
    provider_type: str = Field(
        default="openai", description="LLM provider type ('openai' or 'anthropic')."
    )
    model_name: str = Field(
        default="gpt-4o", description="LLM model name."
    )
    max_retries: int = Field(
        default=3, ge=1, le=10,
        description="Max retries for structure generation loop.",
    )
    max_outer_loops: int = Field(
        default=1, ge=1, le=5,
        description="Max Design->Sim->Review iterations. >1 enables closed-loop retry.",
    )
    output_dir: str = Field(
        default="pipeline_output",
        description="Base directory for generated files.",
    )
    skip_review: bool = Field(
        default=True,
        description="If True, stop after input file generation (DFT not yet run).",
    )
    selector_mode: str = Field(
        default="simple",
        description="Fine selector mode: 'simple' or 'multi_agent'.",
    )
    strict_veto: bool = Field(
        default=True,
        description="If False, halve veto thresholds for metastable material exploration.",
    )
    max_design_retries: int = Field(
        default=2, ge=0, le=5,
        description="Design micro-loop max retries when all candidates are vetoed.",
    )
    api_max_retries: int = Field(
        default=3, ge=1, le=7,
        description="API call retry count for rate-limit resilience.",
    )
    save_state: bool = Field(
        default=True,
        description="Save PipelineResult as JSON when entering AWAITING_DFT.",
    )
    calc_type: str = Field(
        default="relaxation",
        description="VASP calculation type (relaxation, static, band_structure, dos, elastic).",
    )
    accuracy: str = Field(
        default="standard",
        description="VASP accuracy level (standard or precise).",
    )
    vasp_user_incar: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User INCAR overrides (highest priority, applied on top of presets).",
    )


class Pipeline:
    """End-to-end orchestrator connecting Design -> Simulation -> Review.

    Args:
        objective: Natural language target objective.
        backend: Backend name shorthand (overrides config.backend_name).
        config: Pipeline configuration. Uses defaults if not provided.
        llm_provider: Pre-configured LLMProvider. Created from config if None.
        callbacks: Optional list of callbacks invoked after each step.
    """

    def __init__(
        self,
        objective: str,
        backend: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
        callbacks: Optional[List[StepCallback]] = None,
    ):
        self.objective = objective
        self.config = config or PipelineConfig()
        self.callbacks = callbacks or []

        if backend is not None:
            self.config = self.config.model_copy(update={"backend_name": backend})

        self.llm = llm_provider or LLMProvider(
            provider_type=self.config.provider_type,
            model_name=self.config.model_name,
        )
        self.backend = get_backend(self.config.backend_name)

    def run(self) -> PipelineResult:
        """Execute the full pipeline with optional closed-loop retry.

        Returns:
            PipelineResult containing all intermediate outputs and final status.
        """
        previous_feedback = ""

        for iteration in range(1, self.config.max_outer_loops + 1):
            result = self._run_single_iteration(iteration, previous_feedback)

            # If not a review failure or if we can't retry, return as-is
            if result.status != PipelineStatus.FAILED_REVIEW:
                return result
            if iteration >= self.config.max_outer_loops:
                return result

            # Closed-loop: feed review feedback back to design
            if result.review_result is not None:
                previous_feedback = result.review_result.feedback_for_design
                logger.info(
                    "Outer loop iteration %d/%d: feeding review feedback to design.",
                    iteration,
                    self.config.max_outer_loops,
                )

        return result  # type: ignore[possibly-undefined]

    def _run_single_iteration(
        self, iteration: int, previous_feedback: str
    ) -> PipelineResult:
        """Run one full Design -> Simulation -> Review iteration."""
        steps_completed: List[str] = []

        # ------------------------------------------------------------------
        # Step 1: Coarse Selection
        # ------------------------------------------------------------------
        logger.info("[Iter %d] Step 1: Coarse Selection...", iteration)
        try:
            coarse_selector = CoarseSelector(llm_provider=self.llm)
            candidates = coarse_selector.select(self.objective, context=previous_feedback)
        except Exception as e:
            logger.error("Coarse selection failed: %s", e)
            return PipelineResult(
                status=PipelineStatus.FAILED_DESIGN,
                objective=self.objective,
                iteration=iteration,
                error_message=f"Coarse selection failed: {e}",
                steps_completed=steps_completed,
            )

        steps_completed.append("coarse_selection")
        self._notify("coarse_selection", {"candidates": candidates})
        logger.info("Coarse selection: %d candidates.", len(candidates))

        # ------------------------------------------------------------------
        # Step 2: Fine Selection (simple or multi-agent)
        # ------------------------------------------------------------------
        logger.info("[Iter %d] Step 2: Fine Selection (mode=%s)...", iteration, self.config.selector_mode)
        try:
            fine_selector = get_fine_selector(
                self.llm,
                mode=self.config.selector_mode,
                strict_veto=self.config.strict_veto,
                api_max_retries=self.config.api_max_retries,
            )
            if self.config.selector_mode == "multi_agent" and isinstance(fine_selector, MultiAgentFineSelector):
                ranked_material = cast(MultiAgentFineSelector, fine_selector).rank_and_select(
                    self.objective,
                    candidates,
                    coarse_selector=coarse_selector,
                    max_design_retries=self.config.max_design_retries,
                )
            else:
                ranked_material = fine_selector.rank_and_select(self.objective, candidates)
        except Exception as e:
            logger.error("Fine selection failed: %s", e)
            return PipelineResult(
                status=PipelineStatus.FAILED_DESIGN,
                objective=self.objective,
                iteration=iteration,
                candidates=candidates,
                error_message=f"Fine selection failed: {e}",
                steps_completed=steps_completed,
            )

        steps_completed.append("fine_selection")
        self._notify("fine_selection", {"ranked_material": ranked_material})
        logger.info(
            "Fine selection: %s (score=%.2f).",
            ranked_material.candidate.material_name,
            ranked_material.score,
        )

        # ------------------------------------------------------------------
        # Step 3: Structure Generation
        # ------------------------------------------------------------------
        logger.info("[Iter %d] Step 3: Structure Generation...", iteration)
        try:
            # Create VASP configuration (structure-aware auto-detection deferred
            # until atoms are available inside GeometryReviewer).
            vasp_config = get_preset(
                calc_type=CalculationType(self.config.calc_type),
                accuracy=AccuracyLevel(self.config.accuracy),
            )
            if self.config.vasp_user_incar:
                vasp_config.user_incar_settings.update(self.config.vasp_user_incar)

            generator = GeometryGenerator(llm_provider=self.llm)
            reviewer = GeometryReviewer(
                generator=generator,
                max_retries=self.config.max_retries,
                backend=self.backend,
                vasp_config=vasp_config,
            )
            success, atoms, path_or_error = reviewer.run_creation_loop(
                self.objective, ranked_material
            )
        except Exception as e:
            logger.error("Structure generation failed: %s", e)
            return PipelineResult(
                status=PipelineStatus.FAILED_SIMULATION,
                objective=self.objective,
                iteration=iteration,
                candidates=candidates,
                ranked_material=ranked_material,
                error_message=f"Structure generation failed: {e}",
                steps_completed=steps_completed,
            )

        if not success:
            logger.error("Structure generation exhausted retries: %s", path_or_error)
            return PipelineResult(
                status=PipelineStatus.FAILED_SIMULATION,
                objective=self.objective,
                iteration=iteration,
                candidates=candidates,
                ranked_material=ranked_material,
                error_message=path_or_error,
                steps_completed=steps_completed,
            )

        steps_completed.append("structure_generation")
        self._notify("structure_generation", {"path": path_or_error})
        logger.info("Structure generated: %s", path_or_error)

        # ------------------------------------------------------------------
        # Step 4: Review (conditional)
        # ------------------------------------------------------------------
        if self.config.skip_review:
            logger.info(
                "Input files at '%s'. Awaiting DFT execution.", path_or_error
            )
            steps_completed.append("awaiting_dft")
            result = PipelineResult(
                status=PipelineStatus.AWAITING_DFT,
                objective=self.objective,
                iteration=iteration,
                candidates=candidates,
                ranked_material=ranked_material,
                structure_generated=True,
                structure_path=path_or_error,
                steps_completed=steps_completed,
            )
            self._save_state(result)
            return result

        logger.info("[Iter %d] Step 4: Review...", iteration)
        try:
            review_agent = ReviewAgent(llm_provider=self.llm)
            review_result = review_agent.review_with_backend(
                self.objective, path_or_error, self.backend
            )
        except Exception as e:
            logger.error("Review failed: %s", e)
            return PipelineResult(
                status=PipelineStatus.FAILED_REVIEW,
                objective=self.objective,
                iteration=iteration,
                candidates=candidates,
                ranked_material=ranked_material,
                structure_generated=True,
                structure_path=path_or_error,
                error_message=f"Review failed: {e}",
                steps_completed=steps_completed,
            )

        steps_completed.append("review")
        self._notify("review", {"review_result": review_result})

        status = (
            PipelineStatus.COMPLETED
            if review_result.is_successful
            else PipelineStatus.FAILED_REVIEW
        )
        logger.info("Review complete. Successful: %s", review_result.is_successful)

        return PipelineResult(
            status=status,
            objective=self.objective,
            iteration=iteration,
            candidates=candidates,
            ranked_material=ranked_material,
            structure_generated=True,
            structure_path=path_or_error,
            review_result=review_result,
            steps_completed=steps_completed,
        )

    def resume_from_dft(self, result: PipelineResult) -> PipelineResult:
        """Resume a pipeline from AWAITING_DFT state after DFT completes.

        The ReviewAgent will diagnose DFT failures and suggest VASP parameter
        adjustments (AMIX/BMIX, ISMEAR, ALGO, LMAXMIX, etc.) in its
        ``feedback_for_design`` field.

        Args:
            result: A PipelineResult with status AWAITING_DFT.

        Returns:
            Updated PipelineResult with review results.

        Raises:
            ValueError: If result.status is not AWAITING_DFT.
        """
        if result.status != PipelineStatus.AWAITING_DFT:
            raise ValueError(
                f"Can only resume from AWAITING_DFT, got {result.status}"
            )

        structure_path = result.structure_path or ""
        logger.info("Resuming from DFT: reviewing output at '%s'.", structure_path)
        try:
            review_agent = ReviewAgent(llm_provider=self.llm)
            review_result = review_agent.review_with_backend(
                result.objective, structure_path, self.backend
            )
        except Exception as e:
            logger.error("Review after DFT failed: %s", e)
            return result.model_copy(
                update={
                    "status": PipelineStatus.FAILED_REVIEW,
                    "error_message": f"Review after DFT failed: {e}",
                    "steps_completed": result.steps_completed + ["review_failed"],
                }
            )

        status = (
            PipelineStatus.COMPLETED
            if review_result.is_successful
            else PipelineStatus.FAILED_REVIEW
        )
        return result.model_copy(
            update={
                "status": status,
                "review_result": review_result,
                "steps_completed": result.steps_completed + ["review"],
            }
        )

    def _notify(self, step_name: str, step_data: dict) -> None:
        """Invoke all registered callbacks for the given step."""
        for callback in self.callbacks:
            try:
                callback(step_name, step_data)
            except Exception as e:
                logger.warning("Callback error on step '%s': %s", step_name, e)

    def _save_state(self, result: PipelineResult) -> None:
        """Save pipeline state to JSON for later resume (if save_state is enabled)."""
        if not self.config.save_state:
            return
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            state_path = output_dir / "pipeline_state.json"
            state_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Pipeline state saved to '%s'.", state_path)
        except Exception as e:
            logger.warning("Failed to save pipeline state: %s", e)
