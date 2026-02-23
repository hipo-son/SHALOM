"""End-to-end Pipeline Orchestrator connecting Design -> Simulation -> Review.

Usage::

    from shalom import Pipeline

    result = Pipeline(objective="Find 2D HER catalyst").run()
    print(result.status)          # "awaiting_dft"
    print(result.ranked_material) # winner
    print(result.structure_path)  # generated POSCAR path
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

from pydantic import BaseModel, Field

from shalom.agents.design_layer import CoarseSelector, MultiAgentFineSelector, get_fine_selector
from shalom.agents.review_layer import ReviewAgent
from shalom.agents.simulation_layer import GeometryGenerator, GeometryReviewer
from shalom.backends import get_backend
from shalom.backends._physics import AccuracyLevel
from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import (
    MaterialCandidate,
    PipelineResult,
    PipelineStatus,
    PipelineStep,
    RankedMaterial,
)

logger = logging.getLogger(__name__)

# Valid step names for validation
_VALID_STEPS = frozenset(s.value for s in PipelineStep)

# Type alias
StepCallback = Callable[[str, dict], None]


def synthesize_ranked_material(material_name: str) -> RankedMaterial:
    """Create a RankedMaterial from a formula string with physics-aware defaults.

    Uses ``_physics.py`` lookups for magnetic element detection and GGA+U necessity,
    and ASE's ``string2symbols`` for robust formula parsing (not raw regex).

    Args:
        material_name: Chemical formula (e.g., ``"Fe2O3"``, ``"MoS2"``, ``"Si"``).

    Returns:
        A ``RankedMaterial`` with auto-populated ``expected_properties``.
    """
    from ase.symbols import string2symbols

    from shalom.backends._physics import ANION_ELEMENTS, MAGNETIC_ELEMENTS

    elements = list(dict.fromkeys(string2symbols(material_name)))
    element_set = set(elements)

    has_magnetic = bool(element_set & set(MAGNETIC_ELEMENTS))
    has_anion = bool(element_set & ANION_ELEMENTS)
    needs_hubbard_u = has_magnetic and has_anion

    expected_properties: Dict[str, Any] = {}
    if has_magnetic:
        expected_properties["magnetic"] = True
    if needs_hubbard_u:
        expected_properties["needs_hubbard_u"] = True

    candidate = MaterialCandidate(
        material_name=material_name,
        elements=elements,
        reasoning=f"User-specified material: {material_name}",
        expected_properties=expected_properties,
    )
    return RankedMaterial(
        candidate=candidate,
        score=1.0,
        ranking_justification="User-specified (Design layer skipped)",
    )


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
        default="",
        description=(
            "Base directory for pipeline outputs. "
            "Defaults to $SHALOM_WORKSPACE/pipeline (or ~/Desktop/shalom-runs/pipeline)."
        ),
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
    execute: bool = Field(
        default=False,
        description="Run DFT via subprocess after input generation.",
    )
    nprocs: int = Field(
        default=1, ge=1,
        description="MPI process count for DFT execution.",
    )
    mpi_command: str = Field(
        default="mpirun",
        description="MPI launcher command (mpirun, srun, etc.).",
    )
    execution_timeout: int = Field(
        default=86400,
        description="DFT execution timeout in seconds.",
    )
    max_execution_retries: int = Field(
        default=3, ge=0, le=10,
        description="Max error recovery retries during execution.",
    )
    calc_type: str = Field(
        default="relaxation",
        description=(
            "Calculation type. VASP: relaxation/static/band_structure/dos/elastic. "
            "QE: scf/relax/vc-relax/bands/nscf."
        ),
    )
    accuracy: str = Field(
        default="standard",
        description="Accuracy level (standard or precise).",
    )
    vasp_user_incar: Optional[Dict[str, Any]] = Field(
        default=None,
        description="VASP INCAR overrides (highest priority, applied on top of presets).",
    )
    qe_user_settings: Optional[Dict[str, Any]] = Field(
        default=None,
        description="QE namelist overrides (e.g., {'system.ecutwfc': 80}).",
    )

    # --- Flexible pipeline step configuration ---
    steps: Optional[List[str]] = Field(
        default=None,
        description=(
            "Explicit list of pipeline steps to run: 'design', 'simulation', 'review'. "
            "None derives steps from skip_review for backward compatibility."
        ),
    )
    material_name: Optional[str] = Field(
        default=None,
        description="Material formula when skipping Design (e.g., 'MoS2', 'Fe2O3').",
    )
    input_structure_path: Optional[str] = Field(
        default=None,
        description="Path to existing structure/DFT output when skipping Simulation.",
    )
    input_ranked_material: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Serialized RankedMaterial dict for direct injection (preserves full physics context). "
            "Takes priority over material_name."
        ),
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

        # Resolve effective steps
        self._effective_steps = self._resolve_effective_steps()
        self._validate_step_requirements()

        self.llm = llm_provider or LLMProvider(
            provider_type=self.config.provider_type,
            model_name=self.config.model_name,
        )
        self.backend = get_backend(self.config.backend_name)

    def _resolve_effective_steps(self) -> List[str]:
        """Derive effective steps from config.steps or skip_review (backward compat)."""
        if self.config.steps is not None:
            steps = list(self.config.steps)
            if not steps:
                raise ValueError("steps list must not be empty when explicitly provided.")
            invalid = set(steps) - _VALID_STEPS
            if invalid:
                raise ValueError(
                    f"Invalid step name(s): {invalid}. Valid: {sorted(_VALID_STEPS)}"
                )
        else:
            # Backward compatibility: derive from skip_review
            if self.config.skip_review:
                steps = ["design", "simulation"]
            else:
                steps = ["design", "simulation", "review"]

        # Auto-insert execution step if --execute enabled
        if self.config.execute and "execution" not in steps:
            if "review" in steps:
                idx = steps.index("review")
                steps.insert(idx, "execution")
            else:
                steps.append("execution")

        return steps

    def _validate_step_requirements(self) -> None:
        """Validate that required inputs are provided for the selected steps."""
        steps = self._effective_steps

        if "design" not in steps:
            has_material = (
                self.config.material_name is not None
                or self.config.input_ranked_material is not None
            )
            if "simulation" in steps and not has_material:
                raise ValueError(
                    "material_name or input_ranked_material is required "
                    "when skipping 'design' with 'simulation' enabled."
                )

        if "simulation" not in steps and "review" in steps:
            if self.config.input_structure_path is None:
                raise ValueError(
                    "input_structure_path is required when skipping 'simulation' "
                    "with 'review' enabled."
                )

        if "execution" in steps and "simulation" not in steps:
            if self.config.input_structure_path is None:
                raise ValueError(
                    "input_structure_path is required when running 'execution' "
                    "without 'simulation'."
                )

    def run(self) -> PipelineResult:
        """Execute the full pipeline with optional closed-loop retry.

        Returns:
            PipelineResult containing all intermediate outputs and final status.
        """
        start_time = time.monotonic()
        previous_feedback = ""
        config_snapshot = self.config.model_dump()

        for iteration in range(1, self.config.max_outer_loops + 1):
            result = self._run_single_iteration(iteration, previous_feedback)

            # Attach timing and config snapshot
            elapsed = time.monotonic() - start_time
            result = result.model_copy(update={
                "elapsed_seconds": elapsed,
                "config_snapshot": config_snapshot,
            })

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
        """Run one iteration with step-aware execution.

        Steps are determined by ``self._effective_steps`` (resolved from config).
        """
        steps = self._effective_steps
        steps_completed: List[str] = []
        candidates = None
        ranked_material = None
        structure_path: Optional[str] = None
        atoms: Optional[Any] = None
        dft_config: Optional[Any] = None
        exec_wall_time: Optional[float] = None
        corr_history: Optional[List[Dict[str, Any]]] = None
        quality_warnings: List[str] = []

        # ------------------------------------------------------------------
        # Step 1: Design (Coarse + Fine Selection)
        # ------------------------------------------------------------------
        if "design" in steps:
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

            logger.info(
                "[Iter %d] Step 2: Fine Selection (mode=%s)...",
                iteration, self.config.selector_mode,
            )
            try:
                fine_selector = get_fine_selector(
                    self.llm,
                    mode=self.config.selector_mode,
                    strict_veto=self.config.strict_veto,
                    api_max_retries=self.config.api_max_retries,
                )
                if self.config.selector_mode == "multi_agent" and isinstance(
                    fine_selector, MultiAgentFineSelector
                ):
                    ranked_material = cast(
                        MultiAgentFineSelector, fine_selector
                    ).rank_and_select(
                        self.objective,
                        candidates,
                        coarse_selector=coarse_selector,
                        max_design_retries=self.config.max_design_retries,
                    )
                else:
                    ranked_material = fine_selector.rank_and_select(
                        self.objective, candidates
                    )
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
        else:
            # Design skipped — resolve ranked_material from config
            if self.config.input_ranked_material is not None:
                ranked_material = RankedMaterial.model_validate(
                    self.config.input_ranked_material
                )
                logger.info(
                    "Design skipped: using injected RankedMaterial (%s).",
                    ranked_material.candidate.material_name,
                )
            elif self.config.material_name is not None:
                ranked_material = synthesize_ranked_material(self.config.material_name)
                logger.info(
                    "Design skipped: synthesized RankedMaterial for '%s'.",
                    self.config.material_name,
                )
            steps_completed.append("design_skipped")

        # Design-only exit
        if "simulation" not in steps and "review" not in steps:
            logger.info("Design-only mode: returning COMPLETED_DESIGN.")
            return PipelineResult(
                status=PipelineStatus.COMPLETED_DESIGN,
                objective=self.objective,
                iteration=iteration,
                candidates=candidates,
                ranked_material=ranked_material,
                steps_completed=steps_completed,
            )

        # ------------------------------------------------------------------
        # Step 2: Simulation (Structure Generation)
        # ------------------------------------------------------------------
        # Create DFT config (needed for both simulation and execution)
        if "simulation" in steps or "execution" in steps:
            dft_config = self._create_dft_config()

        if "simulation" in steps:
            logger.info("[Iter %d] Structure Generation...", iteration)
            if ranked_material is None:
                return PipelineResult(
                    status=PipelineStatus.FAILED_SIMULATION,
                    objective=self.objective,
                    iteration=iteration,
                    error_message="No ranked_material available for structure generation.",
                    steps_completed=steps_completed,
                )
            try:
                from shalom.mp_client import is_mp_available, fetch_structure
                mp_success = False

                if is_mp_available():
                    try:
                        logger.info("Attempting MP structure fetch for '%s'", ranked_material.candidate.material_name)
                        mp_result = fetch_structure(ranked_material.candidate.material_name)
                        atoms = mp_result.atoms
                        
                        import copy
                        import re
                        from shalom.tools.ase_builder import ASEBuilder
                        from shalom.agents.simulation_layer import _apply_structure_hints

                        output_dir = "generated_structures"
                        mat_name = re.sub(r'[^\w\-.]', '_', ranked_material.candidate.material_name)

                        if self.backend is not None:
                            if self.backend.name == "qe":
                                write_params = {}
                            else:
                                write_params = {"filename": f"POSCAR_{mat_name}"}
                            if dft_config is not None:
                                effective_config = copy.deepcopy(dft_config)
                                _apply_structure_hints(self.backend.name, atoms, effective_config)
                                write_params["config"] = effective_config
                            path_or_error = self.backend.write_input(
                                atoms, output_dir, **write_params
                            )
                        else:
                            path_or_error = ASEBuilder.save_poscar(
                                atoms, filename=f"POSCAR_{mat_name}", directory=output_dir
                            )
                        success = True
                        mp_success = True
                        logger.info("Successfully fetched structure from Materials Project.")
                    except Exception as mp_e:
                        logger.warning("MP structure fetch failed: %s. Falling back to AI Geometry Generator.", mp_e)

                if not mp_success:
                    generator = GeometryGenerator(llm_provider=self.llm)
                    reviewer = GeometryReviewer(
                        generator=generator,
                        max_retries=self.config.max_retries,
                        backend=self.backend,
                        dft_config=dft_config,
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
                logger.error(
                    "Structure generation exhausted retries: %s", path_or_error
                )
                return PipelineResult(
                    status=PipelineStatus.FAILED_SIMULATION,
                    objective=self.objective,
                    iteration=iteration,
                    candidates=candidates,
                    ranked_material=ranked_material,
                    error_message=path_or_error,
                    steps_completed=steps_completed,
                )

            structure_path = path_or_error
            steps_completed.append("structure_generation")
            self._notify("structure_generation", {"path": structure_path})
            logger.info("Structure generated: %s", structure_path)
        else:
            # Simulation skipped — use provided structure path
            structure_path = self.config.input_structure_path
            steps_completed.append("simulation_skipped")
            logger.info("Simulation skipped: using '%s'.", structure_path)

        # ------------------------------------------------------------------
        # Step 2.5: Execution (DFT via subprocess)
        # ------------------------------------------------------------------
        if "execution" in steps:
            logger.info("[Iter %d] Executing DFT...", iteration)

            # Load atoms from file if simulation was skipped
            if atoms is None and structure_path:
                import os
                from ase.io import read as ase_read
                pw_in_path = os.path.join(structure_path, "pw.in")
                if os.path.exists(pw_in_path):
                    loaded = ase_read(pw_in_path)
                    atoms = loaded[0] if isinstance(loaded, list) else loaded

            from shalom.backends.runner import (
                ExecutionConfig, ExecutionRunner, execute_with_recovery,
            )
            from shalom.backends.qe_error_recovery import QEErrorRecoveryEngine

            exec_config = ExecutionConfig(
                nprocs=self.config.nprocs,
                mpi_command=self.config.mpi_command,
                timeout_seconds=self.config.execution_timeout,
            )
            runner = ExecutionRunner(config=exec_config)
            recovery = QEErrorRecoveryEngine()

            exec_result, dft_result, corr_history = execute_with_recovery(
                self.backend, runner, recovery,
                structure_path or "",
                dft_config,
                atoms,
                max_retries=self.config.max_execution_retries,
            )

            exec_wall_time = exec_result.wall_time_seconds
            if dft_result is not None:
                quality_warnings = dft_result.quality_warnings

            if not exec_result.success and (
                dft_result is None or not dft_result.is_converged
            ):
                return PipelineResult(
                    status=PipelineStatus.FAILED_EXECUTION,
                    objective=self.objective,
                    iteration=iteration,
                    candidates=candidates,
                    ranked_material=ranked_material,
                    structure_generated="simulation" in steps,
                    structure_path=structure_path,
                    execution_wall_time=exec_wall_time,
                    correction_history=corr_history,
                    quality_warnings=quality_warnings,
                    error_message=exec_result.error_message or "DFT execution failed",
                    steps_completed=steps_completed,
                )

            steps_completed.append("execution")
            self._notify("execution", {
                "wall_time": exec_wall_time,
                "correction_history": corr_history,
            })
            logger.info(
                "DFT execution completed in %.1fs (%d corrections).",
                exec_wall_time, len(corr_history or []),
            )

        # ------------------------------------------------------------------
        # Step 3: Review
        # ------------------------------------------------------------------
        if "review" not in steps:
            # If execution ran, return COMPLETED; otherwise AWAITING_DFT
            if "execution" in steps:
                logger.info("Execution complete (no review). Returning COMPLETED.")
                return PipelineResult(
                    status=PipelineStatus.COMPLETED,
                    objective=self.objective,
                    iteration=iteration,
                    candidates=candidates,
                    ranked_material=ranked_material,
                    structure_generated="simulation" in steps,
                    structure_path=structure_path,
                    execution_wall_time=exec_wall_time,
                    correction_history=corr_history,
                    quality_warnings=quality_warnings,
                    steps_completed=steps_completed,
                )

            logger.info(
                "Input files at '%s'. Awaiting DFT execution.", structure_path
            )
            steps_completed.append("awaiting_dft")
            result = PipelineResult(
                status=PipelineStatus.AWAITING_DFT,
                objective=self.objective,
                iteration=iteration,
                candidates=candidates,
                ranked_material=ranked_material,
                structure_generated="simulation" in steps,
                structure_path=structure_path,
                steps_completed=steps_completed,
            )
            self._save_state(result)
            return result

        logger.info("[Iter %d] Review...", iteration)
        try:
            review_agent = ReviewAgent(llm_provider=self.llm)
            review_result = review_agent.review_with_backend(
                self.objective, structure_path or "", self.backend
            )
        except Exception as e:
            logger.error("Review failed: %s", e)
            return PipelineResult(
                status=PipelineStatus.FAILED_REVIEW,
                objective=self.objective,
                iteration=iteration,
                candidates=candidates,
                ranked_material=ranked_material,
                structure_generated="simulation" in steps,
                structure_path=structure_path,
                execution_wall_time=exec_wall_time,
                correction_history=corr_history,
                quality_warnings=quality_warnings,
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
            structure_generated="simulation" in steps,
            structure_path=structure_path,
            execution_wall_time=exec_wall_time,
            correction_history=corr_history,
            quality_warnings=quality_warnings,
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

    def _create_dft_config(self) -> Any:
        """Create backend-specific DFT configuration from pipeline settings."""
        if self.config.backend_name.lower() == "vasp":
            from shalom.backends.vasp_config import CalculationType, get_preset
            vasp_config = get_preset(
                calc_type=CalculationType(self.config.calc_type),
                accuracy=AccuracyLevel(self.config.accuracy),
            )
            if self.config.vasp_user_incar:
                vasp_config.user_incar_settings.update(self.config.vasp_user_incar)
            return vasp_config
        elif self.config.backend_name.lower() == "qe":
            from shalom.backends.qe_config import (
                get_qe_preset, VASP_TO_QE_CALC_MAP, QECalculationType,
            )
            qe_calc = VASP_TO_QE_CALC_MAP.get(
                self.config.calc_type,
                QECalculationType(self.config.calc_type)
                if self.config.calc_type in {e.value for e in QECalculationType}
                else QECalculationType.SCF,
            )
            qe_config = get_qe_preset(
                calc_type=qe_calc,
                accuracy=AccuracyLevel(self.config.accuracy),
            )
            if self.config.qe_user_settings:
                qe_config.user_settings.update(self.config.qe_user_settings)
            return qe_config
        logger.warning("Unknown backend '%s', returning None config.", self.config.backend_name)
        return None

    def _notify(self, step_name: str, step_data: dict) -> None:
        """Invoke all registered callbacks for the given step."""
        for callback in self.callbacks:
            try:
                callback(step_name, step_data)
            except Exception as e:
                logger.warning("Callback error on step '%s': %s", step_name, e)

    def _save_state(self, result: PipelineResult) -> None:
        """Save pipeline state and config to JSON for later resume."""
        if not self.config.save_state:
            return
        try:
            from shalom.direct_run import _resolve_workspace
            _base = self.config.output_dir or str(
                Path(_resolve_workspace()) / "pipeline"
            )
            output_dir = Path(_base)
            output_dir.mkdir(parents=True, exist_ok=True)
            state_path = output_dir / "pipeline_state.json"
            state_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            config_path = output_dir / "pipeline_config.json"
            config_path.write_text(
                self.config.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info("Pipeline state saved to '%s'.", state_path)
        except Exception as e:
            logger.warning("Failed to save pipeline state: %s", e)
