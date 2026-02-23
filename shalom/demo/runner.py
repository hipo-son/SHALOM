"""Demo runner — thin wrapper around Pipeline with rich output callbacks."""

from __future__ import annotations

import logging
import time
from typing import Any

from shalom.core.schemas import PipelineResult
from shalom.demo.console import DemoConsole
from shalom.demo.cost_tracker import CostTracker
from shalom.demo.scenarios import Scenario
from shalom.pipeline import Pipeline, PipelineConfig

logger = logging.getLogger(__name__)


class DemoRunner:
    """Orchestrates a demo run: builds Pipeline, hooks callbacks, displays results."""

    def __init__(
        self,
        scenario: Scenario,
        provider_type: str = "openai",
        model_name: str = "gpt-4o",
        dry_run: bool = False,
        quiet: bool = False,
        no_color: bool = False,
        verbose: bool = False,
    ) -> None:
        self.scenario = scenario
        self.dry_run = dry_run
        self.verbose = verbose
        self.tracker = CostTracker()
        self.display = DemoConsole(quiet=quiet, no_color=no_color)
        self._t0 = 0.0

        self.llm: Any  # CannedLLMProvider or LLMProvider
        if dry_run:
            from shalom.demo.canned_provider import CannedLLMProvider, CANNED_RESPONSES
            self.llm = CannedLLMProvider(CANNED_RESPONSES)
            self._provider_type = "canned"
            self._model_name = "dry-run"
        else:
            from shalom.core.llm_provider import LLMProvider
            self.llm = LLMProvider(
                provider_type=provider_type,
                model_name=model_name,
                usage_callback=self.tracker.record,
            )
            self._provider_type = provider_type
            self._model_name = model_name

    def _step_callback(self, step_name: str, step_data: dict) -> None:
        """Pipeline callback for rich display output."""
        elapsed = time.time() - self._t0

        if step_name == "coarse_selection":
            self.display.print_candidates(
                step_data["candidates"], elapsed, self.tracker,
            )
        elif step_name == "fine_selection":
            ranked = step_data["ranked_material"]
            if ranked.evaluation_details:
                candidates = step_data.get(
                    "candidates",
                    [ranked.candidate],  # fallback for simple mode
                )
                # Try to get full candidate list from coarse_selection context
                self.display.print_evaluation_matrix(
                    ranked.evaluation_details, candidates,
                )
                if ranked.evaluation_details.veto_reasons:
                    self.display.print_veto_info(
                        ranked.evaluation_details.veto_reasons,
                    )
            self.display.print_winner(ranked)
        elif step_name == "structure_generation":
            self.display.print_structure_info(step_data["path"])

    def run(self) -> PipelineResult:
        """Execute the demo pipeline and display results.

        Returns:
            PipelineResult from the pipeline.

        Raises:
            SystemExit: On keyboard interrupt (130) or fatal error (1).
        """
        self._t0 = time.time()
        self.display.print_header(
            self.scenario,
            self._provider_type,
            self._model_name,
            dry_run=self.dry_run,
        )

        config = PipelineConfig(
            selector_mode=self.scenario.selector_mode,
            skip_review=True,
        )

        try:
            pipeline = Pipeline(
                objective=self.scenario.objective,
                config=config,
                llm_provider=self.llm,
                callbacks=[self._step_callback],
            )
            result = pipeline.run()
        except KeyboardInterrupt:
            self.display.print_error("Interrupted by user (Ctrl+C)")
            raise SystemExit(130)
        except Exception as e:
            self.display.print_error(str(e), verbose=self.verbose)
            if self.verbose:
                import traceback
                traceback.print_exc()
            raise SystemExit(1)

        elapsed = time.time() - self._t0
        self.display.print_final_report(result, self.tracker, elapsed)
        return result
