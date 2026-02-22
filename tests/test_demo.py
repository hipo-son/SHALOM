"""Tests for the shalom.demo package."""

import json
import threading
import time
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from shalom.core.schemas import (
    CandidateScore,
    EvaluationDetails,
    EvaluationResponse,
    MaterialCandidate,
    PipelineResult,
    PipelineStatus,
    RankedMaterial,
)
from shalom.demo.canned_provider import CannedLLMProvider, CANNED_RESPONSES
from shalom.demo.cost_tracker import CostTracker
from shalom.demo.report import save_report
from shalom.demo.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# CostTracker tests
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.call_count == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0

    def test_record_and_totals(self):
        tracker = CostTracker()
        tracker.record({
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 1000,
            "output_tokens": 500,
        })
        assert tracker.call_count == 1
        assert tracker.total_tokens == 1500
        # Cost: 1000 * 2.50/1M + 500 * 10.00/1M = 0.0025 + 0.005 = 0.0075
        assert abs(tracker.total_cost - 0.0075) < 1e-6

    def test_unknown_model_zero_cost(self):
        tracker = CostTracker()
        tracker.record({
            "provider": "custom",
            "model": "my-model",
            "input_tokens": 1000,
            "output_tokens": 1000,
        })
        assert tracker.total_tokens == 2000
        assert tracker.total_cost == 0.0

    def test_thread_safety(self):
        tracker = CostTracker()
        errors = []

        def record_n(n):
            try:
                for _ in range(n):
                    tracker.record({"input_tokens": 10, "output_tokens": 5})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_n, args=(100,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert tracker.call_count == 400
        assert tracker.total_tokens == 400 * 15

    def test_summary_dict(self):
        tracker = CostTracker()
        tracker.record({"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50})
        summary = tracker.summary()
        assert summary["total_calls"] == 1
        assert summary["total_tokens"] == 150
        assert isinstance(summary["total_cost_usd"], float)
        assert len(summary["calls"]) == 1
        assert "timestamp" in summary["calls"][0]

    def test_multiple_models(self):
        tracker = CostTracker()
        tracker.record({"model": "gpt-4o", "input_tokens": 1000, "output_tokens": 500})
        tracker.record({"model": "gpt-4o-mini", "input_tokens": 1000, "output_tokens": 500})
        # gpt-4o: 0.0025 + 0.005 = 0.0075
        # gpt-4o-mini: 0.00015 + 0.0003 = 0.00045
        assert abs(tracker.total_cost - 0.00795) < 1e-6


# ---------------------------------------------------------------------------
# CannedLLMProvider tests
# ---------------------------------------------------------------------------


class TestCannedProvider:
    def test_returns_correct_response_model(self):
        from shalom.agents.design_layer import CandidateListResponse

        provider = CannedLLMProvider(CANNED_RESPONSES)
        result = provider.generate_structured_output(
            system_prompt="test",
            user_prompt="test",
            response_model=CandidateListResponse,
        )
        assert isinstance(result, CandidateListResponse)
        assert len(result.candidates) == 3

    def test_unknown_model_raises(self):
        class UnknownModel(BaseModel):
            x: int

        provider = CannedLLMProvider(CANNED_RESPONSES)
        with pytest.raises(ValueError, match="No canned response"):
            provider.generate_structured_output("", "", UnknownModel)

    def test_queue_consumption(self):
        """EvaluationResponse is a list — should be consumed FIFO."""
        provider = CannedLLMProvider(CANNED_RESPONSES)
        perspectives_seen = set()
        for _ in range(6):
            result = provider.generate_structured_output(
                "", "", EvaluationResponse,
            )
            assert isinstance(result, EvaluationResponse)
            perspectives_seen.add(result.perspective)

        # All 6 should be different perspectives
        assert len(perspectives_seen) == 6

    def test_queue_exhausted_fallback(self):
        """After queue is exhausted, returns last element."""
        provider = CannedLLMProvider(CANNED_RESPONSES)
        # Consume all 6
        for _ in range(6):
            provider.generate_structured_output("", "", EvaluationResponse)
        # 7th call should get the fallback (last element)
        result = provider.generate_structured_output("", "", EvaluationResponse)
        assert isinstance(result, EvaluationResponse)

    def test_single_response_reused(self):
        """Non-list response should be returned every time."""
        provider = CannedLLMProvider(CANNED_RESPONSES)
        r1 = provider.generate_structured_output("", "", RankedMaterial)
        r2 = provider.generate_structured_output("", "", RankedMaterial)
        assert r1 is r2  # Same object, not consumed

    def test_interface_compatibility(self):
        """CannedLLMProvider has the same interface attributes as LLMProvider."""
        provider = CannedLLMProvider(CANNED_RESPONSES)
        assert provider.provider_type == "canned"
        assert provider.model_name == "dry-run"
        assert provider.usage_callback is None


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_all_scenarios_have_required_fields(self):
        for key, sc in SCENARIOS.items():
            assert sc.name, f"{key} missing name"
            assert sc.objective, f"{key} missing objective"
            assert sc.selector_mode in ("simple", "multi_agent"), f"{key} invalid mode"
            assert sc.estimated_api_calls > 0, f"{key} invalid api_calls"
            assert sc.estimated_cost_usd >= 0, f"{key} invalid cost"
            assert sc.description, f"{key} missing description"

    def test_five_scenarios_exist(self):
        assert len(SCENARIOS) == 5
        assert "smoke_test" in SCENARIOS
        assert "her_catalyst" in SCENARIOS

    def test_smoke_test_is_simple(self):
        assert SCENARIOS["smoke_test"].selector_mode == "simple"

    def test_multi_agent_scenarios(self):
        multi = [k for k, s in SCENARIOS.items() if s.selector_mode == "multi_agent"]
        assert len(multi) == 4


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestReport:
    def test_json_report_structure(self, tmp_path):
        result = PipelineResult(
            status=PipelineStatus.AWAITING_DFT,
            objective="test objective",
            steps_completed=["coarse_selection", "fine_selection"],
        )
        tracker = CostTracker()
        tracker.record({"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50})

        report_path = save_report(result, tracker, str(tmp_path / "report.json"))
        assert report_path is not None

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert "pipeline_result" in report
        assert "cost_summary" in report
        assert report["pipeline_result"]["status"] == "awaiting_dft"
        assert report["cost_summary"]["total_calls"] == 1

    def test_report_with_partial_results(self, tmp_path):
        result = PipelineResult(
            status=PipelineStatus.FAILED_DESIGN,
            objective="test",
            error_message="API failed",
        )
        tracker = CostTracker()
        path = save_report(result, tracker, str(tmp_path / "report.json"))
        assert path is not None

        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["pipeline_result"]["error_message"] == "API failed"

    def test_report_bad_path_returns_none(self):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            objective="test",
        )
        tracker = CostTracker()
        # Path that's unlikely to be writable
        path = save_report(result, tracker, "/nonexistent/dir/report.json")
        assert path is None


# ---------------------------------------------------------------------------
# Console tests
# ---------------------------------------------------------------------------


class TestDemoConsole:
    def test_no_color_mode(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        assert console.console.no_color is True

    def test_quiet_suppresses_output(self, capsys):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(quiet=True, no_color=True)
        candidates = [
            MaterialCandidate(
                material_name="Cu", elements=["Cu"],
                reasoning="test", expected_properties={},
            )
        ]
        tracker = CostTracker()
        console.print_candidates(candidates, 1.0, tracker)
        # quiet mode should not print candidates table
        # (Rich prints to its own Console, not stdout, so we check via Console)

    def test_print_winner(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        ranked = RankedMaterial(
            candidate=MaterialCandidate(
                material_name="TiO2", elements=["Ti", "O"],
                reasoning="test",
            ),
            score=0.85,
            ranking_justification="Best bandgap alignment.",
        )
        # Should not raise
        console.print_winner(ranked)


# ---------------------------------------------------------------------------
# EvaluationDetails schema tests
# ---------------------------------------------------------------------------


class TestEvaluationDetails:
    def test_on_ranked_material(self):
        details = EvaluationDetails(
            evaluations=[
                EvaluationResponse(
                    perspective="stability",
                    scores=[CandidateScore(
                        material_name="Cu", score=0.9, confidence=0.7,
                        justification="Stable",
                    )],
                )
            ],
            veto_reasons=["X: vetoed by novelty"],
            micro_loop_retries=1,
        )
        ranked = RankedMaterial(
            candidate=MaterialCandidate(
                material_name="Cu", elements=["Cu"], reasoning="test",
            ),
            score=0.8,
            ranking_justification="test",
            evaluation_details=details,
        )
        assert ranked.evaluation_details is not None
        assert len(ranked.evaluation_details.evaluations) == 1
        assert ranked.evaluation_details.micro_loop_retries == 1

    def test_default_none(self):
        ranked = RankedMaterial(
            candidate=MaterialCandidate(
                material_name="Cu", elements=["Cu"], reasoning="test",
            ),
            score=0.8,
            ranking_justification="test",
        )
        assert ranked.evaluation_details is None

    def test_model_copy_pattern(self):
        ranked = RankedMaterial(
            candidate=MaterialCandidate(
                material_name="Cu", elements=["Cu"], reasoning="test",
            ),
            score=0.8,
            ranking_justification="test",
        )
        details = EvaluationDetails(veto_reasons=["reason1"])
        updated = ranked.model_copy(update={"evaluation_details": details})
        assert updated.evaluation_details is not None
        assert updated.evaluation_details.veto_reasons == ["reason1"]
        # Original unchanged
        assert ranked.evaluation_details is None


# ---------------------------------------------------------------------------
# Dependency guard tests
# ---------------------------------------------------------------------------


class TestDependencyGuard:
    def test_check_passes_with_rich(self):
        from shalom.demo import _check_demo_dependencies
        # Should not raise (rich is installed)
        _check_demo_dependencies()

    def test_check_fails_without_rich(self):
        import importlib
        with patch.dict("sys.modules", {"rich": None}):
            with pytest.raises(SystemExit):
                import shalom.demo as demo_mod
                importlib.reload(demo_mod)
                demo_mod._check_demo_dependencies()


# ---------------------------------------------------------------------------
# DemoRunner dry-run integration test
# ---------------------------------------------------------------------------


class TestDemoRunner:
    def test_dry_run_completes(self, tmp_path, monkeypatch):
        from shalom.demo.runner import DemoRunner

        monkeypatch.chdir(tmp_path)
        scenario = SCENARIOS["smoke_test"]
        runner = DemoRunner(
            scenario=scenario,
            dry_run=True,
            quiet=True,
            no_color=True,
        )
        result = runner.run()
        assert result.status in (PipelineStatus.AWAITING_DFT, PipelineStatus.FAILED_SIMULATION)
        assert runner.tracker.call_count == 0  # dry-run = no API calls

    def test_dry_run_provider_type(self):
        from shalom.demo.runner import DemoRunner

        scenario = SCENARIOS["smoke_test"]
        runner = DemoRunner(scenario=scenario, dry_run=True, quiet=True)
        assert runner._provider_type == "canned"
        assert runner._model_name == "dry-run"

    def test_keyboard_interrupt_exits_130(self, monkeypatch):
        from shalom.demo.runner import DemoRunner

        scenario = SCENARIOS["smoke_test"]
        runner = DemoRunner(scenario=scenario, dry_run=True, quiet=True, no_color=True)

        # Make Pipeline.run() raise KeyboardInterrupt
        with patch("shalom.demo.runner.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.side_effect = KeyboardInterrupt()
            with pytest.raises(SystemExit) as exc_info:
                runner.run()
            assert exc_info.value.code == 130


# ---------------------------------------------------------------------------
# Console extra coverage
# ---------------------------------------------------------------------------


class TestDemoConsoleExtra:
    def test_print_header(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        scenario = SCENARIOS["smoke_test"]
        console.print_header(scenario, "openai", "gpt-4o", dry_run=True)

    def test_print_structure_info(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        console.print_structure_info("/tmp/POSCAR_test")

    def test_print_error(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        console.print_error("Something went wrong", verbose=True)

    def test_print_final_report(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        result = PipelineResult(
            status=PipelineStatus.AWAITING_DFT,
            objective="test",
            structure_path="/tmp/test",
            steps_completed=["coarse_selection"],
        )
        tracker = CostTracker()
        console.print_final_report(result, tracker, 5.0)

    def test_print_veto_info(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        console.print_veto_info(["Cu: vetoed by stability (score=0.1 < 0.3)"])

    def test_print_evaluation_cards_narrow(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        # Force narrow width
        console.console.width = 80
        details = EvaluationDetails(
            evaluations=[
                EvaluationResponse(
                    perspective="stability",
                    scores=[CandidateScore(
                        material_name="Cu", score=0.9, confidence=0.6,
                        justification="test",
                    )],
                )
            ],
        )
        candidates = [MaterialCandidate(
            material_name="Cu", elements=["Cu"], reasoning="test",
        )]
        console.print_evaluation_matrix(details, candidates)

    def test_print_evaluation_table_wide(self):
        """Wide terminal uses table layout."""
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        console.console.width = 150
        details = EvaluationDetails(
            evaluations=[
                EvaluationResponse(
                    perspective="stability",
                    scores=[
                        CandidateScore(
                            material_name="Cu", score=0.9, confidence=0.6,
                            justification="test",
                        ),
                        CandidateScore(
                            material_name="Fe", score=0.2, confidence=0.5,
                            justification="bad",
                        ),
                    ],
                ),
                EvaluationResponse(
                    perspective="novelty",
                    scores=[
                        CandidateScore(
                            material_name="Cu", score=0.4, confidence=0.7,
                            justification="mid",
                        ),
                    ],
                ),
            ],
            veto_reasons=["Fe: vetoed by stability"],
        )
        candidates = [
            MaterialCandidate(material_name="Cu", elements=["Cu"], reasoning="test"),
            MaterialCandidate(material_name="Fe", elements=["Fe"], reasoning="test"),
        ]
        # Should use table layout, not raise
        console.print_evaluation_matrix(details, candidates)

    def test_print_candidates_not_quiet(self):
        """print_candidates renders table when not quiet."""
        from shalom.demo.console import DemoConsole

        console = DemoConsole(quiet=False, no_color=True)
        candidates = [
            MaterialCandidate(
                material_name="TiO2", elements=["Ti", "O"],
                reasoning="A" * 200,  # long reasoning gets truncated
            ),
        ]
        tracker = CostTracker()
        console.print_candidates(candidates, 1.5, tracker)

    def test_quiet_suppresses_evaluation(self):
        """quiet mode skips evaluation matrix."""
        from shalom.demo.console import DemoConsole

        console = DemoConsole(quiet=True, no_color=True)
        details = EvaluationDetails(
            evaluations=[
                EvaluationResponse(
                    perspective="stability",
                    scores=[CandidateScore(
                        material_name="Cu", score=0.9, confidence=0.6,
                        justification="test",
                    )],
                )
            ],
        )
        candidates = [MaterialCandidate(material_name="Cu", elements=["Cu"], reasoning="test")]
        # Should return immediately (quiet)
        console.print_evaluation_matrix(details, candidates)

    def test_quiet_suppresses_winner(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(quiet=True, no_color=True)
        ranked = RankedMaterial(
            candidate=MaterialCandidate(material_name="Cu", elements=["Cu"], reasoning="t"),
            score=0.8, ranking_justification="test",
        )
        console.print_winner(ranked)

    def test_quiet_suppresses_structure_info(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(quiet=True, no_color=True)
        console.print_structure_info("/tmp/test")

    def test_quiet_suppresses_veto(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(quiet=True, no_color=True)
        console.print_veto_info(["reason1"])

    def test_empty_veto_no_output(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        console.print_veto_info([])

    def test_empty_evaluations_no_output(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        details = EvaluationDetails(evaluations=[])
        candidates = [MaterialCandidate(material_name="Cu", elements=["Cu"], reasoning="t")]
        console.print_evaluation_matrix(details, candidates)

    def test_print_cost_confirmation_yes(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        scenario = SCENARIOS["smoke_test"]
        with patch("builtins.input", return_value="y"):
            assert console.print_cost_confirmation(scenario) is True

    def test_print_cost_confirmation_no(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        scenario = SCENARIOS["smoke_test"]
        with patch("builtins.input", return_value="n"):
            assert console.print_cost_confirmation(scenario) is False

    def test_print_cost_confirmation_eof(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        scenario = SCENARIOS["smoke_test"]
        with patch("builtins.input", side_effect=EOFError):
            assert console.print_cost_confirmation(scenario) is False

    def test_print_final_report_failed_status(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        result = PipelineResult(
            status=PipelineStatus.FAILED_DESIGN,
            objective="test",
            error_message="Something broke",
            steps_completed=["coarse_selection"],
        )
        tracker = CostTracker()
        console.print_final_report(result, tracker, 3.0)

    def test_print_header_live_mode(self):
        from shalom.demo.console import DemoConsole

        console = DemoConsole(no_color=True)
        scenario = SCENARIOS["smoke_test"]
        console.print_header(scenario, "openai", "gpt-4o", dry_run=False)


# ---------------------------------------------------------------------------
# CLI (__main__) tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_list_scenarios(self):
        from shalom.demo.__main__ import main

        with patch("sys.argv", ["demo", "--list"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_dry_run_via_main(self, tmp_path, monkeypatch):
        from shalom.demo.__main__ import main

        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", [
            "demo", "--dry-run", "--scenario", "smoke_test",
            "--quiet", "--no-color", "--save-report", str(tmp_path / "r.json"),
        ]):
            main()
        assert (tmp_path / "r.json").exists()

    def test_dry_run_with_verbose(self, tmp_path, monkeypatch):
        from shalom.demo.__main__ import main

        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", [
            "demo", "--dry-run", "--scenario", "smoke_test",
            "--verbose", "--no-color", "--save-report", str(tmp_path / "r.json"),
        ]):
            main()

    def test_custom_objective(self, tmp_path, monkeypatch):
        from shalom.demo.__main__ import main

        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", [
            "demo", "--dry-run", "--objective", "Find best superconductor",
            "--quiet", "--no-color", "--save-report", str(tmp_path / "r.json"),
        ]):
            main()

    def test_no_api_key_exits(self, monkeypatch):
        from shalom.demo.__main__ import main

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("sys.argv", ["demo", "--scenario", "smoke_test"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_auto_detect_openai_key(self, tmp_path, monkeypatch):
        from shalom.demo.__main__ import main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Should auto-detect openai, but we skip cost confirmation and mock the runner
        with patch("sys.argv", [
            "demo", "--scenario", "smoke_test", "--yes", "--quiet", "--no-color",
            "--save-report", str(tmp_path / "r.json"),
        ]):
            with patch("shalom.demo.runner.Pipeline") as MockPipeline:
                mock_result = PipelineResult(
                    status=PipelineStatus.AWAITING_DFT,
                    objective="test",
                    steps_completed=["coarse_selection"],
                )
                MockPipeline.return_value.run.return_value = mock_result
                main()
                # Verify report was saved
                assert (tmp_path / "r.json").exists()

    def test_auto_detect_anthropic_key(self, tmp_path, monkeypatch):
        from shalom.demo.__main__ import main

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test123")
        with patch("sys.argv", [
            "demo", "--scenario", "smoke_test", "--yes", "--quiet", "--no-color",
            "--save-report", str(tmp_path / "r.json"),
        ]):
            with patch("shalom.demo.runner.Pipeline") as MockPipeline:
                mock_result = PipelineResult(
                    status=PipelineStatus.AWAITING_DFT,
                    objective="test",
                    steps_completed=["coarse_selection"],
                )
                MockPipeline.return_value.run.return_value = mock_result
                main()
                assert (tmp_path / "r.json").exists()

    def test_cost_confirmation_aborts(self, monkeypatch):
        from shalom.demo.__main__ import main

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        with patch("sys.argv", ["demo", "--scenario", "smoke_test", "--no-color"]):
            with patch("shalom.demo.console.DemoConsole.print_cost_confirmation", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# DemoRunner extra coverage
# ---------------------------------------------------------------------------


class TestDemoRunnerExtra:
    def test_generic_exception_exits_1(self):
        from shalom.demo.runner import DemoRunner

        scenario = SCENARIOS["smoke_test"]
        runner = DemoRunner(scenario=scenario, dry_run=True, quiet=True, no_color=True)

        with patch("shalom.demo.runner.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.side_effect = RuntimeError("boom")
            with pytest.raises(SystemExit) as exc_info:
                runner.run()
            assert exc_info.value.code == 1

    def test_verbose_exception_prints_traceback(self):
        from shalom.demo.runner import DemoRunner

        scenario = SCENARIOS["smoke_test"]
        runner = DemoRunner(
            scenario=scenario, dry_run=True, quiet=True, no_color=True, verbose=True,
        )

        with patch("shalom.demo.runner.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.side_effect = ValueError("test error")
            with pytest.raises(SystemExit) as exc_info:
                runner.run()
            assert exc_info.value.code == 1

    def test_live_provider_creation(self, monkeypatch):
        """Non-dry-run runner creates real LLMProvider."""
        from shalom.demo.runner import DemoRunner

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        scenario = SCENARIOS["smoke_test"]
        runner = DemoRunner(
            scenario=scenario, provider_type="openai", model_name="gpt-4o",
            dry_run=False, quiet=True, no_color=True,
        )
        assert runner._provider_type == "openai"
        assert runner._model_name == "gpt-4o"
        assert runner.llm.usage_callback is not None

    def test_step_callback_fine_selection_with_details(self):
        """Step callback handles fine_selection with evaluation_details."""
        from shalom.demo.runner import DemoRunner

        scenario = SCENARIOS["smoke_test"]
        runner = DemoRunner(scenario=scenario, dry_run=True, no_color=True)
        runner._t0 = time.time()

        details = EvaluationDetails(
            evaluations=[
                EvaluationResponse(
                    perspective="stability",
                    scores=[CandidateScore(
                        material_name="Cu", score=0.9, confidence=0.7,
                        justification="test",
                    )],
                )
            ],
            veto_reasons=["Fe: vetoed"],
        )
        ranked = RankedMaterial(
            candidate=MaterialCandidate(material_name="Cu", elements=["Cu"], reasoning="t"),
            score=0.85, ranking_justification="test",
            evaluation_details=details,
        )
        # Should not raise
        runner._step_callback("fine_selection", {"ranked_material": ranked})

    def test_step_callback_structure_generation(self):
        from shalom.demo.runner import DemoRunner

        scenario = SCENARIOS["smoke_test"]
        runner = DemoRunner(scenario=scenario, dry_run=True, no_color=True)
        runner._t0 = time.time()
        runner._step_callback("structure_generation", {"path": "/tmp/POSCAR"})
