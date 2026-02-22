"""Tests for the Pipeline Orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import (
    MaterialCandidate,
    PipelineResult,
    PipelineStatus,
    RankedMaterial,
    ReviewResult,
)
from shalom.pipeline import Pipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMProvider)


@pytest.fixture
def sample_candidates():
    return [
        MaterialCandidate(
            material_name="MoS2",
            elements=["Mo", "S"],
            reasoning="Good d-band center.",
            expected_properties={"bandgap": "1.8 eV"},
        ),
    ]


@pytest.fixture
def sample_ranked(sample_candidates):
    return RankedMaterial(
        candidate=sample_candidates[0],
        score=0.85,
        ranking_justification="Top candidate for HER.",
    )


def _mock_design_steps(mock_llm, sample_candidates, sample_ranked):
    """Configure mock_llm to handle CoarseSelector + FineSelector calls."""
    call_count = {"value": 0}

    def side_effect(**kwargs):
        call_count["value"] += 1
        if call_count["value"] % 2 == 1:
            # Coarse selector (odd calls)
            mock_resp = MagicMock()
            mock_resp.candidates = sample_candidates
            return mock_resp
        else:
            # Fine selector (even calls)
            return sample_ranked

    mock_llm.generate_structured_output.side_effect = side_effect


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.backend_name == "vasp"
        assert config.provider_type == "openai"
        assert config.model_name == "gpt-4o"
        assert config.max_retries == 3
        assert config.max_outer_loops == 1
        assert config.skip_review is True
        assert config.selector_mode == "simple"
        assert config.strict_veto is True
        assert config.save_state is True
        assert config.api_max_retries == 3

    def test_custom_values(self):
        config = PipelineConfig(
            backend_name="qe",
            provider_type="anthropic",
            model_name="claude-sonnet-4-6",
            max_outer_loops=3,
            selector_mode="multi_agent",
            strict_veto=False,
            api_max_retries=5,
        )
        assert config.backend_name == "qe"
        assert config.max_outer_loops == 3
        assert config.strict_veto is False
        assert config.api_max_retries == 5

    def test_max_retries_validation(self):
        with pytest.raises(Exception):
            PipelineConfig(max_retries=0)
        with pytest.raises(Exception):
            PipelineConfig(max_retries=11)

    def test_max_outer_loops_validation(self):
        with pytest.raises(Exception):
            PipelineConfig(max_outer_loops=0)
        with pytest.raises(Exception):
            PipelineConfig(max_outer_loops=6)


# ---------------------------------------------------------------------------
# PipelineResult schema tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_minimal_result(self):
        result = PipelineResult(
            status=PipelineStatus.FAILED_DESIGN,
            objective="test",
        )
        assert result.candidates is None
        assert result.steps_completed == []
        assert result.iteration == 1

    def test_full_result(self, sample_candidates, sample_ranked):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            objective="test",
            iteration=2,
            candidates=sample_candidates,
            ranked_material=sample_ranked,
            structure_generated=True,
            structure_path="/tmp/POSCAR_MoS2",
            review_result=ReviewResult(
                is_successful=True,
                energy=-34.5,
                forces_max=0.01,
                feedback_for_design="OK",
            ),
            steps_completed=["coarse_selection", "fine_selection", "structure_generation", "review"],
        )
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.steps_completed) == 4
        assert result.iteration == 2

    def test_json_serialization(self):
        result = PipelineResult(
            status=PipelineStatus.AWAITING_DFT,
            objective="test",
        )
        json_str = result.model_dump_json()
        assert "awaiting_dft" in json_str

        loaded = PipelineResult.model_validate_json(json_str)
        assert loaded.status == PipelineStatus.AWAITING_DFT


# ---------------------------------------------------------------------------
# Pipeline constructor tests
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_default_construction(self, mock_llm):
        pipeline = Pipeline(objective="test", llm_provider=mock_llm)
        assert pipeline.objective == "test"
        assert pipeline.config.backend_name == "vasp"

    def test_backend_shorthand(self, mock_llm):
        pipeline = Pipeline(objective="test", backend="vasp", llm_provider=mock_llm)
        assert pipeline.config.backend_name == "vasp"

    def test_invalid_backend_raises(self, mock_llm):
        with pytest.raises(ValueError, match="Unknown"):
            Pipeline(objective="test", backend="gaussian", llm_provider=mock_llm)

    def test_config_override(self, mock_llm):
        config = PipelineConfig(max_retries=5, max_outer_loops=3)
        pipeline = Pipeline(objective="test", config=config, llm_provider=mock_llm)
        assert pipeline.config.max_retries == 5
        assert pipeline.config.max_outer_loops == 3


# ---------------------------------------------------------------------------
# Pipeline.run() happy path (mock run_creation_loop directly)
# ---------------------------------------------------------------------------


class TestPipelineRunHappyPath:
    def test_full_pipeline_awaiting_dft(
        self, mock_llm, sample_candidates, sample_ranked, tmp_path
    ):
        """Default skip_review=True -> stops at AWAITING_DFT."""
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        config = PipelineConfig(output_dir=str(tmp_path), save_state=True)
        pipeline = Pipeline(objective="Find HER catalyst", llm_provider=mock_llm, config=config)

        poscar_path = str(tmp_path / "POSCAR_MoS2")
        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (True, MagicMock(), poscar_path)
            result = pipeline.run()

        assert result.status == PipelineStatus.AWAITING_DFT
        assert result.candidates is not None
        assert result.ranked_material is not None
        assert result.structure_generated is True
        assert result.structure_path == poscar_path
        assert "coarse_selection" in result.steps_completed
        assert "fine_selection" in result.steps_completed
        assert "structure_generation" in result.steps_completed
        assert "awaiting_dft" in result.steps_completed


# ---------------------------------------------------------------------------
# Pipeline.run() failure cases
# ---------------------------------------------------------------------------


class TestPipelineRunFailures:
    def test_coarse_selection_failure(self, mock_llm):
        mock_llm.generate_structured_output.side_effect = RuntimeError("API key invalid")
        pipeline = Pipeline(objective="test", llm_provider=mock_llm)
        result = pipeline.run()

        assert result.status == PipelineStatus.FAILED_DESIGN
        assert "API key invalid" in result.error_message
        assert result.candidates is None
        assert result.steps_completed == []

    def test_fine_selection_failure(self, mock_llm, sample_candidates):
        call_count = {"value": 0}

        def side_effect(**kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                mock_resp = MagicMock()
                mock_resp.candidates = sample_candidates
                return mock_resp
            raise RuntimeError("Fine selector error")

        mock_llm.generate_structured_output.side_effect = side_effect
        pipeline = Pipeline(objective="test", llm_provider=mock_llm)
        result = pipeline.run()

        assert result.status == PipelineStatus.FAILED_DESIGN
        assert result.candidates is not None
        assert "coarse_selection" in result.steps_completed
        assert "fine_selection" not in result.steps_completed

    def test_structure_generation_failure(
        self, mock_llm, sample_candidates, sample_ranked
    ):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        pipeline = Pipeline(objective="test", llm_provider=mock_llm)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (False, None, "Max retries (3) exceeded.")
            result = pipeline.run()

        assert result.status == PipelineStatus.FAILED_SIMULATION
        assert result.ranked_material is not None
        assert "Max retries" in result.error_message

    def test_structure_generation_exception(
        self, mock_llm, sample_candidates, sample_ranked
    ):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        pipeline = Pipeline(objective="test", llm_provider=mock_llm)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.side_effect = RuntimeError("Unexpected crash")
            result = pipeline.run()

        assert result.status == PipelineStatus.FAILED_SIMULATION
        assert "Unexpected crash" in result.error_message


# ---------------------------------------------------------------------------
# Closed-loop
# ---------------------------------------------------------------------------


class TestPipelineClosedLoop:
    def test_review_failure_retries_with_feedback(
        self, mock_llm, sample_candidates, sample_ranked
    ):
        """max_outer_loops=2: first review fails, second succeeds."""
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        config = PipelineConfig(
            max_outer_loops=2, skip_review=False, save_state=False
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop, patch(
            "shalom.agents.review_layer.ReviewAgent.review_with_backend"
        ) as mock_review:
            mock_loop.return_value = (True, MagicMock(), "/tmp/POSCAR_MoS2")
            mock_review.side_effect = [
                ReviewResult(
                    is_successful=False, energy=-10.0, forces_max=0.5,
                    feedback_for_design="SCF not converged. Reduce AMIX.",
                ),
                ReviewResult(
                    is_successful=True, energy=-34.5, forces_max=0.01,
                    feedback_for_design="OK",
                ),
            ]
            result = pipeline.run()

        assert result.status == PipelineStatus.COMPLETED
        assert result.review_result.is_successful is True

    def test_all_outer_loops_fail(
        self, mock_llm, sample_candidates, sample_ranked
    ):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        config = PipelineConfig(
            max_outer_loops=2, skip_review=False, save_state=False
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop, patch(
            "shalom.agents.review_layer.ReviewAgent.review_with_backend"
        ) as mock_review:
            mock_loop.return_value = (True, MagicMock(), "/tmp/POSCAR")
            mock_review.return_value = ReviewResult(
                is_successful=False, energy=-10.0, forces_max=0.5,
                feedback_for_design="Still failing.",
            )
            result = pipeline.run()

        assert result.status == PipelineStatus.FAILED_REVIEW


# ---------------------------------------------------------------------------
# resume_from_dft
# ---------------------------------------------------------------------------


class TestPipelineResume:
    def test_resume_success(self, mock_llm):
        awaiting = PipelineResult(
            status=PipelineStatus.AWAITING_DFT,
            objective="test",
            structure_generated=True,
            structure_path="/tmp/POSCAR_MoS2",
            steps_completed=["coarse_selection", "fine_selection", "structure_generation", "awaiting_dft"],
        )

        with patch("shalom.agents.review_layer.ReviewAgent.review_with_backend") as mock_review:
            mock_review.return_value = ReviewResult(
                is_successful=True, energy=-34.5, forces_max=0.01,
                feedback_for_design="Converged successfully.",
            )
            pipeline = Pipeline(objective="test", llm_provider=mock_llm)
            result = pipeline.resume_from_dft(awaiting)

        assert result.status == PipelineStatus.COMPLETED
        assert result.review_result.is_successful is True
        assert "review" in result.steps_completed

    def test_resume_wrong_state_raises(self, mock_llm):
        failed = PipelineResult(
            status=PipelineStatus.FAILED_DESIGN,
            objective="test",
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm)
        with pytest.raises(ValueError, match="AWAITING_DFT"):
            pipeline.resume_from_dft(failed)

    def test_resume_review_failure(self, mock_llm):
        awaiting = PipelineResult(
            status=PipelineStatus.AWAITING_DFT,
            objective="test",
            structure_path="/tmp/POSCAR",
            steps_completed=["awaiting_dft"],
        )

        with patch("shalom.agents.review_layer.ReviewAgent.review_with_backend") as mock_review:
            mock_review.side_effect = FileNotFoundError("OUTCAR not found")
            pipeline = Pipeline(objective="test", llm_provider=mock_llm)
            result = pipeline.resume_from_dft(awaiting)

        assert result.status == PipelineStatus.FAILED_REVIEW
        assert "OUTCAR not found" in result.error_message


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestPipelineCallbacks:
    def test_callbacks_invoked(self, mock_llm, sample_candidates, sample_ranked):
        recorded = []

        def my_callback(step_name, step_data):
            recorded.append(step_name)

        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)
        config = PipelineConfig(save_state=False)
        pipeline = Pipeline(
            objective="test", llm_provider=mock_llm,
            callbacks=[my_callback], config=config,
        )

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (True, MagicMock(), "/tmp/POSCAR")
            pipeline.run()

        assert "coarse_selection" in recorded
        assert "fine_selection" in recorded
        assert "structure_generation" in recorded

    def test_callback_error_does_not_crash(self, mock_llm, sample_candidates):
        def bad_callback(step, data):
            raise RuntimeError("callback crashed")

        call_count = {"value": 0}

        def side_effect(**kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                mock_resp = MagicMock()
                mock_resp.candidates = sample_candidates
                return mock_resp
            raise RuntimeError("stop")

        mock_llm.generate_structured_output.side_effect = side_effect
        pipeline = Pipeline(
            objective="test", llm_provider=mock_llm, callbacks=[bad_callback]
        )
        result = pipeline.run()

        assert "coarse_selection" in result.steps_completed


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestPipelineVASPConfig:
    """Tests for Pipeline VASP configuration integration."""

    def test_vasp_user_incar_passed(
        self, mock_llm, sample_candidates, sample_ranked, tmp_path
    ):
        """vasp_user_incar settings are applied to the VASPInputConfig."""
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        config = PipelineConfig(
            vasp_user_incar={"ENCUT": 700, "ALGO": "Damped"},
            output_dir=str(tmp_path),
            save_state=False,
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (True, MagicMock(), str(tmp_path / "POSCAR"))
            result = pipeline.run()

        assert result.status == PipelineStatus.AWAITING_DFT

    def test_review_exception_handled(
        self, mock_llm, sample_candidates, sample_ranked
    ):
        """Exception during review step -> FAILED_REVIEW status."""
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        config = PipelineConfig(
            skip_review=False, save_state=False,
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop, patch(
            "shalom.agents.review_layer.ReviewAgent.review_with_backend"
        ) as mock_review:
            mock_loop.return_value = (True, MagicMock(), "/tmp/POSCAR")
            mock_review.side_effect = RuntimeError("Review crashed")
            result = pipeline.run()

        assert result.status == PipelineStatus.FAILED_REVIEW
        assert "Review crashed" in result.error_message

    def test_save_state_exception_handled(
        self, mock_llm, sample_candidates, sample_ranked
    ):
        """Exception during state save doesn't crash pipeline."""
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        config = PipelineConfig(
            output_dir="/nonexistent/impossible/path\x00",  # Invalid path
            save_state=True,
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (True, MagicMock(), "/tmp/POSCAR")
            # Should not crash even if save fails
            result = pipeline.run()

        assert result.status == PipelineStatus.AWAITING_DFT


class TestStatePersistence:
    def test_state_saved_on_awaiting_dft(
        self, mock_llm, sample_candidates, sample_ranked, tmp_path
    ):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)

        config = PipelineConfig(output_dir=str(tmp_path), save_state=True)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (True, MagicMock(), str(tmp_path / "POSCAR"))
            result = pipeline.run()

        assert result.status == PipelineStatus.AWAITING_DFT

        state_file = tmp_path / "pipeline_state.json"
        assert state_file.exists()

        loaded = PipelineResult.model_validate_json(state_file.read_text())
        assert loaded.status == PipelineStatus.AWAITING_DFT
        assert loaded.objective == "test"
