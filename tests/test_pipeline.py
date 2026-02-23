"""Tests for the Pipeline Orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import (
    MaterialCandidate,
    PipelineResult,
    PipelineStatus,
    PipelineStep,
    RankedMaterial,
    ReviewResult,
)
from shalom.pipeline import Pipeline, PipelineConfig, synthesize_ranked_material


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


# ---------------------------------------------------------------------------
# synthesize_ranked_material
# ---------------------------------------------------------------------------


class TestSynthesizeRankedMaterial:
    def test_simple_element(self):
        rm = synthesize_ranked_material("Si")
        assert rm.candidate.material_name == "Si"
        assert rm.candidate.elements == ["Si"]
        assert rm.score == 1.0
        assert rm.candidate.expected_properties.get("magnetic") is None

    def test_binary_compound(self):
        rm = synthesize_ranked_material("MoS2")
        assert "Mo" in rm.candidate.elements
        assert "S" in rm.candidate.elements
        assert len(rm.candidate.elements) == 2

    def test_magnetic_detection(self):
        rm = synthesize_ranked_material("Fe")
        assert rm.candidate.expected_properties["magnetic"] is True

    def test_hubbard_u_detection(self):
        """Fe2O3 has magnetic TM (Fe) + anion (O) → needs_hubbard_u."""
        rm = synthesize_ranked_material("Fe2O3")
        assert rm.candidate.expected_properties["magnetic"] is True
        assert rm.candidate.expected_properties["needs_hubbard_u"] is True

    def test_non_magnetic_no_hubbard(self):
        """Si has no magnetic elements → no hubbard_u."""
        rm = synthesize_ranked_material("Si")
        assert "magnetic" not in rm.candidate.expected_properties
        assert "needs_hubbard_u" not in rm.candidate.expected_properties

    def test_complex_formula(self):
        rm = synthesize_ranked_material("LiFePO4")
        elements = rm.candidate.elements
        assert "Li" in elements
        assert "Fe" in elements
        assert "P" in elements
        assert "O" in elements


# ---------------------------------------------------------------------------
# PipelineStep enum
# ---------------------------------------------------------------------------


class TestPipelineStepEnum:
    def test_values(self):
        assert PipelineStep.DESIGN.value == "design"
        assert PipelineStep.SIMULATION.value == "simulation"
        assert PipelineStep.REVIEW.value == "review"

    def test_completed_design_status(self):
        assert PipelineStatus.COMPLETED_DESIGN.value == "completed_design"

    def test_completed_design_serialization(self):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED_DESIGN, objective="test"
        )
        json_str = result.model_dump_json()
        assert "completed_design" in json_str
        loaded = PipelineResult.model_validate_json(json_str)
        assert loaded.status == PipelineStatus.COMPLETED_DESIGN


# ---------------------------------------------------------------------------
# PipelineConfig steps field
# ---------------------------------------------------------------------------


class TestPipelineConfigSteps:
    def test_default_steps_none(self):
        config = PipelineConfig()
        assert config.steps is None

    def test_explicit_steps(self):
        config = PipelineConfig(steps=["design", "simulation"])
        assert config.steps == ["design", "simulation"]

    def test_material_name_field(self):
        config = PipelineConfig(material_name="MoS2")
        assert config.material_name == "MoS2"

    def test_input_structure_path_field(self):
        config = PipelineConfig(input_structure_path="/tmp/POSCAR")
        assert config.input_structure_path == "/tmp/POSCAR"

    def test_input_ranked_material_field(self):
        rm_dict = {
            "candidate": {
                "material_name": "Si",
                "elements": ["Si"],
                "reasoning": "test",
            },
            "score": 0.9,
            "ranking_justification": "test",
        }
        config = PipelineConfig(input_ranked_material=rm_dict)
        assert config.input_ranked_material is not None


# ---------------------------------------------------------------------------
# Pipeline step validation
# ---------------------------------------------------------------------------


class TestPipelineStepValidation:
    def test_invalid_step_name_raises(self, mock_llm):
        config = PipelineConfig(steps=["design", "invalid_step"])
        with pytest.raises(ValueError, match="Invalid step name"):
            Pipeline(objective="test", llm_provider=mock_llm, config=config)

    def test_empty_steps_raises(self, mock_llm):
        config = PipelineConfig(steps=[])
        with pytest.raises(ValueError, match="must not be empty"):
            Pipeline(objective="test", llm_provider=mock_llm, config=config)

    def test_simulation_without_material_raises(self, mock_llm):
        config = PipelineConfig(steps=["simulation"])
        with pytest.raises(ValueError, match="material_name or input_ranked_material"):
            Pipeline(objective="test", llm_provider=mock_llm, config=config)

    def test_review_without_structure_raises(self, mock_llm):
        config = PipelineConfig(steps=["review"])
        with pytest.raises(ValueError, match="input_structure_path"):
            Pipeline(objective="test", llm_provider=mock_llm, config=config)


# ---------------------------------------------------------------------------
# Pipeline skip design
# ---------------------------------------------------------------------------


class TestPipelineSkipDesign:
    def test_simulation_only_with_material_name(self, mock_llm, tmp_path):
        config = PipelineConfig(
            steps=["simulation"],
            material_name="MoS2",
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
        assert result.ranked_material is not None
        assert result.ranked_material.candidate.material_name == "MoS2"
        assert "design_skipped" in result.steps_completed
        assert "structure_generation" in result.steps_completed
        assert result.candidates is None  # design was skipped

    def test_simulation_with_ranked_material_injection(self, mock_llm, tmp_path):
        rm_dict = {
            "candidate": {
                "material_name": "Si",
                "elements": ["Si"],
                "reasoning": "Direct injection test",
                "expected_properties": {"bandgap": "1.1 eV"},
            },
            "score": 0.95,
            "ranking_justification": "Injected",
        }
        config = PipelineConfig(
            steps=["simulation"],
            input_ranked_material=rm_dict,
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
        assert result.ranked_material.candidate.material_name == "Si"
        assert result.ranked_material.score == 0.95

    def test_simulation_and_review_skip_design(self, mock_llm):
        config = PipelineConfig(
            steps=["simulation", "review"],
            material_name="Si",
            save_state=False,
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop, patch(
            "shalom.agents.review_layer.ReviewAgent.review_with_backend"
        ) as mock_review:
            mock_loop.return_value = (True, MagicMock(), "/tmp/POSCAR")
            mock_review.return_value = ReviewResult(
                is_successful=True, energy=-34.5, forces_max=0.01,
                feedback_for_design="OK",
            )
            result = pipeline.run()

        assert result.status == PipelineStatus.COMPLETED
        assert "design_skipped" in result.steps_completed
        assert "review" in result.steps_completed


# ---------------------------------------------------------------------------
# Pipeline skip simulation
# ---------------------------------------------------------------------------


class TestPipelineSkipSimulation:
    def test_design_only(self, mock_llm, sample_candidates, sample_ranked):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)
        config = PipelineConfig(steps=["design"], save_state=False)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)
        result = pipeline.run()

        assert result.status == PipelineStatus.COMPLETED_DESIGN
        assert result.candidates is not None
        assert result.ranked_material is not None
        assert "coarse_selection" in result.steps_completed
        assert "fine_selection" in result.steps_completed

    def test_review_only(self, mock_llm):
        config = PipelineConfig(
            steps=["review"],
            input_structure_path="/tmp/dft_output",
            save_state=False,
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.review_layer.ReviewAgent.review_with_backend"
        ) as mock_review:
            mock_review.return_value = ReviewResult(
                is_successful=True, energy=-34.5, forces_max=0.01,
                feedback_for_design="Converged.",
            )
            result = pipeline.run()

        assert result.status == PipelineStatus.COMPLETED
        assert result.structure_path == "/tmp/dft_output"
        assert "simulation_skipped" in result.steps_completed
        assert "review" in result.steps_completed
        assert result.structure_generated is False

    def test_design_and_review_skip_simulation(
        self, mock_llm, sample_candidates, sample_ranked
    ):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)
        config = PipelineConfig(
            steps=["design", "review"],
            input_structure_path="/tmp/dft_output",
            save_state=False,
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.review_layer.ReviewAgent.review_with_backend"
        ) as mock_review:
            mock_review.return_value = ReviewResult(
                is_successful=True, energy=-34.5, forces_max=0.01,
                feedback_for_design="OK",
            )
            result = pipeline.run()

        assert result.status == PipelineStatus.COMPLETED
        assert "coarse_selection" in result.steps_completed
        assert "simulation_skipped" in result.steps_completed
        assert "review" in result.steps_completed


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestPipelineBackwardCompat:
    def test_default_derives_design_simulation(self, mock_llm):
        """Default config (skip_review=True) → ['design', 'simulation']."""
        pipeline = Pipeline(objective="test", llm_provider=mock_llm)
        assert pipeline._effective_steps == ["design", "simulation"]

    def test_skip_review_false_derives_all(self, mock_llm):
        config = PipelineConfig(skip_review=False)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)
        assert pipeline._effective_steps == ["design", "simulation", "review"]

    def test_explicit_steps_override_skip_review(self, mock_llm):
        """Explicit steps take priority over skip_review."""
        config = PipelineConfig(steps=["design"], skip_review=False)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)
        assert pipeline._effective_steps == ["design"]


# ---------------------------------------------------------------------------
# Config serialization & timing
# ---------------------------------------------------------------------------


class TestPipelineConfigSerialization:
    def test_config_saved_alongside_state(
        self, mock_llm, sample_candidates, sample_ranked, tmp_path
    ):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)
        config = PipelineConfig(output_dir=str(tmp_path), save_state=True)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (True, MagicMock(), str(tmp_path / "POSCAR"))
            pipeline.run()

        config_file = tmp_path / "pipeline_config.json"
        assert config_file.exists()

        loaded = PipelineConfig.model_validate_json(config_file.read_text())
        assert loaded.backend_name == "vasp"
        assert loaded.output_dir == str(tmp_path)

    def test_config_snapshot_in_result(
        self, mock_llm, sample_candidates, sample_ranked, tmp_path
    ):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)
        config = PipelineConfig(
            output_dir=str(tmp_path), save_state=False, calc_type="static"
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (True, MagicMock(), str(tmp_path / "POSCAR"))
            result = pipeline.run()

        assert result.config_snapshot is not None
        assert result.config_snapshot["calc_type"] == "static"
        assert result.config_snapshot["backend_name"] == "vasp"

    def test_config_snapshot_serialization_roundtrip(self):
        result = PipelineResult(
            status=PipelineStatus.AWAITING_DFT,
            objective="test",
            config_snapshot={"backend_name": "qe", "calc_type": "scf"},
        )
        json_str = result.model_dump_json()
        loaded = PipelineResult.model_validate_json(json_str)
        assert loaded.config_snapshot["backend_name"] == "qe"


class TestPipelineTiming:
    def test_elapsed_seconds_populated(
        self, mock_llm, sample_candidates, sample_ranked, tmp_path
    ):
        _mock_design_steps(mock_llm, sample_candidates, sample_ranked)
        config = PipelineConfig(output_dir=str(tmp_path), save_state=False)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)

        with patch(
            "shalom.agents.simulation_layer.GeometryReviewer.run_creation_loop"
        ) as mock_loop:
            mock_loop.return_value = (True, MagicMock(), str(tmp_path / "POSCAR"))
            result = pipeline.run()

        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds >= 0.0

    def test_elapsed_seconds_on_failure(self, mock_llm):
        mock_llm.generate_structured_output.side_effect = RuntimeError("fail")
        pipeline = Pipeline(objective="test", llm_provider=mock_llm)
        result = pipeline.run()

        assert result.status == PipelineStatus.FAILED_DESIGN
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds >= 0.0

    def test_elapsed_seconds_serialization(self):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            objective="test",
            elapsed_seconds=12.345,
        )
        json_str = result.model_dump_json()
        loaded = PipelineResult.model_validate_json(json_str)
        assert abs(loaded.elapsed_seconds - 12.345) < 0.001


# ---------------------------------------------------------------------------
# Execution step integration
# ---------------------------------------------------------------------------


class TestExecutionStepEnum:
    """Test EXECUTION step and FAILED_EXECUTION status."""

    def test_execution_step_value(self):
        assert PipelineStep.EXECUTION.value == "execution"

    def test_failed_execution_status(self):
        assert PipelineStatus.FAILED_EXECUTION.value == "failed_execution"

    def test_failed_execution_serialization(self):
        result = PipelineResult(
            status=PipelineStatus.FAILED_EXECUTION,
            objective="test",
            error_message="DFT timed out",
        )
        json_str = result.model_dump_json()
        loaded = PipelineResult.model_validate_json(json_str)
        assert loaded.status == PipelineStatus.FAILED_EXECUTION

    def test_execution_wall_time_field(self):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            objective="test",
            execution_wall_time=3600.5,
        )
        assert result.execution_wall_time == 3600.5

    def test_correction_history_field(self):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            objective="test",
            correction_history=[{"error_type": "QE_SCF_UNCONVERGED", "step": 0}],
        )
        assert len(result.correction_history) == 1

    def test_quality_warnings_field(self):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            objective="test",
            quality_warnings=["loosely_relaxed"],
        )
        assert "loosely_relaxed" in result.quality_warnings

    def test_quality_warnings_default_empty(self):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED, objective="test",
        )
        assert result.quality_warnings == []

    def test_execution_fields_serialization(self):
        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            objective="test",
            execution_wall_time=120.0,
            correction_history=[{"error": "scf", "step": 0}],
            quality_warnings=["loosely_relaxed"],
        )
        json_str = result.model_dump_json()
        loaded = PipelineResult.model_validate_json(json_str)
        assert loaded.execution_wall_time == 120.0
        assert len(loaded.correction_history) == 1
        assert loaded.quality_warnings == ["loosely_relaxed"]


class TestPipelineExecuteConfig:
    """Test PipelineConfig execution fields."""

    def test_execute_default_false(self):
        config = PipelineConfig()
        assert config.execute is False

    def test_nprocs_default(self):
        config = PipelineConfig()
        assert config.nprocs == 1

    def test_execution_timeout_default(self):
        config = PipelineConfig()
        assert config.execution_timeout == 86400

    def test_max_execution_retries_default(self):
        config = PipelineConfig()
        assert config.max_execution_retries == 3

    def test_execute_flag_true(self):
        config = PipelineConfig(execute=True, nprocs=4, mpi_command="srun")
        assert config.execute is True
        assert config.nprocs == 4
        assert config.mpi_command == "srun"


class TestPipelineResolveStepsWithExecute:
    """Test _resolve_effective_steps with execute=True."""

    def test_execute_auto_inserts_before_review(self, mock_llm):
        config = PipelineConfig(execute=True, skip_review=False)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)
        steps = pipeline._effective_steps
        assert "execution" in steps
        assert steps.index("execution") < steps.index("review")

    def test_execute_appends_when_no_review(self, mock_llm):
        config = PipelineConfig(execute=True, skip_review=True)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)
        steps = pipeline._effective_steps
        assert steps[-1] == "execution"

    def test_no_execute_no_execution_step(self, mock_llm):
        config = PipelineConfig(execute=False)
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)
        assert "execution" not in pipeline._effective_steps

    def test_explicit_execution_step(self, mock_llm):
        config = PipelineConfig(
            steps=["simulation", "execution"],
            material_name="Si",
        )
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)
        assert "execution" in pipeline._effective_steps


class TestPipelineExecutionValidation:
    """Test validation for execution step."""

    def test_execution_without_simulation_needs_path(self, mock_llm):
        config = PipelineConfig(steps=["execution"])
        with pytest.raises(ValueError, match="input_structure_path"):
            Pipeline(objective="test", llm_provider=mock_llm, config=config)

    def test_execution_without_simulation_with_path(self, mock_llm):
        config = PipelineConfig(
            steps=["execution"],
            input_structure_path="/tmp/calc",
        )
        # Should not raise
        Pipeline(objective="test", llm_provider=mock_llm, config=config)

    def test_backward_compat_no_execute(self, mock_llm):
        """Existing tests unaffected — no execution step unless execute=True."""
        config = PipelineConfig()
        pipeline = Pipeline(objective="test", llm_provider=mock_llm, config=config)
        assert "execution" not in pipeline._effective_steps
