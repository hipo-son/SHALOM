import pytest
from pydantic import ValidationError
from shalom.core.schemas import (
    AgentMessage,
    MaterialCandidate,
    RankedMaterial,
    ReviewResult,
    StructureReviewForm,
)


class TestMaterialCandidate:
    """Tests for MaterialCandidate schema."""

    def test_valid_candidate(self):
        candidate = MaterialCandidate(
            material_name="Iron",
            elements=["Fe"],
            reasoning="Strong metallic bonds.",
            expected_properties={"hardness": "high"},
        )
        assert candidate.material_name == "Iron"
        assert "Fe" in candidate.elements

    def test_missing_reasoning_raises(self):
        with pytest.raises(ValidationError):
            MaterialCandidate(material_name="Silver", elements=["Ag"])

    def test_default_expected_properties(self):
        candidate = MaterialCandidate(material_name="Cu", elements=["Cu"], reasoning="test")
        assert candidate.expected_properties == {}

    def test_multi_element(self):
        candidate = MaterialCandidate(
            material_name="MoS2",
            elements=["Mo", "S"],
            reasoning="2D TMD",
            expected_properties={"bandgap": "1.8 eV"},
        )
        assert len(candidate.elements) == 2


class TestRankedMaterial:
    """Tests for RankedMaterial schema with score validation."""

    def _make_candidate(self):
        return MaterialCandidate(
            material_name="Gold", elements=["Au"], reasoning="Noble", expected_properties={}
        )

    def test_valid_score(self):
        ranked = RankedMaterial(
            candidate=self._make_candidate(), score=0.99, ranking_justification="Best."
        )
        assert ranked.score == 0.99

    def test_score_zero(self):
        ranked = RankedMaterial(
            candidate=self._make_candidate(), score=0.0, ranking_justification="Worst."
        )
        assert ranked.score == 0.0

    def test_score_one(self):
        ranked = RankedMaterial(
            candidate=self._make_candidate(), score=1.0, ranking_justification="Perfect."
        )
        assert ranked.score == 1.0

    def test_score_above_one_raises(self):
        with pytest.raises(ValidationError):
            RankedMaterial(
                candidate=self._make_candidate(), score=1.5, ranking_justification="Invalid."
            )

    def test_score_negative_raises(self):
        with pytest.raises(ValidationError):
            RankedMaterial(
                candidate=self._make_candidate(), score=-0.1, ranking_justification="Invalid."
            )


class TestStructureReviewForm:
    """Tests for StructureReviewForm schema."""

    def test_valid_form(self):
        form = StructureReviewForm(
            file_path="POSCAR",
            num_atoms=4,
            cell_volume=64.0,
            minimum_distance=2.5,
            is_valid=True,
            feedback="Looks good.",
        )
        assert form.is_valid is True

    def test_optional_fields_default_none(self):
        form = StructureReviewForm(
            num_atoms=1,
            cell_volume=10.0,
            minimum_distance=2.0,
            is_valid=True,
            feedback="OK",
        )
        assert form.file_path is None
        assert form.vacuum_thickness is None


class TestReviewResult:
    """Tests for ReviewResult schema."""

    def test_failed_result(self):
        result = ReviewResult(
            is_successful=False,
            energy=-25.5,
            forces_max=0.01,
            feedback_for_design="Energy too high.",
        )
        assert result.is_successful is False
        assert result.energy == -25.5

    def test_successful_result(self):
        result = ReviewResult(
            is_successful=True,
            energy=-50.0,
            forces_max=0.001,
            feedback_for_design="Converged.",
        )
        assert result.is_successful is True

    def test_optional_energy_none(self):
        result = ReviewResult(is_successful=False, feedback_for_design="Parsing failed.")
        assert result.energy is None
        assert result.forces_max is None


class TestAgentMessage:
    """Tests for AgentMessage schema."""

    def test_basic_message(self):
        msg = AgentMessage(
            sender="DesignLayer",
            receiver="SimulationLayer",
            content="Start simulation.",
        )
        assert msg.sender == "DesignLayer"
        assert msg.payload is None

    def test_message_with_payload(self):
        msg = AgentMessage(
            sender="ReviewLayer",
            receiver="DesignLayer",
            content="Retry needed.",
            payload={"reason": "energy_too_high", "energy": -25.5},
        )
        assert msg.payload["reason"] == "energy_too_high"
