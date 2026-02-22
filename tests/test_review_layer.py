"""Tests for ReviewAgent — covering all branches including enriched prompts."""

import os
import textwrap

import pytest
from shalom.agents.review_layer import ReviewAgent
from shalom.backends.base import DFTResult
from shalom.backends.vasp import VASPBackend
from shalom.core.schemas import ReviewResult


# ---------------------------------------------------------------------------
# parse_outcar tests
# ---------------------------------------------------------------------------


class TestParseOutcar:
    """Tests for ReviewAgent.parse_outcar()."""

    def test_parse_outcar_basic(self, mock_llm):
        agent = ReviewAgent(llm_provider=mock_llm)
        dummy_path = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_outcar.txt")
        parsed = agent.parse_outcar(dummy_path)
        assert parsed["is_converged"] is True
        assert parsed["energy"] == -34.567890

    def test_parse_outcar_file_not_found(self, mock_llm):
        """FileNotFoundError raised for missing OUTCAR."""
        agent = ReviewAgent(llm_provider=mock_llm)
        with pytest.raises(FileNotFoundError, match="OUTCAR"):
            agent.parse_outcar("/nonexistent/path/OUTCAR")

    def test_parse_outcar_malformed_energy(self, mock_llm, tmp_path):
        """Malformed energy line doesn't crash — energy stays None."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN  =\n"
            " General timing and accounting informations for this job:\n"
        )
        agent = ReviewAgent(llm_provider=mock_llm)
        parsed = agent.parse_outcar(str(outcar))
        assert parsed["energy"] is None
        assert parsed["is_converged"] is True


# ---------------------------------------------------------------------------
# review() backward compatibility
# ---------------------------------------------------------------------------


class TestReviewBackwardCompat:
    """Tests for ReviewAgent.review() with OUTCAR path."""

    def test_review_basic(self, mock_llm):
        agent = ReviewAgent(llm_provider=mock_llm)
        mock_result = ReviewResult(
            is_successful=True, energy=-34.567890, forces_max=0.01,
            feedback_for_design="OK",
        )
        mock_llm.generate_structured_output.return_value = mock_result

        dummy_path = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_outcar.txt")

        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(dummy_path, os.path.join(tmpdir, "OUTCAR"))
            result = agent.review("Find stable material", os.path.join(tmpdir, "OUTCAR"))

        assert result.is_successful is True
        assert mock_llm.generate_structured_output.called


# ---------------------------------------------------------------------------
# review_with_backend() with correction_history
# ---------------------------------------------------------------------------


class TestReviewWithBackendCorrectionHistory:
    """Tests for review_with_backend() passing correction_history to DFTResult."""

    def test_correction_history_passed(self, mock_llm, tmp_path):
        """correction_history parameter is assigned to DFTResult."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(textwrap.dedent("""\
         free  energy   TOTEN  =       -10.0 eV
         General timing and accounting informations for this job:
        """))

        mock_result = ReviewResult(
            is_successful=True, energy=-10.0, forces_max=None,
            feedback_for_design="OK",
        )
        mock_llm.generate_structured_output.return_value = mock_result

        agent = ReviewAgent(llm_provider=mock_llm)
        backend = VASPBackend()
        history = [{"error_type": "BRMIX", "action": "AMIX=0.1"}]
        result = agent.review_with_backend(
            "Test", str(tmp_path), backend, correction_history=history,
        )

        assert result.is_successful is True
        # Verify the user_prompt sent to LLM contains BRMIX
        call_kwargs = mock_llm.generate_structured_output.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt", "")
        assert "BRMIX" in user_prompt


# ---------------------------------------------------------------------------
# _evaluate enriched prompt branches
# ---------------------------------------------------------------------------


class TestEvaluateEnrichedPrompt:
    """Tests for _evaluate() covering all optional field branches."""

    def _call_evaluate(self, mock_llm, dft_result):
        """Helper to call _evaluate and capture the user_prompt."""
        mock_result = ReviewResult(
            is_successful=True, energy=-10.0, forces_max=0.01,
            feedback_for_design="OK",
        )
        mock_llm.generate_structured_output.return_value = mock_result
        agent = ReviewAgent(llm_provider=mock_llm)
        agent._evaluate("Test objective", dft_result)
        call_kwargs = mock_llm.generate_structured_output.call_args
        return call_kwargs.kwargs.get("user_prompt", "")

    def test_bandgap_in_prompt(self, mock_llm):
        """bandgap is included in evaluation prompt when present."""
        dft = DFTResult(energy=-10.0, bandgap=1.5)
        prompt = self._call_evaluate(mock_llm, dft)
        assert "Band Gap: 1.5 eV" in prompt

    def test_magnetization_in_prompt(self, mock_llm):
        """magnetization is included in evaluation prompt when present."""
        dft = DFTResult(energy=-10.0, magnetization=3.2)
        prompt = self._call_evaluate(mock_llm, dft)
        assert "Magnetization: 3.2 muB" in prompt

    def test_entropy_in_prompt(self, mock_llm):
        """entropy_per_atom is included in evaluation prompt when present."""
        dft = DFTResult(energy=-10.0, entropy_per_atom=-0.0005)
        prompt = self._call_evaluate(mock_llm, dft)
        assert "Entropy T*S/atom" in prompt

    def test_correction_history_in_prompt(self, mock_llm):
        """correction_history is included in evaluation prompt when present."""
        dft = DFTResult(
            energy=-10.0,
            correction_history=[
                {"error_type": "SCF_UNCONVERGED", "action": "NELM=200"},
                {"error_type": "BRMIX", "action": "AMIX=0.05"},
            ],
        )
        prompt = self._call_evaluate(mock_llm, dft)
        assert "Error Correction History" in prompt
        assert "SCF_UNCONVERGED" in prompt
        assert "BRMIX" in prompt

    def test_physics_warnings_in_prompt(self, mock_llm):
        """Physics warnings are included in evaluation prompt when present."""
        dft = DFTResult(energy=-10.0, forces_max=0.05)  # > 0.02 threshold
        prompt = self._call_evaluate(mock_llm, dft)
        assert "Physics Validation Warnings" in prompt
        assert "0.02" in prompt

    def test_all_fields_in_prompt(self, mock_llm):
        """All optional fields present in a single evaluation."""
        dft = DFTResult(
            energy=-10.0,
            forces_max=0.05,
            bandgap=0.8,
            magnetization=2.5,
            entropy_per_atom=-0.005,
            correction_history=[{"error_type": "BRMIX", "action": "applied"}],
        )
        prompt = self._call_evaluate(mock_llm, dft)
        assert "Band Gap" in prompt
        assert "Magnetization" in prompt
        assert "Entropy" in prompt
        assert "BRMIX" in prompt
        assert "Physics Validation Warnings" in prompt

    def test_no_optional_fields_clean_prompt(self, mock_llm):
        """No optional fields -> no extra sections in prompt."""
        dft = DFTResult(energy=-10.0, forces_max=0.01, is_converged=True)
        prompt = self._call_evaluate(mock_llm, dft)
        assert "Band Gap" not in prompt
        assert "Magnetization" not in prompt
        assert "Entropy" not in prompt
        assert "Error Correction History" not in prompt
        assert "Physics Validation Warnings" not in prompt
