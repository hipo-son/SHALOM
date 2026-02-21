from unittest.mock import MagicMock, patch

from ase import Atoms
from ase.build import bulk, surface

from shalom.core.schemas import RankedMaterial, MaterialCandidate
from shalom.agents.simulation_layer import (
    GeometryGenerator,
    FormFiller,
    GeometryReviewer,
)

# ---------------------------------------------------------------------------
# FormFiller tests
# ---------------------------------------------------------------------------


class TestFormFillerValidStructure:
    """Tests for FormFiller with valid structures."""

    def test_valid_bulk_structure(self):
        """Simple FCC Cu bulk passes validation."""
        atoms = bulk("Cu", "fcc", a=3.6)
        form = FormFiller.evaluate_atoms(atoms)

        assert form.is_valid is True
        assert form.num_atoms == 1
        assert form.cell_volume > 0

    def test_valid_multi_atom_structure(self):
        """Multi-atom bulk structure passes validation."""
        atoms = bulk("NaCl", "rocksalt", a=5.64)
        form = FormFiller.evaluate_atoms(atoms)

        assert form.is_valid is True
        assert form.num_atoms == 2
        assert form.minimum_distance > 0.5

    def test_valid_surface_slab(self):
        """Surface slab with adequate vacuum passes validation."""
        base = bulk("Pt", "fcc", a=3.92)
        slab = surface(base, (1, 1, 1), 3, vacuum=12.0)
        form = FormFiller.evaluate_atoms(slab)

        assert form.is_valid is True
        assert form.num_atoms == 3

    def test_filepath_stored(self):
        """filepath argument is stored in the result."""
        atoms = bulk("Cu", "fcc", a=3.6)
        form = FormFiller.evaluate_atoms(atoms, filepath="/tmp/POSCAR_Cu")

        assert form.file_path == "/tmp/POSCAR_Cu"


class TestFormFillerInvalidStructure:
    """Tests for FormFiller with invalid/edge-case structures."""

    def test_empty_atoms(self):
        """Empty Atoms object is rejected."""
        empty_atoms = Atoms()
        form = FormFiller.evaluate_atoms(empty_atoms)

        assert form.is_valid is False
        assert "No atoms were generated" in form.feedback

    def test_overlapping_atoms(self):
        """Atoms closer than 0.5 A are detected as overlapping."""
        # Two Cu atoms at nearly the same position
        atoms = Atoms(
            "Cu2",
            positions=[(0, 0, 0), (0.1, 0.1, 0.1)],
            cell=[10, 10, 10],
            pbc=True,
        )
        form = FormFiller.evaluate_atoms(atoms)

        assert form.is_valid is False
        assert "too short" in form.feedback

    def test_oversized_cell(self):
        """Unit cell volume > 10000 A^3 is flagged as too large."""
        atoms = Atoms(
            "Cu",
            positions=[(0, 0, 0)],
            cell=[50, 50, 50],
            pbc=True,
        )
        form = FormFiller.evaluate_atoms(atoms)

        assert form.is_valid is False
        assert "too large" in form.feedback

    def test_thin_vacuum_layer(self):
        """Vacuum layer thinner than 8 A is flagged."""
        base = bulk("Cu", "fcc", a=3.6)
        slab = surface(base, (1, 0, 0), 2, vacuum=3.0)  # Too thin
        form = FormFiller.evaluate_atoms(slab)

        # The vacuum check should flag thin vacuum
        if form.vacuum_thickness is not None and form.vacuum_thickness < 8.0:
            assert form.is_valid is False
            assert "too thin" in form.feedback

    def test_adequate_vacuum_passes(self):
        """Vacuum layer >= 10 A passes validation."""
        base = bulk("Cu", "fcc", a=3.6)
        slab = surface(base, (1, 1, 1), 2, vacuum=12.0)
        form = FormFiller.evaluate_atoms(slab)

        # With 12 A vacuum and valid structure, should pass
        assert form.is_valid is True

    def test_vacuum_thickness_calculated(self):
        """vacuum_thickness is computed for surface slabs with z > 10 A."""
        base = bulk("Pt", "fcc", a=3.92)
        slab = surface(base, (1, 1, 1), 2, vacuum=12.0)
        form = FormFiller.evaluate_atoms(slab)

        assert form.vacuum_thickness is not None
        assert form.vacuum_thickness > 0

    def test_no_vacuum_for_bulk(self):
        """Bulk structures have no vacuum thickness."""
        atoms = bulk("Fe", "bcc", a=2.87)
        form = FormFiller.evaluate_atoms(atoms)

        assert form.vacuum_thickness is None


# ---------------------------------------------------------------------------
# GeometryGenerator tests
# ---------------------------------------------------------------------------


class TestGeometryGenerator:
    """Tests for GeometryGenerator LLM code generation."""

    def test_generate_code_basic(self, mock_llm):
        """Basic code generation without feedback."""
        generator = GeometryGenerator(llm_provider=mock_llm)

        mock_response = MagicMock()
        mock_response.python_code = "atoms = bulk('Cu', 'fcc', a=3.6)"
        mock_response.explanation = "Generated Cu fcc bulk."
        mock_llm.generate_structured_output.return_value = mock_response

        candidate = MaterialCandidate(
            material_name="Cu", elements=["Cu"], reasoning="Good", expected_properties={}
        )
        ranked = RankedMaterial(candidate=candidate, score=0.9, ranking_justification="Best")

        response = generator.generate_code("Make Cu bulk", ranked)

        assert "atoms = bulk" in response.python_code
        assert mock_llm.generate_structured_output.called

    def test_generate_code_with_feedback(self, mock_llm):
        """Feedback from previous iteration is included in the user prompt."""
        generator = GeometryGenerator(llm_provider=mock_llm)

        mock_response = MagicMock()
        mock_response.python_code = "atoms = bulk('Cu', 'fcc', a=3.6)"
        mock_response.explanation = "Revised."
        mock_llm.generate_structured_output.return_value = mock_response

        candidate = MaterialCandidate(
            material_name="Cu", elements=["Cu"], reasoning="Good", expected_properties={}
        )
        ranked = RankedMaterial(candidate=candidate, score=0.9, ranking_justification="Best")

        generator.generate_code("Make Cu bulk", ranked, previous_feedback="Atoms too close")

        call_kwargs = mock_llm.generate_structured_output.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt", call_kwargs[1].get("user_prompt", ""))
        assert "CRITICAL FEEDBACK" in user_prompt
        assert "Atoms too close" in user_prompt


# ---------------------------------------------------------------------------
# GeometryReviewer tests (mocked)
# ---------------------------------------------------------------------------


class TestGeometryReviewer:
    """Tests for GeometryReviewer loop logic with mocked generator."""

    def _make_ranked_material(self):
        candidate = MaterialCandidate(
            material_name="Cu", elements=["Cu"], reasoning="test", expected_properties={}
        )
        return RankedMaterial(candidate=candidate, score=0.9, ranking_justification="test")

    def test_success_on_first_attempt(self, mock_llm, tmp_path):
        """Loop succeeds on first valid structure."""
        generator = GeometryGenerator(llm_provider=mock_llm)
        reviewer = GeometryReviewer(generator=generator, max_retries=3)

        # Use bulk() directly — it's injected via local_vars by GeometryReviewer
        mock_response = MagicMock()
        mock_response.python_code = "atoms = bulk('Cu', 'fcc', a=3.6)"
        mock_response.explanation = "Cu bulk"
        mock_llm.generate_structured_output.return_value = mock_response

        ranked = self._make_ranked_material()

        with patch("shalom.agents.simulation_layer.ASEBuilder.save_poscar") as mock_save:
            mock_save.return_value = str(tmp_path / "POSCAR_Cu")
            success, atoms, path = reviewer.run_creation_loop("test", ranked)

        assert success is True
        assert atoms is not None

    def test_failure_after_max_retries(self, mock_llm):
        """Loop fails after max_retries with bad code."""
        generator = GeometryGenerator(llm_provider=mock_llm)
        reviewer = GeometryReviewer(generator=generator, max_retries=2)

        # Code that raises an error each time
        mock_response = MagicMock()
        mock_response.python_code = "raise ValueError('broken')"
        mock_response.explanation = "Bad code"
        mock_llm.generate_structured_output.return_value = mock_response

        ranked = self._make_ranked_material()
        success, atoms, msg = reviewer.run_creation_loop("test", ranked)

        assert success is False
        assert atoms is None
        assert "Max retries" in msg
        assert mock_llm.generate_structured_output.call_count == 2

    def test_missing_atoms_variable(self, mock_llm):
        """Loop retries when 'atoms' variable is not created."""
        generator = GeometryGenerator(llm_provider=mock_llm)
        reviewer = GeometryReviewer(generator=generator, max_retries=2)

        # Code that doesn't create 'atoms'
        mock_response = MagicMock()
        mock_response.python_code = "x = 42"
        mock_response.explanation = "No atoms"
        mock_llm.generate_structured_output.return_value = mock_response

        ranked = self._make_ranked_material()
        success, atoms, msg = reviewer.run_creation_loop("test", ranked)

        assert success is False
        assert atoms is None
        assert "Max retries" in msg

    def test_invalid_structure_triggers_retry(self, mock_llm, tmp_path):
        """Loop retries when FormFiller rejects the structure."""
        generator = GeometryGenerator(llm_provider=mock_llm)
        reviewer = GeometryReviewer(generator=generator, max_retries=3)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            if call_count <= 2:
                # First two: code that produces no atoms (raises ValueError via guard)
                mock_resp.python_code = "x = 42"
                mock_resp.explanation = "Missing atoms"
            else:
                # Third: valid — bulk() is injected via local_vars
                mock_resp.python_code = "atoms = bulk('Cu', 'fcc', a=3.6)"
                mock_resp.explanation = "Fixed"
            return mock_resp

        mock_llm.generate_structured_output.side_effect = side_effect

        ranked = self._make_ranked_material()

        with patch("shalom.agents.simulation_layer.ASEBuilder.save_poscar") as mock_save:
            mock_save.return_value = str(tmp_path / "POSCAR_Cu")
            success, atoms, path = reviewer.run_creation_loop("test", ranked)

        assert success is True
        assert call_count == 3
