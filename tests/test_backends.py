import os

import pytest
from ase import Atoms
from ase.build import bulk

from shalom.backends import get_backend, DFTBackend, DFTResult
from shalom.backends.vasp import VASPBackend
from shalom.backends.qe import QEBackend
from shalom.backends.vasp_config import (
    CalculationType,
    AccuracyLevel,
    VASPInputConfig,
    get_preset,
)
from shalom.agents.review_layer import ReviewAgent
from shalom.core.schemas import ReviewResult

# ---------------------------------------------------------------------------
# DFTResult tests
# ---------------------------------------------------------------------------


class TestDFTResult:
    """Tests for the DFTResult dataclass."""

    def test_defaults(self):
        result = DFTResult()
        assert result.energy is None
        assert result.forces_max is None
        assert result.is_converged is False
        assert result.raw == {}
        # Phase 1 new fields
        assert result.bandgap is None
        assert result.magnetization is None
        assert result.entropy_per_atom is None
        assert result.stress_tensor is None
        assert result.forces is None
        assert result.ionic_energies is None
        assert result.ionic_forces_max is None
        assert result.correction_history == []

    def test_custom_values(self):
        result = DFTResult(energy=-34.5, forces_max=0.02, is_converged=True, raw={"key": "val"})
        assert result.energy == -34.5
        assert result.forces_max == 0.02
        assert result.is_converged is True
        assert result.raw == {"key": "val"}


# ---------------------------------------------------------------------------
# VASPBackend tests
# ---------------------------------------------------------------------------


class TestVASPBackend:
    """Tests for VASPBackend implementation."""

    def test_implements_protocol(self):
        backend = VASPBackend()
        assert isinstance(backend, DFTBackend)

    def test_name(self):
        assert VASPBackend().name == "vasp"

    def test_write_input_creates_poscar(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = VASPBackend()

        result_dir = backend.write_input(atoms, str(tmp_path))

        assert result_dir == str(tmp_path)
        poscar_path = os.path.join(str(tmp_path), "POSCAR")
        assert os.path.exists(poscar_path)
        with open(poscar_path, "r") as f:
            content = f.read()
        assert "Cu" in content

    def test_write_input_custom_filename(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = VASPBackend()

        backend.write_input(atoms, str(tmp_path), filename="POSCAR_Cu")

        assert os.path.exists(os.path.join(str(tmp_path), "POSCAR_Cu"))

    def test_write_input_creates_directory(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = VASPBackend()
        nested = os.path.join(str(tmp_path), "sub", "dir")

        backend.write_input(atoms, nested)

        assert os.path.exists(os.path.join(nested, "POSCAR"))

    def test_parse_output_with_fixture(self):
        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        # Rename dummy_outcar.txt -> OUTCAR in a temp copy
        backend = VASPBackend()

        # Use the fixture directory directly by creating a symlink-like test
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(
                os.path.join(fixture_dir, "dummy_outcar.txt"),
                os.path.join(tmpdir, "OUTCAR"),
            )
            result = backend.parse_output(tmpdir)

        assert result.is_converged is True
        assert result.energy == -34.567890
        # Old fixture has no TOTAL-FORCE block → forces_max is None
        assert result.forces_max is None
        assert result.raw["energy"] == -34.567890

    def test_parse_output_missing_outcar(self, tmp_path):
        backend = VASPBackend()
        with pytest.raises(FileNotFoundError, match="OUTCAR"):
            backend.parse_output(str(tmp_path))

    def test_parse_output_unconverged(self, tmp_path):
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN  =       -10.123456 eV\n"
            " Some other output without timing section\n"
        )
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))

        assert result.is_converged is False
        assert result.energy == -10.123456


# ---------------------------------------------------------------------------
# QEBackend tests
# ---------------------------------------------------------------------------


class TestQEBackend:
    """Tests for QEBackend stub."""

    def test_implements_protocol(self):
        backend = QEBackend()
        assert isinstance(backend, DFTBackend)

    def test_name(self):
        assert QEBackend().name == "qe"

    def test_write_input_raises(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = QEBackend()
        with pytest.raises(NotImplementedError, match="Phase 3"):
            backend.write_input(atoms, "/tmp")

    def test_parse_output_raises(self):
        backend = QEBackend()
        with pytest.raises(NotImplementedError, match="Phase 3"):
            backend.parse_output("/tmp")


# ---------------------------------------------------------------------------
# get_backend factory tests
# ---------------------------------------------------------------------------


class TestGetBackend:
    """Tests for the get_backend factory function."""

    def test_vasp(self):
        backend = get_backend("vasp")
        assert isinstance(backend, VASPBackend)

    def test_qe(self):
        backend = get_backend("qe")
        assert isinstance(backend, QEBackend)

    def test_default_is_vasp(self):
        backend = get_backend()
        assert isinstance(backend, VASPBackend)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("gaussian")


# ---------------------------------------------------------------------------
# ReviewAgent.review_with_backend integration tests
# ---------------------------------------------------------------------------


class TestReviewWithBackend:
    """Tests for ReviewAgent.review_with_backend() with VASPBackend."""

    def test_review_with_vasp_backend(self, mock_llm):
        agent = ReviewAgent(llm_provider=mock_llm)

        mock_result = ReviewResult(
            is_successful=True,
            energy=-34.567890,
            forces_max=0.01,
            feedback_for_design="Converged successfully.",
        )
        mock_llm.generate_structured_output.return_value = mock_result

        import shutil
        import tempfile

        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(
                os.path.join(fixture_dir, "dummy_outcar.txt"),
                os.path.join(tmpdir, "OUTCAR"),
            )
            backend = VASPBackend()
            result = agent.review_with_backend("Find stable material", tmpdir, backend)

        assert result.is_successful is True
        assert mock_llm.generate_structured_output.called

    def test_review_backward_compatible(self, mock_llm):
        """Existing review() method still works with OUTCAR path."""
        agent = ReviewAgent(llm_provider=mock_llm)

        mock_result = ReviewResult(
            is_successful=True,
            energy=-34.567890,
            forces_max=0.01,
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


# ---------------------------------------------------------------------------
# VASPBackend Config-Based Input Generation (Phase 1.2)
# ---------------------------------------------------------------------------


class TestVASPBackendConfig:
    """Tests for VASPBackend with VASPInputConfig."""

    def test_config_none_poscar_only_backward_compat(self, tmp_path):
        """config=None → POSCAR only (backward compatible via **params)."""
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = VASPBackend()
        backend.write_input(atoms, str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "POSCAR"))
        assert not os.path.exists(os.path.join(str(tmp_path), "INCAR"))
        assert not os.path.exists(os.path.join(str(tmp_path), "KPOINTS"))

    def test_config_generates_four_files(self, tmp_path, sample_bulk_cu):
        """config → POSCAR + INCAR + KPOINTS + POTCAR.spec."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)
        for fname in ["POSCAR", "INCAR", "KPOINTS", "POTCAR.spec"]:
            assert os.path.exists(os.path.join(str(tmp_path), fname)), f"Missing {fname}"

    def test_incar_contains_preset_values(self, tmp_path, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "INCAR"), "r") as f:
            content = f.read()
        assert "ENCUT" in content
        assert "EDIFF" in content
        assert "ISMEAR" in content

    def test_config_plus_filename_both_work(self, tmp_path, sample_bulk_cu):
        """config + filename **params both passed simultaneously."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config, filename="POSCAR_Cu")
        assert os.path.exists(os.path.join(str(tmp_path), "POSCAR_Cu"))
        assert os.path.exists(os.path.join(str(tmp_path), "INCAR"))

    def test_user_incar_override_priority(self, tmp_path, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        config.user_incar_settings["ENCUT"] = 999
        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "INCAR"), "r") as f:
            content = f.read()
        assert "999" in content

    def test_2d_isif_4_in_incar(self, tmp_path, sample_2d_slab):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_2d_slab)
        assert config.is_2d is True
        backend = VASPBackend()
        backend.write_input(sample_2d_slab, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "INCAR"), "r") as f:
            content = f.read()
        assert "ISIF = 4" in content

    def test_2d_kpoints_z_1(self, tmp_path, sample_2d_slab):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_2d_slab)
        backend = VASPBackend()
        backend.write_input(sample_2d_slab, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "KPOINTS"), "r") as f:
            lines = f.readlines()
        # The 4th line (index 3) has the grid: Nx Ny 1
        grid_line = lines[3].strip()
        parts = grid_line.split()
        assert parts[-1] == "1", f"KPOINTS z should be 1, got: {grid_line}"

    def test_metal_ismear_sigma_in_incar(self, tmp_path, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "INCAR"), "r") as f:
            content = f.read()
        assert "ISMEAR = 1" in content
        assert "SIGMA = 0.1" in content


# ---------------------------------------------------------------------------
# Force Parsing Tests (Phase 1.2)
# ---------------------------------------------------------------------------


class TestForcesParsing:
    """Tests for real OUTCAR force/entropy/ionic-step parsing."""

    def test_real_forces_parsed(self, dummy_outcar_forces):
        backend = VASPBackend()
        result = backend.parse_output(dummy_outcar_forces)
        assert result.forces_max is not None
        assert result.forces_max > 0
        # Should be sqrt(0.012^2 + 0.008^2 + 0.035^2) ≈ 0.0380
        assert abs(result.forces_max - 0.038) < 0.002

    def test_forces_array(self, dummy_outcar_forces):
        backend = VASPBackend()
        result = backend.parse_output(dummy_outcar_forces)
        assert result.forces is not None
        assert len(result.forces) == 2
        assert len(result.forces[0]) == 3

    def test_entropy_parsed(self, dummy_outcar_forces):
        backend = VASPBackend()
        result = backend.parse_output(dummy_outcar_forces)
        assert result.entropy_per_atom is not None
        # -0.0015 / 2 atoms = -0.00075
        assert abs(result.entropy_per_atom - (-0.00075)) < 1e-6

    def test_ionic_step_energies(self, dummy_outcar_ionic_steps):
        backend = VASPBackend()
        result = backend.parse_output(dummy_outcar_ionic_steps)
        assert result.ionic_energies is not None
        assert len(result.ionic_energies) == 7
        assert result.ionic_energies[0] == pytest.approx(-30.1, abs=0.01)

    def test_ionic_step_forces(self, dummy_outcar_ionic_steps):
        backend = VASPBackend()
        result = backend.parse_output(dummy_outcar_ionic_steps)
        assert result.ionic_forces_max is not None
        assert len(result.ionic_forces_max) == 7

    def test_incomplete_outcar_graceful(self, tmp_path):
        """OUTCAR with partial data doesn't crash."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(" free  energy   TOTEN  =       -5.0 eV\n")
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.energy == -5.0
        assert result.is_converged is False
        assert result.forces_max is None


# ---------------------------------------------------------------------------
# POTCAR.spec Tests (Phase 1.2)
# ---------------------------------------------------------------------------


class TestPOTCARSpec:
    """Tests for POTCAR.spec file generation."""

    def test_multi_element_spec(self, tmp_path, sample_tmo_feo):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_tmo_feo)
        backend = VASPBackend()
        backend.write_input(sample_tmo_feo, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "POTCAR.spec"), "r", encoding="utf-8") as f:
            content = f.read()
        # Fe uses "Fe_pv" in vasp_recommended (MP standard)
        assert "Fe_pv\n" in content
        assert "O\n" in content

    def test_vasp_recommended_in_spec(self, tmp_path, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "POTCAR.spec"), "r", encoding="utf-8") as f:
            content = f.read()
        assert "vasp_recommended" in content
        assert "Cu\n" in content

    def test_mp_default_preset(self, tmp_path, sample_bulk_fe):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_fe)
        config.potcar_preset = "mp_default"
        backend = VASPBackend()
        backend.write_input(sample_bulk_fe, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "POTCAR.spec"), "r", encoding="utf-8") as f:
            content = f.read()
        assert "Fe_pv\n" in content


# ---------------------------------------------------------------------------
# Double Relaxation Tests (Phase 1.2)
# ---------------------------------------------------------------------------


class TestDoubleRelaxation:
    """Tests for double relaxation input generation."""

    def test_creates_two_directories(self, tmp_path, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        dirs = backend.write_double_relaxation(sample_bulk_cu, str(tmp_path), config)
        assert len(dirs) == 2
        assert os.path.isdir(dirs[0])
        assert os.path.isdir(dirs[1])

    def test_step1_coarse_settings(self, tmp_path, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        dirs = backend.write_double_relaxation(sample_bulk_cu, str(tmp_path), config)
        with open(os.path.join(dirs[0], "INCAR"), "r") as f:
            content = f.read()
        assert "-0.05" in content  # EDIFFG coarse

    def test_step2_fine_settings(self, tmp_path, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        dirs = backend.write_double_relaxation(sample_bulk_cu, str(tmp_path), config)
        with open(os.path.join(dirs[1], "INCAR"), "r") as f:
            content = f.read()
        # Fine step uses original EDIFFG from config
        assert "-0.02" in content

    def test_both_steps_have_four_files(self, tmp_path, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        dirs = backend.write_double_relaxation(sample_bulk_cu, str(tmp_path), config)
        for d in dirs:
            for fname in ["POSCAR", "INCAR", "KPOINTS", "POTCAR.spec"]:
                assert os.path.exists(os.path.join(d, fname)), f"Missing {fname} in {d}"


# ---------------------------------------------------------------------------
# KPOINTS Mode Tests (Phase 1.2 coverage)
# ---------------------------------------------------------------------------


class TestKPOINTSModes:
    """Tests for KPOINTS generation in line and explicit modes."""

    def test_line_mode_kpoints(self, tmp_path, sample_bulk_cu):
        """kpoints.mode='line' generates band structure line-mode KPOINTS."""
        config = get_preset(CalculationType.BAND_STRUCTURE, AccuracyLevel.STANDARD, sample_bulk_cu)
        # Force line mode
        config.kpoints.mode = "line"
        config.kpoints.num_kpts_per_segment = 40

        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)

        with open(os.path.join(str(tmp_path), "KPOINTS"), "r", encoding="utf-8") as f:
            content = f.read()

        assert "K-Points along high symmetry lines" in content
        assert "40" in content
        assert "Line-mode" in content
        assert "Reciprocal" in content

    def test_explicit_mode_kpoints(self, tmp_path, sample_bulk_cu):
        """kpoints.mode='explicit' with grid generates Gamma-centered mesh."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        config.kpoints.mode = "explicit"
        config.kpoints.grid = [6, 6, 6]

        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)

        with open(os.path.join(str(tmp_path), "KPOINTS"), "r", encoding="utf-8") as f:
            content = f.read()

        assert "Automatic mesh" in content
        assert "Gamma" in content
        assert "6  6  6" in content

    def test_automatic_mode_default(self, tmp_path, sample_bulk_cu):
        """Default automatic mode uses computed grid."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        assert config.kpoints.mode == "automatic"

        backend = VASPBackend()
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)

        with open(os.path.join(str(tmp_path), "KPOINTS"), "r", encoding="utf-8") as f:
            content = f.read()

        assert "Automatic mesh" in content
        assert "Gamma" in content


# ---------------------------------------------------------------------------
# Magnetization Parsing Tests (Phase 1.2 coverage)
# ---------------------------------------------------------------------------


class TestMagnetizationParsing:
    """Tests for magnetization parsing from OUTCAR."""

    def test_magnetization_parsed(self, tmp_path):
        """Magnetization is parsed from OUTCAR 'number of electron ... magnetization' line."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN  =       -15.000000 eV\n"
            "\n"
            "   number of electron      16.0000000 magnetization       2.1234567\n"
            "\n"
            " General timing and accounting informations for this job:\n"
        )
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.magnetization is not None
        assert abs(result.magnetization - 2.1234567) < 1e-5

    def test_negative_magnetization(self, tmp_path):
        """Negative magnetization is parsed correctly."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN  =       -15.000000 eV\n"
            "   number of electron      16.0000000 magnetization      -1.5000000\n"
            " General timing and accounting informations for this job:\n"
        )
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.magnetization == pytest.approx(-1.5)

    def test_no_magnetization_line(self, tmp_path):
        """No magnetization line -> magnetization is None."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN  =       -15.000000 eV\n"
            " General timing and accounting informations for this job:\n"
        )
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.magnetization is None


# ---------------------------------------------------------------------------
# Parse Edge Cases (Phase 1.2 coverage)
# ---------------------------------------------------------------------------


class TestParseEdgeCases:
    """Tests for graceful handling of malformed OUTCAR lines."""

    def test_malformed_energy_line(self, tmp_path):
        """Malformed TOTEN line doesn't crash (IndexError/ValueError caught)."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN  =       CORRUPTED eV\n"
            " General timing and accounting informations for this job:\n"
        )
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.energy is None
        assert result.is_converged is True

    def test_malformed_force_line(self, tmp_path):
        """Malformed force values in TOTAL-FORCE block don't crash."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN  =       -10.0 eV\n"
            "\n"
            " TOTAL-FORCE (eV/Angst)\n"
            " ---------------------------\n"
            "      0.000  0.000  0.000    BAD  VALUE  DATA\n"
            " ---------------------------\n"
            "\n"
            " General timing and accounting informations for this job:\n"
        )
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.energy == -10.0
        # Forces block had bad data -> no forces parsed
        assert result.forces_max is None

    def test_malformed_entropy_line(self, tmp_path):
        """Malformed EENTRO value doesn't crash (ValueError caught)."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN  =       -10.0 eV\n"
            " EENTRO  =        NaN_BAD\n"
            " number of atoms/cell =      2\n"
            " General timing and accounting informations for this job:\n"
        )
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.entropy_per_atom is None

    def test_truncated_energy_line(self, tmp_path):
        """Energy line with too few tokens doesn't crash (IndexError caught)."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " free  energy   TOTEN\n"
            " General timing and accounting informations for this job:\n"
        )
        backend = VASPBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.energy is None


# ---------------------------------------------------------------------------
# Pymatgen Path Tests (Phase 1.2 coverage — mocked)
# ---------------------------------------------------------------------------


class TestPymatgenParsePath:
    """Tests for _parse_with_pymatgen using mocks (pymatgen not installed)."""

    def test_pymatgen_path_with_vasprun(self, tmp_path):
        """When _PYMATGEN_AVAILABLE=True and vasprun.xml exists, uses pymatgen."""
        from unittest.mock import patch, MagicMock

        # Create OUTCAR and vasprun.xml
        (tmp_path / "OUTCAR").write_text(
            " General timing and accounting informations for this job:\n"
        )
        (tmp_path / "vasprun.xml").write_text("<xml>dummy</xml>")

        mock_outcar = MagicMock()
        mock_outcar.final_energy = -25.0
        mock_outcar.magnetization = [{"tot": {"tot": 3.5}}]

        mock_vasprun = MagicMock()
        mock_vasprun.final_energy = -25.5
        mock_vasprun.converged = True
        mock_bs = MagicMock()
        mock_bs.get_band_gap.return_value = {"energy": 1.2}
        mock_vasprun.get_band_structure.return_value = mock_bs

        backend = VASPBackend()
        with patch("shalom.backends.vasp._PYMATGEN_AVAILABLE", True), \
             patch("shalom.backends.vasp.PmgOutcar", create=True, return_value=mock_outcar), \
             patch("shalom.backends.vasp.Vasprun", create=True, return_value=mock_vasprun):
            result = backend.parse_output(str(tmp_path))

        assert result.energy == -25.5
        assert result.is_converged is True
        assert result.bandgap == 1.2
        assert result.raw["source"] == "pymatgen"

    def test_pymatgen_path_outcar_only(self, tmp_path):
        """Pymatgen path without vasprun.xml falls back to OUTCAR data."""
        from unittest.mock import patch, MagicMock

        (tmp_path / "OUTCAR").write_text(
            " General timing and accounting informations for this job:\n"
        )

        mock_outcar = MagicMock()
        mock_outcar.final_energy = -30.0
        mock_outcar.magnetization = []

        backend = VASPBackend()
        with patch("shalom.backends.vasp._PYMATGEN_AVAILABLE", True), \
             patch("shalom.backends.vasp.PmgOutcar", create=True, return_value=mock_outcar):
            result = backend.parse_output(str(tmp_path))

        assert result.energy == -30.0
        assert result.is_converged is True
        assert result.bandgap is None

    def test_pymatgen_vasprun_exception(self, tmp_path):
        """Vasprun parsing exception is caught gracefully."""
        from unittest.mock import patch, MagicMock

        (tmp_path / "OUTCAR").write_text(
            " General timing and accounting informations for this job:\n"
        )
        (tmp_path / "vasprun.xml").write_text("<xml>corrupt</xml>")

        mock_outcar = MagicMock()
        mock_outcar.final_energy = -20.0
        mock_outcar.magnetization = []

        backend = VASPBackend()
        with patch("shalom.backends.vasp._PYMATGEN_AVAILABLE", True), \
             patch("shalom.backends.vasp.PmgOutcar", create=True, return_value=mock_outcar), \
             patch("shalom.backends.vasp.Vasprun", create=True, side_effect=Exception("parse error")):
            result = backend.parse_output(str(tmp_path))

        # Falls back to OUTCAR energy
        assert result.energy == -20.0
        assert result.is_converged is True

    def test_pymatgen_outcar_magnetization(self, tmp_path):
        """Pymatgen OUTCAR magnetization is extracted."""
        from unittest.mock import patch, MagicMock

        (tmp_path / "OUTCAR").write_text(
            " General timing and accounting informations for this job:\n"
        )

        mock_outcar = MagicMock()
        mock_outcar.final_energy = -15.0
        mock_outcar.magnetization = [{"tot": {"tot": 4.2}}]

        backend = VASPBackend()
        with patch("shalom.backends.vasp._PYMATGEN_AVAILABLE", True), \
             patch("shalom.backends.vasp.PmgOutcar", create=True, return_value=mock_outcar):
            result = backend.parse_output(str(tmp_path))

        assert result.magnetization == 4.2

    def test_pymatgen_outcar_magnetization_exception(self, tmp_path):
        """Malformed magnetization in pymatgen path doesn't crash."""
        from unittest.mock import patch, MagicMock

        (tmp_path / "OUTCAR").write_text(
            " General timing and accounting informations for this job:\n"
        )

        mock_outcar = MagicMock()
        mock_outcar.final_energy = -15.0
        mock_outcar.magnetization = ["bad_data"]

        backend = VASPBackend()
        with patch("shalom.backends.vasp._PYMATGEN_AVAILABLE", True), \
             patch("shalom.backends.vasp.PmgOutcar", create=True, return_value=mock_outcar):
            result = backend.parse_output(str(tmp_path))

        assert result.magnetization is None

    def test_pymatgen_convergence_fallback(self, tmp_path):
        """When vasprun not converged, checks OUTCAR timing section."""
        from unittest.mock import patch, MagicMock

        (tmp_path / "OUTCAR").write_text(
            " NOT converged via vasprun\n"
        )

        mock_outcar = MagicMock()
        mock_outcar.final_energy = -10.0
        mock_outcar.magnetization = []

        backend = VASPBackend()
        with patch("shalom.backends.vasp._PYMATGEN_AVAILABLE", True), \
             patch("shalom.backends.vasp.PmgOutcar", create=True, return_value=mock_outcar):
            result = backend.parse_output(str(tmp_path))

        assert result.is_converged is False
