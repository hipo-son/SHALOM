"""Tests for shalom.backends.qe (QEBackend) implementation."""

import os

import pytest
from ase import Atoms
from ase.build import bulk

from shalom.backends.qe import QEBackend, RY_TO_EV, _format_qe_value
from shalom.backends.qe_config import (
    QECalculationType,
    QEKPointsConfig,
    get_qe_preset,
)
from shalom.backends._physics import AccuracyLevel


# ---------------------------------------------------------------------------
# write_input tests
# ---------------------------------------------------------------------------


class TestQEWriteInput:
    def test_creates_pw_in(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = QEBackend()
        result_dir = backend.write_input(atoms, str(tmp_path))
        assert result_dir == str(tmp_path)
        assert os.path.exists(os.path.join(str(tmp_path), "pw.in"))

    def test_custom_filename(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), filename="scf.in")
        assert os.path.exists(os.path.join(str(tmp_path), "scf.in"))

    def test_creates_directory(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = QEBackend()
        nested = os.path.join(str(tmp_path), "sub", "dir")
        backend.write_input(atoms, nested)
        assert os.path.exists(os.path.join(nested, "pw.in"))

    def test_config_none_creates_default(self, tmp_path):
        """config=None generates a default SCF config with auto-detection."""
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path))
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "ecutwfc" in content
        assert "ATOMIC_SPECIES" in content

    def test_namelists_present(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "&CONTROL" in content
        assert "&SYSTEM" in content
        assert "&ELECTRONS" in content

    def test_vc_relax_has_ions_cell(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = get_qe_preset(QECalculationType.VC_RELAX, AccuracyLevel.STANDARD, atoms)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "&IONS" in content
        assert "&CELL" in content

    def test_atomic_species_card(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "ATOMIC_SPECIES" in content
        assert "Cu" in content
        assert ".UPF" in content or ".upf" in content

    def test_atomic_positions_crystal(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path))
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "ATOMIC_POSITIONS crystal" in content

    def test_kpoints_automatic(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path))
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "K_POINTS automatic" in content

    def test_cell_parameters_angstrom(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path))
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "CELL_PARAMETERS angstrom" in content

    def test_prefix_outdir_present(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "prefix" in content
        assert "outdir" in content

    def test_pseudo_dir_in_control(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms)
        config.pseudo_dir = "/path/to/pseudos"
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "/path/to/pseudos" in content

    def test_multi_element(self, tmp_path):
        """Multi-element system generates correct ATOMIC_SPECIES."""
        atoms = Atoms("FeO", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4], pbc=True)
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms)
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "Fe" in content
        assert "O" in content
        assert "nat = 2" in content
        assert "ntyp = 2" in content


# ---------------------------------------------------------------------------
# parse_output tests
# ---------------------------------------------------------------------------


class TestQEParseOutput:
    def test_converged_scf(self, tmp_path):
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -31.50000000 Ry\n"
            "     convergence has been achieved in   8 iterations\n"
            "     Total force =     0.000100\n"
            "     total magnetization       =     2.50 Bohr mag/cell\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.is_converged is True
        assert result.energy is not None
        assert abs(result.energy - (-31.5 * RY_TO_EV)) < 0.1
        assert result.magnetization == 2.5

    def test_unconverged(self, tmp_path):
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -10.00000000 Ry\n"
            "     SCF iteration did not converge\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.is_converged is False
        assert result.energy is not None

    def test_missing_output_raises(self, tmp_path):
        backend = QEBackend()
        with pytest.raises(FileNotFoundError, match="No pw.x output"):
            backend.parse_output(str(tmp_path))

    def test_alternative_filenames(self, tmp_path):
        """Accepts pwscf.out or espresso.out."""
        (tmp_path / "espresso.out").write_text(
            "!    total energy              =     -5.00000000 Ry\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.is_converged is True
        assert result.energy is not None

    def test_ionic_energies(self, tmp_path):
        """Multiple energy lines → ionic_energies list."""
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -30.00000000 Ry\n"
            "!    total energy              =     -31.00000000 Ry\n"
            "!    total energy              =     -31.50000000 Ry\n"
            "     convergence has been achieved in   5 iterations\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.ionic_energies is not None
        assert len(result.ionic_energies) == 3
        # Last energy should be the final one
        assert abs(result.energy - (-31.5 * RY_TO_EV)) < 0.1

    def test_per_atom_forces(self, tmp_path):
        """Per-atom forces parsed from force block."""
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -31.50000000 Ry\n"
            "     Forces acting on atoms (cartesian axes, Ry/au):\n"
            "     atom    1 type  1   force =     0.001000    0.002000    0.003000\n"
            "     atom    2 type  1   force =    -0.001000   -0.002000   -0.003000\n"
            "     Total force =     0.005000\n"
            "     convergence has been achieved in   5 iterations\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.forces is not None
        assert len(result.forces) == 2
        assert len(result.forces[0]) == 3

    def test_no_magnetization(self, tmp_path):
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -5.00000000 Ry\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.magnetization is None

    def test_raw_contains_source(self, tmp_path):
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -5.00000000 Ry\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.raw["source"] == "qe_regex"

    def test_parse_scientific_notation_energy(self, tmp_path):
        """Energy in scientific notation (e.g. small systems)."""
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -3.15000000E+02 Ry\n"
            "     convergence has been achieved in   5 iterations\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.energy is not None
        assert abs(result.energy - (-315.0 * RY_TO_EV)) < 0.1

    def test_parse_scientific_notation_forces(self, tmp_path):
        """Per-atom forces in scientific notation."""
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -10.00000000 Ry\n"
            "     Forces acting on atoms (cartesian axes, Ry/au):\n"
            "     atom    1 type  1   force =     1.23E-04    2.34E-05   -3.45E-06\n"
            "     Total force =     1.50E-04\n"
            "     convergence has been achieved in   5 iterations\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.forces is not None
        assert len(result.forces) == 1
        assert result.forces_max is not None

    def test_parse_fallback_generic_out_file(self, tmp_path):
        """Falls back to any .out file if standard names not found."""
        (tmp_path / "custom_run.out").write_text(
            "!    total energy              =     -5.00000000 Ry\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.is_converged is True
        assert result.energy is not None

    def test_parse_single_energy_ionic_list(self, tmp_path):
        """Single energy → ionic_energies has 1 entry."""
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -10.00000000 Ry\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.ionic_energies is not None
        assert len(result.ionic_energies) == 1

    def test_parse_empty_force_block(self, tmp_path):
        """Force header with no atom lines → no crash."""
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -10.00000000 Ry\n"
            "     Forces acting on atoms (cartesian axes, Ry/au):\n"
            "\n"
            "     Total force =     0.000000\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.is_converged is True

    def test_parse_multiple_magnetization(self, tmp_path):
        """Multiple magnetization lines → last value wins."""
        (tmp_path / "pw.out").write_text(
            "!    total energy              =     -10.00000000 Ry\n"
            "     total magnetization       =     1.50 Bohr mag/cell\n"
            "!    total energy              =     -10.50000000 Ry\n"
            "     total magnetization       =     2.00 Bohr mag/cell\n"
            "     JOB DONE.\n"
        )
        backend = QEBackend()
        result = backend.parse_output(str(tmp_path))
        assert result.magnetization == 2.0


# ---------------------------------------------------------------------------
# K-points mode tests
# ---------------------------------------------------------------------------


class TestKPointsModes:
    def test_kpoints_gamma_mode(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms)
        config.kpoints = QEKPointsConfig(mode="gamma")
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "K_POINTS gamma" in content

    def test_kpoints_crystal_b_mode(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = get_qe_preset(QECalculationType.BANDS, AccuracyLevel.STANDARD, atoms)
        config.kpoints = QEKPointsConfig(mode="crystal_b")
        backend = QEBackend()
        backend.write_input(atoms, str(tmp_path), config=config)
        with open(os.path.join(str(tmp_path), "pw.in"), "r") as f:
            content = f.read()
        assert "K_POINTS crystal_b" in content
        assert "seekpath" in content


# ---------------------------------------------------------------------------
# _format_qe_value tests
# ---------------------------------------------------------------------------


class TestFormatQEValue:
    def test_prequoted_string(self):
        assert _format_qe_value("'cold'") == "'cold'"

    def test_unquoted_string(self):
        assert _format_qe_value("cold") == "'cold'"

    def test_bool_true(self):
        assert _format_qe_value(True) == ".true."

    def test_bool_false(self):
        assert _format_qe_value(False) == ".false."

    def test_int(self):
        assert _format_qe_value(42) == "42"

    def test_float_normal(self):
        assert _format_qe_value(0.5) == "0.5"

    def test_float_small_scientific(self):
        result = _format_qe_value(7.35e-7)
        assert "E" in result or "e" in result

    def test_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            _format_qe_value(None)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            _format_qe_value([1, 2, 3])

    def test_dict_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            _format_qe_value({"key": "val"})
