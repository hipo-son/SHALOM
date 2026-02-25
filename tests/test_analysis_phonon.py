"""Tests for shalom.analysis phonon module.

All tests are pure unit tests — no DFT execution required.
Uses Si diamond structure with synthetic forces for validation.
"""

import numpy as np
import pytest
from unittest.mock import patch

from shalom.analysis._base import PhononResult
from shalom.analysis.phonon import (
    is_phonopy_available,
    generate_phonon_displacements,
    analyze_phonon,
    analyze_phonon_from_force_constants,
    _ensure_phonopy_available,
    _ase_to_phonopy,
    _phonopy_to_ase,
    _normalize_label,
    _run_phonon_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _si_atoms():
    """Return Si diamond primitive cell."""
    from ase.build import bulk

    return bulk("Si", "diamond", a=5.43)


def _si_supercell_matrix():
    """Return 2x2x2 diagonal supercell matrix."""
    return [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def _si_force_sets(n_disps: int, n_atoms: int, seed: int = 42):
    """Return synthetic force sets for testing."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n_atoms, 3)) * 0.1 for _ in range(n_disps)]


# ---------------------------------------------------------------------------
# PhononResult dataclass
# ---------------------------------------------------------------------------


class TestPhononResult:
    def test_default_fields(self):
        r = PhononResult()
        assert r.band_qpoints is None
        assert r.band_distances is None
        assert r.band_frequencies is None
        assert r.band_labels == {}
        assert r.dos_frequencies is None
        assert r.dos_density is None
        assert r.thermal_temperatures is None
        assert r.thermal_free_energy is None
        assert r.thermal_entropy is None
        assert r.thermal_cv is None
        assert r.min_frequency_THz is None
        assert r.is_stable is False
        assert r.imaginary_modes == []
        assert r.n_atoms == 0
        assert r.n_branches == 0
        assert r.force_constants is None
        assert r.raw is None
        assert r.metadata == {}

    def test_field_assignment(self):
        freqs = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = PhononResult(band_frequencies=freqs, n_atoms=2, n_branches=6)
        assert r.band_frequencies is freqs
        assert r.n_atoms == 2
        assert r.n_branches == 6

    def test_metadata_mutable(self):
        r = PhononResult()
        r.metadata["source"] = "test"
        assert r.metadata["source"] == "test"

    def test_imaginary_modes_list(self):
        r = PhononResult(imaginary_modes=[(0, 1, -0.5), (10, 2, -0.3)])
        assert len(r.imaginary_modes) == 2
        assert r.imaginary_modes[0] == (0, 1, -0.5)

    def test_is_stable_default_false(self):
        r = PhononResult()
        assert r.is_stable is False


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


class TestAvailabilityGuard:
    def test_is_phonopy_available(self):
        assert is_phonopy_available() is True

    def test_ensure_phonopy_available_passes(self):
        _ensure_phonopy_available()  # should not raise

    def test_ensure_raises_when_unavailable(self):
        with patch("shalom.analysis.phonon._PHONOPY_AVAILABLE", False):
            with pytest.raises(ImportError, match="phonopy"):
                _ensure_phonopy_available()

    def test_is_phonopy_available_false(self):
        with patch("shalom.analysis.phonon._PHONOPY_AVAILABLE", False):
            assert is_phonopy_available() is False


# ---------------------------------------------------------------------------
# ASE ↔ phonopy conversion
# ---------------------------------------------------------------------------


class TestConversion:
    def test_ase_to_phonopy_symbols(self):
        atoms = _si_atoms()
        pa = _ase_to_phonopy(atoms)
        assert list(pa.symbols) == atoms.get_chemical_symbols()

    def test_ase_to_phonopy_cell(self):
        atoms = _si_atoms()
        pa = _ase_to_phonopy(atoms)
        np.testing.assert_allclose(pa.cell, atoms.cell.array, atol=1e-10)

    def test_round_trip(self):
        atoms = _si_atoms()
        pa = _ase_to_phonopy(atoms)
        atoms2 = _phonopy_to_ase(pa)
        assert atoms2.get_chemical_symbols() == atoms.get_chemical_symbols()
        np.testing.assert_allclose(atoms2.cell.array, atoms.cell.array, atol=1e-10)
        np.testing.assert_allclose(
            atoms2.get_scaled_positions(),
            atoms.get_scaled_positions(),
            atol=1e-10,
        )
        assert all(atoms2.pbc)


# ---------------------------------------------------------------------------
# generate_phonon_displacements
# ---------------------------------------------------------------------------


class TestGenerateDisplacements:
    def test_si_displacement_count(self):
        atoms = _si_atoms()
        disps, ph = generate_phonon_displacements(atoms, _si_supercell_matrix())
        assert len(disps) >= 1  # Si has high symmetry, typically 1 displacement

    def test_displacement_cell_size(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        # 2x2x2 supercell of Si (2 atoms) = 16 atoms
        assert len(disps[0]) == 16

    def test_displacement_pbc(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        assert all(disps[0].pbc)

    def test_invalid_supercell_shape(self):
        atoms = _si_atoms()
        with pytest.raises(ValueError, match="3x3"):
            generate_phonon_displacements(atoms, [[2, 2], [2, 2]])

    def test_returns_phonopy_object(self):
        atoms = _si_atoms()
        _, ph = generate_phonon_displacements(atoms, _si_supercell_matrix())
        from phonopy import Phonopy
        assert isinstance(ph, Phonopy)


# ---------------------------------------------------------------------------
# analyze_phonon (from force sets)
# ---------------------------------------------------------------------------


class TestAnalyzePhonon:
    def test_returns_phonon_result(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_max=300, t_step=100,
        )
        assert isinstance(result, PhononResult)

    def test_band_structure_shape(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_max=300, t_step=100,
        )
        assert result.band_frequencies is not None
        # 2 atoms in primitive cell → 6 branches
        assert result.band_frequencies.shape[1] == 6
        assert result.n_branches == 6
        assert result.n_atoms == 2

    def test_dos_computed(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_max=300, t_step=100,
        )
        assert result.dos_frequencies is not None
        assert result.dos_density is not None
        assert len(result.dos_frequencies) == len(result.dos_density)

    def test_thermal_properties(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_min=0, t_max=300, t_step=100,
        )
        assert result.thermal_temperatures is not None
        assert result.thermal_cv is not None
        assert result.thermal_entropy is not None
        assert result.thermal_free_energy is not None

    def test_band_labels_present(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_max=100, t_step=50,
        )
        assert len(result.band_labels) > 0
        # G (Gamma) should appear in labels
        assert "G" in result.band_labels.values()

    def test_force_constants_stored(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_max=100, t_step=50,
        )
        assert result.force_constants is not None

    def test_raw_phonopy_object(self):
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_max=100, t_step=50,
        )
        assert result.raw is not None

    def test_empty_force_sets(self):
        atoms = _si_atoms()
        with pytest.raises(ValueError, match="empty"):
            analyze_phonon(atoms, _si_supercell_matrix(), [])

    def test_invalid_force_shape(self):
        atoms = _si_atoms()
        with pytest.raises(ValueError, match="shape"):
            analyze_phonon(atoms, _si_supercell_matrix(), [[[1, 2]]])

    def test_invalid_supercell(self):
        atoms = _si_atoms()
        with pytest.raises(ValueError, match="3x3"):
            analyze_phonon(atoms, [[2, 2]], [np.zeros((16, 3))])


# ---------------------------------------------------------------------------
# analyze_phonon_from_force_constants
# ---------------------------------------------------------------------------


class TestAnalyzeFromForceConstants:
    def test_returns_result(self):
        atoms = _si_atoms()
        # Generate force constants via phonopy
        disps, ph = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16)
        ph.forces = np.array(force_sets)
        ph.produce_force_constants()
        fc = ph.force_constants

        result = analyze_phonon_from_force_constants(
            atoms, fc, _si_supercell_matrix(),
            mesh=[4, 4, 4], band_npoints=5, t_max=100, t_step=50,
        )
        assert isinstance(result, PhononResult)
        assert result.band_frequencies is not None

    def test_invalid_fc_shape(self):
        atoms = _si_atoms()
        with pytest.raises(ValueError, match="shape"):
            analyze_phonon_from_force_constants(
                atoms, np.zeros((3, 3)), _si_supercell_matrix(),
            )


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class TestStability:
    def test_random_forces_unstable(self):
        """Random forces produce imaginary modes → unstable."""
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16, seed=0)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_max=100, t_step=50,
        )
        # Random forces will almost certainly produce imaginary modes
        assert result.min_frequency_THz is not None
        assert result.min_frequency_THz < 0
        assert result.is_stable is False
        assert len(result.imaginary_modes) > 0

    def test_imaginary_modes_format(self):
        """Imaginary modes list has correct tuple format."""
        atoms = _si_atoms()
        disps, _ = generate_phonon_displacements(atoms, _si_supercell_matrix())
        force_sets = _si_force_sets(len(disps), 16, seed=0)
        result = analyze_phonon(
            atoms, _si_supercell_matrix(), force_sets,
            mesh=[4, 4, 4], band_npoints=5, t_max=100, t_step=50,
        )
        if result.imaginary_modes:
            q_idx, b_idx, freq = result.imaginary_modes[0]
            assert isinstance(q_idx, int)
            assert isinstance(b_idx, int)
            assert isinstance(freq, float)
            assert freq < 0


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


class TestNormalizeLabel:
    def test_gamma_dollar(self):
        assert _normalize_label("$\\Gamma$") == "G"

    def test_gamma_plain(self):
        assert _normalize_label("\\Gamma") == "G"

    def test_gamma_upper(self):
        assert _normalize_label("GAMMA") == "G"

    def test_mathrm(self):
        assert _normalize_label("$\\mathrm{X}$") == "X"

    def test_mathrm_multi(self):
        assert _normalize_label("$\\mathrm{K}$") == "K"

    def test_plain_label(self):
        assert _normalize_label("L") == "L"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    def test_analyze_phonon_parser(self):
        from shalom.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "analyze", "phonon",
            "--structure", "POSCAR",
            "--supercell", "2x2x2",
            "--generate-displacements",
        ])
        assert args.command == "analyze"
        assert args.analyze_type == "phonon"
        assert args.structure == "POSCAR"
        assert args.supercell == "2x2x2"
        assert args.generate_displacements is True

    def test_analyze_phonon_force_sets_parser(self):
        from shalom.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "analyze", "phonon",
            "--structure", "POSCAR",
            "--supercell", "2x2x2",
            "--force-sets", "FORCE_SETS",
        ])
        assert args.force_sets == "FORCE_SETS"

    def test_analyze_no_type_shows_phonon(self):
        from shalom.__main__ import build_parser, cmd_analyze

        parser = build_parser()
        args = parser.parse_args(["analyze"])
        rc = cmd_analyze(args)
        assert rc == 1

    def test_parse_supercell_nxnxn(self):
        from shalom.__main__ import _parse_supercell

        sc = _parse_supercell("2x2x2")
        assert sc == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    def test_parse_supercell_json(self):
        from shalom.__main__ import _parse_supercell
        import json

        sc = _parse_supercell(json.dumps([[3, 0, 0], [0, 3, 0], [0, 0, 3]]))
        assert sc == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

    def test_parse_supercell_invalid(self):
        from shalom.__main__ import _parse_supercell

        assert _parse_supercell("bad") is None

    def test_generate_displacements_cli(self, tmp_path):
        """Test displacement generation via CLI."""
        from shalom.__main__ import build_parser
        from ase.build import bulk
        from ase.io import write as ase_write

        # Write a POSCAR
        atoms = _si_atoms()
        poscar = tmp_path / "POSCAR"
        ase_write(str(poscar), atoms, format="vasp")

        parser = build_parser()
        args = parser.parse_args([
            "analyze", "phonon",
            "--structure", str(poscar),
            "--supercell", "2x2x2",
            "--generate-displacements",
            "-o", str(tmp_path / "disps"),
        ])

        from shalom.__main__ import _cmd_analyze_phonon
        rc = _cmd_analyze_phonon(args)
        assert rc == 0

        import os
        disp_files = [f for f in os.listdir(tmp_path / "disps") if f.startswith("disp-")]
        assert len(disp_files) >= 1


# ---------------------------------------------------------------------------
# MCP tool integration
# ---------------------------------------------------------------------------


def _import_mcp_tool():
    """Import analyze_phonon_properties from mcp_server, skipping if mcp not installed."""
    try:
        from shalom.mcp_server import analyze_phonon_properties
        return analyze_phonon_properties
    except (ImportError, SystemExit):
        pytest.skip("MCP package not installed")


class TestMCPTool:
    def test_missing_input(self):
        tool = _import_mcp_tool()
        result = tool(structure_file="nonexistent.vasp")
        assert result["success"] is False

    def test_invalid_supercell(self):
        tool = _import_mcp_tool()
        result = tool(
            structure_file="nonexistent.vasp",
            supercell="bad",
        )
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_analysis_init_exports(self):
        from shalom.analysis import (
            PhononResult,
            analyze_phonon,
            analyze_phonon_from_force_constants,
            generate_phonon_displacements,
            is_phonopy_available,
        )
        assert callable(analyze_phonon)
        assert callable(analyze_phonon_from_force_constants)
        assert callable(generate_phonon_displacements)
        assert callable(is_phonopy_available)
        assert PhononResult is not None

    def test_analysis_all(self):
        import shalom.analysis
        assert "PhononResult" in shalom.analysis.__all__
        assert "analyze_phonon" in shalom.analysis.__all__
        assert "analyze_phonon_from_force_constants" in shalom.analysis.__all__
        assert "generate_phonon_displacements" in shalom.analysis.__all__
        assert "is_phonopy_available" in shalom.analysis.__all__

    def test_plotting_exports(self):
        from shalom.plotting import PhononBandPlotter, PhononDOSPlotter
        assert PhononBandPlotter is not None
        assert PhononDOSPlotter is not None
