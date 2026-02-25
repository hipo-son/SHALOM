"""Tests for shalom.analysis.xrd module.

All tests are pure unit tests -- no DFT execution required.
Uses real Si and Cu structures from ASE for validation.
"""

import numpy as np
import pytest
from unittest.mock import patch

from shalom.analysis._base import XRDResult
from shalom.analysis.xrd import (
    is_xrd_available,
    _ensure_xrd_available,
    _ase_to_pymatgen_structure,
    calculate_xrd,
)


def _mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _si_atoms():
    """Return Si diamond primitive cell."""
    from ase.build import bulk

    return bulk("Si", "diamond", a=5.43)


def _cu_atoms():
    """Return Cu FCC primitive cell."""
    from ase.build import bulk

    return bulk("Cu", "fcc", a=3.61)


# ---------------------------------------------------------------------------
# XRDResult dataclass
# ---------------------------------------------------------------------------


class TestXRDResult:
    def test_default_fields(self):
        r = XRDResult()
        assert r.two_theta is None
        assert r.intensities is None
        assert r.hkl_indices == []
        assert r.d_spacings is None
        assert r.wavelength == "CuKa"
        assert r.wavelength_angstrom is None
        assert r.n_peaks == 0
        assert r.raw is None
        assert r.metadata == {}

    def test_field_assignment(self):
        theta = np.array([28.4, 47.3])
        r = XRDResult(two_theta=theta, n_peaks=2, wavelength="MoKa")
        assert r.two_theta is theta
        assert r.n_peaks == 2
        assert r.wavelength == "MoKa"

    def test_metadata_mutable(self):
        r = XRDResult()
        r.metadata["source"] = "test"
        assert r.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


class TestAvailabilityGuard:
    def test_is_xrd_available(self):
        assert is_xrd_available() is True

    def test_ensure_xrd_available_passes(self):
        _ensure_xrd_available()  # should not raise

    def test_ensure_raises_when_unavailable(self):
        with patch("shalom.analysis.xrd._XRD_AVAILABLE", False):
            with pytest.raises(ImportError, match="pymatgen"):
                _ensure_xrd_available()


# ---------------------------------------------------------------------------
# ASE to pymatgen conversion
# ---------------------------------------------------------------------------


class TestASEToPymatgen:
    def test_si_conversion_symbols(self):
        atoms = _si_atoms()
        structure = _ase_to_pymatgen_structure(atoms)
        ase_symbols = sorted(atoms.get_chemical_symbols())
        pmg_symbols = sorted([str(sp) for sp in structure.species])
        assert ase_symbols == pmg_symbols

    def test_si_conversion_cell(self):
        atoms = _si_atoms()
        structure = _ase_to_pymatgen_structure(atoms)
        np.testing.assert_allclose(
            structure.lattice.matrix,
            atoms.cell.array,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# calculate_xrd
# ---------------------------------------------------------------------------


class TestCalculateXRD:
    def test_si_peaks_exist(self):
        """Si diamond should produce XRD peaks."""
        atoms = _si_atoms()
        result = calculate_xrd(atoms)
        assert result.n_peaks > 0
        assert len(result.two_theta) > 0

    def test_intensities_normalized(self):
        """Peak intensities should be normalised to 0-100 range."""
        atoms = _si_atoms()
        result = calculate_xrd(atoms)
        assert np.max(result.intensities) == pytest.approx(100.0, abs=0.1)
        assert np.all(result.intensities >= 0)
        assert np.all(result.intensities <= 100.0 + 0.1)

    def test_d_spacings_positive(self):
        """All d-spacings should be positive."""
        atoms = _si_atoms()
        result = calculate_xrd(atoms)
        assert np.all(result.d_spacings > 0)

    def test_hkl_count_matches(self):
        """Number of hkl indices should match number of peaks."""
        atoms = _si_atoms()
        result = calculate_xrd(atoms)
        assert len(result.hkl_indices) == result.n_peaks

    def test_n_peaks_field(self):
        """n_peaks should match length of two_theta array."""
        atoms = _si_atoms()
        result = calculate_xrd(atoms)
        assert result.n_peaks == len(result.two_theta)
        assert result.n_peaks == len(result.intensities)
        assert result.n_peaks == len(result.d_spacings)

    def test_wavelength_stored(self):
        """Wavelength label and angstrom value should be stored."""
        atoms = _si_atoms()
        result = calculate_xrd(atoms)
        assert result.wavelength == "CuKa"
        assert result.wavelength_angstrom is not None
        assert abs(result.wavelength_angstrom - 1.5406) < 0.01

    def test_different_wavelength(self):
        """Using MoKa should produce different peak positions."""
        atoms = _si_atoms()
        result_cu = calculate_xrd(atoms, wavelength="CuKa")
        result_mo = calculate_xrd(atoms, wavelength="MoKa")
        assert result_mo.wavelength == "MoKa"
        assert result_mo.wavelength_angstrom is not None
        assert abs(result_mo.wavelength_angstrom - 0.7107) < 0.01
        # MoKa has shorter wavelength -> peaks at smaller 2theta
        if result_cu.n_peaks > 0 and result_mo.n_peaks > 0:
            assert np.mean(result_mo.two_theta) < np.mean(result_cu.two_theta)

    def test_theta_range_filter(self):
        """Restricting 2theta range should filter peaks."""
        atoms = _si_atoms()
        result_full = calculate_xrd(atoms, two_theta_range=(0, 90))
        result_narrow = calculate_xrd(atoms, two_theta_range=(20, 40))
        # Narrow range should have fewer or equal peaks
        assert result_narrow.n_peaks <= result_full.n_peaks
        if result_narrow.n_peaks > 0:
            assert np.all(result_narrow.two_theta >= 20.0 - 0.1)
            assert np.all(result_narrow.two_theta <= 40.0 + 0.1)


# ---------------------------------------------------------------------------
# CLI integration (Phase 2)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_analyze_xrd_parser(self):
        from shalom.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["analyze", "xrd", "--structure", "POSCAR"])
        assert args.command == "analyze"
        assert args.analyze_type == "xrd"


# ---------------------------------------------------------------------------
# MCP tool integration (Phase 2)
# ---------------------------------------------------------------------------


class TestMCPTool:
    @pytest.mark.skipif(
        not _mcp_available(), reason="mcp package not installed"
    )
    def test_missing_file(self):
        from shalom.mcp_server import analyze_xrd_pattern

        result = analyze_xrd_pattern(structure_file="nonexistent.vasp")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Package imports (Phase 2)
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_analysis_init_exports(self):
        from shalom.analysis import (
            XRDResult,
            calculate_xrd,
            is_xrd_available,
        )
        assert callable(calculate_xrd)
        assert callable(is_xrd_available)
        assert XRDResult is not None

    def test_analysis_all(self):
        import shalom.analysis
        assert "XRDResult" in shalom.analysis.__all__
        assert "calculate_xrd" in shalom.analysis.__all__
        assert "is_xrd_available" in shalom.analysis.__all__
