"""Tests for shalom.analysis.symmetry module.

All tests are pure unit tests — no DFT execution required.
Uses well-known crystal structures built via ASE for validation.
"""

import pytest
from unittest.mock import patch

from shalom.analysis._base import SymmetryResult
from shalom.analysis.symmetry import (
    analyze_symmetry,
    is_spglib_available,
    _ensure_spglib_available,
    _ase_to_spglib_cell,
    _space_group_to_crystal_system,
)


# ---------------------------------------------------------------------------
# Helper — build test structures
# ---------------------------------------------------------------------------


def _si_diamond():
    """Si diamond structure (sg=227, Fd-3m, cubic)."""
    from ase.build import bulk
    return bulk("Si", "diamond", a=5.43)


def _cu_fcc():
    """Cu FCC structure (sg=225, Fm-3m, cubic)."""
    from ase.build import bulk
    return bulk("Cu", "fcc", a=3.6)


def _fe_bcc():
    """Fe BCC structure (sg=229, Im-3m, cubic)."""
    from ase.build import bulk
    return bulk("Fe", "bcc", a=2.87)


def _nacl_rocksalt():
    """NaCl rocksalt structure (sg=225, Fm-3m, cubic)."""
    from ase.build import bulk
    return bulk("NaCl", "rocksalt", a=5.64)


# ---------------------------------------------------------------------------
# SymmetryResult dataclass
# ---------------------------------------------------------------------------


class TestSymmetryResult:
    def test_default_fields(self):
        r = SymmetryResult()
        assert r.space_group_number == 0
        assert r.space_group_symbol == ""
        assert r.point_group == ""
        assert r.crystal_system == ""
        assert r.lattice_type == ""
        assert r.hall_symbol == ""
        assert r.wyckoff_letters == []
        assert r.equivalent_atoms == []
        assert r.n_operations == 0
        assert r.is_primitive is False
        assert r.raw is None
        assert r.metadata == {}

    def test_metadata_mutable(self):
        r = SymmetryResult()
        r.metadata["source"] = "test"
        assert r.metadata["source"] == "test"

    def test_metadata_independent(self):
        """Metadata dict is independent between instances (default_factory)."""
        r1 = SymmetryResult()
        r2 = SymmetryResult()
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


class TestAvailabilityGuard:
    def test_is_spglib_available(self):
        assert is_spglib_available() is True

    def test_ensure_spglib_available_passes(self):
        _ensure_spglib_available()  # should not raise

    def test_ensure_raises_when_unavailable(self):
        with patch("shalom.analysis.symmetry._SPGLIB_AVAILABLE", False):
            with pytest.raises(ImportError, match="spglib"):
                _ensure_spglib_available()


# ---------------------------------------------------------------------------
# _ase_to_spglib_cell
# ---------------------------------------------------------------------------


class TestASEToSpglib:
    def test_cell_tuple_shape(self):
        si = _si_diamond()
        cell = _ase_to_spglib_cell(si)
        assert len(cell) == 3
        lattice, positions, numbers = cell
        assert lattice.shape == (3, 3)
        assert positions.shape == (len(si), 3)
        assert numbers.shape == (len(si),)

    def test_atomic_numbers_preserved(self):
        si = _si_diamond()
        _, _, numbers = _ase_to_spglib_cell(si)
        # Si atomic number is 14
        assert all(z == 14 for z in numbers)


# ---------------------------------------------------------------------------
# analyze_symmetry — standard crystal structures
# ---------------------------------------------------------------------------


class TestAnalyzeSymmetry:
    def test_si_space_group_number(self):
        result = analyze_symmetry(_si_diamond())
        assert result.space_group_number == 227

    def test_si_space_group_symbol(self):
        result = analyze_symmetry(_si_diamond())
        assert result.space_group_symbol == "Fd-3m"

    def test_si_crystal_system(self):
        result = analyze_symmetry(_si_diamond())
        assert result.crystal_system == "cubic"

    def test_cu_space_group(self):
        result = analyze_symmetry(_cu_fcc())
        assert result.space_group_number == 225

    def test_fe_space_group(self):
        result = analyze_symmetry(_fe_bcc())
        assert result.space_group_number == 229

    def test_wyckoff_count_matches_atoms(self):
        si = _si_diamond()
        result = analyze_symmetry(si)
        assert len(result.wyckoff_letters) == len(si)

    def test_equivalent_atoms_count(self):
        si = _si_diamond()
        result = analyze_symmetry(si)
        assert len(result.equivalent_atoms) == len(si)

    def test_n_operations_positive(self):
        result = analyze_symmetry(_si_diamond())
        assert result.n_operations > 0

    def test_point_group(self):
        result = analyze_symmetry(_si_diamond())
        assert result.point_group == "m-3m"

    def test_is_primitive(self):
        """ASE bulk('Si', 'diamond') returns the primitive cell (2 atoms)."""
        result = analyze_symmetry(_si_diamond())
        assert result.is_primitive is True

    def test_hall_symbol_nonempty(self):
        result = analyze_symmetry(_si_diamond())
        assert len(result.hall_symbol) > 0

    def test_nacl_space_group(self):
        result = analyze_symmetry(_nacl_rocksalt())
        assert result.space_group_number == 225
        assert result.space_group_symbol == "Fm-3m"

    def test_lattice_type_si(self):
        result = analyze_symmetry(_si_diamond())
        assert result.lattice_type == "F"

    def test_lattice_type_fe(self):
        result = analyze_symmetry(_fe_bcc())
        assert result.lattice_type == "I"

    def test_raw_dict_has_number(self):
        result = analyze_symmetry(_si_diamond())
        assert result.raw is not None
        assert "number" in result.raw
        assert result.raw["number"] == 227


# ---------------------------------------------------------------------------
# _space_group_to_crystal_system — all 7 crystal systems
# ---------------------------------------------------------------------------


class TestCrystalSystemMapping:
    def test_triclinic(self):
        assert _space_group_to_crystal_system(1) == "triclinic"
        assert _space_group_to_crystal_system(2) == "triclinic"

    def test_monoclinic(self):
        assert _space_group_to_crystal_system(3) == "monoclinic"
        assert _space_group_to_crystal_system(15) == "monoclinic"

    def test_orthorhombic(self):
        assert _space_group_to_crystal_system(16) == "orthorhombic"
        assert _space_group_to_crystal_system(74) == "orthorhombic"

    def test_tetragonal(self):
        assert _space_group_to_crystal_system(75) == "tetragonal"
        assert _space_group_to_crystal_system(142) == "tetragonal"

    def test_trigonal(self):
        assert _space_group_to_crystal_system(143) == "trigonal"
        assert _space_group_to_crystal_system(167) == "trigonal"

    def test_hexagonal(self):
        assert _space_group_to_crystal_system(168) == "hexagonal"
        assert _space_group_to_crystal_system(194) == "hexagonal"

    def test_cubic(self):
        assert _space_group_to_crystal_system(195) == "cubic"
        assert _space_group_to_crystal_system(230) == "cubic"


# ---------------------------------------------------------------------------
# Custom symprec
# ---------------------------------------------------------------------------


class TestCustomSymprec:
    def test_loose_symprec_preserves_or_raises_symmetry(self):
        """A looser symprec should find equal or higher symmetry."""
        si = _si_diamond()
        tight = analyze_symmetry(si, symprec=1e-7)
        loose = analyze_symmetry(si, symprec=1e-2)
        # With a perfect lattice, both should find the same or loose finds
        # at least as high symmetry (>= operations).
        assert loose.n_operations >= tight.n_operations


# ---------------------------------------------------------------------------
# Package imports — Phase 2 integration
# ---------------------------------------------------------------------------


class TestPackageImports:
    @pytest.mark.skip(reason="Phase 2: symmetry not yet exported from __init__")
    def test_analysis_init_exports_symmetry(self):
        from shalom.analysis import (  # noqa: F401
            SymmetryResult,
            analyze_symmetry,
            is_spglib_available,
        )

    @pytest.mark.skip(reason="Phase 2: symmetry not yet in __all__")
    def test_analysis_all_contains_symmetry(self):
        import shalom.analysis
        assert "SymmetryResult" in shalom.analysis.__all__
        assert "analyze_symmetry" in shalom.analysis.__all__
        assert "is_spglib_available" in shalom.analysis.__all__
