"""Tests for shalom.backends._physics shared module."""

from ase.build import bulk, fcc111

from shalom.backends._physics import (
    AccuracyLevel,
    MAGNETIC_ELEMENTS,
    DEFAULT_MAGMOM,
    HUBBARD_U_VALUES,
    ANION_ELEMENTS,
    _is_pure_metal,
    _get_lmaxmix,
    detect_2d,
    compute_kpoints_grid,
)


class TestAccuracyLevel:
    def test_standard(self):
        assert AccuracyLevel("standard") == AccuracyLevel.STANDARD

    def test_precise(self):
        assert AccuracyLevel("precise") == AccuracyLevel.PRECISE


class TestMagneticElements:
    def test_fe_is_magnetic(self):
        assert "Fe" in MAGNETIC_ELEMENTS

    def test_si_not_magnetic(self):
        assert "Si" not in MAGNETIC_ELEMENTS

    def test_default_magmom_for_fe(self):
        assert "Fe" in DEFAULT_MAGMOM
        assert DEFAULT_MAGMOM["Fe"] > 0


class TestHubbardU:
    def test_fe_has_u_value(self):
        assert "Fe" in HUBBARD_U_VALUES
        l_val, u_val, j_val = HUBBARD_U_VALUES["Fe"]
        assert l_val == 2  # d-orbital
        assert u_val > 0

    def test_anion_elements(self):
        assert "O" in ANION_ELEMENTS
        assert "S" in ANION_ELEMENTS


class TestIsPureMetal:
    def test_cu_is_metal(self):
        assert _is_pure_metal(["Cu"])

    def test_fe_is_metal(self):
        assert _is_pure_metal(["Fe"])

    def test_feo_not_pure_metal(self):
        assert not _is_pure_metal(["Fe", "O"])


class TestGetLmaxmix:
    def test_d_block(self):
        assert _get_lmaxmix([26]) == 4  # Fe: Z=26

    def test_f_block(self):
        assert _get_lmaxmix([58]) == 6  # Ce: Z=58

    def test_sp_block(self):
        assert _get_lmaxmix([14]) is None  # Si: Z=14, no d/f


class TestDetect2D:
    def test_bulk_not_2d(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        assert detect_2d(atoms) is False

    def test_slab_is_2d(self):
        atoms = fcc111("Cu", size=(1, 1, 3), vacuum=15.0)
        assert detect_2d(atoms) is True


class TestComputeKpointsGrid:
    def test_bulk_kpoints(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        grid = compute_kpoints_grid(atoms, kpr=30.0)
        assert len(grid) == 3
        assert all(k >= 1 for k in grid)

    def test_2d_kz_is_1(self):
        atoms = fcc111("Cu", size=(1, 1, 3), vacuum=15.0)
        grid = compute_kpoints_grid(atoms, kpr=30.0, is_2d=True)
        assert grid[2] == 1


class TestBackwardCompat:
    """Verify backward-compatible imports from vasp_config still work."""

    def test_import_accuracy_level_from_vasp_config(self):
        from shalom.backends.vasp_config import AccuracyLevel as AL
        assert AL.STANDARD == AccuracyLevel.STANDARD

    def test_import_magnetic_elements_from_vasp_config(self):
        from shalom.backends.vasp_config import MAGNETIC_ELEMENTS as ME
        assert "Fe" in ME

    def test_import_detect_2d_from_vasp_config(self):
        from shalom.backends.vasp_config import detect_2d as d2d
        atoms = bulk("Cu", "fcc", a=3.6)
        assert d2d(atoms) is False


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_is_pure_metal_empty_list(self):
        """Empty element list → vacuous truth (all-of-nothing)."""
        assert _is_pure_metal([]) is True

    def test_detect_2d_empty_atoms(self):
        """Empty Atoms → not 2D."""
        from ase import Atoms
        atoms = Atoms()
        assert detect_2d(atoms) is False
