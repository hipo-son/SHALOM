"""Tests for shalom.analysis magnetic module.

All tests are pure unit tests -- no DFT execution required.
Uses mock QE pw.out content for site magnetization and Lowdin charge parsing.
"""

import pytest

from shalom.backends.base import DFTResult
from shalom.analysis._base import MagneticResult
from shalom.analysis.magnetic import (
    is_magnetic_available,
    _ensure_magnetic_available,
    extract_site_magnetization,
    extract_lowdin_charges,
    analyze_magnetism,
)


# ---------------------------------------------------------------------------
# Mock QE output content
# ---------------------------------------------------------------------------

MOCK_PW_OUT_MAGNETIC = """
     Program PWSCF v.7.2 starts on ...
     ...
     total magnetization       =     4.0000 Bohr mag/cell
     ...
     atom:    1    charge:   6.1234    magn:   2.1000    constr:   0.0000
     atom:    2    charge:   6.0987    magn:   1.9000    constr:   0.0000
     atom:    3    charge:   3.5678    magn:   0.0100    constr:   0.0000
     atom:    4    charge:   3.5432    magn:  -0.0100    constr:   0.0000
     ...
     JOB DONE.
"""

MOCK_PW_OUT_NONMAGNETIC = """
     Program PWSCF v.7.2 starts on ...
     ...
     convergence has been achieved in  10 iterations
     ...
     JOB DONE.
"""

MOCK_PW_OUT_MULTI_STEP = """
     Program PWSCF v.7.2 starts on ...

     total magnetization       =     3.5000 Bohr mag/cell

     atom:    1    charge:   6.0000    magn:   1.8000    constr:   0.0000
     atom:    2    charge:   6.0000    magn:   1.7000    constr:   0.0000
     atom:    3    charge:   3.5000    magn:   0.0200    constr:   0.0000
     atom:    4    charge:   3.5000    magn:  -0.0200    constr:   0.0000

     convergence has been achieved in   8 iterations

     total magnetization       =     4.0000 Bohr mag/cell

     atom:    1    charge:   6.1234    magn:   2.1000    constr:   0.0000
     atom:    2    charge:   6.0987    magn:   1.9000    constr:   0.0000
     atom:    3    charge:   3.5678    magn:   0.0100    constr:   0.0000
     atom:    4    charge:   3.5432    magn:  -0.0100    constr:   0.0000

     JOB DONE.
"""

MOCK_LOWDIN_OUTPUT = """
     Lowdin Charges:

     Atom #   1: total charge =   6.1234, s =  0.3500, p =  0.0100, d =  5.7634
     Atom #   2: total charge =   6.0987, s =  0.3400, p =  0.0200, d =  5.7387
     Atom #   3: total charge =   3.5678, s =  0.9800, p =  2.5878
     Atom #   4: total charge =   3.5432, s =  0.9700, p =  2.5732
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mock(tmp_path, content, name="pw.out"):
    """Write mock content to a temp file and return its path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _fe2o3_atoms():
    """Return a 4-atom Fe2O2 stand-in (2 Fe + 2 O)."""
    from ase import Atoms

    return Atoms(
        "Fe2O2",
        positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
        cell=[8, 8, 8],
        pbc=True,
    )


def _si_atoms():
    """Return a simple Si bulk structure."""
    from ase.build import bulk

    return bulk("Si", "diamond", a=5.43)


# ---------------------------------------------------------------------------
# MagneticResult dataclass
# ---------------------------------------------------------------------------


class TestMagneticResult:
    def test_default_fields(self):
        r = MagneticResult()
        assert r.total_magnetization is None
        assert r.is_magnetic is False
        assert r.is_spin_polarized is False
        assert r.site_magnetizations is None
        assert r.site_charges is None
        assert r.lowdin_charges is None
        assert r.magnetic_elements == []
        assert r.dominant_moment_element is None
        assert r.raw is None
        assert r.metadata == {}

    def test_metadata_mutable(self):
        r = MagneticResult()
        r.metadata["source"] = "test"
        assert r.metadata["source"] == "test"

    def test_is_magnetic_default_false(self):
        r = MagneticResult()
        assert r.is_magnetic is False


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


class TestAvailabilityGuard:
    def test_is_magnetic_available_true(self):
        assert is_magnetic_available() is True

    def test_ensure_magnetic_available_passes(self):
        _ensure_magnetic_available()  # should not raise


# ---------------------------------------------------------------------------
# extract_site_magnetization
# ---------------------------------------------------------------------------


class TestExtractSiteMagnetization:
    def test_parses_magnetic_pw_out(self, tmp_path):
        path = _write_mock(tmp_path, MOCK_PW_OUT_MAGNETIC)
        mags = extract_site_magnetization(path)
        assert mags is not None
        assert len(mags) == 4
        assert abs(mags[0] - 2.1) < 1e-6
        assert abs(mags[1] - 1.9) < 1e-6
        assert abs(mags[2] - 0.01) < 1e-6
        assert abs(mags[3] - (-0.01)) < 1e-6

    def test_returns_none_for_nonmagnetic(self, tmp_path):
        path = _write_mock(tmp_path, MOCK_PW_OUT_NONMAGNETIC)
        mags = extract_site_magnetization(path)
        assert mags is None

    def test_returns_none_for_missing_file(self):
        mags = extract_site_magnetization("/nonexistent/path/pw.out")
        assert mags is None

    def test_multiple_ionic_steps_takes_last(self, tmp_path):
        path = _write_mock(tmp_path, MOCK_PW_OUT_MULTI_STEP)
        mags = extract_site_magnetization(path)
        assert mags is not None
        assert len(mags) == 4
        # Last block values (2.1, 1.9, 0.01, -0.01), not first (1.8, 1.7, ...)
        assert abs(mags[0] - 2.1) < 1e-6
        assert abs(mags[1] - 1.9) < 1e-6


# ---------------------------------------------------------------------------
# extract_lowdin_charges
# ---------------------------------------------------------------------------


class TestExtractLowdinCharges:
    def test_parses_lowdin_section(self, tmp_path):
        path = _write_mock(tmp_path, MOCK_LOWDIN_OUTPUT)
        result = extract_lowdin_charges(path)
        assert result is not None
        assert "total_charges" in result
        assert len(result["total_charges"]) == 4
        assert abs(result["total_charges"][0] - 6.1234) < 1e-4

    def test_returns_none_if_no_section(self, tmp_path):
        path = _write_mock(tmp_path, MOCK_PW_OUT_NONMAGNETIC)
        result = extract_lowdin_charges(path)
        assert result is None

    def test_spd_charges_populated(self, tmp_path):
        path = _write_mock(tmp_path, MOCK_LOWDIN_OUTPUT)
        result = extract_lowdin_charges(path)
        assert result is not None
        spd = result["spd_charges"]
        assert len(spd) == 4

        # Atom 1: s=0.35, p=0.01, d=5.7634
        assert abs(spd[0]["s"] - 0.35) < 1e-4
        assert abs(spd[0]["p"] - 0.01) < 1e-4
        assert abs(spd[0]["d"] - 5.7634) < 1e-4

        # Atom 3: s=0.98, p=2.5878, no d channel
        assert abs(spd[2]["s"] - 0.98) < 1e-4
        assert abs(spd[2]["p"] - 2.5878) < 1e-4
        assert "d" not in spd[2]


# ---------------------------------------------------------------------------
# analyze_magnetism
# ---------------------------------------------------------------------------


class TestAnalyzeMagnetism:
    def test_fe2o3_like_magnetic(self, tmp_path):
        """Fe2O2 with site magnetization -- should be magnetic."""
        path = _write_mock(tmp_path, MOCK_PW_OUT_MAGNETIC)
        atoms = _fe2o3_atoms()
        dft = DFTResult(magnetization=4.0, is_converged=True)

        result = analyze_magnetism(dft, atoms, pw_out_path=path)

        assert result.is_magnetic is True
        assert result.is_spin_polarized is True
        assert result.total_magnetization == 4.0
        assert result.site_magnetizations is not None
        assert len(result.site_magnetizations) == 4

    def test_si_nonmagnetic(self):
        """Si without magnetization -- nonmagnetic."""
        atoms = _si_atoms()
        dft = DFTResult(magnetization=None, is_converged=True)

        result = analyze_magnetism(dft, atoms)

        assert result.is_magnetic is False
        assert result.is_spin_polarized is False
        assert result.total_magnetization is None
        assert result.site_magnetizations is None

    def test_with_pw_out_and_site_data(self, tmp_path):
        """Verify site magnetizations are populated from pw.out."""
        path = _write_mock(tmp_path, MOCK_PW_OUT_MAGNETIC)
        atoms = _fe2o3_atoms()
        dft = DFTResult(magnetization=4.0, is_converged=True)

        result = analyze_magnetism(dft, atoms, pw_out_path=path)

        assert result.site_magnetizations is not None
        assert abs(result.site_magnetizations[0] - 2.1) < 1e-6

    def test_without_pw_out(self):
        """Without pw.out path, site data should be None."""
        atoms = _fe2o3_atoms()
        dft = DFTResult(magnetization=4.0, is_converged=True)

        result = analyze_magnetism(dft, atoms)

        assert result.site_magnetizations is None
        assert result.lowdin_charges is None

    def test_magnetic_elements_detected(self, tmp_path):
        """Fe should be detected as magnetic, O should not."""
        path = _write_mock(tmp_path, MOCK_PW_OUT_MAGNETIC)
        atoms = _fe2o3_atoms()
        dft = DFTResult(magnetization=4.0, is_converged=True)

        result = analyze_magnetism(dft, atoms, pw_out_path=path)

        assert "Fe" in result.magnetic_elements
        # O has |magn| = 0.01 which is below 0.05 threshold
        assert "O" not in result.magnetic_elements

    def test_dominant_moment_element(self, tmp_path):
        """Fe should be the dominant moment element."""
        path = _write_mock(tmp_path, MOCK_PW_OUT_MAGNETIC)
        atoms = _fe2o3_atoms()
        dft = DFTResult(magnetization=4.0, is_converged=True)

        result = analyze_magnetism(dft, atoms, pw_out_path=path)

        assert result.dominant_moment_element == "Fe"

    def test_spin_polarized_flag(self):
        """spin_polarized is True when magnetization is not None."""
        atoms = _fe2o3_atoms()
        dft = DFTResult(magnetization=0.0, is_converged=True)

        result = analyze_magnetism(dft, atoms)

        # magnetization=0.0 means nspin=2 was used but result is 0
        assert result.is_spin_polarized is True
        assert result.is_magnetic is False  # |0.0| < 0.01

    def test_threshold_boundary(self):
        """Magnetization exactly at 0.01 is not magnetic (> not >=)."""
        atoms = _fe2o3_atoms()
        dft = DFTResult(magnetization=0.01, is_converged=True)

        result = analyze_magnetism(dft, atoms)

        assert result.is_spin_polarized is True
        assert result.is_magnetic is False  # abs(0.01) > 0.01 is False


# ---------------------------------------------------------------------------
# Package imports (Phase 2 — marked skip)
# ---------------------------------------------------------------------------


class TestPackageImports:
    @pytest.mark.skip(reason="Phase 2: MagneticResult not yet exported from __init__")
    def test_analysis_init_exports_magnetic_result(self):
        from shalom.analysis import MagneticResult as MR  # noqa: F811

        assert MR is not None

    @pytest.mark.skip(reason="Phase 2: analyze_magnetism not yet exported from __init__")
    def test_analysis_all_includes_magnetic(self):
        import shalom.analysis

        assert "MagneticResult" in shalom.analysis.__all__
        assert "analyze_magnetism" in shalom.analysis.__all__
