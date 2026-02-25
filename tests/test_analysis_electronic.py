"""Tests for shalom.analysis.electronic module.

All tests are pure unit tests -- no DFT execution required.
Uses synthetic BandStructureData/DOSData built from numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from shalom.backends.base import BandStructureData, DOSData
from shalom.analysis._base import ElectronicResult
from shalom.analysis.electronic import (
    analyze_band_structure,
    is_electronic_available,
    _ensure_electronic_available,
    _find_band_gap,
    _estimate_effective_mass,
    _dos_at_fermi,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic band structure builders
# ---------------------------------------------------------------------------


def _make_semiconductor(
    gap: float = 1.1,
    n_kpts: int = 50,
    n_bands: int = 8,
    direct: bool = False,
) -> BandStructureData:
    """Build a synthetic semiconductor BandStructureData.

    Creates 4 valence bands (parabolic, curving down from a VBM) and 4
    conduction bands (parabolic, curving up from a CBM) with a specified gap.
    The Fermi energy is placed in the middle of the gap.

    Args:
        gap: Band gap in eV.
        n_kpts: Number of k-points.
        n_bands: Total number of bands (half valence, half conduction).
        direct: If True, VBM and CBM are at the same k-point (centre).
                If False, VBM is at centre and CBM is offset.
    """
    n_val = n_bands // 2
    n_con = n_bands - n_val

    # k-path distances: 0 to 2 (1/Angstrom)
    kdist = np.linspace(0.0, 2.0, n_kpts)
    k_mid = n_kpts // 2

    eigenvalues = np.zeros((n_kpts, n_bands))

    # Valence bands: parabolic with maximum at k_mid
    for b in range(n_val):
        offset = -0.5 * b  # lower bands shifted down
        eigenvalues[:, b] = (
            offset - 0.3 * (kdist - kdist[k_mid]) ** 2
        )

    vbm_val = 0.0  # VBM is at k_mid, band index n_val-1

    # Conduction bands: parabolic with minimum
    if direct:
        # CBM also at k_mid
        cbm_center = k_mid
    else:
        # CBM offset by ~1/4 of the path
        cbm_center = k_mid + n_kpts // 4
        if cbm_center >= n_kpts:
            cbm_center = n_kpts - 1

    cbm_val = vbm_val + gap

    for b in range(n_con):
        offset = 0.5 * b  # upper bands shifted up
        eigenvalues[:, n_val + b] = (
            cbm_val + offset + 0.3 * (kdist - kdist[cbm_center]) ** 2
        )

    fermi = vbm_val + gap / 2.0

    kcoords = np.zeros((n_kpts, 3))
    kcoords[:, 0] = np.linspace(0.0, 0.5, n_kpts)

    return BandStructureData(
        eigenvalues=eigenvalues,
        kpoint_coords=kcoords,
        kpath_distances=kdist,
        fermi_energy=fermi,
        nkpts=n_kpts,
        nbands=n_bands,
        source="test",
    )


def _make_metal(n_kpts: int = 50, n_bands: int = 8) -> BandStructureData:
    """Build a synthetic metallic BandStructureData.

    Constructs bands such that no clean gap exists: the maximum occupied
    eigenvalue (VBM) across all k exceeds the minimum unoccupied eigenvalue
    (CBM) across all k.  This is done by having two bands that both cross
    the Fermi level but at different k-points, creating band overlap.
    """
    kdist = np.linspace(0.0, 2.0, n_kpts)
    eigenvalues = np.zeros((n_kpts, n_bands))

    # Deep valence bands (always below Fermi)
    eigenvalues[:, 0] = -4.0
    eigenvalues[:, 1] = -3.0
    eigenvalues[:, 2] = -2.0

    # Band 3: rises from -1 to +1 across the path (crosses Fermi at centre)
    # At k_end: eigenvalue = +1.0 (above Fermi → unoccupied)
    # VBM contribution: ~+1.0 * (n_kpts-2)/(n_kpts-1) ≈ +0.96 (occupied)
    eigenvalues[:, 3] = np.linspace(-1.0, 1.0, n_kpts)

    # Band 4: falls from +1 to -1 across the path (crosses Fermi at centre)
    # At k_start: eigenvalue = +1.0 (above Fermi → unoccupied)
    # CBM contribution: first unoccupied value ≈ +tiny (just after crossing)
    # But at k_end: eigenvalue = -1.0 (below Fermi → occupied)
    # So band 4 is occupied at high k where band 3 is also occupied with
    # eigenvalue ~+1.  Meanwhile band 4 is unoccupied at low k at ~+1.
    # The minimum unoccupied from band 4 is ~ +tiny (just above Fermi).
    # The maximum occupied from band 3 is ~ +1 (just below the end).
    # VBM (+0.96) > CBM (+0.04) → overlap → metal.
    eigenvalues[:, 4] = np.linspace(1.0, -1.0, n_kpts)

    # High conduction bands (always above Fermi)
    eigenvalues[:, 5] = 2.0
    eigenvalues[:, 6] = 3.0
    eigenvalues[:, 7] = 4.0

    fermi = 0.0

    kcoords = np.zeros((n_kpts, 3))
    kcoords[:, 0] = np.linspace(0.0, 0.5, n_kpts)

    return BandStructureData(
        eigenvalues=eigenvalues,
        kpoint_coords=kcoords,
        kpath_distances=kdist,
        fermi_energy=fermi,
        nkpts=n_kpts,
        nbands=n_bands,
        source="test",
    )


def _make_dos(
    fermi: float = 0.0,
    is_metal: bool = True,
) -> DOSData:
    """Build a synthetic DOSData.

    Args:
        fermi: Fermi energy.
        is_metal: If True, DOS is nonzero at Fermi level.
    """
    energies = np.linspace(-10.0, 10.0, 1000)
    if is_metal:
        # Smooth DOS with nonzero value at Fermi level
        dos = 2.0 + np.sin(energies * 0.5)
    else:
        # Semiconductor-like DOS: gap around Fermi level
        dos = np.where(np.abs(energies - fermi) < 0.5, 0.0, 1.5)

    integrated = np.cumsum(dos) * (energies[1] - energies[0])

    return DOSData(
        energies=energies,
        dos=dos,
        integrated_dos=integrated,
        fermi_energy=fermi,
        source="test",
    )


# ---------------------------------------------------------------------------
# TestElectronicResult
# ---------------------------------------------------------------------------


class TestElectronicResult:
    def test_default_fields(self):
        r = ElectronicResult()
        assert r.bandgap_eV is None
        assert r.is_direct is False
        assert r.is_metal is False
        assert r.vbm_energy is None
        assert r.cbm_energy is None
        assert r.vbm_k_index is None
        assert r.cbm_k_index is None
        assert r.effective_mass_electron is None
        assert r.effective_mass_hole is None
        assert r.dos_at_fermi is None
        assert r.n_occupied_bands == 0
        assert r.raw is None
        assert r.metadata == {}

    def test_metadata_mutable(self):
        r = ElectronicResult()
        r.metadata["source"] = "test"
        assert r.metadata["source"] == "test"

    def test_fields_assignable(self):
        r = ElectronicResult(
            bandgap_eV=1.5,
            is_direct=True,
            is_metal=False,
            vbm_energy=-0.5,
            cbm_energy=1.0,
            vbm_k_index=10,
            cbm_k_index=25,
            effective_mass_electron=0.3,
            effective_mass_hole=0.5,
            dos_at_fermi=0.0,
            n_occupied_bands=4,
            raw={"test": True},
            metadata={"source": "vasp"},
        )
        assert r.bandgap_eV == 1.5
        assert r.is_direct is True
        assert r.cbm_k_index == 25
        assert r.raw == {"test": True}


# ---------------------------------------------------------------------------
# TestAvailabilityGuard
# ---------------------------------------------------------------------------


class TestAvailabilityGuard:
    def test_is_electronic_available(self):
        assert is_electronic_available() is True

    def test_ensure_electronic_available_passes(self):
        _ensure_electronic_available()  # should not raise


# ---------------------------------------------------------------------------
# TestBandGapDetection
# ---------------------------------------------------------------------------


class TestBandGapDetection:
    def test_semiconductor_gap(self):
        """Detect ~1.1 eV indirect gap in synthetic semiconductor."""
        band = _make_semiconductor(gap=1.1, direct=False)
        gap, is_direct, vbm_e, cbm_e, vbm_k, cbm_k, n_occ = _find_band_gap(
            band.eigenvalues, band.fermi_energy
        )
        assert gap is not None
        assert abs(gap - 1.1) < 0.05
        assert is_direct is False
        assert vbm_e is not None
        assert cbm_e is not None
        assert vbm_k is not None
        assert cbm_k is not None
        assert vbm_k != cbm_k

    def test_metal_no_gap(self):
        """Metal should return None gap."""
        band = _make_metal()
        gap, is_direct, vbm_e, cbm_e, vbm_k, cbm_k, n_occ = _find_band_gap(
            band.eigenvalues, band.fermi_energy
        )
        assert gap is None
        assert vbm_e is None
        assert cbm_e is None

    def test_direct_gap(self):
        """Direct gap: VBM and CBM at same k-point."""
        band = _make_semiconductor(gap=2.0, direct=True)
        gap, is_direct, vbm_e, cbm_e, vbm_k, cbm_k, n_occ = _find_band_gap(
            band.eigenvalues, band.fermi_energy
        )
        assert gap is not None
        assert abs(gap - 2.0) < 0.05
        assert is_direct is True
        assert vbm_k == cbm_k

    def test_indirect_gap(self):
        """Indirect gap: VBM and CBM at different k-points."""
        band = _make_semiconductor(gap=0.5, direct=False)
        gap, is_direct, vbm_e, cbm_e, vbm_k, cbm_k, n_occ = _find_band_gap(
            band.eigenvalues, band.fermi_energy
        )
        assert gap is not None
        assert is_direct is False
        assert vbm_k != cbm_k

    def test_n_occupied_bands_positive(self):
        """Semiconductor should have positive n_occupied_bands."""
        band = _make_semiconductor(gap=1.0)
        gap, is_direct, vbm_e, cbm_e, vbm_k, cbm_k, n_occ = _find_band_gap(
            band.eigenvalues, band.fermi_energy
        )
        assert n_occ > 0


# ---------------------------------------------------------------------------
# TestEffectiveMass
# ---------------------------------------------------------------------------


class TestEffectiveMass:
    def test_parabolic_band_known_mass(self):
        """Parabolic band E = hbar^2 k^2 / (2 * m*) should recover m*."""
        # Create a simple parabolic band with m* = 1.0 m_e
        from shalom.analysis.electronic import HBAR_EV_S, M_E_KG, ANG_TO_M

        n_kpts = 101
        # k in 1/Angstrom centred on 0
        k_dist = np.linspace(-0.5, 0.5, n_kpts)
        k_center = n_kpts // 2

        # E(k) = hbar^2 * k^2 / (2 * m_e) in eV
        # k in 1/Angstrom → 1/m: k_m = k * 1e10
        # hbar in J*s: hbar_js = HBAR_EV_S * 1.602e-19
        hbar_js = HBAR_EV_S * 1.602176634e-19
        target_mass = 1.0  # m_e
        m_kg = target_mass * M_E_KG
        # E in eV = hbar^2 * (k*1e10)^2 / (2 * m_kg) / 1.602e-19
        e_eV = (
            hbar_js**2
            * (k_dist * 1.0 / ANG_TO_M) ** 2
            / (2.0 * m_kg)
            / 1.602176634e-19
        )

        eigenvalues = e_eV.reshape(n_kpts, 1)

        m_star = _estimate_effective_mass(
            eigenvalues, k_dist, band_index=0, k_center=k_center, n_points=10
        )
        assert m_star is not None
        # Should recover ~1.0 m_e within 10%
        assert abs(m_star - 1.0) < 0.1, f"Expected ~1.0 m_e, got {m_star}"

    def test_flat_band_returns_none(self):
        """Flat band (zero curvature) should return None."""
        n_kpts = 50
        k_dist = np.linspace(0.0, 2.0, n_kpts)
        eigenvalues = np.ones((n_kpts, 1)) * 5.0  # perfectly flat

        m_star = _estimate_effective_mass(
            eigenvalues, k_dist, band_index=0, k_center=25, n_points=10
        )
        assert m_star is None

    def test_insufficient_points(self):
        """Only 2 k-points (< 3 required) should return None."""
        k_dist = np.array([0.0, 1.0])
        eigenvalues = np.array([[0.0], [1.0]])

        m_star = _estimate_effective_mass(
            eigenvalues, k_dist, band_index=0, k_center=0, n_points=0
        )
        assert m_star is None


# ---------------------------------------------------------------------------
# TestDOSAtFermi
# ---------------------------------------------------------------------------


class TestDOSAtFermi:
    def test_metal_nonzero_dos(self):
        """Metal DOS at Fermi should be nonzero."""
        dos = _make_dos(fermi=0.0, is_metal=True)
        val = _dos_at_fermi(dos)
        assert val is not None
        assert val > 0.0

    def test_semiconductor_near_zero_dos(self):
        """Semiconductor DOS at Fermi (in gap) should be near zero."""
        dos = _make_dos(fermi=0.0, is_metal=False)
        val = _dos_at_fermi(dos)
        assert val is not None
        assert abs(val) < 0.01

    def test_no_dos_data(self):
        """When dos_data is None, analyze_band_structure returns None."""
        band = _make_semiconductor(gap=1.0)
        result = analyze_band_structure(band, dos_data=None)
        assert result.dos_at_fermi is None


# ---------------------------------------------------------------------------
# TestAnalyzeBandStructure
# ---------------------------------------------------------------------------


class TestAnalyzeBandStructure:
    def test_full_semiconductor(self):
        """Full analysis of a synthetic semiconductor."""
        band = _make_semiconductor(gap=1.1, direct=False)
        result = analyze_band_structure(band)

        assert isinstance(result, ElectronicResult)
        assert result.is_metal is False
        assert result.bandgap_eV is not None
        assert abs(result.bandgap_eV - 1.1) < 0.05
        assert result.is_direct is False
        assert result.vbm_energy is not None
        assert result.cbm_energy is not None
        assert result.vbm_k_index is not None
        assert result.cbm_k_index is not None
        assert result.n_occupied_bands > 0
        assert "fermi_energy" in result.metadata
        assert result.metadata["source"] == "test"

    def test_full_metal(self):
        """Full analysis of a synthetic metal."""
        band = _make_metal()
        result = analyze_band_structure(band)

        assert isinstance(result, ElectronicResult)
        assert result.is_metal is True
        assert result.bandgap_eV is None
        assert result.vbm_energy is None
        assert result.cbm_energy is None
        assert result.effective_mass_electron is None
        assert result.effective_mass_hole is None

    def test_with_dos(self):
        """Analysis with DOS data should populate dos_at_fermi."""
        band = _make_metal()
        dos = _make_dos(fermi=band.fermi_energy, is_metal=True)
        result = analyze_band_structure(band, dos_data=dos)

        assert result.dos_at_fermi is not None
        assert result.dos_at_fermi > 0.0

    def test_spin_polarized(self):
        """Spin-polarised analysis picks the narrower gap."""
        n_kpts = 50
        n_bands = 8
        kdist = np.linspace(0.0, 2.0, n_kpts)
        kcoords = np.zeros((n_kpts, 3))
        kcoords[:, 0] = np.linspace(0.0, 0.5, n_kpts)

        # Spin-up: 2.0 eV gap
        eigs_up = np.zeros((n_kpts, n_bands))
        for b in range(4):
            eigs_up[:, b] = -1.0 - 0.3 * b
        for b in range(4):
            eigs_up[:, 4 + b] = 1.0 + 0.3 * b

        # Spin-down: 1.0 eV gap (narrower)
        eigs_dn = np.zeros((n_kpts, n_bands))
        for b in range(4):
            eigs_dn[:, b] = -0.5 - 0.3 * b
        for b in range(4):
            eigs_dn[:, 4 + b] = 0.5 + 0.3 * b

        fermi = 0.0

        # Combined eigenvalues (not used for spin-polarized, but required field)
        eigs = eigs_up.copy()

        band = BandStructureData(
            eigenvalues=eigs,
            kpoint_coords=kcoords,
            kpath_distances=kdist,
            fermi_energy=fermi,
            spin_up=eigs_up,
            spin_down=eigs_dn,
            is_spin_polarized=True,
            nkpts=n_kpts,
            nbands=n_bands,
            source="test",
        )

        result = analyze_band_structure(band)

        assert isinstance(result, ElectronicResult)
        assert result.is_metal is False
        assert result.bandgap_eV is not None
        # Should pick the narrower gap (~1.0 eV from spin-down)
        assert abs(result.bandgap_eV - 1.0) < 0.05
        assert result.metadata.get("spin_channel") == "spin_down"

    def test_empty_eigenvalues_raises(self):
        """Empty eigenvalues should raise ValueError."""
        band = BandStructureData(
            eigenvalues=np.array([]).reshape(0, 0),
            kpoint_coords=np.array([]).reshape(0, 3),
            kpath_distances=np.array([]),
            fermi_energy=0.0,
        )
        with pytest.raises(ValueError, match="empty"):
            analyze_band_structure(band)

    def test_none_eigenvalues_raises(self):
        """None eigenvalues should raise ValueError."""
        band = BandStructureData(
            eigenvalues=None,
            kpoint_coords=np.zeros((10, 3)),
            kpath_distances=np.zeros(10),
            fermi_energy=0.0,
        )
        with pytest.raises(ValueError, match="None"):
            analyze_band_structure(band)


# ---------------------------------------------------------------------------
# TestPackageImports (Phase 2 integration pending)
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_analysis_exports(self):
        from shalom.analysis import (
            ElectronicResult,
            analyze_band_structure,
            is_electronic_available,
        )
        assert callable(analyze_band_structure)
        assert callable(is_electronic_available)
        assert ElectronicResult is not None

    def test_analysis_all(self):
        import shalom.analysis
        assert "ElectronicResult" in shalom.analysis.__all__
        assert "analyze_band_structure" in shalom.analysis.__all__
        assert "is_electronic_available" in shalom.analysis.__all__
