"""Electronic structure analysis from band structure and DOS data.

Analyzes ``BandStructureData`` and ``DOSData`` (from ``shalom.backends.base``)
to extract band gaps, effective masses, metallicity, and related electronic
properties.  Only requires numpy — no optional dependencies.

Example::

    from shalom.backends.base import BandStructureData
    from shalom.analysis.electronic import analyze_band_structure

    band_data = BandStructureData(
        eigenvalues=eigenvalues_array,
        kpoint_coords=kcoords,
        kpath_distances=kdists,
        fermi_energy=5.0,
    )
    result = analyze_band_structure(band_data)
    print(f"Band gap: {result.bandgap_eV:.3f} eV, direct={result.is_direct}")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from shalom.analysis._base import ElectronicResult
    from shalom.backends.base import BandStructureData, DOSData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

HBAR_EV_S = 6.582119569e-16  # hbar in eV*s
M_E_KG = 9.1093837015e-31  # electron mass in kg
ANG_TO_M = 1e-10  # Angstrom to meter
_EV_TO_J = 1.602176634e-19  # eV to Joules
_HBAR_JS = HBAR_EV_S * _EV_TO_J  # hbar in J*s


# ---------------------------------------------------------------------------
# Availability guard — follows elastic.py / phonon.py pattern
# ---------------------------------------------------------------------------


def is_electronic_available() -> bool:
    """Always True -- only requires numpy."""
    return True


def _ensure_electronic_available() -> None:
    """No optional dependencies needed for electronic analysis."""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_band_gap(
    eigenvalues: Any,
    fermi_energy: float,
) -> Tuple[
    Optional[float],  # gap
    bool,  # is_direct
    Optional[float],  # vbm_energy
    Optional[float],  # cbm_energy
    Optional[int],  # vbm_k_index
    Optional[int],  # cbm_k_index
    int,  # n_occupied_bands (at VBM k-point)
]:
    """Find the band gap from eigenvalues and Fermi energy.

    For each k-point, determine the highest occupied eigenvalue (at or below
    the Fermi level) and the lowest unoccupied eigenvalue (above the Fermi
    level).  The VBM is the global maximum of occupied states, and the CBM is
    the global minimum of unoccupied states.

    Metal detection uses two criteria:
    1. Standard: VBM >= CBM (bands overlap in energy).
    2. Band-crossing: any single band has eigenvalues both above and below
       the Fermi level at different k-points, indicating partial occupation.

    Args:
        eigenvalues: Shape ``(nkpts, nbands)`` array of band energies in eV.
        fermi_energy: Fermi energy in eV.

    Returns:
        Tuple of ``(gap, is_direct, vbm_energy, cbm_energy, vbm_k_index,
        cbm_k_index, n_occupied_bands)``.  Returns ``(None, False, None, None,
        None, None, 0)`` for metals (no gap).
    """
    eigs = np.asarray(eigenvalues, dtype=float)
    nkpts, nbands = eigs.shape

    # Small tolerance for Fermi level comparison to handle numerical noise
    tol = 1e-6

    # ---- Band-crossing metal detection ----
    # A band that has eigenvalues both above and below the Fermi level
    # at different k-points is partially occupied → metal.
    for b in range(nbands):
        band_vals = eigs[:, b]
        has_below = bool(np.any(band_vals < fermi_energy - tol))
        has_above = bool(np.any(band_vals > fermi_energy + tol))
        if has_below and has_above:
            return None, False, None, None, None, None, 0

    # ---- Standard VBM/CBM analysis ----
    vbm = -np.inf
    cbm = np.inf
    vbm_k = None
    cbm_k = None
    n_occ_at_vbm = 0

    has_occupied = False
    has_unoccupied = False

    for k_idx in range(nkpts):
        occupied_mask = eigs[k_idx] <= fermi_energy + tol
        unoccupied_mask = eigs[k_idx] > fermi_energy + tol

        if np.any(occupied_mask):
            has_occupied = True
            max_occ = float(np.max(eigs[k_idx][occupied_mask]))
            if max_occ > vbm:
                vbm = max_occ
                vbm_k = k_idx
                n_occ_at_vbm = int(np.sum(occupied_mask))

        if np.any(unoccupied_mask):
            has_unoccupied = True
            min_unocc = float(np.min(eigs[k_idx][unoccupied_mask]))
            if min_unocc < cbm:
                cbm = min_unocc
                cbm_k = k_idx

    # Metal: no gap exists
    if not has_occupied or not has_unoccupied:
        return None, False, None, None, None, None, 0

    gap = cbm - vbm
    if gap <= 0:
        # Overlapping bands → metal
        return None, False, None, None, None, None, 0

    is_direct = vbm_k == cbm_k

    return gap, is_direct, vbm, cbm, vbm_k, cbm_k, n_occ_at_vbm


def _estimate_effective_mass(
    eigenvalues: Any,
    kpath_distances: Any,
    band_index: int,
    k_center: int,
    n_points: int,
) -> Optional[float]:
    """Estimate effective mass from parabolic fit near a band extremum.

    Fits ``E(k) = a0 + a1*k + a2*k^2`` to ``n_points`` around ``k_center``
    on the specified band, then computes ``m* = hbar^2 / (2 * |a2|)`` in
    units of the free electron mass.

    Args:
        eigenvalues: Shape ``(nkpts, nbands)`` array of band energies in eV.
        kpath_distances: Shape ``(nkpts,)`` cumulative k-path distances
            in 1/Angstrom.
        band_index: Band index to fit.
        k_center: Central k-point index for the fit window.
        n_points: Number of k-points to use on each side of k_center.

    Returns:
        Effective mass in units of m_e, or None if the fit fails
        (insufficient points, zero curvature, or numerical issues).
    """
    eigs = np.asarray(eigenvalues, dtype=float)
    kdist = np.asarray(kpath_distances, dtype=float)
    nkpts = eigs.shape[0]

    # Determine fit window
    k_start = max(0, k_center - n_points)
    k_end = min(nkpts, k_center + n_points + 1)
    indices = np.arange(k_start, k_end)

    if len(indices) < 3:
        logger.debug(
            "Effective mass fit: insufficient points (%d < 3) near k=%d",
            len(indices),
            k_center,
        )
        return None

    k_vals = kdist[indices]
    e_vals = eigs[indices, band_index]

    # Convert k from 1/Angstrom to 1/m for SI calculation
    k_si = k_vals * (1.0 / ANG_TO_M)  # 1/Angstrom → 1/m
    # Convert E from eV to Joules
    e_joules = e_vals * _EV_TO_J

    try:
        coeffs = np.polyfit(k_si, e_joules, 2)
    except (np.linalg.LinAlgError, ValueError):
        logger.debug("Effective mass fit: polyfit failed near k=%d", k_center)
        return None

    a2 = coeffs[0]  # quadratic coefficient (J * m^2)

    if abs(a2) < 1e-60:
        logger.debug(
            "Effective mass fit: near-zero curvature (a2=%.3e) near k=%d",
            a2,
            k_center,
        )
        return None

    # m* = hbar^2 / (2 * |a2|)
    m_star_kg = _HBAR_JS**2 / (2.0 * abs(a2))
    m_star_me = m_star_kg / M_E_KG

    # Sanity check: reject unphysical masses
    if m_star_me > 1000 or m_star_me < 1e-6:
        logger.debug(
            "Effective mass fit: unphysical value %.3e m_e near k=%d",
            m_star_me,
            k_center,
        )
        return None

    return float(m_star_me)


def _dos_at_fermi(dos_data: "DOSData") -> Optional[float]:
    """Interpolate DOS at the Fermi level.

    Args:
        dos_data: DOSData with energies, dos, and fermi_energy fields.

    Returns:
        DOS value at the Fermi energy in states/eV, or None if
        interpolation fails.
    """
    try:
        energies = np.asarray(dos_data.energies, dtype=float)
        dos = np.asarray(dos_data.dos, dtype=float)

        if len(energies) == 0 or len(dos) == 0:
            return None

        return float(np.interp(dos_data.fermi_energy, energies, dos))
    except (ValueError, TypeError) as exc:
        logger.warning("Could not interpolate DOS at Fermi level: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_band_structure(
    band_data: "BandStructureData",
    dos_data: Optional["DOSData"] = None,
    n_edge_points: int = 5,
) -> "ElectronicResult":
    """Analyze electronic band structure.

    Extracts band gap, VBM/CBM positions, metallicity, effective masses
    (via parabolic fit near band edges), and DOS at the Fermi level.

    For spin-polarised calculations (``band_data.is_spin_polarized=True``
    with ``spin_up`` and ``spin_down`` populated), both spin channels are
    analysed independently and the narrower gap is reported.

    Args:
        band_data: :class:`~shalom.backends.base.BandStructureData` with
            ``eigenvalues``, ``fermi_energy``, and ``kpath_distances``.
        dos_data: Optional :class:`~shalom.backends.base.DOSData` for
            computing the DOS at the Fermi level.
        n_edge_points: Number of k-points on each side of the band edge
            to include in the parabolic effective mass fit.

    Returns:
        :class:`ElectronicResult` with all derived electronic properties.

    Raises:
        ValueError: If ``eigenvalues`` is None or has zero size.
    """
    _ensure_electronic_available()

    from shalom.analysis._base import ElectronicResult

    # --- Validate input ---
    eigs = band_data.eigenvalues
    if eigs is None:
        raise ValueError("band_data.eigenvalues must not be None")

    eigs = np.asarray(eigs, dtype=float)
    if eigs.size == 0:
        raise ValueError("band_data.eigenvalues must not be empty")

    fermi = band_data.fermi_energy
    kdist = np.asarray(band_data.kpath_distances, dtype=float)

    # --- Handle spin-polarised data ---
    if band_data.is_spin_polarized and band_data.spin_up is not None:
        return _analyze_spin_polarized(band_data, dos_data, n_edge_points)

    # --- Band gap ---
    gap, is_direct, vbm_e, cbm_e, vbm_k, cbm_k, n_occ = _find_band_gap(
        eigs, fermi
    )

    is_metal = gap is None

    # --- Effective masses (only for semiconductors/insulators) ---
    m_e = None
    m_h = None
    if not is_metal and vbm_k is not None and cbm_k is not None:
        # Find the band indices for VBM and CBM
        tol = 1e-6
        vbm_bands = np.where(
            np.abs(eigs[vbm_k] - vbm_e) < tol  # type: ignore[operator]
        )[0]
        cbm_bands = np.where(
            np.abs(eigs[cbm_k] - cbm_e) < tol  # type: ignore[operator]
        )[0]

        if len(vbm_bands) > 0:
            m_h = _estimate_effective_mass(
                eigs, kdist, int(vbm_bands[-1]), vbm_k, n_edge_points
            )

        if len(cbm_bands) > 0:
            m_e = _estimate_effective_mass(
                eigs, kdist, int(cbm_bands[0]), cbm_k, n_edge_points
            )

    # --- DOS at Fermi ---
    dos_fermi = None
    if dos_data is not None:
        dos_fermi = _dos_at_fermi(dos_data)

    logger.info(
        "Electronic analysis: %s, gap=%.3f eV, direct=%s",
        "metal" if is_metal else "semiconductor",
        gap if gap is not None else 0.0,
        is_direct,
    )

    return ElectronicResult(
        bandgap_eV=gap,
        is_direct=is_direct,
        is_metal=is_metal,
        vbm_energy=vbm_e,
        cbm_energy=cbm_e,
        vbm_k_index=vbm_k,
        cbm_k_index=cbm_k,
        effective_mass_electron=m_e,
        effective_mass_hole=m_h,
        dos_at_fermi=dos_fermi,
        n_occupied_bands=n_occ,
        metadata={
            "fermi_energy": fermi,
            "source": band_data.source,
            "n_edge_points": n_edge_points,
        },
    )


def _analyze_spin_polarized(
    band_data: "BandStructureData",
    dos_data: Optional["DOSData"],
    n_edge_points: int,
) -> "ElectronicResult":
    """Analyze spin-polarised band structure.

    Analyses spin-up and spin-down channels independently and reports the
    narrower gap.  Called internally by :func:`analyze_band_structure` when
    ``band_data.is_spin_polarized`` is True.
    """
    from shalom.analysis._base import ElectronicResult

    fermi = band_data.fermi_energy
    kdist = band_data.kpath_distances

    spin_up = np.asarray(band_data.spin_up, dtype=float)
    spin_down = np.asarray(band_data.spin_down, dtype=float)

    # Analyse each spin channel
    gap_up, direct_up, vbm_e_up, cbm_e_up, vbm_k_up, cbm_k_up, n_occ_up = (
        _find_band_gap(spin_up, fermi)
    )
    gap_dn, direct_dn, vbm_e_dn, cbm_e_dn, vbm_k_dn, cbm_k_dn, n_occ_dn = (
        _find_band_gap(spin_down, fermi)
    )

    # Select the narrower gap (or metal if either channel is metallic)
    if gap_up is None and gap_dn is None:
        # Both metallic
        gap, is_direct = None, False
        vbm_e, cbm_e, vbm_k, cbm_k, n_occ = None, None, None, None, 0
        chosen = "both_metal"
    elif gap_up is None:
        gap, is_direct = None, False
        vbm_e, cbm_e, vbm_k, cbm_k, n_occ = None, None, None, None, 0
        chosen = "up_metal"
    elif gap_dn is None:
        gap, is_direct = None, False
        vbm_e, cbm_e, vbm_k, cbm_k, n_occ = None, None, None, None, 0
        chosen = "down_metal"
    elif gap_up <= gap_dn:
        gap, is_direct = gap_up, direct_up
        vbm_e, cbm_e = vbm_e_up, cbm_e_up
        vbm_k, cbm_k = vbm_k_up, cbm_k_up
        n_occ = n_occ_up
        chosen = "spin_up"
    else:
        gap, is_direct = gap_dn, direct_dn
        vbm_e, cbm_e = vbm_e_dn, cbm_e_dn
        vbm_k, cbm_k = vbm_k_dn, cbm_k_dn
        n_occ = n_occ_dn
        chosen = "spin_down"

    is_metal = gap is None

    # Effective masses from the chosen channel
    m_e = None
    m_h = None
    if not is_metal and vbm_k is not None and cbm_k is not None:
        chosen_eigs = spin_up if chosen == "spin_up" else spin_down
        tol = 1e-6

        vbm_bands = np.where(
            np.abs(chosen_eigs[vbm_k] - vbm_e) < tol  # type: ignore[operator]
        )[0]
        cbm_bands = np.where(
            np.abs(chosen_eigs[cbm_k] - cbm_e) < tol  # type: ignore[operator]
        )[0]

        if len(vbm_bands) > 0:
            m_h = _estimate_effective_mass(
                chosen_eigs, kdist, int(vbm_bands[-1]), vbm_k, n_edge_points
            )
        if len(cbm_bands) > 0:
            m_e = _estimate_effective_mass(
                chosen_eigs, kdist, int(cbm_bands[0]), cbm_k, n_edge_points
            )

    # DOS at Fermi
    dos_fermi = None
    if dos_data is not None:
        dos_fermi = _dos_at_fermi(dos_data)

    logger.info(
        "Spin-polarised analysis: %s, gap=%.3f eV (channel=%s), direct=%s",
        "metal" if is_metal else "semiconductor",
        gap if gap is not None else 0.0,
        chosen,
        is_direct,
    )

    return ElectronicResult(
        bandgap_eV=gap,
        is_direct=is_direct,
        is_metal=is_metal,
        vbm_energy=vbm_e,
        cbm_energy=cbm_e,
        vbm_k_index=vbm_k,
        cbm_k_index=cbm_k,
        effective_mass_electron=m_e,
        effective_mass_hole=m_h,
        dos_at_fermi=dos_fermi,
        n_occupied_bands=n_occ,
        metadata={
            "fermi_energy": fermi,
            "source": band_data.source,
            "n_edge_points": n_edge_points,
            "spin_channel": chosen,
            "gap_spin_up": gap_up,
            "gap_spin_down": gap_dn,
        },
    )
