"""X-ray diffraction pattern analysis using pymatgen.

Wraps ``pymatgen.analysis.diffraction.xrd`` to compute powder XRD patterns
from crystal structures, providing peak positions (2theta), intensities,
Miller indices, and d-spacings.

Requires pymatgen::

    pip install shalom[analysis]

Example::

    from ase.build import bulk
    from shalom.analysis.xrd import calculate_xrd

    atoms = bulk("Si", "diamond", a=5.43)
    result = calculate_xrd(atoms)
    print(f"Number of peaks: {result.n_peaks}")
    print(f"Strongest peak at 2theta = {result.two_theta[0]:.2f} deg")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ase import Atoms
    from shalom.analysis._base import XRDResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional dependency guard -- follows elastic.py pattern
# ---------------------------------------------------------------------------

try:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.core import Lattice, Structure

    _XRD_AVAILABLE = True
except ImportError:
    _XRD_AVAILABLE = False


def is_xrd_available() -> bool:
    """Check if pymatgen XRD module is installed."""
    return _XRD_AVAILABLE


def _ensure_xrd_available() -> None:
    """Raise ImportError if pymatgen XRD is not available."""
    if not _XRD_AVAILABLE:
        raise ImportError(
            "XRD analysis requires pymatgen. "
            "Install with: pip install shalom[analysis]"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Common X-ray wavelengths in Angstrom (fallback lookup)
_WAVELENGTH_ANGSTROM: Dict[str, float] = {
    "CuKa": 1.5406,
    "CuKa2": 1.5444,
    "CuKa1": 1.5405,
    "CuKb1": 1.3922,
    "MoKa": 0.7107,
    "MoKa2": 0.7136,
    "MoKa1": 0.7093,
    "MoKb1": 0.6323,
    "CrKa": 2.2910,
    "CrKa2": 2.2936,
    "CrKa1": 2.2897,
    "FeKa": 1.9373,
    "FeKa2": 1.9400,
    "FeKa1": 1.9360,
    "CoKa": 1.7903,
    "CoKa2": 1.7929,
    "CoKa1": 1.7890,
    "AgKa": 0.5594,
    "AgKa2": 0.5638,
    "AgKa1": 0.5594,
}


def _ase_to_pymatgen_structure(atoms: "Atoms") -> "Structure":
    """Convert ASE Atoms to pymatgen Structure.

    Args:
        atoms: ASE Atoms object with cell and positions.

    Returns:
        Equivalent pymatgen Structure.
    """
    return Structure(
        lattice=Lattice(atoms.cell.array),
        species=atoms.get_chemical_symbols(),
        coords=atoms.get_scaled_positions(),
    )


def _extract_hkl(hkl_entry: Any) -> Tuple[int, int, int]:
    """Extract (h, k, l) tuple from a single hkls entry.

    Handles two pymatgen DiffractionPattern formats:

    - **Modern** (pymatgen >= 2024): list of dicts with ``"hkl"`` key,
      e.g. ``[{"hkl": (1,1,1), "multiplicity": 8}, ...]``
    - **Legacy**: dict mapping ``(h,k,l)`` tuple to multiplicity,
      e.g. ``{(1,1,1): 8, ...}``

    Returns:
        Miller indices as ``(h, k, l)`` tuple of ints.
    """
    if isinstance(hkl_entry, (list, tuple)) and len(hkl_entry) > 0:
        first = hkl_entry[0]
        if isinstance(first, dict) and "hkl" in first:
            # Modern format: [{"hkl": (h,k,l), "multiplicity": m}, ...]
            hkl = first["hkl"]
            return (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        elif isinstance(first, dict):
            # Legacy format wrapped in list: [{(h,k,l): m}, ...]
            key = next(iter(first))
            if isinstance(key, (tuple, list)):
                return (int(key[0]), int(key[1]), int(key[2]))
    elif isinstance(hkl_entry, dict):
        if "hkl" in hkl_entry:
            hkl = hkl_entry["hkl"]
            return (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        # Legacy dict: {(h,k,l): multiplicity}
        key = next(iter(hkl_entry))
        if isinstance(key, (tuple, list)):
            return (int(key[0]), int(key[1]), int(key[2]))
    return (0, 0, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_xrd(
    atoms: "Atoms",
    wavelength: str = "CuKa",
    two_theta_range: Tuple[float, float] = (0, 90),
) -> "XRDResult":
    """Calculate powder X-ray diffraction pattern for a crystal structure.

    Args:
        atoms: Crystal structure as ASE Atoms.
        wavelength: X-ray radiation source label recognised by pymatgen
            (e.g. ``"CuKa"``, ``"MoKa"``, ``"CoKa"``).
        two_theta_range: ``(min, max)`` 2-theta range in degrees.

    Returns:
        :class:`XRDResult` with peak positions, intensities, Miller
        indices, and d-spacings.

    Raises:
        ImportError: If pymatgen is not installed.
        ValueError: If the structure cannot be processed.
    """
    _ensure_xrd_available()

    import numpy as np

    from shalom.analysis._base import XRDResult

    structure = _ase_to_pymatgen_structure(atoms)

    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=two_theta_range)

    two_theta = np.array(pattern.x, dtype=float)
    intensities = np.array(pattern.y, dtype=float)
    d_spacings = np.array(pattern.d_hkls, dtype=float)

    # Extract Miller indices from pattern.hkls.
    # pymatgen returns a list-of-lists-of-dicts:
    #   [[{"hkl": (h,k,l), "multiplicity": m}, ...], ...]
    # Older versions may return list-of-dicts with (h,k,l) tuple keys.
    # We extract the first hkl tuple from each peak entry.
    hkl_indices: List[Tuple[int, int, int]] = []
    for hkl_entry in pattern.hkls:
        hkl_tuple = _extract_hkl(hkl_entry)
        hkl_indices.append(hkl_tuple)

    # Determine wavelength in Angstrom
    wavelength_ang: Optional[float] = None
    try:
        wavelength_ang = float(calc.wavelength)
    except (AttributeError, TypeError, ValueError):
        wavelength_ang = _WAVELENGTH_ANGSTROM.get(wavelength)

    n_peaks = len(two_theta)

    logger.info(
        "XRD pattern computed: %d peaks in 2theta range (%.1f, %.1f) deg, "
        "wavelength=%s (%.4f A)",
        n_peaks,
        two_theta_range[0],
        two_theta_range[1],
        wavelength,
        wavelength_ang or 0.0,
    )

    return XRDResult(
        two_theta=two_theta,
        intensities=intensities,
        hkl_indices=hkl_indices,
        d_spacings=d_spacings,
        wavelength=wavelength,
        wavelength_angstrom=wavelength_ang,
        n_peaks=n_peaks,
        raw=pattern,
        metadata={
            "two_theta_range": two_theta_range,
            "formula": structure.composition.reduced_formula,
        },
    )
