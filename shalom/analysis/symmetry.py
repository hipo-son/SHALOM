"""Structural symmetry analysis using spglib.

Wraps ``spglib`` to determine space group, point group, Wyckoff positions,
crystal system, and symmetry operations from an ASE ``Atoms`` object.

Requires spglib::

    pip install shalom[symmetry]

Example::

    from ase.build import bulk
    from shalom.analysis.symmetry import analyze_symmetry

    si = bulk("Si", "diamond", a=5.43)
    result = analyze_symmetry(si)
    print(f"Space group: {result.space_group_symbol} (#{result.space_group_number})")
    print(f"Crystal system: {result.crystal_system}")
    print(f"Point group: {result.point_group}")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ase import Atoms
    from shalom.analysis._base import SymmetryResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard — follows elastic.py / phonon.py pattern
# ---------------------------------------------------------------------------

try:
    import spglib

    _SPGLIB_AVAILABLE = True
except ImportError:
    _SPGLIB_AVAILABLE = False


def is_spglib_available() -> bool:
    """Check if spglib is installed."""
    return _SPGLIB_AVAILABLE


def _ensure_spglib_available() -> None:
    """Raise ImportError if spglib is not available."""
    if not _SPGLIB_AVAILABLE:
        raise ImportError(
            "Symmetry analysis requires spglib. "
            "Install with: pip install shalom[symmetry]"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ase_to_spglib_cell(atoms: "Atoms") -> tuple:
    """Convert ASE Atoms to the spglib cell tuple format.

    Returns:
        ``(lattice, scaled_positions, atomic_numbers)`` where:

        - *lattice*: (3, 3) array — row vectors of the unit cell.
        - *scaled_positions*: (N, 3) array — fractional coordinates.
        - *atomic_numbers*: (N,) array of integers.
    """
    return (
        atoms.cell.array,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )


def _space_group_to_crystal_system(sg_number: int) -> str:
    """Map an international space group number (1--230) to the crystal system.

    Args:
        sg_number: Space group number in the range 1--230.

    Returns:
        Crystal system name as a lowercase string.

    Raises:
        ValueError: If *sg_number* is outside the valid range.
    """
    if sg_number < 1 or sg_number > 230:
        raise ValueError(
            f"Space group number must be 1-230, got {sg_number}"
        )
    if sg_number <= 2:
        return "triclinic"
    if sg_number <= 15:
        return "monoclinic"
    if sg_number <= 74:
        return "orthorhombic"
    if sg_number <= 142:
        return "tetragonal"
    if sg_number <= 167:
        return "trigonal"
    if sg_number <= 194:
        return "hexagonal"
    return "cubic"


def _extract_lattice_type(space_group_symbol: str) -> str:
    """Extract the Bravais lattice type letter from the international symbol.

    The first character of the Hermann-Mauguin symbol encodes the lattice
    centering: P (primitive), I (body-centred), F (face-centred), C (base-
    centred), R (rhombohedral), or A (base-centred, alternate axis).

    Args:
        space_group_symbol: International (Hermann-Mauguin) space group symbol.

    Returns:
        Single uppercase letter (e.g. ``"F"`` for ``"Fm-3m"``).
        Returns ``""`` if the symbol is empty.
    """
    if not space_group_symbol:
        return ""
    return space_group_symbol[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_symmetry(
    atoms: "Atoms",
    symprec: float = 1e-5,
) -> "SymmetryResult":
    """Analyze the crystal symmetry of an ASE Atoms object using spglib.

    Determines the space group, point group, Wyckoff positions, crystal
    system, number of symmetry operations, and whether the cell is primitive.

    Args:
        atoms: ASE ``Atoms`` object with periodic boundary conditions
            (``pbc=True``).
        symprec: Symmetry tolerance in Angstrom.  Larger values allow more
            distortion to be treated as symmetric.  Default ``1e-5``.

    Returns:
        :class:`SymmetryResult` populated with all symmetry information.

    Raises:
        ImportError: If spglib is not installed.
        ValueError: If spglib fails to determine the symmetry (returns None).
    """
    _ensure_spglib_available()

    import numpy as np  # noqa: F811 — lazy import follows elastic.py pattern

    from shalom.analysis._base import SymmetryResult

    cell = _ase_to_spglib_cell(atoms)
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    if dataset is None:
        raise ValueError(
            "spglib could not determine symmetry for the given structure. "
            "The structure may be too distorted or the tolerance (symprec) "
            "may need adjustment."
        )

    # Use attribute access (spglib >= 2.1 deprecates dict interface).
    # Fall back to dict access for older spglib versions.
    def _get(key: str) -> Any:
        try:
            return getattr(dataset, key)
        except AttributeError:
            return dataset[key]  # type: ignore[index]

    sg_number = int(_get("number"))
    sg_symbol = str(_get("international"))
    hall_symbol = str(_get("hall"))
    point_group = str(_get("pointgroup"))

    # Wyckoff letters — spglib returns a list of single-char strings
    wyckoff_letters = list(_get("wyckoffs"))

    # Equivalent atoms — spglib returns an integer array
    equivalent_atoms = [int(x) for x in _get("equivalent_atoms")]

    # Number of symmetry operations
    n_operations = len(_get("rotations"))

    # Crystal system from space group number
    crystal_system = _space_group_to_crystal_system(sg_number)

    # Lattice type from first letter of international symbol
    lattice_type = _extract_lattice_type(sg_symbol)

    # Check if the input cell is already primitive.
    # find_primitive returns (lattice, positions, numbers) or None.
    # If the primitive cell has the same number of atoms, the cell IS primitive.
    prim_cell = spglib.find_primitive(cell, symprec=symprec)
    if prim_cell is not None and prim_cell[1] is not None:
        n_prim = len(prim_cell[1])
        is_prim = n_prim == len(atoms)
    else:
        # find_primitive failed — conservatively assume not primitive
        is_prim = False

    # Convert dataset to a plain dict for storage in raw.
    # spglib >= 2.1 uses attribute interface; fall back to dict iteration.
    raw_dict: Dict[str, Any] = {}
    _raw_keys = [
        "number", "hall_number", "international", "hall", "choice",
        "transformation_matrix", "origin_shift", "rotations", "translations",
        "wyckoffs", "site_symmetry_symbols", "crystallographic_orbits",
        "equivalent_atoms", "primitive_lattice", "mapping_to_primitive",
        "std_lattice", "std_positions", "std_types",
        "std_rotation_matrix", "std_mapping_to_primitive", "pointgroup",
    ]
    for key in _raw_keys:
        try:
            val = getattr(dataset, key, None)
            if val is None:
                val = dataset[key]  # type: ignore[index]
        except (KeyError, TypeError):
            continue
        if isinstance(val, np.ndarray):
            raw_dict[key] = val.tolist()
        else:
            raw_dict[key] = val

    logger.info(
        "Symmetry: %s (#%d), %s, %s, %d ops, primitive=%s",
        sg_symbol,
        sg_number,
        crystal_system,
        point_group,
        n_operations,
        is_prim,
    )

    return SymmetryResult(
        space_group_number=sg_number,
        space_group_symbol=sg_symbol,
        point_group=point_group,
        crystal_system=crystal_system,
        lattice_type=lattice_type,
        hall_symbol=hall_symbol,
        wyckoff_letters=wyckoff_letters,
        equivalent_atoms=equivalent_atoms,
        n_operations=n_operations,
        is_primitive=is_prim,
        raw=raw_dict,
        metadata={},
    )
