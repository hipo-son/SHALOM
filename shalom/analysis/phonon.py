"""Phonon analysis using phonopy.

Wraps ``phonopy`` to compute phonon band structures, density of states,
thermal properties, and dynamical stability from force sets or force
constants.

Requires phonopy::

    pip install shalom[phonon]

Example::

    from ase.build import bulk
    from shalom.analysis import generate_phonon_displacements, analyze_phonon

    atoms = bulk("Si", "diamond", a=5.43)
    disps, ph = generate_phonon_displacements(
        atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    # ... run DFT on each displaced supercell, collect forces ...
    # result = analyze_phonon(atoms, [[2,0,0],[0,2,0],[0,0,2]], force_sets)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ase import Atoms
    from shalom.analysis._base import PhononResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard — follows elastic.py pattern
# ---------------------------------------------------------------------------

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    _PHONOPY_AVAILABLE = True
except ImportError:
    _PHONOPY_AVAILABLE = False


def is_phonopy_available() -> bool:
    """Check if phonopy is installed."""
    return _PHONOPY_AVAILABLE


def _ensure_phonopy_available() -> None:
    """Raise ImportError if phonopy is not available."""
    if not _PHONOPY_AVAILABLE:
        raise ImportError(
            "Phonon analysis requires phonopy. "
            "Install with: pip install shalom[phonon]"
        )


# ---------------------------------------------------------------------------
# ASE ↔ phonopy converters
# ---------------------------------------------------------------------------

_IMAGINARY_THRESHOLD_THZ = -0.1


def _ase_to_phonopy(atoms: "Atoms") -> Any:
    """Convert ASE Atoms to PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def _phonopy_to_ase(pa: Any) -> "Atoms":
    """Convert PhonopyAtoms to ASE Atoms."""
    from ase import Atoms as ASEAtoms

    return ASEAtoms(
        symbols=pa.symbols,
        cell=pa.cell,
        scaled_positions=pa.scaled_positions,
        pbc=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_phonon_displacements(
    atoms: "Atoms",
    supercell_matrix: List[List[int]],
    distance: float = 0.01,
) -> Tuple[List["Atoms"], Any]:
    """Generate displaced supercells for phonon force calculations.

    Args:
        atoms: Unit cell as ASE Atoms.
        supercell_matrix: 3x3 supercell matrix, e.g. ``[[2,0,0],[0,2,0],[0,0,2]]``.
        distance: Displacement distance in Angstrom (default: 0.01).

    Returns:
        Tuple of ``(displaced_supercells, phonopy_object)`` where
        *displaced_supercells* is a list of ASE Atoms and *phonopy_object*
        is the :class:`phonopy.Phonopy` instance (needed later to set forces).

    Raises:
        ImportError: If phonopy is not installed.
        ValueError: If supercell_matrix is not 3x3.
    """
    _ensure_phonopy_available()

    import numpy as np

    sc = np.array(supercell_matrix, dtype=int)
    if sc.shape != (3, 3):
        raise ValueError(f"supercell_matrix must be 3x3, got shape {sc.shape}")

    unitcell = _ase_to_phonopy(atoms)
    ph = Phonopy(unitcell, sc)
    ph.generate_displacements(distance=distance)

    supercells = ph.supercells_with_displacements
    if supercells is None:
        return [], ph

    return [_phonopy_to_ase(sc_atom) for sc_atom in supercells], ph


def analyze_phonon(
    atoms: "Atoms",
    supercell_matrix: List[List[int]],
    force_sets: List[Any],
    mesh: Optional[List[int]] = None,
    band_npoints: int = 51,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    t_step: float = 10.0,
) -> "PhononResult":
    """Compute phonon properties from force sets.

    Args:
        atoms: Unit cell as ASE Atoms.
        supercell_matrix: 3x3 supercell matrix.
        force_sets: List of N_displacement arrays, each (N_atoms_supercell, 3)
            forces in eV/Angstrom.
        mesh: Monkhorst-Pack q-point mesh for DOS, e.g. ``[20, 20, 20]``.
            Defaults to ``[20, 20, 20]``.
        band_npoints: Number of q-points per segment for band structure.
        t_min: Minimum temperature for thermal properties (K).
        t_max: Maximum temperature for thermal properties (K).
        t_step: Temperature step for thermal properties (K).

    Returns:
        :class:`~shalom.analysis._base.PhononResult` with band structure,
        DOS, thermal properties, and stability assessment.

    Raises:
        ImportError: If phonopy is not installed.
        ValueError: If inputs have wrong shapes.
    """
    _ensure_phonopy_available()

    import numpy as np

    if mesh is None:
        mesh = [20, 20, 20]

    sc = np.array(supercell_matrix, dtype=int)
    if sc.shape != (3, 3):
        raise ValueError(f"supercell_matrix must be 3x3, got shape {sc.shape}")

    if len(force_sets) == 0:
        raise ValueError("force_sets must not be empty")

    unitcell = _ase_to_phonopy(atoms)
    ph = Phonopy(unitcell, sc)
    ph.generate_displacements(distance=0.01)

    forces_array = np.array(force_sets, dtype=float)
    if forces_array.ndim != 3 or forces_array.shape[2] != 3:
        raise ValueError(
            f"force_sets must have shape (n_displacements, n_atoms, 3), "
            f"got {forces_array.shape}"
        )

    ph.forces = forces_array
    ph.produce_force_constants()

    return _run_phonon_analysis(ph, mesh, band_npoints, t_min, t_max, t_step)


def analyze_phonon_from_force_constants(
    atoms: "Atoms",
    force_constants: Any,
    supercell_matrix: List[List[int]],
    mesh: Optional[List[int]] = None,
    band_npoints: int = 51,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    t_step: float = 10.0,
) -> "PhononResult":
    """Compute phonon properties from precomputed force constants.

    Args:
        atoms: Unit cell as ASE Atoms.
        force_constants: (N, N, 3, 3) force constant matrix in eV/Ang^2,
            as a nested list or numpy array.
        supercell_matrix: 3x3 supercell matrix.
        mesh: Q-point mesh for DOS.  Defaults to ``[20, 20, 20]``.
        band_npoints: Points per segment for band path.
        t_min: Min temperature for thermal properties (K).
        t_max: Max temperature for thermal properties (K).
        t_step: Temperature step (K).

    Returns:
        :class:`~shalom.analysis._base.PhononResult`.

    Raises:
        ImportError: If phonopy is not installed.
        ValueError: If inputs have wrong shapes.
    """
    _ensure_phonopy_available()

    import numpy as np

    if mesh is None:
        mesh = [20, 20, 20]

    sc = np.array(supercell_matrix, dtype=int)
    if sc.shape != (3, 3):
        raise ValueError(f"supercell_matrix must be 3x3, got shape {sc.shape}")

    fc = np.array(force_constants, dtype=float)
    if fc.ndim != 4 or fc.shape[2] != 3 or fc.shape[3] != 3:
        raise ValueError(
            f"force_constants must have shape (N, N, 3, 3), got {fc.shape}"
        )

    unitcell = _ase_to_phonopy(atoms)
    ph = Phonopy(unitcell, sc)
    ph.force_constants = fc

    return _run_phonon_analysis(ph, mesh, band_npoints, t_min, t_max, t_step)


# ---------------------------------------------------------------------------
# Internal: shared analysis pipeline
# ---------------------------------------------------------------------------


def _run_phonon_analysis(
    ph: Any,
    mesh: List[int],
    band_npoints: int,
    t_min: float,
    t_max: float,
    t_step: float,
) -> "PhononResult":
    """Run band structure, DOS, and thermal analysis on a Phonopy object."""
    import numpy as np

    from shalom.analysis._base import PhononResult

    n_atoms = len(ph.primitive.symbols)
    n_branches = n_atoms * 3

    # ---- Band structure ----
    ph.auto_band_structure(npoints=band_npoints, plot=False)
    bs = ph.get_band_structure_dict()

    # Flatten segmented arrays into contiguous arrays
    all_qpoints = []
    all_distances = []
    all_frequencies = []
    labels: Dict[int, str] = {}
    offset = 0

    for seg_idx, (seg_q, seg_d, seg_f) in enumerate(
        zip(bs["qpoints"], bs["distances"], bs["frequencies"])
    ):
        seg_q = np.array(seg_q)
        seg_d = np.array(seg_d)
        seg_f = np.array(seg_f)
        all_qpoints.append(seg_q)
        all_distances.append(seg_d)
        all_frequencies.append(seg_f)
        offset += len(seg_q)

    # Extract labels from phonopy BandStructure object
    labels = _build_band_labels(ph, bs)

    band_qpoints = np.concatenate(all_qpoints, axis=0) if all_qpoints else None
    band_distances = np.concatenate(all_distances, axis=0) if all_distances else None
    band_frequencies = (
        np.concatenate(all_frequencies, axis=0) if all_frequencies else None
    )

    # ---- DOS ----
    dos_frequencies = None
    dos_density = None
    ph.run_mesh(mesh)
    try:
        ph.run_total_dos()
        dos_dict = ph.get_total_dos_dict()
        dos_frequencies = np.array(dos_dict["frequency_points"])
        dos_density = np.array(dos_dict["total_dos"])
    except (ValueError, RuntimeError) as exc:
        logger.warning("Could not compute phonon DOS: %s", exc)

    # ---- Thermal properties ----
    ph.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    tp = ph.get_thermal_properties_dict()
    thermal_temps = np.array(tp["temperatures"])
    thermal_fe = np.array(tp["free_energy"])
    thermal_entropy = np.array(tp["entropy"])
    thermal_cv = np.array(tp["heat_capacity"])

    # ---- Stability analysis ----
    min_freq = float(band_frequencies.min()) if band_frequencies is not None else 0.0
    imaginary_modes: List[Tuple[int, int, float]] = []

    if band_frequencies is not None:
        imag_mask = band_frequencies < _IMAGINARY_THRESHOLD_THZ
        qi, bi = np.where(imag_mask)
        for q_idx, b_idx in zip(qi, bi):
            imaginary_modes.append(
                (int(q_idx), int(b_idx), float(band_frequencies[q_idx, b_idx]))
            )

    is_stable = min_freq > _IMAGINARY_THRESHOLD_THZ

    # ---- Force constants ----
    fc = ph.force_constants if ph.force_constants is not None else None

    return PhononResult(
        band_qpoints=band_qpoints,
        band_distances=band_distances,
        band_frequencies=band_frequencies,
        band_labels=labels,
        dos_frequencies=dos_frequencies,
        dos_density=dos_density,
        thermal_temperatures=thermal_temps,
        thermal_free_energy=thermal_fe,
        thermal_entropy=thermal_entropy,
        thermal_cv=thermal_cv,
        min_frequency_THz=min_freq,
        is_stable=is_stable,
        imaginary_modes=imaginary_modes,
        n_atoms=n_atoms,
        n_branches=n_branches,
        force_constants=fc,
        raw=ph,
    )


def _build_band_labels(ph: Any, bs_dict: Dict[str, Any]) -> Dict[int, str]:
    """Build q-point index → label mapping from phonopy BandStructure."""
    import numpy as np

    labels_out: Dict[int, str] = {}

    try:
        raw_labels = ph.band_structure.labels
        connections = ph.band_structure.path_connections
    except (AttributeError, TypeError):
        return labels_out

    if raw_labels is None:
        return labels_out

    seg_lengths = [len(np.array(seg)) for seg in bs_dict["qpoints"]]
    n_segments = len(seg_lengths)

    # Map labels to cumulative q-point indices.
    # phonopy labels list follows: for each group of connected segments,
    # the first segment has start+end labels, and each additional connected
    # segment adds one more label (its end).
    # Disconnected boundary → two separate labels (prev end, next start).
    label_idx = 0
    q_offset = 0

    for seg_i in range(n_segments):
        seg_len = seg_lengths[seg_i]

        # Start label for this segment
        is_first_seg = seg_i == 0
        is_after_gap = seg_i > 0 and not connections[seg_i - 1]

        if is_first_seg or is_after_gap:
            if label_idx < len(raw_labels):
                lbl = _normalize_label(raw_labels[label_idx])
                if lbl:
                    labels_out[q_offset] = lbl
                label_idx += 1

        # End label for this segment
        if label_idx < len(raw_labels):
            lbl = _normalize_label(raw_labels[label_idx])
            if lbl:
                labels_out[q_offset + seg_len - 1] = lbl
            label_idx += 1

        q_offset += seg_len

    return labels_out


def _normalize_label(label: str) -> str:
    r"""Normalize phonopy label to plain text.

    Converts LaTeX-style labels (``$\\Gamma$``, ``$\\mathrm{X}$``)
    to plain text (``G``, ``X``).
    """
    import re

    label = label.replace("$", "").strip()
    if label in ("\\Gamma", "GAMMA", "\\gamma", "Gamma"):
        return "G"
    # \mathrm{X} → X
    m = re.match(r"\\mathrm\{(\w+)\}", label)
    if m:
        return m.group(1)
    return label
