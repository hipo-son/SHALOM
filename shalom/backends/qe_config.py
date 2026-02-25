"""Quantum ESPRESSO input configuration and preset system.

Mirrors the VASP config architecture (``vasp_config.py``): composable presets,
structure-aware auto-detection, dynamic ecutwfc/ecutrho allocation from SSSP
metadata, and K-point grid computation.

Pseudopotential library:
    SSSP Efficiency v1.3.0 (PBE)
    Ref: Prandini et al. npj Comput. Mater. 4, 72 (2018).
    Ref: Bosoni et al. Nat. Rev. Phys. 6, 45 (2024).

Hubbard U:
    Wang et al. PRB 73, 195107 (2006), PBE-fitted.
    NOTE: These U values were derived with PAW projectors in VASP. When used
    with QE's ortho-atomic projectors, quantitative differences are expected.
    For rigorous studies of strongly-correlated systems, re-derive U with hp.x.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from ase import Atoms

from shalom._config_loader import load_config
from shalom.backends._physics import (
    AccuracyLevel,
    MAGNETIC_ELEMENTS,
    DEFAULT_MAGMOM,
    HUBBARD_U_VALUES,
    ANION_ELEMENTS,
    RY_TO_EV,
    EV_TO_RY,
    _is_pure_metal,
    detect_2d,
    compute_kpoints_grid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QECalculationType(str, Enum):
    """Supported QE pw.x calculation types."""

    SCF = "scf"
    RELAX = "relax"
    VC_RELAX = "vc-relax"
    BANDS = "bands"
    NSCF = "nscf"


# VASP calc_type string → QE calc_type mapping.
# Key correction: VASP "relaxation" (ISIF=3, cell+atoms) → QE vc-relax, not relax.
VASP_TO_QE_CALC_MAP: Dict[str, QECalculationType] = {
    "relaxation": QECalculationType.VC_RELAX,
    "static": QECalculationType.SCF,
    "band_structure": QECalculationType.BANDS,
    "dos": QECalculationType.NSCF,
}


# ---------------------------------------------------------------------------
# KPoints Config
# ---------------------------------------------------------------------------


@dataclass
class QEKPointsConfig:
    """K-point mesh configuration for QE.

    Attributes:
        mode: K_POINTS card mode — automatic, gamma, or crystal_b.
        grid: Explicit grid [Nx, Ny, Nz].
        shift: Grid offset [sx, sy, sz] (0 or 1).
        kpath_points: High-symmetry k-path for ``crystal_b`` mode.
            Each entry is ``(crystal_coords, npts_on_segment)`` where
            ``crystal_coords`` is a 3-element list in fractional primitive-cell
            coordinates and ``npts_on_segment`` is the number of k-points on
            the segment (the last point conventionally uses ``npts=1``).
            Example: ``[([0,0,0], 40), ([0.5,0,0.5], 40), ([0.5,0.5,0.5], 1)]``
        kpath_labels: Map from *segment index* (0-based) to high-symmetry label,
            e.g. ``{0: "G", 1: "X", 2: "L"}``.  Used by plotters to annotate
            band-structure axes.
    """

    mode: Literal["automatic", "gamma", "crystal_b"] = "automatic"
    grid: Optional[List[int]] = None
    shift: List[int] = field(default_factory=lambda: [0, 0, 0])
    kpath_points: Optional[List[Tuple[List[float], int]]] = None
    kpath_labels: Optional[Dict[int, str]] = None


# ---------------------------------------------------------------------------
# QE Input Config
# ---------------------------------------------------------------------------


@dataclass
class QEInputConfig:
    """Complete QE pw.x input configuration.

    Holds namelist dicts (&CONTROL, &SYSTEM, &ELECTRONS, &IONS, &CELL),
    user overrides, K-point config, and pseudopotential mapping.
    """

    calc_type: QECalculationType = QECalculationType.SCF
    accuracy: AccuracyLevel = AccuracyLevel.STANDARD
    control: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    electrons: Dict[str, Any] = field(default_factory=dict)
    ions: Dict[str, Any] = field(default_factory=dict)
    cell: Dict[str, Any] = field(default_factory=dict)
    user_settings: Dict[str, Any] = field(default_factory=dict)
    kpoints: QEKPointsConfig = field(default_factory=QEKPointsConfig)
    # Resolved at runtime by direct_run._create_qe_config() via $SHALOM_PSEUDO_DIR
    # or ~/pseudopotentials. Only used as-is when QEInputConfig is built directly.
    pseudo_dir: str = "./"
    pseudo_map: Dict[str, str] = field(default_factory=dict)
    functional: str = "PBE"
    is_2d: bool = False

    def get_merged_namelists(self) -> Dict[str, Dict[str, Any]]:
        """Return namelists with user_settings overrides applied.

        user_settings keys support dot notation (e.g. ``"system.ecutwfc"``).
        Keys without a dot are placed in &SYSTEM by default.
        Only &IONS is included for relax/vc-relax; &CELL only for vc-relax.
        """
        result: Dict[str, Dict[str, Any]] = {
            "CONTROL": dict(self.control),
            "SYSTEM": dict(self.system),
            "ELECTRONS": dict(self.electrons),
        }
        if self.calc_type in (QECalculationType.RELAX, QECalculationType.VC_RELAX):
            result["IONS"] = dict(self.ions)
        if self.calc_type == QECalculationType.VC_RELAX:
            result["CELL"] = dict(self.cell)

        for key, val in self.user_settings.items():
            if "." in key:
                ns, param = key.split(".", 1)
                ns_upper = ns.upper()
                if ns_upper in result:
                    result[ns_upper][param] = val
            else:
                result.setdefault("SYSTEM", {})[key] = val
        return result


# ---------------------------------------------------------------------------
# SSSP Pseudopotential Metadata
# ---------------------------------------------------------------------------

_sssp_cfg = load_config("sssp_metadata")
SSSP_ELEMENTS: Dict[str, Dict[str, Any]] = _sssp_cfg["elements"]


def get_pseudo_filename(element: str) -> str:
    """Return the SSSP pseudopotential filename for an element."""
    entry = SSSP_ELEMENTS.get(element)
    if entry is not None:
        return entry["pseudo"]
    return f"{element}.UPF"


def get_z_valence(element: str) -> int:
    """Return the number of valence electrons from SSSP metadata."""
    entry = SSSP_ELEMENTS.get(element)
    if entry is not None:
        return entry["z_valence"]
    return 1  # conservative fallback


def compute_ecutwfc(elements: List[str], accuracy: AccuracyLevel) -> float:
    """Compute ecutwfc from SSSP per-element hints.

    STANDARD: max(hints), minimum 45 Ry.
    PRECISE:  max(hints) * 1.2, minimum 60 Ry.
    """
    hints = [SSSP_ELEMENTS[el]["ecutwfc"] for el in set(elements) if el in SSSP_ELEMENTS]
    if not hints:
        return 60.0 if accuracy == AccuracyLevel.PRECISE else 45.0
    max_hint = max(hints)
    if accuracy == AccuracyLevel.PRECISE:
        return max(60.0, math.ceil(max_hint * 1.2))
    return max(45.0, float(max_hint))


def compute_ecutrho(elements: List[str], accuracy: AccuracyLevel) -> float:
    """Compute ecutrho from SSSP per-element metadata (NOT blanket 8x).

    Uses max(SSSP ecutrho per element). This respects PP-type dependent ratios:
    NC → ~4x, USPP → ~8-12x, PAW → ~8-12x.
    """
    hints = [SSSP_ELEMENTS[el]["ecutrho"] for el in set(elements) if el in SSSP_ELEMENTS]
    if not hints:
        ecutwfc = compute_ecutwfc(elements, accuracy)
        return ecutwfc * 8.0  # fallback
    max_rho = max(hints)
    if accuracy == AccuracyLevel.PRECISE:
        return max(max_rho, math.ceil(max_rho * 1.2))
    return float(max_rho)


# ---------------------------------------------------------------------------
# Structure-Aware Auto-Detection
# ---------------------------------------------------------------------------


def detect_and_apply_qe_hints(
    atoms: Atoms, config: QEInputConfig,
) -> QEInputConfig:
    """Apply structure-aware auto-detection rules to a QEInputConfig.

    Modifies the config in-place:
    1. Magnetic elements → nspin=2, starting_magnetization = MAGMOM / z_valence
    2. GGA+U (PRECISE only) → lda_plus_u, Hubbard_U(i)
    3. 2D detection → assume_isolated='2D', vdw_corr, cell_dofree='2Dxy'
    4. Pure metal → smearing='methfessel-paxton', degauss=0.00735 Ry
    5. ecutwfc / ecutrho from SSSP
    6. K-points grid (shared compute_kpoints_grid)
    7. ibrav=0, nat, ntyp

    Args:
        atoms: ASE Atoms object.
        config: QEInputConfig to modify.

    Returns:
        The modified config (same object).
    """
    if len(atoms) == 0:
        return config

    symbols = list(atoms.get_chemical_symbols())
    unique_elements = list(set(symbols))
    # Species order preserving first occurrence (for QE type indexing)
    species_order = list(dict.fromkeys(symbols))

    # 1. Magnetic element detection
    has_magnetic = any(el in MAGNETIC_ELEMENTS for el in unique_elements)
    if has_magnetic:
        config.system["nspin"] = 2
        # starting_magnetization(i) = MAGMOM / z_valence
        # Uses z_valence from SSSP metadata, NOT /10.
        for i, el in enumerate(species_order, 1):
            if el in DEFAULT_MAGMOM:
                z_val = get_z_valence(el)
                start_mag = min(DEFAULT_MAGMOM[el] / z_val, 1.0)
                config.system[f"starting_magnetization({i})"] = round(start_mag, 4)

    # 2. GGA+U auto-detection (PRECISE only, same policy as VASP)
    has_anion = any(el in ANION_ELEMENTS for el in unique_elements)
    magnetic_tm = [el for el in unique_elements if el in HUBBARD_U_VALUES]
    if has_magnetic and has_anion and magnetic_tm:
        u_elements = [el for el in magnetic_tm if HUBBARD_U_VALUES[el][1] > 0.0]
        if u_elements and config.accuracy == AccuracyLevel.PRECISE:
            config.system["lda_plus_u"] = True
            for i, el in enumerate(species_order, 1):
                if el in HUBBARD_U_VALUES:
                    _, u_val, _ = HUBBARD_U_VALUES[el]
                    config.system[f"Hubbard_U({i})"] = u_val
            if config.functional != "PBE":
                logger.warning(
                    "GGA+U enabled with functional='%s', but Hubbard U "
                    "values are fitted for PBE only (Wang et al. PRB 73, "
                    "195107). Results may be unreliable.",
                    config.functional,
                )
        elif u_elements and config.accuracy != AccuracyLevel.PRECISE:
            logger.warning(
                "GGA+U disabled at STANDARD accuracy for elements %s. "
                "For transition metal oxides/chalcogenides, use accuracy='precise'.",
                u_elements,
            )

    # 3. 2D detection
    if detect_2d(atoms):
        config.is_2d = True
        config.system["assume_isolated"] = "2D"
        # DFT-D3 with BJ damping to match VASP IVDW=12
        # Ref: Grimme et al. J. Comput. Chem. 32, 1456 (2011)
        config.system["vdw_corr"] = "dft-d3"
        config.system["dftd3_version"] = 4  # BJ damping
        if config.calc_type == QECalculationType.VC_RELAX:
            config.cell["cell_dofree"] = "2Dxy"

    # 4. Occupation / smearing based on material class
    if _is_pure_metal(unique_elements):
        # Methfessel-Paxton order 1, degauss = VASP SIGMA(0.1 eV) / 13.6057
        config.system["occupations"] = "smearing"
        config.system["smearing"] = "methfessel-paxton"
        config.system["degauss"] = 0.00735
        config.electrons["mixing_beta"] = 0.4  # Lower for metallic SCF stability
    elif not has_magnetic:
        # Non-magnetic non-metal (semiconductor/insulator): reduce broadening
        # 0.02 eV / 13.6057 = 0.00147 Ry — negligible for wide-gap materials
        config.system["degauss"] = 0.00147

    # 5. ecutwfc / ecutrho from SSSP metadata
    ecutwfc = compute_ecutwfc(unique_elements, config.accuracy)
    ecutrho = compute_ecutrho(unique_elements, config.accuracy)
    config.system["ecutwfc"] = ecutwfc
    config.system["ecutrho"] = ecutrho

    # 6. K-points grid (reuses shared function)
    config.kpoints.grid = compute_kpoints_grid(
        atoms, kpr=30.0, is_2d=config.is_2d,
    )

    # 7. Pseudo mapping
    config.pseudo_map = {el: get_pseudo_filename(el) for el in species_order}

    # 8. ibrav=0, nat, ntyp
    config.system["ibrav"] = 0
    config.system["nat"] = len(atoms)
    config.system["ntyp"] = len(species_order)

    return config


# ---------------------------------------------------------------------------
# Preset Factory
# ---------------------------------------------------------------------------

_qe_preset_cfg = load_config("qe_presets")
_PRESET_QE: Dict[tuple, Dict[str, Any]] = {}
for _k, _v in _qe_preset_cfg.items():
    _parts = _k.split(":")
    _calc_str, _acc_str = _parts[0], _parts[1]
    _qe_calc = QECalculationType(_calc_str)
    _PRESET_QE[(_qe_calc, AccuracyLevel(_acc_str))] = _v


def get_qe_preset(
    calc_type: QECalculationType = QECalculationType.SCF,
    accuracy: AccuracyLevel = AccuracyLevel.STANDARD,
    atoms: Optional[Atoms] = None,
) -> QEInputConfig:
    """Create a QEInputConfig from preset defaults.

    If ``atoms`` is provided, structure-aware auto-detection is applied.

    Args:
        calc_type: QE calculation type (scf, relax, vc-relax, bands, nscf).
        accuracy: Accuracy level (standard or precise).
        atoms: Optional ASE Atoms for structure-aware configuration.

    Returns:
        Configured QEInputConfig.
    """
    key = (calc_type, accuracy)
    preset = _PRESET_QE.get(key, _PRESET_QE.get(
        (QECalculationType.SCF, accuracy), {}
    ))

    config = QEInputConfig(
        calc_type=calc_type,
        accuracy=accuracy,
        control=dict(preset.get("control", {})),
        system=dict(preset.get("system", {})),
        electrons=dict(preset.get("electrons", {})),
        ions=dict(preset.get("ions", {})),
        cell=dict(preset.get("cell", {})),
    )

    if atoms is not None:
        detect_and_apply_qe_hints(atoms, config)

    return config


# ---------------------------------------------------------------------------
# Band path generation
# ---------------------------------------------------------------------------

# Hardcoded high-symmetry paths for common Bravais lattices.
# Used as fallback when seekpath is not installed.
# Key: ASE Bravais-lattice name → list of (label, crystal_coords) segments
_FALLBACK_KPATHS: Dict[str, List[Tuple[str, List[float]]]] = {
    "FCC":  [("G", [0.0, 0.0, 0.0]), ("X", [0.5, 0.0, 0.5]),
             ("U", [0.625, 0.25, 0.625]), ("K", [0.375, 0.375, 0.75]),
             ("G", [0.0, 0.0, 0.0]), ("L", [0.5, 0.5, 0.5]),
             ("W", [0.5, 0.25, 0.75]), ("X", [0.5, 0.0, 0.5])],
    "BCC":  [("G", [0.0, 0.0, 0.0]), ("H", [0.5, -0.5, 0.5]),
             ("N", [0.0, 0.0, 0.5]), ("G", [0.0, 0.0, 0.0]),
             ("P", [0.25, 0.25, 0.25])],
    "HEX":  [("G", [0.0, 0.0, 0.0]), ("M", [0.5, 0.0, 0.0]),
             ("K", [1.0/3, 1.0/3, 0.0]), ("G", [0.0, 0.0, 0.0]),
             ("A", [0.0, 0.0, 0.5])],
    "ORC":  [("G", [0.0, 0.0, 0.0]), ("X", [0.5, 0.0, 0.0]),
             ("S", [0.5, 0.5, 0.0]), ("G", [0.0, 0.0, 0.0]),
             ("Z", [0.0, 0.0, 0.5])],
    # Simple cubic and default fallback
    "CUB":  [("G", [0.0, 0.0, 0.0]), ("X", [0.5, 0.0, 0.0]),
             ("M", [0.5, 0.5, 0.0]), ("G", [0.0, 0.0, 0.0]),
             ("R", [0.5, 0.5, 0.5])],
    # Simple tetragonal (a=b≠c)
    "TET":  [("G", [0.0, 0.0, 0.0]), ("X", [0.5, 0.0, 0.0]),
             ("M", [0.5, 0.5, 0.0]), ("G", [0.0, 0.0, 0.0]),
             ("Z", [0.0, 0.0, 0.5]), ("R", [0.5, 0.0, 0.5]),
             ("A", [0.5, 0.5, 0.5])],
    # Body-centred tetragonal — BCT1 convention (c/a < 1)
    "BCT":  [("G", [0.0, 0.0, 0.0]), ("M", [-0.5, 0.5, 0.5]),
             ("X", [0.0, 0.0, 0.5]), ("G", [0.0, 0.0, 0.0]),
             ("N", [0.0, 0.5, 0.0])],
    # Rhombohedral (trigonal)
    "RHL":  [("G", [0.0, 0.0, 0.0]), ("L", [0.5, 0.0, 0.0]),
             ("Z", [0.5, 0.5, 0.5]), ("G", [0.0, 0.0, 0.0]),
             ("F", [0.5, 0.5, 0.0])],
}


def get_band_calc_atoms(atoms: Atoms, is_2d: bool = False) -> Optional[Atoms]:
    """Return seekpath's standardized primitive cell for band-structure DFT runs.

    Seekpath returns k-coordinates in the reciprocal frame of its own
    standardized primitive cell.  SCF, bands, and NSCF all share the same
    charge density in QE and therefore **must** use this same primitive cell so
    that ``crystal_b`` k-coordinates are interpreted correctly.

    2D materials are excluded because seekpath cannot model vacuum layers.

    Args:
        atoms: Input structure (any cell; seekpath standardizes internally).
        is_2d: If ``True``, returns ``None`` immediately — 2D materials keep
            their vacuum cell and should not be seekpath-standardized.

    Returns:
        ASE ``Atoms`` of the seekpath primitive cell, or ``None`` if
        ``is_2d=True``, seekpath is unavailable, or standardization fails.
    """
    if is_2d:
        return None
    try:
        import seekpath  # type: ignore[import]

        spg_structure = (
            atoms.cell[:].tolist(),
            atoms.get_scaled_positions().tolist(),
            atoms.get_atomic_numbers().tolist(),
        )
        path_data = seekpath.get_path(spg_structure)
        return Atoms(
            numbers=list(path_data["primitive_types"]),
            scaled_positions=list(path_data["primitive_positions"]),
            cell=path_data["primitive_lattice"],
            pbc=atoms.pbc,
        )
    except ImportError:
        logger.debug("seekpath not available — get_band_calc_atoms returns None")
        return None
    except Exception as exc:
        logger.warning("get_band_calc_atoms: seekpath failed (%s); returning None", exc)
        return None


def generate_band_kpath(
    atoms: Atoms,
    npoints: int = 40,
    is_2d: bool = False,
) -> QEKPointsConfig:
    """Generate a high-symmetry k-path for a band-structure (``crystal_b``) run.

    Three-tier fallback strategy:

    1. **seekpath** (primary): structure-specific, handles all space groups.
       Discontinuities (jumps in the Brillouin zone path, e.g. X→U then K→G)
       are encoded as composite labels ``"X|K"`` at the break point with
       ``npts=1`` so that no spurious band segment is drawn.  The start of
       the new segment is added as an unlabelled entry.
    2. **ASE** ``atoms.cell.bandpath()`` (secondary): covers all 14 Bravais
       lattice types when seekpath is unavailable.  Comma-separated segments
       in the ASE path string are treated as discontinuities.
    3. **Hardcoded table** ``_FALLBACK_KPATHS`` (tertiary): eight common
       lattice types (CUB, FCC, BCC, HEX, ORC, TET, BCT, RHL).

    The ``atoms`` argument **should** already be the seekpath standardized
    primitive cell (obtained via :func:`get_band_calc_atoms`) so that the
    returned crystal-coordinate k-points are consistent with the cell written
    to ``CELL_PARAMETERS`` in pw.in.

    2-D correction:
        If ``is_2d=True``, all k-point z-coordinates are forced to 0.0.

    Args:
        atoms: ASE ``Atoms`` object.
        npoints: k-points per segment (default 40).  Break-point entries and
            the final endpoint use ``npts=1``.
        is_2d: Enforce kz = 0 on all k-points.

    Returns:
        ``QEKPointsConfig`` with ``mode='crystal_b'`` and path data filled in.
    """
    seg_labels: List[str] = []
    seg_coords: List[List[float]] = []
    break_indices: set = set()   # indices where npts must be 1 (segment ends)

    # ------------------------------------------------------------------
    # Tier 1: seekpath
    # ------------------------------------------------------------------
    _tier1_ok = False
    try:
        import seekpath  # type: ignore[import]

        spg_structure = (
            atoms.cell[:].tolist(),
            atoms.get_scaled_positions().tolist(),
            atoms.get_atomic_numbers().tolist(),
        )
        path_data = seekpath.get_path(spg_structure)
        point_coords: Dict[str, List[float]] = path_data["point_coords"]
        path_segments: List[Tuple[str, str]] = path_data["path"]

        prev_end_lbl: Optional[str] = None
        for start_lbl, end_lbl in path_segments:
            if prev_end_lbl is None:
                # First segment — add start point normally
                seg_labels.append(start_lbl)
                seg_coords.append(list(point_coords[start_lbl]))
            elif start_lbl == prev_end_lbl:
                # Continuous path — previous end already appended; skip duplicate
                pass
            else:
                # Discontinuity: path jumps from prev_end_lbl to start_lbl.
                # Mark the previous endpoint as a break (npts=1), combine its
                # label with the new-segment start as "prev|new", then add the
                # new start as an unlabelled k-point (its coords are needed for
                # QE to sample the correct direction in the next segment).
                break_indices.add(len(seg_labels) - 1)
                seg_labels[-1] = f"{prev_end_lbl}|{start_lbl}"
                seg_labels.append("")  # unlabelled — coordinates carry direction
                seg_coords.append(list(point_coords[start_lbl]))

            seg_labels.append(end_lbl)
            seg_coords.append(list(point_coords[end_lbl]))
            prev_end_lbl = end_lbl

        _tier1_ok = True

    except ImportError:
        logger.debug("seekpath not available — trying ASE bandpath (Tier 2)")
    except Exception as exc:
        logger.warning("seekpath failed (%s) — trying ASE bandpath (Tier 2)", exc)

    # ------------------------------------------------------------------
    # Tier 2: ASE atoms.cell.bandpath() — covers all 14 Bravais types
    # ------------------------------------------------------------------
    if not _tier1_ok:
        try:
            bp = atoms.cell.bandpath(npoints=0)
            path_str: str = bp.path           # e.g. "GXMG,RL" — comma = break
            special_pts = bp.special_points   # label -> ndarray([x,y,z])

            prev_last_lbl: Optional[str] = None
            for seg_str in path_str.split(","):
                if not seg_str:
                    continue
                for i, lbl in enumerate(seg_str):
                    if not seg_labels:
                        seg_labels.append(lbl)
                        seg_coords.append(list(special_pts[lbl]))
                    elif i == 0 and prev_last_lbl is not None:
                        # Comma discontinuity: combine last label with new start
                        break_indices.add(len(seg_labels) - 1)
                        seg_labels[-1] = f"{prev_last_lbl}|{lbl}"
                        seg_labels.append("")
                        seg_coords.append(list(special_pts[lbl]))
                    else:
                        seg_labels.append(lbl)
                        seg_coords.append(list(special_pts[lbl]))
                prev_last_lbl = seg_str[-1]

        except Exception as exc2:
            logger.debug("ASE bandpath failed (%s) — using hardcoded table (Tier 3)", exc2)
            # Discard any partial results so Tier 3 receives an empty seg_labels
            seg_labels.clear()
            seg_coords.clear()
            break_indices.clear()

    # ------------------------------------------------------------------
    # Tier 3: hardcoded _FALLBACK_KPATHS table
    # ------------------------------------------------------------------
    if not seg_labels:
        lattice_name = _detect_bravais(atoms)
        path_pts = _FALLBACK_KPATHS.get(lattice_name, _FALLBACK_KPATHS["CUB"])
        for lbl, coords in path_pts:
            seg_labels.append(lbl)
            seg_coords.append(list(coords))

    # ------------------------------------------------------------------
    # 2D correction: set kz = 0 for all points
    # ------------------------------------------------------------------
    if is_2d:
        seg_coords = [[c[0], c[1], 0.0] for c in seg_coords]

    # ------------------------------------------------------------------
    # Build QE crystal_b format
    # Break-point entries get npts=1 (no band drawn to next segment start).
    # Unlabelled entries (new-segment starts after a break) also get npoints.
    # ------------------------------------------------------------------
    kpath_points: List[Tuple[List[float], int]] = []
    kpath_labels: Dict[int, str] = {}
    n = len(seg_coords)
    for i, (coords, lbl) in enumerate(zip(seg_coords, seg_labels)):
        is_last = (i == n - 1)
        is_break = (i in break_indices)
        npts = 1 if (is_last or is_break) else npoints
        kpath_points.append((coords, npts))
        if lbl:  # skip unlabelled entries (new-segment starts after a break)
            kpath_labels[i] = lbl

    return QEKPointsConfig(
        mode="crystal_b",
        kpath_points=kpath_points,
        kpath_labels=kpath_labels,
    )


def _detect_bravais(atoms: Atoms) -> str:
    """Detect the Bravais lattice type for the fallback k-path table.

    Returns one of the keys in ``_FALLBACK_KPATHS``.
    """
    try:
        bravais = atoms.cell.get_bravais_lattice()
        name = bravais.name  # e.g. "FCC", "BCC", "HEX", "ORC", "CUB", ...
        if name in _FALLBACK_KPATHS:
            return name
    except Exception:
        pass
    return "CUB"  # safe default: G→X→M→G→R path
