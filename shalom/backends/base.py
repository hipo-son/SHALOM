from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ase import Atoms


def compute_forces_max(forces: List[List[float]]) -> float:
    """Compute maximum per-atom force magnitude from an Nx3 forces list.

    Args:
        forces: List of [fx, fy, fz] per atom (eV/Angstrom).

    Returns:
        Maximum force magnitude across all atoms.
    """
    return max(math.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2) for f in forces)


@dataclass
class DFTResult:
    """Unified result container for any DFT backend.

    Attributes:
        energy: Final total energy in eV.
        forces_max: Maximum force component in eV/Angstrom.
        is_converged: Whether the SCF/ionic cycle converged.
        bandgap: Band gap in eV (None if not computed or metallic).
        magnetization: Total magnetization in Bohr magneton.
        entropy_per_atom: Electronic entropy T*S per atom in eV (for SIGMA validation).
        stress_tensor: Voigt stress [xx, yy, zz, xy, yz, xz] in kBar.
        forces: Per-atom forces as N x 3 list (eV/Angstrom).
        ionic_energies: Energy at each ionic step (for false convergence detection).
        ionic_forces_max: Max force at each ionic step.
        correction_history: Error recovery actions applied (JSON-serializable).
        raw: Backend-specific parsed data for advanced inspection.
    """

    energy: Optional[float] = None
    forces_max: Optional[float] = None
    is_converged: bool = False
    bandgap: Optional[float] = None
    magnetization: Optional[float] = None
    entropy_per_atom: Optional[float] = None
    stress_tensor: Optional[List[float]] = None
    forces: Optional[List[List[float]]] = None
    ionic_energies: Optional[List[float]] = None
    ionic_forces_max: Optional[List[float]] = None
    error_log: Optional[str] = None  # Compressed log — not for scan_for_errors().
    quality_warnings: List[str] = field(default_factory=list)
    correction_history: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BandStructureData:
    """Backend-agnostic band structure representation.

    Populated by the QE parser (qe_parser.py) or a future VASP parser.
    Stored in ``DFTResult.raw["band_structure"]`` by convention.

    Attributes:
        eigenvalues: Shape ``(nkpts, nbands)`` array of band energies in eV.
        kpoint_coords: Shape ``(nkpts, 3)`` array of k-point coordinates in
            crystal (fractional) units.
        kpath_distances: Shape ``(nkpts,)`` cumulative distances along the
            k-path in reciprocal-space (Cartesian, 1/Angstrom).
        fermi_energy: Fermi energy in eV. Prefer the value from a dense-mesh
            NSCF calculation over the SCF value (especially for metals).
        high_sym_labels: Map from k-point *index* to high-symmetry label,
            e.g. ``{0: "G", 40: "X", 80: "L"}``. Integer keys are used to
            avoid IEEE-754 floating-point comparison issues.
        spin_up: Shape ``(nkpts, nbands)`` spin-up eigenvalues in eV (only
            populated for spin-polarised calculations).
        spin_down: Shape ``(nkpts, nbands)`` spin-down eigenvalues in eV.
        is_spin_polarized: Whether the calculation used ``nspin=2``.
        nbands: Number of bands.
        nkpts: Number of k-points along the path.
        projections: Optional orbital-projection data for future VASP PROCAR
            support.  Shape and key conventions are backend-specific.
        source: Backend that produced this data (``"qe"`` or ``"vasp"``).
    """

    eigenvalues: Any  # np.ndarray (nkpts, nbands), eV
    kpoint_coords: Any  # np.ndarray (nkpts, 3), crystal coords
    kpath_distances: Any  # np.ndarray (nkpts,), 1/Angstrom
    fermi_energy: float = 0.0
    high_sym_labels: Dict[int, str] = field(default_factory=dict)
    spin_up: Optional[Any] = None  # np.ndarray (nkpts, nbands), eV
    spin_down: Optional[Any] = None
    is_spin_polarized: bool = False
    nbands: int = 0
    nkpts: int = 0
    projections: Optional[Dict[Tuple, Any]] = None
    source: str = "unknown"


@dataclass
class DOSData:
    """Backend-agnostic density of states representation.

    Produced by ``dos.x`` (QE) or a future VASP DOSCAR parser.
    Stored in ``DFTResult.raw["dos"]`` by convention.

    Attributes:
        energies: Shape ``(npts,)`` energy grid in eV.
        dos: Shape ``(npts,)`` total DOS in states/eV.
        integrated_dos: Shape ``(npts,)`` integrated DOS (number of states).
        fermi_energy: Fermi energy in eV.
        dos_up: Spin-up partial DOS in states/eV (spin-polarised only).
        dos_down: Spin-down partial DOS in states/eV.
        is_spin_polarized: Whether the calculation used ``nspin=2``.
        dos_orbital: Optional l-decomposed or orbital-projected DOS for future
            VASP DOSCAR support. Keys are ``(atom_idx, orbital_label)``.
        source: Backend that produced this data (``"qe"`` or ``"vasp"``).
    """

    energies: Any  # np.ndarray (npts,), eV
    dos: Any  # np.ndarray (npts,), states/eV
    integrated_dos: Any  # np.ndarray (npts,)
    fermi_energy: float = 0.0
    dos_up: Optional[Any] = None
    dos_down: Optional[Any] = None
    is_spin_polarized: bool = False
    dos_orbital: Optional[Dict[Tuple, Any]] = None
    source: str = "unknown"


@dataclass
class MDTrajectoryData:
    """Backend-agnostic MD trajectory container.

    Produced by VASP AIMD (XDATCAR), LAMMPS (dump file), or QE MD.
    Stored in ``DFTResult.raw["md_trajectory"]`` by convention.

    Attributes:
        positions: Shape ``(n_frames, n_atoms, 3)`` atomic positions in Angstrom.
        energies: Shape ``(n_frames,)`` total energies in eV.
        temperatures: Shape ``(n_frames,)`` instantaneous temperatures in K.
        times: Shape ``(n_frames,)`` simulation times in fs.
        species: Element symbols per atom (length ``n_atoms``).
        cell_vectors: Shape ``(n_frames, 3, 3)`` or ``(3, 3)`` cell vectors in Angstrom.
        pressures: Shape ``(n_frames,)`` pressures in kBar.
        forces: Shape ``(n_frames, n_atoms, 3)`` forces in eV/Angstrom.
        velocities: Shape ``(n_frames, n_atoms, 3)`` velocities in Angstrom/fs.
        kinetic_energies: Shape ``(n_frames,)`` kinetic energies in eV.
        potential_energies: Shape ``(n_frames,)`` potential energies in eV.
        ensemble: Thermodynamic ensemble (``"NVE"``, ``"NVT"``, ``"NPT"``).
        timestep_fs: MD timestep in femtoseconds.
        source: Backend that produced this data (``"vasp"``, ``"lammps"``, ``"qe"``).
    """

    positions: Any  # np.ndarray (n_frames, n_atoms, 3), Å
    energies: Any  # np.ndarray (n_frames,), eV
    temperatures: Any  # np.ndarray (n_frames,), K
    times: Any  # np.ndarray (n_frames,), fs
    species: List[str]  # (n_atoms,)
    cell_vectors: Any = None  # (n_frames, 3, 3) or (3, 3), Å
    pressures: Optional[Any] = None  # (n_frames,), kBar
    forces: Optional[Any] = None  # (n_frames, n_atoms, 3), eV/Å
    velocities: Optional[Any] = None  # (n_frames, n_atoms, 3), Å/fs
    kinetic_energies: Optional[Any] = None  # (n_frames,), eV
    potential_energies: Optional[Any] = None  # (n_frames,), eV
    ensemble: str = "NVT"
    timestep_fs: float = 1.0
    source: str = "unknown"

    @property
    def n_frames(self) -> int:
        """Number of trajectory frames."""
        return len(self.energies) if self.energies is not None else 0

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return len(self.species)


@runtime_checkable
class DFTBackend(Protocol):
    """Protocol defining the interface every DFT backend must implement.

    Implementations must provide ``name``, ``write_input``, and ``parse_output``.
    Use ``get_backend()`` from ``shalom.backends`` to instantiate backends by name.
    """

    name: str

    def write_input(self, atoms: Atoms, directory: str, **params: Any) -> str:
        """Write DFT input files for the given structure.

        Args:
            atoms: ASE Atoms object representing the structure.
            directory: Target directory for the input files.
            **params: Backend-specific parameters (e.g. INCAR settings for VASP).

        Returns:
            The directory path where input files were written.
        """
        ...

    def parse_output(self, directory: str) -> DFTResult:
        """Parse DFT output files and return a unified result.

        Args:
            directory: Directory containing the DFT output files.

        Returns:
            DFTResult with energy, forces, and convergence information.
        """
        ...
