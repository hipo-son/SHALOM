from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ase import Atoms


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
    correction_history: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


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
