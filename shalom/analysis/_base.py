"""Result dataclasses for post-DFT analysis modules.

Each analysis type (elastic, phonon, etc.) has its own result dataclass here.
These are analogous to ``BandStructureData`` / ``DOSData`` in ``backends.base``
but for derived physical properties rather than raw DFT outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ElasticResult:
    """Elastic properties derived from a Voigt elastic tensor.

    All moduli are in GPa.  The ``raw`` field holds the pymatgen
    ``ElasticTensor`` object for advanced inspection.

    Attributes:
        elastic_tensor: 6x6 Voigt elastic tensor in GPa (np.ndarray).
        bulk_modulus_vrh: Voigt-Reuss-Hill bulk modulus (GPa).
        shear_modulus_vrh: VRH shear modulus (GPa).
        youngs_modulus: Young's modulus (GPa).
        poisson_ratio: Homogeneous Poisson's ratio (dimensionless).
        is_stable: True if elastic tensor satisfies Born stability criteria
            (all eigenvalues of the Voigt matrix are positive).
        stability_violations: Human-readable descriptions of violated criteria.
        universal_anisotropy: Universal elastic anisotropy index A_U.
        compliance_tensor: 6x6 compliance tensor in 1/GPa (np.ndarray).
        raw: pymatgen ``ElasticTensor`` object (None if pymatgen unavailable).
        metadata: Arbitrary key-value pairs for provenance tracking.
    """

    elastic_tensor: Any  # np.ndarray (6, 6), GPa
    bulk_modulus_vrh: Optional[float] = None
    shear_modulus_vrh: Optional[float] = None
    youngs_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    is_stable: bool = False
    stability_violations: List[str] = field(default_factory=list)
    universal_anisotropy: Optional[float] = None
    compliance_tensor: Optional[Any] = None  # np.ndarray (6, 6), 1/GPa
    raw: Optional[Any] = None  # pymatgen ElasticTensor
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhononResult:
    """Phonon properties derived from force sets or force constants.

    All frequencies are in THz.  Thermal properties use standard conventions
    (kJ/mol for free energy, J/K/mol for entropy and heat capacity).

    Attributes:
        band_qpoints: (n_qpoints, 3) q-point coordinates in fractional.
        band_distances: (n_qpoints,) cumulative distances along q-path.
        band_frequencies: (n_qpoints, n_branches) phonon frequencies in THz.
        band_labels: Map from q-point index to high-symmetry label.
        dos_frequencies: (n_dos,) frequency grid in THz.
        dos_density: (n_dos,) phonon DOS in states/THz.
        thermal_temperatures: (n_T,) temperature grid in K.
        thermal_free_energy: (n_T,) Helmholtz free energy in kJ/mol.
        thermal_entropy: (n_T,) vibrational entropy in J/K/mol.
        thermal_cv: (n_T,) heat capacity at constant volume in J/K/mol.
        min_frequency_THz: Minimum phonon frequency (negative → imaginary).
        is_stable: True if no significant imaginary frequencies
            (tolerance: -0.1 THz for acoustic near Gamma).
        imaginary_modes: List of (q-index, branch-index, frequency) for
            modes below the stability threshold.
        n_atoms: Number of atoms in the primitive cell.
        n_branches: Number of phonon branches (3 * n_atoms).
        force_constants: (N, N, 3, 3) force constant matrix in eV/Ang^2.
        raw: The phonopy ``Phonopy`` object for advanced inspection.
        metadata: Arbitrary key-value pairs for provenance tracking.
    """

    band_qpoints: Optional[Any] = None
    band_distances: Optional[Any] = None
    band_frequencies: Optional[Any] = None
    band_labels: Dict[int, str] = field(default_factory=dict)
    dos_frequencies: Optional[Any] = None
    dos_density: Optional[Any] = None
    thermal_temperatures: Optional[Any] = None
    thermal_free_energy: Optional[Any] = None
    thermal_entropy: Optional[Any] = None
    thermal_cv: Optional[Any] = None
    min_frequency_THz: Optional[float] = None
    is_stable: bool = False
    imaginary_modes: List[Tuple[int, int, float]] = field(default_factory=list)
    n_atoms: int = 0
    n_branches: int = 0
    force_constants: Optional[Any] = None
    raw: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
