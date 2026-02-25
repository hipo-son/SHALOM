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


@dataclass
class ElectronicResult:
    """Electronic structure properties from band structure / DOS analysis.

    Attributes:
        bandgap_eV: Fundamental band gap in eV (None if not computed).
        is_direct: True if VBM and CBM occur at the same k-point.
        is_metal: True if the system is metallic (no gap or VBM >= CBM).
        vbm_energy: Valence band maximum energy in eV.
        cbm_energy: Conduction band minimum energy in eV.
        vbm_k_index: K-point index of the VBM.
        cbm_k_index: K-point index of the CBM.
        effective_mass_electron: Electron effective mass in units of m_e.
        effective_mass_hole: Hole effective mass in units of m_e.
        dos_at_fermi: DOS at the Fermi level in states/eV.
        n_occupied_bands: Number of occupied bands at the VBM k-point.
        raw: Backend-specific or intermediate data for advanced inspection.
        metadata: Arbitrary key-value pairs for provenance tracking.
    """

    bandgap_eV: Optional[float] = None
    is_direct: bool = False
    is_metal: bool = False
    vbm_energy: Optional[float] = None
    cbm_energy: Optional[float] = None
    vbm_k_index: Optional[int] = None
    cbm_k_index: Optional[int] = None
    effective_mass_electron: Optional[float] = None
    effective_mass_hole: Optional[float] = None
    dos_at_fermi: Optional[float] = None
    n_occupied_bands: int = 0
    raw: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XRDResult:
    """X-ray diffraction pattern result.

    Attributes:
        two_theta: Peak positions in degrees (n_peaks,).
        intensities: Relative intensities normalised to 0-100 (n_peaks,).
        hkl_indices: Miller indices for each peak as (h, k, l) tuples.
        d_spacings: Inter-planar d-spacings in Angstrom (n_peaks,).
        wavelength: Radiation source label (e.g. ``"CuKa"``).
        wavelength_angstrom: Wavelength in Angstrom.
        n_peaks: Number of diffraction peaks.
        raw: pymatgen ``DiffractionPattern`` object for advanced inspection.
        metadata: Arbitrary key-value pairs for provenance tracking.
    """

    two_theta: Any = None
    intensities: Any = None
    hkl_indices: List[Tuple[int, int, int]] = field(default_factory=list)
    d_spacings: Any = None
    wavelength: str = "CuKa"
    wavelength_angstrom: Optional[float] = None
    n_peaks: int = 0
    raw: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymmetryResult:
    """Crystal symmetry analysis result from spglib.

    Attributes:
        space_group_number: International space group number (1--230).
        space_group_symbol: International (Hermann-Mauguin) symbol, e.g. "Fd-3m".
        point_group: Point group symbol, e.g. "m-3m".
        crystal_system: One of triclinic / monoclinic / orthorhombic /
            tetragonal / trigonal / hexagonal / cubic.
        lattice_type: Bravais lattice letter (P, I, F, C, R, A).
        hall_symbol: Hall symbol string.
        wyckoff_letters: Per-atom Wyckoff letter list (length = n_atoms).
        equivalent_atoms: Per-atom symmetry-equivalence mapping.
        n_operations: Number of symmetry operations in the space group.
        is_primitive: True if the input cell is already a primitive cell.
        raw: Full spglib symmetry dataset dict for advanced inspection.
        metadata: Arbitrary key-value pairs for provenance tracking.
    """

    space_group_number: int = 0
    space_group_symbol: str = ""
    point_group: str = ""
    crystal_system: str = ""
    lattice_type: str = ""
    hall_symbol: str = ""
    wyckoff_letters: List[str] = field(default_factory=list)
    equivalent_atoms: List[int] = field(default_factory=list)
    n_operations: int = 0
    is_primitive: bool = False
    raw: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MagneticResult:
    """Magnetic and charge analysis result from DFT output.

    Attributes:
        total_magnetization: Total cell magnetization in Bohr magneton.
        is_magnetic: True if ``|total_magnetization| > 0.01``.
        is_spin_polarized: True if the DFT calculation reported magnetization.
        site_magnetizations: Per-atom magnetic moments in Bohr magneton.
        site_charges: Per-atom Lowdin total charges (if available).
        lowdin_charges: Full Lowdin decomposition from ``projwfc.x``.
        magnetic_elements: Sorted list of elements with significant moments.
        dominant_moment_element: Element with the largest average moment.
        raw: Reserved for backend-specific objects.
        metadata: Arbitrary key-value pairs for provenance tracking.
    """

    total_magnetization: Optional[float] = None
    is_magnetic: bool = False
    is_spin_polarized: bool = False
    site_magnetizations: Optional[List[float]] = None
    site_charges: Optional[List[float]] = None
    lowdin_charges: Optional[Dict[str, Any]] = None
    magnetic_elements: List[str] = field(default_factory=list)
    dominant_moment_element: Optional[str] = None
    raw: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
