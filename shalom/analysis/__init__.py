"""Post-DFT analysis tools wrapping mature external libraries.

All sub-modules use optional dependencies with lazy imports.
Install analysis dependencies with::

    pip install shalom[analysis]   # elastic (pymatgen), XRD
    pip install shalom[phonon]     # phonon (phonopy)
    pip install shalom[symmetry]   # symmetry (spglib)

Available modules
-----------------
- **elastic**: Mechanical properties from elastic tensors (pymatgen).
- **phonon**: Phonon band structure, DOS, thermal properties (phonopy).
- **electronic**: Band gap, effective mass, metallicity (numpy only).
- **xrd**: X-ray diffraction patterns (pymatgen).
- **symmetry**: Space group, Wyckoff positions, crystal system (spglib).
- **magnetic**: Site magnetization and Lowdin charges (QE output parsing).
- **md**: MD trajectory analysis — RDF, MSD, VACF, diffusion (numpy only).

Usage::

    from shalom.analysis import analyze_elastic_tensor, is_elastic_available

    if is_elastic_available():
        result = analyze_elastic_tensor(voigt_6x6)
        print(f"Bulk modulus: {result.bulk_modulus_vrh:.1f} GPa")
"""

from shalom.analysis._base import (
    ElasticResult,
    ElectronicResult,
    MDResult,
    MagneticResult,
    PhononResult,
    SymmetryResult,
    XRDResult,
    save_result_json,
)
from shalom.analysis.elastic import (
    analyze_elastic_tensor,
    analyze_stress_strain,
    is_elastic_available,
)
from shalom.analysis.electronic import (
    analyze_band_structure,
    is_electronic_available,
)
from shalom.analysis.magnetic import (
    analyze_magnetism,
    extract_lowdin_charges,
    extract_site_magnetization,
    extract_total_magnetization,
    is_magnetic_available,
)
from shalom.analysis.phonon import (
    analyze_phonon,
    analyze_phonon_from_force_constants,
    generate_phonon_displacements,
    is_phonopy_available,
)
from shalom.analysis.symmetry import (
    analyze_symmetry,
    is_spglib_available,
)
from shalom.analysis.md import (
    analyze_md_trajectory,
    compute_msd,
    compute_rdf,
    compute_vacf,
    compute_diffusion_coefficient,
    detect_equilibration,
)
from shalom.analysis.xrd import (
    calculate_xrd,
    is_xrd_available,
)

__all__ = [
    # Elastic
    "ElasticResult",
    "analyze_elastic_tensor",
    "analyze_stress_strain",
    "is_elastic_available",
    # Phonon
    "PhononResult",
    "analyze_phonon",
    "analyze_phonon_from_force_constants",
    "generate_phonon_displacements",
    "is_phonopy_available",
    # Electronic
    "ElectronicResult",
    "analyze_band_structure",
    "is_electronic_available",
    # XRD
    "XRDResult",
    "calculate_xrd",
    "is_xrd_available",
    # Symmetry
    "SymmetryResult",
    "analyze_symmetry",
    "is_spglib_available",
    # MD
    "MDResult",
    "analyze_md_trajectory",
    "compute_rdf",
    "compute_msd",
    "compute_vacf",
    "compute_diffusion_coefficient",
    "detect_equilibration",
    # Magnetic
    "MagneticResult",
    "analyze_magnetism",
    "extract_site_magnetization",
    "extract_total_magnetization",
    "extract_lowdin_charges",
    "is_magnetic_available",
    # Serialization
    "save_result_json",
]
