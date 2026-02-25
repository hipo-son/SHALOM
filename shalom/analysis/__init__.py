"""Post-DFT analysis tools wrapping mature external libraries.

All sub-modules use optional dependencies with lazy imports.
Install analysis dependencies with::

    pip install shalom[analysis]   # elastic (pymatgen)
    pip install shalom[phonon]     # phonon (phonopy)

Available modules
-----------------
- **elastic**: Mechanical properties from elastic tensors (pymatgen).
- **phonon**: Phonon band structure, DOS, thermal properties (phonopy).

Usage::

    from shalom.analysis import analyze_elastic_tensor, is_elastic_available

    if is_elastic_available():
        result = analyze_elastic_tensor(voigt_6x6)
        print(f"Bulk modulus: {result.bulk_modulus_vrh:.1f} GPa")
"""

from shalom.analysis._base import ElasticResult, PhononResult
from shalom.analysis.elastic import (
    analyze_elastic_tensor,
    analyze_stress_strain,
    is_elastic_available,
)
from shalom.analysis.phonon import (
    analyze_phonon,
    analyze_phonon_from_force_constants,
    generate_phonon_displacements,
    is_phonopy_available,
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
]
