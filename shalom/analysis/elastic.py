"""Elastic properties analysis using pymatgen.

Wraps ``pymatgen.analysis.elasticity`` to derive mechanical properties
(bulk modulus, shear modulus, Young's modulus, Poisson ratio, stability)
from a 6x6 Voigt elastic tensor or stress-strain data.

Requires pymatgen::

    pip install shalom[analysis]

Example::

    from shalom.analysis import analyze_elastic_tensor

    # Si cubic elastic constants (GPa)
    si_tensor = [
        [165.7, 63.9, 63.9,  0.0,  0.0,  0.0],
        [ 63.9, 165.7, 63.9,  0.0,  0.0,  0.0],
        [ 63.9,  63.9, 165.7,  0.0,  0.0,  0.0],
        [  0.0,   0.0,  0.0, 79.6,  0.0,  0.0],
        [  0.0,   0.0,  0.0,  0.0, 79.6,  0.0],
        [  0.0,   0.0,  0.0,  0.0,  0.0, 79.6],
    ]
    result = analyze_elastic_tensor(si_tensor)
    print(f"Bulk modulus: {result.bulk_modulus_vrh:.1f} GPa")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from shalom.analysis._base import ElasticResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard — follows mp_client.py pattern
# ---------------------------------------------------------------------------

try:
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pymatgen.analysis.elasticity.strain import Strain
    from pymatgen.analysis.elasticity.stress import Stress

    _ELASTIC_AVAILABLE = True
except ImportError:
    _ELASTIC_AVAILABLE = False


def is_elastic_available() -> bool:
    """Check if pymatgen elasticity module is installed."""
    return _ELASTIC_AVAILABLE


def _ensure_elastic_available() -> None:
    """Raise ImportError if pymatgen elasticity is not available."""
    if not _ELASTIC_AVAILABLE:
        raise ImportError(
            "Elastic analysis requires pymatgen. "
            "Install with: pip install shalom[analysis]"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PA_TO_GPA = 1e-9


def analyze_elastic_tensor(
    voigt_6x6: List[List[float]],
) -> "ElasticResult":
    """Analyze a 6x6 Voigt elastic tensor and derive mechanical properties.

    Args:
        voigt_6x6: 6x6 elastic tensor in GPa (Voigt notation).
            Rows/columns follow the standard Voigt order:
            xx, yy, zz, yz, xz, xy.

    Returns:
        :class:`~shalom.analysis._base.ElasticResult` with all derived
        mechanical properties.

    Raises:
        ImportError: If pymatgen is not installed.
        ValueError: If the tensor shape is not 6x6.
    """
    _ensure_elastic_available()

    import numpy as np

    from shalom.analysis._base import ElasticResult

    c_ij = np.array(voigt_6x6, dtype=float)
    if c_ij.shape != (6, 6):
        raise ValueError(
            f"Elastic tensor must be 6x6, got shape {c_ij.shape}"
        )

    et = ElasticTensor.from_voigt(c_ij)

    # VRH averages
    bulk = float(et.k_vrh)
    shear = float(et.g_vrh)

    # Young's modulus — pymatgen returns Pa, convert to GPa
    youngs: Optional[float] = None
    if et.y_mod is not None:
        youngs = float(et.y_mod) * _PA_TO_GPA

    # Poisson's ratio
    poisson: Optional[float] = None
    if et.homogeneous_poisson is not None:
        poisson = float(et.homogeneous_poisson)

    # Universal anisotropy
    anisotropy: Optional[float] = None
    try:
        anisotropy = float(et.universal_anisotropy)
    except Exception:
        pass

    # Born stability: all eigenvalues of the Voigt matrix must be positive
    eigenvalues = np.linalg.eigvalsh(c_ij)
    violations: List[str] = []
    neg_eigs = eigenvalues[eigenvalues <= 0]
    if len(neg_eigs) > 0:
        violations.append(
            f"Non-positive eigenvalue(s): "
            f"{[round(float(v), 4) for v in neg_eigs]}"
        )

    # Compliance tensor
    compliance = None
    try:
        compliance = et.compliance_tensor.voigt
    except Exception:
        pass

    return ElasticResult(
        elastic_tensor=c_ij,
        bulk_modulus_vrh=bulk,
        shear_modulus_vrh=shear,
        youngs_modulus=youngs,
        poisson_ratio=poisson,
        is_stable=len(violations) == 0,
        stability_violations=violations,
        universal_anisotropy=anisotropy,
        compliance_tensor=compliance,
        raw=et,
    )


def analyze_stress_strain(
    stresses: List[List[float]],
    strains: List[List[float]],
    stress_unit: str = "GPa",
) -> "ElasticResult":
    """Derive elastic tensor from stress-strain pairs.

    Typical use case: after VASP ``IBRION=6`` or finite-displacement elastic
    calculations.

    Args:
        stresses: List of 3x3 stress tensors (each a list of 3 lists of 3
            floats). Units determined by ``stress_unit``.
        strains: List of 3x3 strain tensors (each a list of 3 lists of 3
            floats). Dimensionless (engineering strain).
        stress_unit: Unit of stress input. ``"GPa"`` (default) or ``"kBar"``.
            kBar values are converted to GPa (1 kBar = 0.1 GPa).

    Returns:
        :class:`~shalom.analysis._base.ElasticResult` with the fitted elastic
        tensor and derived properties.

    Raises:
        ImportError: If pymatgen is not installed.
        ValueError: If stresses and strains have different lengths or wrong shapes.
    """
    _ensure_elastic_available()

    import numpy as np

    if len(stresses) != len(strains):
        raise ValueError(
            f"stresses ({len(stresses)}) and strains ({len(strains)}) "
            f"must have the same length"
        )
    if len(stresses) == 0:
        raise ValueError("At least one stress-strain pair is required")

    kbar_to_gpa = 0.1

    stress_objs = []
    for s in stresses:
        arr = np.array(s, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError(
                f"Each stress tensor must be 3x3, got shape {arr.shape}"
            )
        if stress_unit == "kBar":
            arr = arr * kbar_to_gpa
        stress_objs.append(Stress(arr))

    strain_objs = []
    for s in strains:
        arr = np.array(s, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError(
                f"Each strain tensor must be 3x3, got shape {arr.shape}"
            )
        strain_objs.append(Strain(arr))

    et = ElasticTensor.from_pseudoinverse(strain_objs, stress_objs)
    return analyze_elastic_tensor(et.voigt.tolist())
