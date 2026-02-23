"""DFT backend abstraction layer.

Provides a unified interface for different DFT solvers (VASP, Quantum ESPRESSO)
through the ``DFTBackend`` protocol. Use ``get_backend()`` to instantiate a
backend by name.

Example::

    from shalom.backends import get_backend

    backend = get_backend("vasp")
    backend.write_input(atoms, "./calc_dir")
    result = backend.parse_output("./calc_dir")
    print(result.energy, result.is_converged)
"""

from typing import Union

from shalom.backends.base import DFTBackend, DFTResult, BandStructureData, DOSData
from shalom.backends.vasp import VASPBackend
from shalom.backends.qe import QEBackend
from shalom.backends.vasp_config import (
    CalculationType,
    AccuracyLevel,
    VASPInputConfig,
    KPointsConfig,
    get_preset,
)
from shalom.backends.error_recovery import ErrorRecoveryEngine, VASPError, Correction
from shalom.backends.qe_error_recovery import (
    QEErrorRecoveryEngine, QEError, QECorrection, compute_safe_dt,
)
from shalom.backends.qe_config import (
    QECalculationType,
    QEInputConfig,
    QEKPointsConfig,
    get_qe_preset,
    generate_band_kpath,
)
from shalom.backends.qe_parser import (
    parse_xml_bands,
    parse_dos_file,
    find_xml_path,
    extract_fermi_energy,
    compute_nbnd,
    HA_TO_EV,
    QE_XML_NS,
)
from shalom.backends.runner import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionRunner,
    execute_with_recovery,
)

__all__ = [
    "DFTBackend", "DFTResult", "BandStructureData", "DOSData",
    "VASPBackend", "QEBackend", "get_backend",
    "CalculationType", "AccuracyLevel", "VASPInputConfig", "KPointsConfig", "get_preset",
    "QECalculationType", "QEInputConfig", "QEKPointsConfig", "get_qe_preset",
    "generate_band_kpath",
    "parse_xml_bands", "parse_dos_file", "find_xml_path",
    "extract_fermi_energy", "compute_nbnd", "HA_TO_EV", "QE_XML_NS",
    "ErrorRecoveryEngine", "VASPError", "Correction",
    "QEErrorRecoveryEngine", "QEError", "QECorrection", "compute_safe_dt",
    "ExecutionConfig", "ExecutionResult", "ExecutionRunner", "execute_with_recovery",
]


def get_backend(name: str = "vasp") -> Union[VASPBackend, QEBackend]:
    """Factory function to instantiate a DFT backend by name.

    Args:
        name: Backend identifier. Supported values: "vasp", "qe".

    Returns:
        An instance implementing the DFTBackend protocol.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    if name == "vasp":
        return VASPBackend()
    elif name == "qe":
        return QEBackend()
    raise ValueError(f"Unknown backend: '{name}'. Supported backends: ['vasp', 'qe']")
