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

from shalom.backends.base import (
    DFTBackend, DFTResult, BandStructureData, DOSData, MDTrajectoryData,
)
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
    get_band_calc_atoms,
    resolve_pseudo_dir,
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
    create_runner,
)
from shalom.backends.slurm import SlurmConfig, SlurmRunner
from shalom.backends.lammps import LAMMPSBackend
from shalom.backends.lammps_config import (
    LAMMPSInputConfig,
    detect_force_field,
    get_lammps_preset,
    resolve_potential_dir as resolve_lammps_potential_dir,
)

__all__ = [
    "DFTBackend", "DFTResult", "BandStructureData", "DOSData", "MDTrajectoryData",
    "VASPBackend", "QEBackend", "LAMMPSBackend", "get_backend",
    "CalculationType", "AccuracyLevel", "VASPInputConfig", "KPointsConfig", "get_preset",
    "QECalculationType", "QEInputConfig", "QEKPointsConfig", "get_qe_preset",
    "generate_band_kpath", "get_band_calc_atoms", "resolve_pseudo_dir",
    "parse_xml_bands", "parse_dos_file", "find_xml_path",
    "extract_fermi_energy", "compute_nbnd", "HA_TO_EV", "QE_XML_NS",
    "ErrorRecoveryEngine", "VASPError", "Correction",
    "QEErrorRecoveryEngine", "QEError", "QECorrection", "compute_safe_dt",
    "ExecutionConfig", "ExecutionResult", "ExecutionRunner", "execute_with_recovery",
    "create_runner",
    "SlurmConfig", "SlurmRunner",
    "LAMMPSInputConfig", "detect_force_field", "get_lammps_preset",
    "resolve_lammps_potential_dir",
]


def get_backend(
    name: str = "vasp",
) -> Union[VASPBackend, QEBackend, LAMMPSBackend]:
    """Factory function to instantiate a simulation backend by name.

    Args:
        name: Backend identifier. Supported values: ``"vasp"``, ``"qe"``,
            ``"lammps"``.

    Returns:
        An instance implementing the DFTBackend protocol.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    if name == "vasp":
        return VASPBackend()
    elif name == "qe":
        return QEBackend()
    elif name == "lammps":
        return LAMMPSBackend()
    raise ValueError(
        f"Unknown backend: '{name}'. Supported backends: ['vasp', 'qe', 'lammps']"
    )
