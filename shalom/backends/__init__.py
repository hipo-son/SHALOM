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

from shalom.backends.base import DFTBackend, DFTResult
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

__all__ = [
    "DFTBackend", "DFTResult", "VASPBackend", "QEBackend", "get_backend",
    "CalculationType", "AccuracyLevel", "VASPInputConfig", "KPointsConfig", "get_preset",
    "ErrorRecoveryEngine", "VASPError", "Correction",
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
