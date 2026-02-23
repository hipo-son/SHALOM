"""DFT workflow orchestration for SHALOM.

Provides:

- :class:`~shalom.workflows.standard.StandardWorkflow` — 5-step sequential
  QE pipeline (vc-relax → scf → bands → nscf → dos.x → plots).
- :class:`~shalom.workflows.convergence.CutoffConvergence` — ecutwfc sweep.
- :class:`~shalom.workflows.convergence.KpointConvergence` — k-point density sweep.
- :class:`~shalom.workflows.base.ConvergenceWorkflow` — abstract base class.
- :class:`~shalom.workflows.base.ConvergenceResult` — single-run result dataclass.
- :class:`~shalom.workflows.base.ConvergenceTestResult` — full-sweep result dataclass.
"""

from shalom.workflows.base import (
    ConvergenceWorkflow,
    ConvergenceResult,
    ConvergenceTestResult,
)
from shalom.workflows.convergence import CutoffConvergence, KpointConvergence
from shalom.workflows.standard import StandardWorkflow

__all__ = [
    "ConvergenceWorkflow",
    "ConvergenceResult",
    "ConvergenceTestResult",
    "CutoffConvergence",
    "KpointConvergence",
    "StandardWorkflow",
]
