"""Abstract base class for SHALOM convergence workflow tests.

``ConvergenceWorkflow`` provides shared infrastructure (parameter sweep loop,
per-atom convergence detection, result dataclasses) for concrete subclasses
``CutoffConvergence`` and ``KpointConvergence`` defined in ``convergence.py``.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceResult:
    """Result of a single parameter-value DFT run inside a sweep.

    Attributes:
        parameter_value: The parameter value tested (e.g. ``ecutwfc`` in Ry or
            k-point resolution in Bohr^-1).
        energy: Total energy in eV, or ``None`` if the run failed.
        is_converged: ``True`` if ``pw.x`` reached self-consistency (SCF
            converged).
        calc_dir: Absolute path to the calculation directory.
        error_message: Human-readable error description if the run failed.
    """
    parameter_value: float
    energy: Optional[float]
    is_converged: bool
    calc_dir: str = ""
    error_message: str = ""


@dataclass
class ConvergenceTestResult:
    """Aggregated result of a full convergence sweep.

    Attributes:
        parameter_name: Name of the swept parameter (e.g. ``"ecutwfc"``).
        results: List of ``ConvergenceResult`` for each value tested, in the
            order they were run.
        converged_value: The lowest parameter value that satisfies the
            convergence threshold (per-atom energy criterion).  ``None`` if
            convergence was not achieved within the tested range.
        reference_value: The highest parameter value tested, used as the
            reference energy for threshold comparison.
    """
    parameter_name: str
    results: List[ConvergenceResult] = field(default_factory=list)
    converged_value: Optional[float] = None
    reference_value: Optional[float] = None

    @property
    def converged_results(self) -> List[ConvergenceResult]:
        """Only SCF-converged results (excludes failed runs)."""
        return [r for r in self.results if r.is_converged and r.energy is not None]

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.converged_value is not None:
            return (
                f"{self.parameter_name}: converged at {self.converged_value} "
                f"(ref={self.reference_value})"
            )
        return f"{self.parameter_name}: NOT converged in tested range"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ConvergenceWorkflow(ABC):
    """Abstract base for ecutwfc and k-point convergence sweeps.

    Subclasses must implement :meth:`_run_single` and :meth:`_parameter_name`.

    Args:
        atoms: ASE ``Atoms`` object for the structure under test.
        output_dir: Root directory; sub-directories per value are created
            inside it automatically.
        pseudo_dir: Override for pseudopotential directory.  Uses the SHALOM
            default (``~/pseudopotentials``) when ``None``.
        nprocs: Number of MPI processes passed to ``pw.x``.
        timeout: Maximum wall time per single run (seconds).
        mpi_command: MPI launcher command (e.g. ``"mpirun"``).
        accuracy: QE preset accuracy level — ``"standard"`` or ``"precise"``.
        parallel: If ``True``, run all parameter values concurrently using
            ``multiprocessing.Pool``.  Use with care; each process spawns its
            own ``pw.x``.
        threshold_per_atom: Energy convergence criterion in eV/atom.  Default
            is 1 meV/atom (1e-3 eV/atom), matching SSSP verification standards.
    """

    _parameter_name: str = "unknown"

    def __init__(
        self,
        atoms: "Atoms",
        output_dir: str,
        pseudo_dir: Optional[str] = None,
        nprocs: int = 1,
        timeout: int = 3600,
        mpi_command: str = "mpirun",
        accuracy: str = "standard",
        parallel: bool = False,
        threshold_per_atom: float = 1e-3,
        wsl: bool = False,
        slurm_config: Optional[Any] = None,
    ) -> None:
        self.atoms = atoms
        self.output_dir = os.path.abspath(output_dir)
        from shalom.backends.qe_config import resolve_pseudo_dir

        self.pseudo_dir = resolve_pseudo_dir(pseudo_dir)
        self.nprocs = nprocs
        self.timeout = timeout
        self.mpi_command = mpi_command
        self.accuracy = accuracy
        self.parallel = parallel
        self.threshold_per_atom = threshold_per_atom
        self.wsl = wsl
        self.slurm_config = slurm_config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _values(self) -> List[float]:
        """Sorted list of parameter values to sweep (ascending)."""
        ...

    @abstractmethod
    def _run_single(self, param_value: float, calc_dir: str) -> ConvergenceResult:
        """Run a single SCF calculation with the given parameter value.

        Args:
            param_value: The parameter value to use (ecutwfc in Ry, or k-point
                resolution in Bohr^-1).
            calc_dir: Absolute path to the directory for this run.

        Returns:
            A ``ConvergenceResult`` populated with the run outcome.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _find_converged_value(
        self,
        results: List[ConvergenceResult],
    ) -> Optional[float]:
        """Return the lowest parameter value satisfying the convergence threshold.

        Compares each result against the highest-parameter reference energy on
        a per-atom basis::

            |E_n - E_ref| / n_atoms < threshold_per_atom

        Args:
            results: List of ``ConvergenceResult`` objects from the sweep.

        Returns:
            Converged parameter value, or ``None`` if no value meets the
            criterion.
        """
        n_atoms = len(self.atoms)
        converged = [r for r in results if r.is_converged and r.energy is not None]
        if len(converged) < 2:
            logger.warning(
                "Need ≥2 converged runs to determine convergence; got %d.", len(converged)
            )
            return None

        ref_energy: float = converged[-1].energy  # type: ignore[assignment]
        for r in converged[:-1]:
            diff_per_atom = abs(r.energy - ref_energy) / n_atoms  # type: ignore[operator]
            if diff_per_atom < self.threshold_per_atom:
                logger.info(
                    "Convergence at %s=%.4g  (ΔE/atom=%.3g eV)",
                    self._parameter_name, r.parameter_value, diff_per_atom,
                )
                return r.parameter_value

        logger.warning(
            "No convergence within threshold %.3g eV/atom in tested range.",
            self.threshold_per_atom,
        )
        return None

    def _make_calc_dir(self, param_value: float) -> str:
        """Create and return the directory for a single parameter run."""
        tag = f"{param_value:.4g}".replace(".", "p")
        calc_dir = os.path.join(self.output_dir, f"{self._parameter_name}_{tag}")
        os.makedirs(calc_dir, exist_ok=True)
        return calc_dir

    def _run_sequential(self) -> List[ConvergenceResult]:
        """Run all parameter values one at a time."""
        results: List[ConvergenceResult] = []
        for val in sorted(self._values):
            calc_dir = self._make_calc_dir(val)
            logger.info("Running %s = %.4g in %s", self._parameter_name, val, calc_dir)
            result = self._run_single(val, calc_dir)
            results.append(result)
        return results

    def _run_parallel(self) -> List[ConvergenceResult]:
        """Run all parameter values concurrently using multiprocessing."""
        import multiprocessing as mp

        items = [(val, self._make_calc_dir(val)) for val in sorted(self._values)]
        with mp.Pool(processes=len(items)) as pool:
            results = pool.starmap(self._run_single, items)
        return results

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceTestResult:
        """Execute the full convergence sweep and determine the converged value.

        Returns:
            ``ConvergenceTestResult`` containing all individual results and the
            determined ``converged_value``.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            "Starting %s convergence sweep: %s",
            self._parameter_name, sorted(self._values),
        )

        if self.parallel:
            results = self._run_parallel()
        else:
            results = self._run_sequential()

        converged_value = self._find_converged_value(results)
        reference_value = max(self._values) if self._values else None

        return ConvergenceTestResult(
            parameter_name=self._parameter_name,
            results=sorted(results, key=lambda r: r.parameter_value),
            converged_value=converged_value,
            reference_value=reference_value,
        )

    def plot(
        self,
        result: ConvergenceTestResult,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot energy vs. parameter value convergence curve.

        Args:
            result: Output of :meth:`run`.
            output_path: Save path.  If ``None``, defaults to
                ``{output_dir}/{parameter_name}_convergence.png``.

        Returns:
            Absolute path to the saved plot, or ``None`` if matplotlib is not
            installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; convergence plot skipped.")
            return None

        if output_path is None:
            output_path = os.path.join(
                self.output_dir, f"{self._parameter_name}_convergence.png"
            )

        converged_results = result.converged_results
        if not converged_results:
            logger.warning("No converged results to plot.")
            return None

        n_atoms = len(self.atoms)
        xs = [r.parameter_value for r in converged_results]
        ref_energy: float = converged_results[-1].energy  # type: ignore[assignment]
        ys = [abs(r.energy - ref_energy) / n_atoms * 1000  # meV/atom
              for r in converged_results]  # type: ignore[operator]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.semilogy(xs, ys, "o-", color="royalblue", lw=1.5, ms=6)
        ax.axhline(self.threshold_per_atom * 1000, color="grey",
                   lw=0.8, linestyle="--", alpha=0.7,
                   label=f"Threshold {self.threshold_per_atom * 1000:.1f} meV/atom")

        if result.converged_value is not None:
            ax.axvline(result.converged_value, color="crimson",
                       lw=1.0, linestyle=":", alpha=0.8,
                       label=f"Converged: {result.converged_value:.4g}")

        ax.set_xlabel(self._parameter_name, fontsize=12)
        ax.set_ylabel("ΔE/atom (meV)", fontsize=12)
        ax.set_title(f"{self._parameter_name} Convergence", fontsize=13)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return os.path.abspath(output_path)
