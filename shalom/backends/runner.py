"""DFT execution runner with error recovery loop.

Provides subprocess-based DFT execution (pw.x) with:
- Cross-platform timeout via threading.Timer
- SIGINT/KeyboardInterrupt cleanup (no zombie processes)
- Prerequisite validation (executable, input file, pseudopotentials)
- Progressive error recovery via QEErrorRecoveryEngine
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ase import Atoms

from shalom.backends.base import DFTBackend, DFTResult
from shalom.backends.qe_error_recovery import QEErrorRecoveryEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """Configuration for DFT execution."""
    command: str = "pw.x"
    input_file: str = "pw.in"
    output_file: str = "pw.out"
    nprocs: int = 1
    mpi_command: str = "mpirun"
    timeout_seconds: int = 86400
    env_overrides: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of a DFT execution."""
    success: bool
    return_code: Optional[int] = None
    wall_time_seconds: float = 0.0
    timed_out: bool = False
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    stderr_tail: Optional[str] = None


# ---------------------------------------------------------------------------
# ExecutionRunner
# ---------------------------------------------------------------------------

class ExecutionRunner:
    """Runs DFT calculations via subprocess.

    Uses Popen with file-based stdin/stdout (no shell=True) and
    threading.Timer for cross-platform timeout.

    Usage::

        runner = ExecutionRunner(ExecutionConfig(nprocs=4))
        result = runner.run("/path/to/calc_dir")
        if result.success:
            print(f"Done in {result.wall_time_seconds:.1f}s")
    """

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        self.config = config or ExecutionConfig()
        self._process: Optional[subprocess.Popen] = None

    @staticmethod
    def build_command(config: ExecutionConfig) -> List[str]:
        """Build the command list for subprocess.

        Args:
            config: Execution configuration.

        Returns:
            Command as list of strings (e.g., ["mpirun", "-np", "4", "pw.x"]).
        """
        if config.nprocs <= 1:
            return [config.command]
        return [config.mpi_command, "-np", str(config.nprocs), config.command]

    def validate_prerequisites(self, directory: str) -> List[str]:
        """Check that all prerequisites are met before execution.

        Args:
            directory: Calculation directory.

        Returns:
            List of error messages (empty if all OK).
        """
        errors: List[str] = []

        # Check executable
        if shutil.which(self.config.command) is None:
            errors.append(
                f"Executable '{self.config.command}' not found on PATH.\n"
                f"  Install: sudo apt install quantum-espresso  (Ubuntu/Debian)\n"
                f"           conda install -c conda-forge qe     (conda)\n"
                f"  Windows: Run from inside WSL2, not native Windows.\n"
                f"  Check:   python -m shalom setup-qe"
            )

        # Check MPI if parallel
        if self.config.nprocs > 1:
            if shutil.which(self.config.mpi_command) is None:
                errors.append(
                    f"MPI launcher '{self.config.mpi_command}' not found on PATH.\n"
                    f"  Install: sudo apt install openmpi-bin  (Ubuntu/Debian)\n"
                    f"  Check:   python -m shalom setup-qe"
                )

        # Check input file
        input_path = os.path.join(directory, self.config.input_file)
        if not os.path.exists(input_path):
            errors.append(
                f"Input file not found: {input_path}"
            )
        else:
            # Check pseudo_dir and pseudopotential files
            pseudo_errors = self._check_pseudopotentials(input_path)
            errors.extend(pseudo_errors)

        return errors

    def _check_pseudopotentials(self, input_path: str) -> List[str]:
        """Parse pseudo_dir from pw.in and check UPF files exist."""
        errors: List[str] = []
        try:
            with open(input_path, "r") as f:
                content = f.read()
            # Extract pseudo_dir
            match = re.search(r"pseudo_dir\s*=\s*['\"]([^'\"]+)['\"]", content)
            if not match:
                return errors  # No pseudo_dir — not a QE input
            pseudo_dir = match.group(1)
            if not os.path.isabs(pseudo_dir):
                pseudo_dir = os.path.join(
                    os.path.dirname(input_path), pseudo_dir
                )
            if not os.path.isdir(pseudo_dir):
                errors.append(f"pseudo_dir not found: {pseudo_dir}")
                return errors  # Can't check files if dir missing

            # Parse ATOMIC_SPECIES card for required UPF files
            species_match = re.search(
                r"ATOMIC_SPECIES\s*\n((?:\s+\S+\s+[\d.eE+-]+\s+\S+.*\n)+)",
                content,
            )
            if species_match:
                for line in species_match.group(1).strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        element = parts[0]
                        upf_file = parts[2]
                        upf_path = os.path.join(pseudo_dir, upf_file)
                        if not os.path.exists(upf_path):
                            # Case-insensitive fallback
                            found = False
                            try:
                                for entry in os.listdir(pseudo_dir):
                                    if entry.lower() == upf_file.lower():
                                        found = True
                                        break
                            except OSError:
                                pass
                            if not found:
                                errors.append(
                                    f"Pseudopotential not found: {upf_path}\n"
                                    f"  Download: python -m shalom setup-qe "
                                    f"--elements {element} --download"
                                )
        except OSError:
            pass  # Input file read error handled by caller
        return errors

    def run(self, directory: str) -> ExecutionResult:
        """Execute DFT calculation in the given directory.

        Args:
            directory: Directory containing pw.in.

        Returns:
            ExecutionResult with success status, timing, and error info.
        """
        cmd = self.build_command(self.config)
        input_path = os.path.join(directory, self.config.input_file)
        output_path = os.path.join(directory, self.config.output_file)

        logger.info("Running: %s (cwd=%s)", " ".join(cmd), directory)

        t0 = time.monotonic()
        timed_out = False
        return_code = None
        stderr_data = b""

        try:
            with open(input_path, "r") as stdin_f, \
                 open(output_path, "w") as stdout_f:
                proc = subprocess.Popen(
                    cmd,
                    stdin=stdin_f,
                    stdout=stdout_f,
                    stderr=subprocess.PIPE,
                    cwd=directory,
                    env={**os.environ, **self.config.env_overrides}
                    if self.config.env_overrides else None,
                )
                self._process = proc

                # Cross-platform timeout via threading.Timer
                timer = threading.Timer(
                    self.config.timeout_seconds, self._kill_process, [proc],
                )
                timer.start()
                try:
                    _, stderr_data = proc.communicate()
                except KeyboardInterrupt:
                    proc.kill()
                    proc.wait()
                    raise
                finally:
                    timer.cancel()

                return_code = proc.returncode
                timed_out = return_code == -9 or (
                    return_code is not None and return_code < 0
                    and not stderr_data
                )
        except FileNotFoundError as e:
            wall_time = time.monotonic() - t0
            return ExecutionResult(
                success=False,
                wall_time_seconds=wall_time,
                error_message=f"Command not found: {e}",
            )
        except OSError as e:
            wall_time = time.monotonic() - t0
            return ExecutionResult(
                success=False,
                wall_time_seconds=wall_time,
                error_message=str(e),
            )
        finally:
            self._process = None

        wall_time = time.monotonic() - t0
        stderr_text = stderr_data.decode("utf-8", errors="replace").strip()
        stderr_tail = stderr_text[-2000:] if stderr_text else None

        # Determine success: return code 0 is required
        success = return_code == 0 and not timed_out

        error_msg = None
        if timed_out:
            error_msg = (
                f"Execution timed out after {self.config.timeout_seconds}s"
            )
        elif return_code != 0:
            error_msg = f"Process exited with code {return_code}"

        return ExecutionResult(
            success=success,
            return_code=return_code,
            wall_time_seconds=wall_time,
            timed_out=timed_out,
            output_file=output_path if os.path.exists(output_path) else None,
            error_message=error_msg,
            stderr_tail=stderr_tail,
        )

    @staticmethod
    def _kill_process(proc: subprocess.Popen) -> None:
        """Kill a process (used by timeout timer)."""
        try:
            proc.kill()
        except OSError:
            pass

    def cancel(self) -> None:
        """Cancel a running execution."""
        proc = self._process
        if proc is not None:
            try:
                proc.kill()
                proc.wait()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# execute_with_recovery
# ---------------------------------------------------------------------------

def execute_with_recovery(
    backend: DFTBackend,
    runner: ExecutionRunner,
    recovery_engine: QEErrorRecoveryEngine,
    directory: str,
    config: Any,
    atoms: Optional[Atoms] = None,
    max_retries: int = 3,
) -> Tuple[ExecutionResult, Optional[DFTResult], List[Dict[str, Any]]]:
    """Execute DFT with automatic error recovery loop.

    Runs pw.x, parses output, detects errors, applies progressive corrections,
    and retries. Implements timeout-first check, fatal-first scan, S_MATRIX
    diagnostic branching, quality warning accumulation, and wall-clock decay.

    Args:
        backend: DFT backend (for write_input/parse_output).
        runner: ExecutionRunner instance.
        recovery_engine: QEErrorRecoveryEngine for error detection/correction.
        directory: Calculation directory containing pw.in.
        config: QEInputConfig (modified in-place during recovery).
        atoms: ASE Atoms (for S_MATRIX diagnostic and light atom dt).
        max_retries: Maximum number of recovery attempts.

    Returns:
        Tuple of (ExecutionResult, DFTResult or None, correction_history).
    """
    quality_warnings: List[str] = []
    correction_history: List[Dict[str, Any]] = []
    original_timeout = runner.config.timeout_seconds

    for attempt in range(max_retries + 1):
        logger.info(
            "Execution attempt %d/%d (dir=%s)",
            attempt + 1, max_retries + 1, directory,
        )

        # Run
        exec_result = runner.run(directory)

        # TIMEOUT CHECK FIRST: do not parse truncated output
        if exec_result.timed_out:
            logger.error("Execution timed out on attempt %d", attempt + 1)
            exec_result.error_message = (
                f"Timed out after {runner.config.timeout_seconds}s "
                f"(attempt {attempt + 1}/{max_retries + 1})"
            )
            return exec_result, None, correction_history

        # Parse output
        dft_result: Optional[DFTResult] = None
        try:
            dft_result = backend.parse_output(directory)
        except Exception as e:
            logger.warning("Failed to parse output: %s", e)

        # Converged → success
        if dft_result is not None and dft_result.is_converged:
            dft_result.quality_warnings.extend(quality_warnings)
            dft_result.correction_history = correction_history
            exec_result.success = True
            return exec_result, dft_result, correction_history

        # No more retries
        if attempt >= max_retries:
            logger.warning("Max retries (%d) exhausted", max_retries)
            break

        # Read pw.out for error scanning
        output_path = os.path.join(directory, runner.config.output_file)
        try:
            with open(output_path, "r") as f:
                output_text = f.read()
        except OSError:
            logger.error("Cannot read output file: %s", output_path)
            break

        # Scan for errors
        errors = recovery_engine.scan_for_errors(output_text)

        if not errors:
            logger.warning("No known errors detected; cannot recover")
            break

        # FATAL-FIRST: check all errors for fatal before applying corrections
        fatal_errors = [e for e in errors if e.severity.value == "fatal"]
        if fatal_errors:
            logger.error(
                "Fatal error(s) detected: %s",
                [e.error_type for e in fatal_errors],
            )
            # Record in history
            for fe in fatal_errors:
                recovery_engine.get_correction(fe)  # records in history
            break

        # Get correction for the primary (first) error
        primary_error = errors[0]
        correction = recovery_engine.get_correction(primary_error, atoms=atoms)

        if correction is None:
            logger.warning(
                "Correction strategies exhausted for %s",
                primary_error.error_type,
            )
            break

        # Backup old output
        backup_path = f"{output_path}.retry_{attempt}"
        try:
            shutil.copy2(output_path, backup_path)
            logger.info("Backed up %s → %s", output_path, backup_path)
        except OSError as e:
            logger.warning("Failed to backup output: %s", e)

        # Apply correction
        warnings = recovery_engine.apply_correction_to_config(config, correction)
        quality_warnings.extend(warnings)

        # Record in correction_history
        correction_history.append({
            "attempt": attempt + 1,
            "error_type": primary_error.error_type,
            "step": correction.step,
            "namelist_updates": correction.namelist_updates,
            "quality_warnings": warnings,
        })

        # Rewrite input
        try:
            backend.write_input(atoms, directory, config=config)  # type: ignore[arg-type]
            logger.info("Rewrote input with correction step %d", correction.step)
        except Exception as e:
            logger.error("Failed to rewrite input: %s", e)
            break

        # Wall-clock decay: reduce timeout for subsequent attempts
        runner.config.timeout_seconds = max(
            7200, original_timeout // (attempt + 2),
        )

    # Failed after all attempts
    final_result = exec_result if "exec_result" in dir() else ExecutionResult(
        success=False, error_message="No execution attempted",
    )
    if dft_result is not None:
        dft_result.quality_warnings.extend(quality_warnings)
        dft_result.correction_history = correction_history
    return final_result, dft_result, correction_history
