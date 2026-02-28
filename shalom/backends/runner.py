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
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ase import Atoms

from shalom.backends._physics import DEFAULT_TIMEOUT_SECONDS
from shalom.backends.base import DFTBackend, DFTResult
from shalom.backends.qe_error_recovery import QEErrorRecoveryEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runner constants
# ---------------------------------------------------------------------------

MIN_RECOVERY_TIMEOUT: int = 7200                  # 2h floor for error recovery
STDERR_TAIL_CHARS: int = 2000                     # max stderr capture characters

# Regex for Windows drive letter paths (e.g. C:/Users/... or C:\Users\...)
_WIN_PATH_RE = re.compile(r"[A-Za-z]:[/\\]")


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
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    env_overrides: Dict[str, str] = field(default_factory=dict)
    wsl: bool = False  # Run via WSL on Windows (wsl -e pw.x)


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

def _windows_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path to its WSL ``/mnt/`` equivalent.

    ``C:/Users/Foo`` → ``/mnt/c/Users/Foo``
    ``D:\\data\\pseudo`` → ``/mnt/d/data/pseudo``

    If the path is already a Unix path (starts with ``/``), return it unchanged.
    """
    if not win_path or win_path.startswith("/"):
        return win_path
    # Normalise backslashes
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]  # starts with /
        return f"/mnt/{drive}{rest}"
    return p


def _patch_input_paths_for_wsl(directory: str, input_file: str = "pw.in") -> None:
    """Rewrite Windows paths inside a QE input file to WSL ``/mnt/`` paths.

    Patches ``pseudo_dir``, ``outdir``, and any other path-valued keys that
    contain Windows drive letters (e.g. ``C:/Users/...``) so that pw.x or
    dos.x running inside WSL can resolve them.  Also converts the
    ``cwd``-relative ``'./tmp'`` form to an absolute WSL path so that
    chained calculations (bands/nscf reusing SCF charge density) work.

    Works with both ``pw.in`` and ``dos.in`` input files.
    """
    input_path = os.path.join(directory, input_file)
    if not os.path.isfile(input_path):
        return

    with open(input_path, "r") as f:
        content = f.read()

    changed = False
    for key in ("pseudo_dir", "outdir"):
        # Match  key = 'value'  or  key = "value"
        pattern = re.compile(
            rf"({key}\s*=\s*)(['\"])(.+?)\2", re.IGNORECASE,
        )
        match = pattern.search(content)
        if match:
            old_val = match.group(3)
            if _WIN_PATH_RE.match(old_val):
                new_val = _windows_to_wsl_path(old_val)
                content = content[:match.start(3)] + new_val + content[match.end(3):]
                changed = True
                logger.debug("WSL path patch: %s = '%s' → '%s'", key, old_val, new_val)
            elif old_val.startswith("./"):
                # Convert relative paths to absolute WSL paths so that
                # chained calculations (bands/nscf reusing SCF charge
                # density) resolve correctly regardless of WSL CWD.
                abs_dir_wsl = _windows_to_wsl_path(os.path.abspath(directory))
                rel_part = old_val[2:]  # strip "./"
                new_val = f"{abs_dir_wsl}/{rel_part}"
                content = content[:match.start(3)] + new_val + content[match.end(3):]
                changed = True
                logger.debug(
                    "WSL path patch (relative): %s = '%s' → '%s'",
                    key, old_val, new_val,
                )

    if changed:
        with open(input_path, "w") as f:
            f.write(content)


def detect_wsl_executable(command: str = "pw.x") -> bool:
    """Check whether *command* is available inside WSL (Windows only).

    Accepts either a bare command name (``"pw.x"``) or a full WSL path
    (``"/opt/micromamba/envs/qe/bin/pw.x"``).  Returns ``False`` immediately
    on non-Windows platforms or when WSL is not installed.
    """
    if sys.platform != "win32":
        return False
    try:
        # Full path → test -x; bare name → which
        if command.startswith("/"):
            check_cmd = ["wsl", "-e", "test", "-x", command]
        else:
            check_cmd = ["wsl", "-e", "which", command]
        result = subprocess.run(check_cmd, capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


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
            When ``config.wsl`` is True, the command is prefixed with
            ``["wsl", "-e"]`` so that the executable runs inside WSL.
        """
        if config.wsl:
            # Use "wsl -e bash -c ..." to avoid MSYS2 path mangling
            # (Git Bash rewrites /opt/... to C:/Program Files/Git/opt/...)
            if config.nprocs <= 1:
                shell_cmd = config.command
            else:
                shell_cmd = (
                    f"{config.mpi_command} --allow-run-as-root"
                    f" -np {config.nprocs} {config.command}"
                )
            return ["wsl", "-e", "bash", "-c", shell_cmd]

        if config.nprocs <= 1:
            return [config.command]
        cmd = [config.mpi_command]
        # Open MPI ≥5 refuses to run as root without explicit opt-in.
        if hasattr(os, "getuid") and os.getuid() == 0:
            cmd.append("--allow-run-as-root")
        cmd.extend(["-np", str(config.nprocs), config.command])
        return cmd

    def validate_prerequisites(self, directory: str) -> List[str]:
        """Check that all prerequisites are met before execution.

        Args:
            directory: Calculation directory.

        Returns:
            List of error messages (empty if all OK).
        """
        errors: List[str] = []

        # Check executable — WSL mode checks inside WSL
        if self.config.wsl:
            if not detect_wsl_executable(self.config.command):
                errors.append(
                    f"Executable '{self.config.command}' not found in WSL.\n"
                    f"  Install: wsl -e sudo apt install quantum-espresso"
                )
        elif shutil.which(self.config.command) is None:
            errors.append(
                f"Executable '{self.config.command}' not found on PATH.\n"
                f"  Install: sudo apt install quantum-espresso  (Ubuntu/Debian)\n"
                f"           conda install -c conda-forge qe     (conda)\n"
                f"  Windows: Run from inside WSL2, not native Windows.\n"
                f"  Check:   python -m shalom setup-qe"
            )

        # Check MPI if parallel
        if self.config.nprocs > 1:
            if self.config.wsl:
                if not detect_wsl_executable(self.config.mpi_command):
                    errors.append(
                        f"MPI launcher '{self.config.mpi_command}' not found in WSL.\n"
                        f"  Install: wsl -e sudo apt install openmpi-bin"
                    )
            elif shutil.which(self.config.mpi_command) is None:
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
            # In WSL mode, pseudo_dir is a WSL path — skip Windows-side check
            if not self.config.wsl:
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
        # WSL path patching: convert Windows paths in input file so pw.x/dos.x can find them
        if self.config.wsl:
            _patch_input_paths_for_wsl(directory, self.config.input_file)

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
        stderr_tail = stderr_text[-STDERR_TAIL_CHARS:] if stderr_text else None

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
            MIN_RECOVERY_TIMEOUT, original_timeout // (attempt + 2),
        )

    # Failed after all attempts
    final_result = exec_result if "exec_result" in dir() else ExecutionResult(
        success=False, error_message="No execution attempted",
    )
    if dft_result is not None:
        dft_result.quality_warnings.extend(quality_warnings)
        dft_result.correction_history = correction_history
    return final_result, dft_result, correction_history


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_runner(
    exec_config: Optional[ExecutionConfig] = None,
    slurm_config: Optional[Any] = None,
) -> Any:
    """Create a local or Slurm runner based on configuration.

    Args:
        exec_config: Execution configuration (command, nprocs, etc.).
        slurm_config: If provided (a :class:`~shalom.backends.slurm.SlurmConfig`
            instance), returns a :class:`~shalom.backends.slurm.SlurmRunner`.

    Returns:
        ``ExecutionRunner`` for local execution or ``SlurmRunner`` for Slurm.
    """
    if slurm_config is not None:
        from shalom.backends.slurm import SlurmRunner
        return SlurmRunner(exec_config=exec_config, slurm_config=slurm_config)
    return ExecutionRunner(config=exec_config)
