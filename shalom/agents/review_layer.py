import os
from typing import Any, Dict, List, Optional

from shalom._config_loader import load_prompt
from shalom.backends.base import DFTBackend, DFTResult
from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import ReviewResult


class ReviewAgent:
    """Review Layer agent for evaluating simulation results.

    Parses and analyzes DFT output (VASP OUTCAR, QE XML, etc.) to determine
    whether the target objective was achieved. Performs quantitative physics
    validation and generates specific feedback for the Design Layer.

    Supports any backend implementing the ``DFTBackend`` protocol via
    ``review_with_backend()``. The convenience method ``review()`` defaults
    to VASP for backward compatibility.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.system_prompt = load_prompt("review_agent")

    def parse_outcar(self, filepath: str) -> Dict[str, Any]:
        """Simple rule-based OUTCAR parser.

        For production use, consider ``pymatgen.io.vasp.outputs.Outcar``.

        Args:
            filepath: Path to the VASP OUTCAR file.

        Returns:
            Parsed data dict with keys: energy, is_converged, forces_max.

        Raises:
            FileNotFoundError: If the OUTCAR file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"OUTCAR file not found: {filepath}")

        energy = None
        is_converged = False

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "free  energy   TOTEN" in line:
                    try:
                        energy = float(line.split()[-2])
                    except (IndexError, ValueError):
                        pass
                if "General timing and accounting informations for this job" in line:
                    is_converged = True

        return {
            "energy": energy,
            "is_converged": is_converged,
            "forces_max": None,
        }

    def review_with_backend(
        self,
        target_objective: str,
        directory: str,
        backend: DFTBackend,
        correction_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ReviewResult:
        """Evaluate DFT results using any backend implementing the DFTBackend protocol.

        Args:
            target_objective: The original natural language objective.
            directory: Directory containing the DFT output files.
            backend: A DFTBackend instance (e.g. VASPBackend, QEBackend).
            correction_history: Error recovery actions applied during calculation.

        Returns:
            ReviewResult with success status, metrics, and feedback.
        """
        dft_result = backend.parse_output(directory)
        if correction_history:
            dft_result.correction_history = correction_history
        return self._evaluate(target_objective, dft_result)

    def review(self, target_objective: str, outcar_path: str) -> ReviewResult:
        """Read and evaluate a VASP OUTCAR file against the target objective.

        Convenience method that parses the given OUTCAR file directly.
        For other backends, use ``review_with_backend()`` with a directory.

        Args:
            target_objective: The original natural language objective.
            outcar_path: Path to the VASP OUTCAR file.

        Returns:
            ReviewResult with success status, metrics, and feedback.
        """
        parsed = self.parse_outcar(outcar_path)
        dft_result = DFTResult(
            energy=parsed["energy"],
            forces_max=parsed["forces_max"],
            is_converged=parsed["is_converged"],
            raw=parsed,
        )
        return self._evaluate(target_objective, dft_result)

    def _evaluate(self, target_objective: str, dft_result: DFTResult) -> ReviewResult:
        """Send parsed DFT results + physics checks to the LLM for evaluation.

        Args:
            target_objective: The original natural language objective.
            dft_result: Unified DFT result from any backend.

        Returns:
            ReviewResult with success status, metrics, and feedback.
        """
        physics_warnings = self._run_physics_checks(dft_result)

        # Build enriched prompt with all available data
        lines = [
            f"Target Objective: {target_objective}",
            "",
            "Parsed DFT Data:",
            f"- Converged: {dft_result.is_converged}",
            f"- Final Energy: {dft_result.energy} eV",
            f"- Max Force: {dft_result.forces_max} eV/A",
        ]
        if dft_result.bandgap is not None:
            lines.append(f"- Band Gap: {dft_result.bandgap} eV")
        if dft_result.magnetization is not None:
            lines.append(f"- Magnetization: {dft_result.magnetization} muB")
        if dft_result.entropy_per_atom is not None:
            lines.append(f"- Entropy T*S/atom: {dft_result.entropy_per_atom:.6f} eV")

        if dft_result.correction_history:
            lines.append("")
            lines.append("Error Correction History:")
            for entry in dft_result.correction_history:
                lines.append(f"  - {entry.get('error_type', 'unknown')}: {entry.get('action', '')}")

        if physics_warnings:
            lines.append("")
            lines.append("Physics Validation Warnings:")
            for w in physics_warnings:
                lines.append(f"  - {w}")

        lines.append("")
        lines.append("Please analyze these results and output the ReviewResult structure.")

        user_prompt = "\n".join(lines)

        return self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=ReviewResult,
        )

    @staticmethod
    def _run_physics_checks(dft_result: DFTResult) -> List[str]:
        """Run quantitative physics validation checks.

        Args:
            dft_result: Parsed DFT result.

        Returns:
            List of warning strings (empty if all checks pass).
        """
        warnings: List[str] = []

        # Force convergence check
        if dft_result.forces_max is not None and dft_result.forces_max > 0.02:
            warnings.append(
                f"Max force ({dft_result.forces_max:.4f} eV/A) exceeds 0.02 eV/A threshold."
            )

        # Entropy check (SIGMA over-smearing)
        if dft_result.entropy_per_atom is not None:
            if abs(dft_result.entropy_per_atom) > 0.001:  # 1 meV/atom
                warnings.append(
                    f"Entropy T*S/atom ({dft_result.entropy_per_atom:.6f} eV) exceeds 1 meV/atom. "
                    "Consider reducing SIGMA."
                )

        # BRMIX in correction history
        if dft_result.correction_history:
            brmix_count = sum(
                1 for h in dft_result.correction_history
                if h.get("error_type") == "BRMIX"
            )
            if brmix_count > 0:
                warnings.append(
                    f"BRMIX error corrected {brmix_count} time(s). "
                    "Indicates charge density sloshing — verify geometry."
                )

        return warnings
