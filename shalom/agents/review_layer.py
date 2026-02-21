import os
from typing import Any, Dict

from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import ReviewResult


class ReviewAgent:
    """Review Layer agent for evaluating simulation results.

    Parses and analyzes simulation output (e.g. VASP OUTCAR) to determine
    whether the target objective was achieved. On failure, generates specific
    chemical/physical feedback for the Design Layer's next iteration.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.system_prompt = """[v1.0.0]
        You are the "Review Agent".
        You compare the parsed key data (energy, max forces, convergence status, etc.)
        from VASP OUTCAR results against the user's original Target Objective.

        [Evaluation Guidelines]
        1. Check whether the simulation steps converged normally.
        2. Determine if the physical properties aligned with the target objective
           (energy stability, etc.).
        3. If the simulation failed or needs improvement, provide specific
           chemical/physical feedback (feedback_for_design) so the Design Layer
           can select better materials on the next iteration.
        """

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
            "forces_max": 0.05,  # Placeholder — use a proper parser in production
        }

    def review(self, target_objective: str, outcar_path: str) -> ReviewResult:
        """Read and evaluate a VASP OUTCAR file against the target objective.

        Args:
            target_objective: The original natural language objective.
            outcar_path: Path to the VASP OUTCAR file.

        Returns:
            ReviewResult with success status, metrics, and feedback.
        """
        parsed_data = self.parse_outcar(outcar_path)

        user_prompt = (
            f"Target Objective: {target_objective}\n\n"
            f"Parsed OUTCAR Data:\n"
            f"- Converged: {parsed_data['is_converged']}\n"
            f"- Final Energy: {parsed_data['energy']} eV\n"
            f"- Max Force: {parsed_data['forces_max']} eV/A\n\n"
            "Please analyze these results and output the ReviewResult structure."
        )

        return self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=ReviewResult,
        )
