from typing import Any

from ase import Atoms

from shalom.backends.base import DFTResult


class QEBackend:
    """Quantum ESPRESSO DFT backend (stub).

    This backend will be fully implemented in Phase 3. It will handle
    ``pw.x`` input file generation and XML output parsing.
    """

    name: str = "qe"

    def write_input(self, atoms: Atoms, directory: str, **params: Any) -> str:
        """Write Quantum ESPRESSO input files.

        Not yet implemented. Will generate ``pw.x`` input files from the
        ASE Atoms object in Phase 3.

        Raises:
            NotImplementedError: Always, until Phase 3 implementation.
        """
        raise NotImplementedError(
            "Quantum ESPRESSO backend will be implemented in Phase 3. "
            "Use 'vasp' backend for now."
        )

    def parse_output(self, directory: str) -> DFTResult:
        """Parse Quantum ESPRESSO XML output.

        Not yet implemented. Will parse ``pw.x`` XML output files in Phase 3.

        Raises:
            NotImplementedError: Always, until Phase 3 implementation.
        """
        raise NotImplementedError(
            "Quantum ESPRESSO backend will be implemented in Phase 3. "
            "Use 'vasp' backend for now."
        )
