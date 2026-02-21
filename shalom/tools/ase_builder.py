import os
from typing import Any, Dict, Optional, Tuple

from ase import Atoms
from ase.build import bulk, surface
from ase.io import write


class ASEBuilder:
    """
    A collection of ASE-related helper tools to be called by the Geometry Generator agent in the Simulation Layer.
    Encourages the LLM to assemble these tools rather than writing the entire ASE syntax directly.
    """

    @staticmethod
    def construct_bulk(symbol: str, crystalstructure: str, a: Optional[float] = None) -> Atoms:
        """
        Constructs a basic Bulk structure.

        Args:
            symbol (str): Element symbol (e.g., 'Cu').
            crystalstructure (str): Crystal structure ('fcc', 'bcc', 'hcp', 'sc', etc.).
            a (float, optional): Lattice constant. Uses ASE default if not provided.

        Returns:
            Atoms: The generated structure.
        """
        if a is None:
            return bulk(symbol, crystalstructure=crystalstructure)
        return bulk(symbol, crystalstructure=crystalstructure, a=a)

    @staticmethod
    def construct_surface(
        atoms: Atoms, indices: Tuple[int, int, int], layers: int, vacuum: float = 10.0
    ) -> Atoms:
        """
        Constructs a surface slab from the given lattice.

        Args:
            atoms (Atoms): Base bulk structure.
            indices (Tuple[int,int,int]): Miller indices (e.g., (1, 1, 1)).
            layers (int): Number of slab layers.
            vacuum (float): Vacuum thickness (Angstrom) to add above and below the slab.

        Returns:
            Atoms: The surface structure.
        """
        # Build surface slab from bulk
        slab = surface(atoms, indices, layers)
        slab.center(vacuum=vacuum, axis=2)  # Center along z-axis and add vacuum
        return slab

    @staticmethod
    def save_poscar(atoms: Atoms, filename: str = "POSCAR", directory: str = ".") -> str:
        """
        Saves the generated structure in VASP POSCAR format.

        Args:
            atoms (Atoms): Structure to save.
            filename (str): Name of the file (default: POSCAR).
            directory (str): Directory to save in (default: current directory).

        Returns:
            str: Full path of the saved file.
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

        # Explicitly specify format to avoid VASP version issues
        write(filepath, atoms, format="vasp")
        return filepath

    @staticmethod
    def analyze_structure(atoms: Atoms) -> Dict[str, Any]:
        """
        Returns basic structural information to be used by the Form Filler agent.

        Args:
            atoms (Atoms): Structure to analyze.

        Returns:
            dict: Structural information (number of atoms, volume, minimum distance, etc.).
        """
        num_atoms = len(atoms)
        try:
            volume = atoms.get_volume()
        except ValueError:
            volume = 0.0

        # Nearest-neighbor distance (O(N^2); optimize for large cells)
        min_dist = float("inf")
        distances = atoms.get_all_distances()
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if distances[i][j] < min_dist:
                    min_dist = distances[i][j]

        return {
            "num_atoms": num_atoms,
            "cell_volume": volume,
            "minimum_distance": min_dist if num_atoms > 1 else 999.0,
            "cell_parameters": atoms.get_cell().tolist() if hasattr(atoms, "get_cell") else [],
        }
