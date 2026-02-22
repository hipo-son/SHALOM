"""Materials Project API client for structure fetching.

Provides unified access to Materials Project structures by MP ID (e.g. mp-19717)
or chemical formula (e.g. Fe2O3). Returns ASE Atoms objects ready for DFT input
generation.

Requires ``mp-api`` and ``pymatgen`` (optional dependencies)::

    pip install mp-api pymatgen

API key: set ``MP_API_KEY`` environment variable.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ase import Atoms

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation.
try:
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

_MP_ID_PATTERN = re.compile(r"^m(p|vc)-\d+$")


def is_mp_available() -> bool:
    """Check if mp-api and pymatgen are installed."""
    return _MP_AVAILABLE


def is_mp_id(spec: str) -> bool:
    """Check if a string looks like a Materials Project ID (mp-NNN or mvc-NNN)."""
    if not spec or not isinstance(spec, str):
        return False
    return bool(_MP_ID_PATTERN.match(spec.strip()))


@dataclass
class MPStructureResult:
    """Result from a Materials Project structure fetch."""

    atoms: Atoms
    mp_id: str
    formula: str
    energy_above_hull: Optional[float] = None
    space_group: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _get_api_key() -> str:
    """Get MP API key from environment."""
    key = os.environ.get("MP_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "MP_API_KEY environment variable not set. "
            "Get your key at https://next-gen.materialsproject.org/api#api-key "
            "and set it: export MP_API_KEY='your-key-here'"
        )
    return key


def _ensure_mp_available() -> None:
    """Raise ImportError with install instructions if mp-api is not available."""
    if not _MP_AVAILABLE:
        raise ImportError(
            "Materials Project client requires 'mp-api' and 'pymatgen'. "
            "Install with: pip install mp-api pymatgen"
        )


def fetch_by_mp_id(mp_id: str) -> MPStructureResult:
    """Fetch a structure from Materials Project by MP ID.

    Args:
        mp_id: Materials Project ID (e.g. 'mp-19717').

    Returns:
        MPStructureResult with ASE Atoms and metadata.

    Raises:
        ImportError: If mp-api is not installed.
        EnvironmentError: If MP_API_KEY is not set.
        ValueError: If the MP ID is not found.
    """
    _ensure_mp_available()
    api_key = _get_api_key()

    mp_id = mp_id.strip()
    logger.info("Fetching structure for %s from Materials Project...", mp_id)

    with MPRester(api_key) as mpr:
        try:
            doc = mpr.materials.summary.get_data_by_id(mp_id)
        except Exception as e:
            raise ValueError(
                f"Could not fetch '{mp_id}' from Materials Project: {e}. "
                "Check the ID at https://next-gen.materialsproject.org/"
            ) from e

    structure = doc.structure
    atoms = AseAtomsAdaptor.get_atoms(structure)

    return MPStructureResult(
        atoms=atoms,
        mp_id=str(doc.material_id),
        formula=str(structure.composition.reduced_formula),
        energy_above_hull=getattr(doc, "energy_above_hull", None),
        space_group=getattr(doc, "symmetry", {}).get("symbol") if hasattr(doc, "symmetry") else None,
        metadata={
            "nsites": len(structure),
            "volume": structure.volume,
        },
    )


def search_by_formula(formula: str, max_results: int = 5) -> List[MPStructureResult]:
    """Search Materials Project by chemical formula.

    Returns structures sorted by energy above hull (most stable first).

    Args:
        formula: Chemical formula (e.g. 'Fe2O3', 'Cu').
        max_results: Maximum number of results to return.

    Returns:
        List of MPStructureResult, sorted by stability.
    """
    _ensure_mp_available()
    api_key = _get_api_key()

    logger.info("Searching Materials Project for formula '%s'...", formula)

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "structure", "energy_above_hull", "symmetry"],
        )

    if not docs:
        raise ValueError(
            f"No structures found for formula '{formula}' in Materials Project. "
            "Check the formula or try a different composition."
        )

    # Sort by energy_above_hull (most stable first)
    docs.sort(key=lambda d: getattr(d, "energy_above_hull", float("inf")) or float("inf"))

    results = []
    for doc in docs[:max_results]:
        structure = doc.structure
        atoms = AseAtomsAdaptor.get_atoms(structure)
        results.append(MPStructureResult(
            atoms=atoms,
            mp_id=str(doc.material_id),
            formula=str(structure.composition.reduced_formula),
            energy_above_hull=getattr(doc, "energy_above_hull", None),
            space_group=getattr(doc, "symmetry", {}).get("symbol") if hasattr(doc, "symmetry") else None,
            metadata={
                "nsites": len(structure),
                "volume": structure.volume,
            },
        ))

    return results


def fetch_structure(material_spec: str) -> MPStructureResult:
    """Fetch a structure by MP ID or chemical formula (unified entry point).

    If ``material_spec`` matches the MP ID pattern (mp-NNN), fetches by ID.
    Otherwise, searches by formula and returns the most stable structure.

    Args:
        material_spec: MP ID (e.g. 'mp-19717') or formula (e.g. 'Cu', 'Fe2O3').

    Returns:
        MPStructureResult with ASE Atoms and metadata.
    """
    if not material_spec or not material_spec.strip():
        raise ValueError("material_spec cannot be empty.")

    if is_mp_id(material_spec):
        return fetch_by_mp_id(material_spec)

    results = search_by_formula(material_spec, max_results=1)
    return results[0]
