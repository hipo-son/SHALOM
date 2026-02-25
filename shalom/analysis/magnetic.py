"""Magnetic and charge analysis from QE DFT output.

Parses per-site magnetization and Lowdin charge data from ``pw.x`` and
``projwfc.x`` output files, then classifies magnetic behaviour per element.

No optional dependencies required (pure text parsing + stdlib).

Example::

    from ase.build import bulk
    from shalom.backends.base import DFTResult
    from shalom.analysis.magnetic import analyze_magnetism

    atoms = bulk("Fe", "bcc", a=2.87)
    dft = DFTResult(magnetization=2.2, is_converged=True)
    result = analyze_magnetism(dft, atoms, pw_out_path="pw.out")
    print(f"Magnetic: {result.is_magnetic}, dominant: {result.dominant_moment_element}")
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from shalom.analysis._base import MagneticResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability guard (no-op for API consistency with elastic/phonon)
# ---------------------------------------------------------------------------


def is_magnetic_available() -> bool:
    """Always True -- magnetic analysis has no optional dependencies."""
    return True


def _ensure_magnetic_available() -> None:
    """No-op for API consistency."""
    pass


# ---------------------------------------------------------------------------
# Regex patterns for QE output parsing
# ---------------------------------------------------------------------------

# QE pw.out site magnetization line:
# "     atom:    1    charge:   3.0123    magn:   0.1234    constr:   0.0000"
_SITE_MAG_RE = re.compile(
    r"atom:\s+\d+\s+charge:\s+([-\d.]+)\s+magn:\s+([-\d.]+)"
)

# QE total magnetization:
# "     total magnetization       =     2.0000 Bohr mag/cell"
_TOTAL_MAG_RE = re.compile(
    r"total magnetization\s+=\s+([-\d.]+)\s+Bohr mag/cell"
)

# Lowdin charges (projwfc.x output):
# "Atom #   1: total charge =   3.8765, s =  0.3456, p =  2.1234, d =  1.4075"
_LOWDIN_RE = re.compile(
    r"Atom\s*#\s*\d+.*?total charge\s*=\s*([-\d.]+)"
    r"(?:,\s*s\s*=\s*([-\d.]+))?"
    r"(?:,\s*p\s*=\s*([-\d.]+))?"
    r"(?:,\s*d\s*=\s*([-\d.]+))?"
    r"(?:,\s*f\s*=\s*([-\d.]+))?"
)

# Threshold for classifying an element as "magnetic" based on average
# absolute site moment (Bohr magneton).
_MAGNETIC_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_total_magnetization(pw_out_path: str) -> Optional[float]:
    """Extract total cell magnetization from QE pw.out.

    Parses lines like::

        total magnetization       =     4.0000 Bohr mag/cell

    If multiple ionic steps exist, returns the **last** value.

    Args:
        pw_out_path: Path to QE ``pw.out`` file.

    Returns:
        Total magnetization in Bohr magneton per cell, or ``None`` if
        the file doesn't exist or no total magnetization data was found.
    """
    if not os.path.isfile(pw_out_path):
        return None

    try:
        with open(pw_out_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None

    matches = _TOTAL_MAG_RE.findall(text)
    if not matches:
        return None

    return float(matches[-1])


def extract_site_magnetization(pw_out_path: str) -> Optional[List[float]]:
    """Extract per-site magnetic moments from QE pw.out.

    Parses lines like::

        atom:    1    charge:   3.0123    magn:   0.1234    constr:   0.0000

    If multiple ionic steps exist, returns the **last** set (final converged
    values).

    Args:
        pw_out_path: Path to QE ``pw.out`` file.

    Returns:
        List of per-atom magnetic moments in Bohr magneton, or ``None`` if
        the file doesn't exist or no site magnetization data was found.
    """
    if not os.path.isfile(pw_out_path):
        logger.warning("pw.out file not found: %s", pw_out_path)
        return None

    try:
        with open(pw_out_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        logger.warning("Cannot read pw.out: %s", exc)
        return None

    # Find all site magnetization matches.  They appear in blocks after each
    # SCF convergence.  We want the last complete block.
    matches = _SITE_MAG_RE.findall(text)
    if not matches:
        return None

    # Determine block size: the first block starts at the beginning and ends
    # when atom indices reset (i.e. the next block starts).  We detect the
    # block boundaries by re-scanning for the full "atom:  N" pattern.
    atom_lines = list(re.finditer(
        r"atom:\s+(\d+)\s+charge:\s+([-\d.]+)\s+magn:\s+([-\d.]+)", text
    ))
    if not atom_lines:
        return None

    # Split into blocks: a new block begins when atom index resets to 1
    # (or decreases relative to the previous line).
    blocks: List[List[float]] = []
    current_block: List[float] = []
    prev_idx = 0

    for m in atom_lines:
        atom_idx = int(m.group(1))
        magn = float(m.group(3))

        if atom_idx <= prev_idx and current_block:
            # New block detected -- save previous block and start fresh
            blocks.append(current_block)
            current_block = []

        current_block.append(magn)
        prev_idx = atom_idx

    # Don't forget the last block
    if current_block:
        blocks.append(current_block)

    if not blocks:
        return None

    # Return the last (most converged) block
    last_block = blocks[-1]
    logger.debug(
        "Extracted %d site magnetizations from %s (last of %d blocks)",
        len(last_block), pw_out_path, len(blocks),
    )
    return last_block


def extract_lowdin_charges(pw_out_path: str) -> Optional[Dict[str, Any]]:
    """Extract Lowdin charge analysis from QE ``projwfc.x`` output.

    Parses sections like::

        Lowdin Charges:

        Atom #   1: total charge =   3.8765, s =  0.3456, p =  2.1234, d =  1.4075

    Args:
        pw_out_path: Path to ``projwfc.x`` output or ``pw.out`` with Lowdin
            data.

    Returns:
        Dict with keys:

        - ``"total_charges"``: ``[float, ...]`` per atom
        - ``"spd_charges"``: ``[{"s": float, "p": float, "d": float, ...}, ...]``
          per atom (keys present only for channels found in the output)

        Returns ``None`` if the Lowdin section is not found.
    """
    if not os.path.isfile(pw_out_path):
        logger.warning("Output file not found: %s", pw_out_path)
        return None

    try:
        with open(pw_out_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        logger.warning("Cannot read output file: %s", exc)
        return None

    # Look for the Lowdin Charges header (case insensitive)
    if not re.search(r"lowdin\s+charges", text, re.IGNORECASE):
        return None

    # Parse all Lowdin atom lines
    matches = _LOWDIN_RE.findall(text)
    if not matches:
        return None

    total_charges: List[float] = []
    spd_charges: List[Dict[str, float]] = []

    for m in matches:
        total_charges.append(float(m[0]))

        spd: Dict[str, float] = {}
        channel_names = ["s", "p", "d", "f"]
        for i, name in enumerate(channel_names):
            val = m[i + 1]  # groups 1-4 are s, p, d, f
            if val:
                spd[name] = float(val)

        spd_charges.append(spd)

    logger.debug("Extracted Lowdin charges for %d atoms", len(total_charges))
    return {
        "total_charges": total_charges,
        "spd_charges": spd_charges,
    }


def analyze_magnetism(
    dft_result: Any,
    atoms: Any,
    pw_out_path: Optional[str] = None,
) -> "MagneticResult":
    """Analyze magnetic properties from DFT result and optional pw.out.

    Combines total magnetization from :class:`~shalom.backends.base.DFTResult`
    with site-resolved data parsed from the QE output file.

    Args:
        dft_result: :class:`~shalom.backends.base.DFTResult` with
            ``magnetization`` field from ``parse_output()``.
        atoms: ASE ``Atoms`` object for element identification per site.
        pw_out_path: Optional path to ``pw.out`` for site-resolved
            magnetization and charge data.

    Returns:
        :class:`MagneticResult` with total/site magnetization, charge data,
        and element analysis.
    """
    from shalom.analysis._base import MagneticResult

    total_mag = dft_result.magnetization
    is_spin_pol = total_mag is not None
    is_mag = is_spin_pol and abs(total_mag) > 0.01

    site_mags: Optional[List[float]] = None
    site_charges: Optional[List[float]] = None
    lowdin: Optional[Dict[str, Any]] = None

    if pw_out_path:
        site_mags = extract_site_magnetization(pw_out_path)
        lowdin = extract_lowdin_charges(pw_out_path)
        if lowdin:
            site_charges = lowdin.get("total_charges")

    # Identify magnetic elements from site-resolved data
    symbols = atoms.get_chemical_symbols()
    mag_elements: List[str] = []
    dominant: Optional[str] = None

    if site_mags and len(site_mags) == len(symbols):
        element_moments: Dict[str, List[float]] = {}
        for sym, mag in zip(symbols, site_mags):
            element_moments.setdefault(sym, []).append(abs(mag))

        for elem, moments in element_moments.items():
            avg = sum(moments) / len(moments)
            if avg > _MAGNETIC_THRESHOLD:
                mag_elements.append(elem)

        if element_moments:
            dominant = max(
                element_moments,
                key=lambda e: sum(element_moments[e]) / len(element_moments[e]),
            )
    elif site_mags and len(site_mags) != len(symbols):
        logger.warning(
            "Site magnetization count (%d) does not match atom count (%d); "
            "skipping per-element analysis",
            len(site_mags), len(symbols),
        )

    result = MagneticResult(
        total_magnetization=total_mag,
        is_magnetic=is_mag,
        is_spin_polarized=is_spin_pol,
        site_magnetizations=site_mags,
        site_charges=site_charges,
        lowdin_charges=lowdin,
        magnetic_elements=sorted(mag_elements),
        dominant_moment_element=dominant,
    )

    logger.info(
        "Magnetic analysis: total=%.4f, is_magnetic=%s, spin_pol=%s, "
        "mag_elements=%s, dominant=%s",
        total_mag if total_mag is not None else 0.0,
        is_mag, is_spin_pol, result.magnetic_elements, dominant,
    )

    return result
