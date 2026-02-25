"""Quantum ESPRESSO output parsers for band structure and DOS data.

Parses ``data-file-schema.xml`` (eigenvalues/k-points from a ``bands`` or
``nscf`` pw.x run) and ``pwscf.dos`` (total DOS from ``dos.x``), returning
the backend-agnostic ``BandStructureData`` and ``DOSData`` dataclasses
defined in ``shalom.backends.base``.

Unit conventions
----------------
* QE 7.x ``data-file-schema.xml`` stores eigenvalues in **Hartree** (Ha).
  Conversion: ``1 Ha = 27.2114 eV``.
* ``pw.out`` stdout reports energies in **Ry**.
  Conversion: ``1 Ry = 13.6057 eV``.
* ``dos.x`` input namelist (``dos.in``) expects energies in **eV**.
* ``pwscf.dos`` output file reports energies in **eV** (dos.x converts).

XML namespace
-------------
QE uses the XML namespace ``http://www.quantum-espresso.org/ns/qes/qes-1.0``.
Python's ``xml.etree.ElementTree`` does **not** strip namespaces automatically,
so every ``find`` / ``findall`` call must use the ``ns`` dict defined here.
"""

from __future__ import annotations

import glob
import math
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms

from shalom.backends._physics import HA_TO_EV, RY_TO_EV, EV_TO_RY
from shalom.backends.base import BandStructureData, DOSData

# QE XML namespace for ElementTree queries.
QE_XML_NS: Dict[str, str] = {
    "qes": "http://www.quantum-espresso.org/ns/qes/qes-1.0",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_xml_bands(
    xml_path: str,
    fermi_energy: float = 0.0,
) -> BandStructureData:
    """Parse band structure eigenvalues from a QE ``data-file-schema.xml`` file.

    Reads ``output/band_structure/ks_energies`` elements for eigenvalues and
    k-point coordinates, and ``output/basis_set/reciprocal_lattice`` for the
    reciprocal lattice vectors (needed to compute Cartesian k-path distances).

    Args:
        xml_path: Absolute or relative path to ``data-file-schema.xml``.
        fermi_energy: Fermi energy in eV.  Should preferably come from the
            dense-mesh NSCF ``pw.out`` (``extract_fermi_energy``), not the SCF
            output, to ensure accuracy for metallic systems.

    Returns:
        ``BandStructureData`` with ``eigenvalues`` in eV, ``kpoint_coords`` in
        crystal (fractional) coordinates, and ``kpath_distances`` in 1/Angstrom.

    Raises:
        FileNotFoundError: If ``xml_path`` does not exist.
        ValueError: If the XML does not contain expected band-structure tags,
            or if it cannot be parsed as valid XML.
    """
    import xml.etree.ElementTree as ET
    import numpy as np

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"QE XML not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed XML in {xml_path}: {exc}") from exc

    root = tree.getroot()
    ns = QE_XML_NS

    # QE XML files may use either a default namespace (xmlns=) where all
    # child elements inherit the namespace, or a prefixed namespace
    # (xmlns:qes=) where unprefixed children are in *no* namespace.
    # Detect which variant we have by testing a common element.
    _use_ns = root.find(".//qes:band_structure", ns) is not None

    def _find(parent, tag):  # type: ignore[no-untyped-def]
        """Find element, trying namespaced then bare tag."""
        if _use_ns:
            elem = parent.find(f"qes:{tag}", ns)
            if elem is not None:
                return elem
        elem = parent.find(tag)
        if elem is not None:
            return elem
        return parent.find(f".//{{{ns['qes']}}}{tag}")

    def _findall(parent, tag):  # type: ignore[no-untyped-def]
        """Find all elements, trying namespaced then bare tag."""
        if _use_ns:
            elems = parent.findall(f"qes:{tag}", ns)
            if elems:
                return elems
        elems = parent.findall(tag)
        if elems:
            return elems
        return parent.findall(f".//{{{ns['qes']}}}{tag}")

    # ------------------------------------------------------------------
    # Reciprocal lattice vectors (needed for Cartesian kpath distances)
    # Stored in Angstrom^-1 in the XML (2π/a units).
    # ------------------------------------------------------------------
    b_matrix = np.eye(3)  # fallback: identity
    recip_elem = root.find(".//qes:reciprocal_lattice", ns)
    if recip_elem is None:
        recip_elem = root.find(".//reciprocal_lattice")
    if recip_elem is None:
        recip_elem = root.find(f".//{{{ns['qes']}}}reciprocal_lattice")
    if recip_elem is not None:
        b_vecs: List[List[float]] = []
        for tag in ("b1", "b2", "b3"):
            vec_elem = _find(recip_elem, tag)
            if vec_elem is not None and vec_elem.text:
                b_vecs.append([float(x) for x in vec_elem.text.split()])
        if len(b_vecs) == 3:
            b_matrix = np.array(b_vecs)  # shape (3, 3), rows = b1,b2,b3
        else:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "reciprocal_lattice incomplete in XML; "
                "kpath distances may be inaccurate."
            )
    else:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "reciprocal_lattice not found in XML; "
            "kpath distances may be inaccurate."
        )

    # ------------------------------------------------------------------
    # Spin polarisation flag
    # ------------------------------------------------------------------
    lsda_elem = root.find(".//qes:lsda", ns)
    if lsda_elem is None:
        lsda_elem = root.find(".//lsda")
    is_spin_polarized = False
    if lsda_elem is not None and lsda_elem.text:
        is_spin_polarized = lsda_elem.text.strip().lower() in ("true", "1", ".true.")

    # ------------------------------------------------------------------
    # Collect per-k-point data
    # ------------------------------------------------------------------
    ks_list = root.findall(".//qes:ks_energies", ns)
    if not ks_list:
        ks_list = root.findall(".//ks_energies")
    if not ks_list:
        raise ValueError(
            f"No <ks_energies> elements found in {xml_path}. "
            "Ensure the calculation type is 'bands' or 'nscf'."
        )

    kcoords: List[List[float]] = []
    evals_up: List[List[float]] = []
    evals_dn: List[List[float]] = []

    for ks in ks_list:
        # k-point coordinates (crystal fractional)
        kpt_elem = _find(ks, "k_point")
        if kpt_elem is not None and kpt_elem.text:
            kcoords.append([float(x) for x in kpt_elem.text.split()])
        else:
            kcoords.append([0.0, 0.0, 0.0])

        # eigenvalues — stored in Hartree; convert to eV
        eig_elems = _findall(ks, "eigenvalues")

        if is_spin_polarized and len(eig_elems) >= 2:
            evals_up.append([float(x) * HA_TO_EV for x in eig_elems[0].text.split()])
            evals_dn.append([float(x) * HA_TO_EV for x in eig_elems[1].text.split()])
        elif eig_elems:
            evals_up.append([float(x) * HA_TO_EV for x in eig_elems[0].text.split()])

    kcoords_arr = np.array(kcoords)                   # (nkpts, 3)
    evals_arr = np.array(evals_up)                    # (nkpts, nbands)

    nkpts, nbands = evals_arr.shape

    # ------------------------------------------------------------------
    # Compute cumulative kpath distances in Cartesian reciprocal space
    # ------------------------------------------------------------------
    k_cart = kcoords_arr @ b_matrix                   # (nkpts, 3), 1/Angstrom
    dk = np.diff(k_cart, axis=0)                      # (nkpts-1, 3)
    seg_lengths = np.linalg.norm(dk, axis=1)          # (nkpts-1,)
    kpath_distances = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    result = BandStructureData(
        eigenvalues=evals_arr,
        kpoint_coords=kcoords_arr,
        kpath_distances=kpath_distances,
        fermi_energy=fermi_energy,
        is_spin_polarized=is_spin_polarized,
        nbands=nbands,
        nkpts=nkpts,
        source="qe",
    )

    if is_spin_polarized and evals_dn:
        result.spin_up = evals_arr
        result.spin_down = np.array(evals_dn)

    return result


def parse_dos_file(dos_path: str) -> DOSData:
    """Parse a ``pwscf.dos`` file produced by QE's ``dos.x`` utility.

    The file format (energies already in eV in the output):

    Non-spin-polarised::

        # E(eV)  dos(E)  idos(E)
        -6.0000   0.000   0.000
        ...

    Spin-polarised::

        # E(eV)  dosup(E)  dosdw(E)  idos(E)
        -6.0000   0.000   0.000   0.000
        ...

    Note:
        The Fermi energy is **not** stored in the ``pwscf.dos`` file itself.
        Use ``extract_fermi_energy()`` on the NSCF (or SCF) ``pw.out`` to
        obtain it separately and set ``DOSData.fermi_energy`` afterwards.

    Args:
        dos_path: Path to ``pwscf.dos``.

    Returns:
        ``DOSData`` with ``energies``, ``dos``, ``integrated_dos`` in eV
        and states/eV units.  ``fermi_energy`` is 0.0 and should be set
        by the caller.

    Raises:
        FileNotFoundError: If ``dos_path`` does not exist.
        ValueError: If the file cannot be parsed.
    """
    import numpy as np

    if not os.path.exists(dos_path):
        raise FileNotFoundError(f"pwscf.dos not found: {dos_path}")

    # Skip comment/header lines starting with '#'
    try:
        data = np.loadtxt(dos_path, comments="#")
    except Exception as exc:
        raise ValueError(f"Cannot parse DOS file {dos_path}: {exc}") from exc

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]
    # 3 columns → non-spin: E, dos, idos
    # 4 columns → spin: E, dos_up, dos_dw, idos
    if ncols < 3:
        raise ValueError(
            f"Expected ≥3 columns in {dos_path}, got {ncols}."
        )

    energies = data[:, 0]
    is_spin = ncols >= 4

    if is_spin:
        dos_up = data[:, 1]
        dos_dw = data[:, 2]
        idos = data[:, 3]
        dos_total = dos_up + dos_dw
        return DOSData(
            energies=energies,
            dos=dos_total,
            integrated_dos=idos,
            dos_up=dos_up,
            dos_down=dos_dw,
            is_spin_polarized=True,
            source="qe",
        )
    else:
        return DOSData(
            energies=energies,
            dos=data[:, 1],
            integrated_dos=data[:, 2],
            is_spin_polarized=False,
            source="qe",
        )


def find_xml_path(calc_dir: str, prefix: str = "shalom") -> Optional[str]:
    """Locate ``data-file-schema.xml`` inside a QE calculation directory.

    Search order:
    1. ``{calc_dir}/{prefix}.save/data-file-schema.xml``
    2. ``{calc_dir}/tmp/{prefix}.save/data-file-schema.xml``  (outdir=./tmp)
    3. Glob fallback: ``{calc_dir}/**/*.save/data-file-schema.xml``

    Args:
        calc_dir: Directory containing QE output files.
        prefix: QE ``prefix`` used in the calculation (default: ``"shalom"``).

    Returns:
        Absolute path to ``data-file-schema.xml``, or ``None`` if not found.
    """
    candidates = [
        os.path.join(calc_dir, "data-file-schema.xml"),
        os.path.join(calc_dir, f"{prefix}.save", "data-file-schema.xml"),
        os.path.join(calc_dir, "tmp", f"{prefix}.save", "data-file-schema.xml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    # Glob fallback
    pattern = os.path.join(calc_dir, "**", "*.save", "data-file-schema.xml")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return os.path.abspath(matches[0])

    return None


def extract_fermi_energy(pw_out_path: str) -> Optional[float]:
    """Extract the Fermi energy (in eV) from a QE ``pw.out`` output file.

    Matches the line::

        the Fermi energy is     X.XXXX ev

    If the file contains multiple Fermi energy lines (e.g. from a restarted
    calculation), the **last** value is returned as it corresponds to the
    final converged result.

    Args:
        pw_out_path: Path to ``pw.out``.

    Returns:
        Fermi energy in eV, or ``None`` if the pattern is not found.
    """
    if not os.path.isfile(pw_out_path):
        return None
    pattern = re.compile(r"the Fermi energy is\s+([-\d.]+)\s+ev", re.IGNORECASE)
    last_fermi: Optional[float] = None
    try:
        with open(pw_out_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = pattern.search(line)
                if m:
                    last_fermi = float(m.group(1))
    except OSError:
        return None
    return last_fermi


def compute_nbnd(atoms: Any, multiplier: float = 1.3) -> int:
    """Estimate a suitable ``nbnd`` (number of bands) for a band-structure run.

    Uses the SSSP valence-electron count to determine the number of occupied
    bands, then multiplies by ``multiplier`` to include enough empty bands for
    a meaningful band structure above the Fermi level.

    Formula::

        n_occupied = sum(z_valence[el] for all atoms) / 2  (spin-paired)
        nbnd = max(20, ceil(n_occupied * multiplier))

    Args:
        atoms: ASE ``Atoms`` object representing the structure.
        multiplier: Scale factor applied to the occupied-band count
            (default 1.3 → 30% extra empty bands).

    Returns:
        Recommended ``nbnd`` value (integer, minimum 20).
    """
    from shalom.backends.qe_config import SSSP_ELEMENTS

    symbols = atoms.get_chemical_symbols()
    z_total = 0
    for sym in symbols:
        entry = SSSP_ELEMENTS.get(sym)
        z_total += entry["z_valence"] if entry else 1

    n_occupied = z_total / 2.0  # spin-paired assumption
    return max(20, math.ceil(n_occupied * multiplier))
