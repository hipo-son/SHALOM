import os
import textwrap

import pytest
from ase import Atoms
from ase.build import bulk
from unittest.mock import MagicMock

from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import MaterialCandidate


@pytest.fixture
def mock_llm() -> MagicMock:
    """Provides a mocked LLMProvider for isolated testing."""
    return MagicMock(spec=LLMProvider)


@pytest.fixture
def sample_candidates():
    """Provides a default list of MaterialCandidates for testing selectors."""
    return [
        MaterialCandidate(
            material_name="Copper (111)",
            elements=["Cu"],
            reasoning="d-band center is appropriate.",
            expected_properties={"surface_energy": "low"},
        ),
        MaterialCandidate(
            material_name="Platinum",
            elements=["Pt"],
            reasoning="High catalytic activity.",
            expected_properties={"cost": "high"},
        ),
    ]


# ---------------------------------------------------------------------------
# Phase 1 Fixtures: Structures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bulk_cu():
    """Simple Cu FCC bulk structure."""
    return bulk("Cu", "fcc", a=3.6)


@pytest.fixture
def sample_bulk_fe():
    """BCC Fe bulk — magnetic element."""
    return bulk("Fe", "bcc", a=2.87)


@pytest.fixture
def sample_2d_slab():
    """2D slab with ~15 Angstrom vacuum in z-direction."""
    atoms = Atoms(
        symbols=["Mo", "S", "S"],
        positions=[[0, 0, 7.5], [1.59, 0.92, 8.9], [1.59, 0.92, 6.1]],
        cell=[[3.18, 0, 0], [-1.59, 2.76, 0], [0, 0, 20.0]],
        pbc=True,
    )
    return atoms


@pytest.fixture
def sample_tmo_feo():
    """FeO rocksalt — transition metal oxide for GGA+U testing."""
    atoms = Atoms(
        symbols=["Fe", "O"],
        positions=[[0, 0, 0], [2.15, 0, 0]],
        cell=[[4.3, 0, 0], [0, 4.3, 0], [0, 0, 4.3]],
        pbc=True,
    )
    return atoms


@pytest.fixture
def sample_si_diamond():
    """Silicon diamond — non-magnetic semiconductor."""
    return bulk("Si", "diamond", a=5.43)


# ---------------------------------------------------------------------------
# Phase 1 Fixtures: OUTCAR Files
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_outcar_forces(tmp_path):
    """OUTCAR with complete TOTAL-FORCE block and entropy."""
    content = textwrap.dedent("""\
     free  energy   TOTEN  =       -34.567890 eV

     TOTAL-FORCE (eV/Angst)
     ---------------------------
          0.00000    0.00000    0.00000     0.01200   -0.00800    0.03500
          1.80000    1.80000    0.00000    -0.01200    0.00800   -0.03500
     ---------------------------

     EENTRO  =        -0.00150000

     number of atoms/cell =      2

     General timing and accounting informations for this job:
    """)
    outcar_path = tmp_path / "OUTCAR"
    outcar_path.write_text(content)
    return str(tmp_path)


@pytest.fixture
def dummy_outcar_ionic_steps(tmp_path):
    """OUTCAR with multiple ionic steps for false convergence testing."""
    steps = []
    energies = [-30.1, -30.5, -30.8, -30.82, -30.81, -30.83, -30.82]
    forces_max = [0.5, 0.3, 0.1, 0.025, 0.018, 0.022, 0.019]
    for i, (e, f) in enumerate(zip(energies, forces_max)):
        steps.append(textwrap.dedent(f"""\
         free  energy   TOTEN  =       {e:.6f} eV

         TOTAL-FORCE (eV/Angst)
         ---------------------------
              0.000  0.000  0.000    {f:.4f}  0.0000  0.0000
              1.800  1.800  0.000   -{f:.4f}  0.0000  0.0000
         ---------------------------
        """))
    content = "\n".join(steps) + (
        "\n General timing and accounting informations for this job:\n"
    )
    outcar_path = tmp_path / "OUTCAR"
    outcar_path.write_text(content)
    return str(tmp_path)


@pytest.fixture
def dummy_outcar_scf_error(tmp_path):
    """OUTCAR with SCF_UNCONVERGED error (NELM reached)."""
    content = textwrap.dedent("""\
     free  energy   TOTEN  =       -10.123456 eV

     NELM reached
     Some other output without timing section
    """)
    outcar_path = tmp_path / "OUTCAR"
    outcar_path.write_text(content)
    return str(tmp_path)


@pytest.fixture
def mock_band_data():
    """Synthetic band structure data (Si-like, 20 k-points, 8 bands)."""
    import numpy as np
    from shalom.backends.base import BandStructureData

    nkpts, nbands = 20, 8
    rng = np.random.default_rng(42)
    return BandStructureData(
        eigenvalues=rng.standard_normal((nkpts, nbands)) * 2 + 5.0,
        kpoint_coords=np.zeros((nkpts, 3)),
        kpath_distances=np.linspace(0, 1, nkpts),
        fermi_energy=5.12,
        high_sym_labels={0: "G", 10: "X", 19: "L"},
        nbands=nbands,
        nkpts=nkpts,
        source="qe",
    )


@pytest.fixture
def mock_dos_data():
    """Synthetic non-spin DOS data (100 energy points)."""
    import numpy as np
    from shalom.backends.base import DOSData

    energies = np.linspace(-10, 10, 100)
    dos = np.exp(-((energies - 0) ** 2) / 2.0)  # Gaussian
    idos = np.cumsum(dos) * (energies[1] - energies[0])
    return DOSData(
        energies=energies,
        dos=dos,
        integrated_dos=idos,
        fermi_energy=0.0,
        is_spin_polarized=False,
        source="qe",
    )


@pytest.fixture
def mock_bands_xml_path():
    """Path to the mock bands XML fixture file."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "mock_bands_xml.xml")


@pytest.fixture
def mock_dos_path():
    """Path to the mock DOS fixture file."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "mock_dos.dat")


@pytest.fixture
def mock_dos_spin_path():
    """Path to the mock spin-polarised DOS fixture file."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "mock_dos_spin.dat")


@pytest.fixture
def dummy_outcar_brmix(tmp_path):
    """OUTCAR with BRMIX error."""
    content = textwrap.dedent("""\
     free  energy   TOTEN  =       -10.123456 eV

     BRMIX: very serious problems
     the old and the new charge density differ
    """)
    outcar_path = tmp_path / "OUTCAR"
    outcar_path.write_text(content)
    return str(tmp_path)
