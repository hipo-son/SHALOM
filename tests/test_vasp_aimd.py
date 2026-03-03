"""Tests for VASP AIMD support — CalculationType.AIMD, presets, XDATCAR parser.

Tests verify:
- CalculationType.AIMD enum value exists
- AIMD INCAR presets (standard, precise) have correct MD tags
- CALC_TYPE_ALIASES include "aimd" and "md"
- XDATCAR parser produces correct MDTrajectoryData
- OSZICAR temperature/energy extraction
- parse_md_output integrates OUTCAR + XDATCAR
"""

import os
import textwrap

import numpy as np
import pytest

from shalom.backends.vasp import VASPBackend
from shalom.backends.vasp_config import CalculationType, get_preset
from shalom.backends._physics import AccuracyLevel
from shalom.backends.base import MDTrajectoryData
from shalom.direct_run import CALC_TYPE_ALIASES


# ---------------------------------------------------------------------------
# CalculationType.AIMD
# ---------------------------------------------------------------------------


class TestCalculationTypeAIMD:
    """Test AIMD enum and preset loading."""

    def test_aimd_enum_exists(self):
        assert CalculationType.AIMD == "aimd"

    def test_aimd_enum_in_list(self):
        all_types = [ct.value for ct in CalculationType]
        assert "aimd" in all_types

    def test_aimd_preset_standard(self):
        preset = get_preset(CalculationType.AIMD, AccuracyLevel.STANDARD)
        incar = preset.incar_settings
        assert incar["IBRION"] == 0
        assert incar["ISYM"] == 0
        assert incar["NSW"] == 5000
        assert abs(incar["POTIM"] - 1.0) < 0.01
        assert incar["SMASS"] == 0
        assert incar["TEBEG"] == 300
        assert incar["TEEND"] == 300
        assert incar["ISIF"] == 2
        assert incar["LWAVE"] is False
        assert incar["LCHARG"] is False

    def test_aimd_preset_precise(self):
        preset = get_preset(CalculationType.AIMD, AccuracyLevel.PRECISE)
        incar = preset.incar_settings
        assert incar["IBRION"] == 0
        assert incar["NSW"] == 10000
        assert abs(incar["POTIM"] - 0.5) < 0.01
        assert incar["EDIFF"] == 1.0e-6

    def test_aimd_preset_with_atoms(self):
        from ase.build import bulk
        atoms = bulk("Fe", "bcc", a=2.87)
        preset = get_preset(CalculationType.AIMD, atoms=atoms)
        incar = preset.incar_settings
        # IBRION=0 should be preserved (not overwritten by auto-detect)
        assert incar["IBRION"] == 0
        # Should still auto-detect magnetic Fe
        assert incar.get("ISPIN") == 2 or "MAGMOM" in incar


# ---------------------------------------------------------------------------
# CALC_TYPE_ALIASES
# ---------------------------------------------------------------------------


class TestCalcTypeAliases:
    """Test AIMD aliases in direct_run."""

    def test_aimd_alias(self):
        assert CALC_TYPE_ALIASES["aimd"] == "aimd"

    def test_md_alias(self):
        assert CALC_TYPE_ALIASES["md"] == "aimd"


# ---------------------------------------------------------------------------
# XDATCAR Parser
# ---------------------------------------------------------------------------


class TestParseXDATCAR:
    """Test VASP XDATCAR trajectory parsing."""

    @pytest.fixture
    def vasp_backend(self):
        return VASPBackend()

    def _write_xdatcar(self, directory, n_frames=2, n_atoms=2):
        """Write a minimal XDATCAR for testing."""
        os.makedirs(directory, exist_ok=True)
        lines = []
        # Header
        lines.append("Si2")
        lines.append("   1.00000000000000")
        lines.append("     5.43000    0.00000    0.00000")
        lines.append("     0.00000    5.43000    0.00000")
        lines.append("     0.00000    0.00000    5.43000")
        lines.append("   Si")
        lines.append(f"     {n_atoms}")
        # Frames
        for frame in range(n_frames):
            lines.append(f"Direct configuration=     {frame + 1}")
            for a in range(n_atoms):
                x = 0.0 + a * 0.25 + frame * 0.001
                y = 0.0 + a * 0.25 + frame * 0.002
                z = 0.0 + a * 0.25 + frame * 0.003
                lines.append(f"  {x:.6f}  {y:.6f}  {z:.6f}")
        with open(os.path.join(directory, "XDATCAR"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def _write_oszicar(self, directory, n_steps=2):
        """Write a minimal OSZICAR with MD step data."""
        os.makedirs(directory, exist_ok=True)
        lines = []
        for i in range(1, n_steps + 1):
            # SCF iteration lines
            lines.append(f"       N       E                     dE             d eps")
            lines.append(f"DAV:   1    -0.10000000E+02   -0.10E+02   -0.50E+01")
            lines.append(f"DAV:   2    -0.10500000E+02   -0.50E-01   -0.10E-01")
            # Ionic step summary line
            t = 300.0 + i * 5.0
            e = -10.5 + i * 0.01
            lines.append(f"   {i} T= {t:.2f} E= {e:.5E} F= {e:.5E} E0= {e:.5E}")
        with open(os.path.join(directory, "OSZICAR"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def _write_incar(self, directory, potim=1.0):
        """Write a minimal INCAR."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "INCAR"), "w") as f:
            f.write(f"IBRION = 0\nPOTIM = {potim}\nNSW = 5000\n")

    def test_parse_xdatcar_basic(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "aimd_test")
        self._write_xdatcar(calc_dir, n_frames=3, n_atoms=2)
        self._write_oszicar(calc_dir, n_steps=3)
        self._write_incar(calc_dir, potim=1.0)

        traj = vasp_backend.parse_xdatcar(calc_dir)
        assert isinstance(traj, MDTrajectoryData)
        assert traj.n_frames == 3
        assert traj.n_atoms == 2
        assert traj.source == "vasp"
        assert traj.positions.shape == (3, 2, 3)

    def test_parse_xdatcar_species(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "species_test")
        self._write_xdatcar(calc_dir, n_frames=1, n_atoms=2)
        self._write_oszicar(calc_dir, n_steps=1)
        traj = vasp_backend.parse_xdatcar(calc_dir)
        assert traj.species == ["Si", "Si"]

    def test_parse_xdatcar_cell(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "cell_test")
        self._write_xdatcar(calc_dir, n_frames=1, n_atoms=2)
        self._write_oszicar(calc_dir, n_steps=1)
        traj = vasp_backend.parse_xdatcar(calc_dir)
        # Cell should be 5.43 x 5.43 x 5.43 diagonal
        assert abs(traj.cell_vectors[0, 0] - 5.43) < 0.01
        assert abs(traj.cell_vectors[1, 1] - 5.43) < 0.01
        assert abs(traj.cell_vectors[2, 2] - 5.43) < 0.01

    def test_parse_xdatcar_temperatures(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "temp_test")
        self._write_xdatcar(calc_dir, n_frames=2, n_atoms=2)
        self._write_oszicar(calc_dir, n_steps=2)
        traj = vasp_backend.parse_xdatcar(calc_dir)
        # T = 305.0, 310.0 from OSZICAR
        assert len(traj.temperatures) == 2
        assert abs(traj.temperatures[0] - 305.0) < 0.1
        assert abs(traj.temperatures[1] - 310.0) < 0.1

    def test_parse_xdatcar_energies(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "energy_test")
        self._write_xdatcar(calc_dir, n_frames=2, n_atoms=2)
        self._write_oszicar(calc_dir, n_steps=2)
        traj = vasp_backend.parse_xdatcar(calc_dir)
        assert len(traj.energies) == 2
        assert abs(traj.energies[0] - (-10.49)) < 0.01

    def test_parse_xdatcar_times(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "time_test")
        self._write_xdatcar(calc_dir, n_frames=3, n_atoms=2)
        self._write_oszicar(calc_dir, n_steps=3)
        self._write_incar(calc_dir, potim=2.0)
        traj = vasp_backend.parse_xdatcar(calc_dir)
        assert abs(traj.timestep_fs - 2.0) < 0.01
        assert abs(traj.times[0] - 0.0) < 0.01
        assert abs(traj.times[1] - 2.0) < 0.01
        assert abs(traj.times[2] - 4.0) < 0.01

    def test_parse_xdatcar_missing_raises(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "no_xdatcar")
        os.makedirs(calc_dir)
        with pytest.raises(FileNotFoundError):
            vasp_backend.parse_xdatcar(calc_dir)

    def test_parse_xdatcar_no_oszicar_defaults(self, vasp_backend, tmp_path):
        """Without OSZICAR, temperatures default to 300.0."""
        calc_dir = str(tmp_path / "no_oszicar")
        self._write_xdatcar(calc_dir, n_frames=2, n_atoms=2)
        traj = vasp_backend.parse_xdatcar(calc_dir)
        assert abs(traj.temperatures[0] - 300.0) < 0.01

    def test_parse_xdatcar_frac_to_cart(self, vasp_backend, tmp_path):
        """Fractional coords should be converted to Cartesian."""
        calc_dir = str(tmp_path / "cart_test")
        os.makedirs(calc_dir, exist_ok=True)
        # Write XDATCAR with known fractional coords
        lines = [
            "Test", "1.0",
            "  10.0  0.0  0.0", "  0.0  10.0  0.0", "  0.0  0.0  10.0",
            "Si", "1",
            "Direct configuration=     1",
            "  0.500000  0.500000  0.500000",
        ]
        with open(os.path.join(calc_dir, "XDATCAR"), "w") as f:
            f.write("\n".join(lines) + "\n")
        traj = vasp_backend.parse_xdatcar(calc_dir)
        # (0.5, 0.5, 0.5) * 10.0 = (5.0, 5.0, 5.0)
        assert abs(traj.positions[0, 0, 0] - 5.0) < 0.01
        assert abs(traj.positions[0, 0, 1] - 5.0) < 0.01
        assert abs(traj.positions[0, 0, 2] - 5.0) < 0.01


# ---------------------------------------------------------------------------
# Multi-element XDATCAR
# ---------------------------------------------------------------------------


class TestParseXDATCARMultiElement:
    """Test XDATCAR parsing with multiple element types."""

    @pytest.fixture
    def vasp_backend(self):
        return VASPBackend()

    def test_multi_element_species(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "multi_elem")
        os.makedirs(calc_dir, exist_ok=True)
        lines = [
            "Fe2O3", "1.0",
            "  5.0  0.0  0.0", "  0.0  5.0  0.0", "  0.0  0.0  14.0",
            "Fe   O", "2   3",
            "Direct configuration=     1",
            "  0.0  0.0  0.0",
            "  0.5  0.5  0.5",
            "  0.1  0.1  0.1",
            "  0.2  0.2  0.2",
            "  0.3  0.3  0.3",
        ]
        with open(os.path.join(calc_dir, "XDATCAR"), "w") as f:
            f.write("\n".join(lines) + "\n")
        traj = vasp_backend.parse_xdatcar(calc_dir)
        assert traj.species == ["Fe", "Fe", "O", "O", "O"]
        assert traj.n_atoms == 5
        assert traj.n_frames == 1


# ---------------------------------------------------------------------------
# parse_md_output
# ---------------------------------------------------------------------------


class TestParseMDOutput:
    """Test combined OUTCAR + XDATCAR parsing."""

    @pytest.fixture
    def vasp_backend(self):
        return VASPBackend()

    def _write_minimal_outcar(self, directory):
        """Write a minimal OUTCAR that parse_output can handle."""
        os.makedirs(directory, exist_ok=True)
        content = textwrap.dedent("""\
            VASP output
            energy  without entropy=      -10.50000  energy(sigma->0) =      -10.50000
            General timing and accounting informations for this job
        """)
        with open(os.path.join(directory, "OUTCAR"), "w") as f:
            f.write(content)

    def _write_xdatcar(self, directory):
        os.makedirs(directory, exist_ok=True)
        lines = [
            "Si2", "1.0",
            "  5.43  0.0  0.0", "  0.0  5.43  0.0", "  0.0  0.0  5.43",
            "Si", "2",
            "Direct configuration=     1",
            "  0.0  0.0  0.0",
            "  0.25  0.25  0.25",
        ]
        with open(os.path.join(directory, "XDATCAR"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_parse_md_output_has_trajectory(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "md_out")
        self._write_minimal_outcar(calc_dir)
        self._write_xdatcar(calc_dir)
        result = vasp_backend.parse_md_output(calc_dir)
        assert result.is_converged is True
        assert "md_trajectory" in result.raw
        traj = result.raw["md_trajectory"]
        assert isinstance(traj, MDTrajectoryData)
        assert traj.n_frames == 1

    def test_parse_md_output_no_xdatcar(self, vasp_backend, tmp_path):
        """Without XDATCAR, result should still work but no trajectory."""
        calc_dir = str(tmp_path / "md_no_xdat")
        self._write_minimal_outcar(calc_dir)
        result = vasp_backend.parse_md_output(calc_dir)
        assert result.is_converged is True
        assert "md_trajectory" not in result.raw


# ---------------------------------------------------------------------------
# _read_potim
# ---------------------------------------------------------------------------


class TestReadPOTIM:
    """Test POTIM extraction from INCAR."""

    @pytest.fixture
    def vasp_backend(self):
        return VASPBackend()

    def test_reads_potim(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "potim_test")
        os.makedirs(calc_dir)
        with open(os.path.join(calc_dir, "INCAR"), "w") as f:
            f.write("POTIM = 2.0\nISIF = 2\n")
        assert abs(vasp_backend._read_potim(calc_dir) - 2.0) < 0.01

    def test_default_potim(self, vasp_backend, tmp_path):
        """Without INCAR, default to 1.0 fs."""
        calc_dir = str(tmp_path / "no_incar")
        os.makedirs(calc_dir)
        assert abs(vasp_backend._read_potim(calc_dir) - 1.0) < 0.01

    def test_potim_with_comment(self, vasp_backend, tmp_path):
        calc_dir = str(tmp_path / "potim_comment")
        os.makedirs(calc_dir)
        with open(os.path.join(calc_dir, "INCAR"), "w") as f:
            f.write("POTIM = 0.5 ! timestep in fs\n")
        assert abs(vasp_backend._read_potim(calc_dir) - 0.5) < 0.01
