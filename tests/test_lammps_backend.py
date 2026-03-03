"""Tests for LAMMPS backend — write_input, parse_output, parse_trajectory.

Tests verify:
- Data file generation (positions, box, masses, triclinic)
- Input script generation (pair_style, fix, thermo, dump, run)
- Log file parsing (thermo data extraction)
- Dump file parsing (positions, velocities, forces)
- get_backend("lammps") factory integration
"""

import os
import textwrap

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from shalom.backends import get_backend
from shalom.backends.lammps import LAMMPSBackend
from shalom.backends.lammps_config import LAMMPSInputConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fe_atoms():
    return bulk("Fe", "bcc", a=2.87)


@pytest.fixture
def si_atoms():
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture
def lammps_backend():
    return LAMMPSBackend()


@pytest.fixture
def calc_dir(tmp_path):
    return str(tmp_path / "lammps_calc")


# ---------------------------------------------------------------------------
# get_backend integration
# ---------------------------------------------------------------------------


class TestGetBackend:
    """Test backend factory for LAMMPS."""

    def test_get_backend_lammps(self):
        backend = get_backend("lammps")
        assert isinstance(backend, LAMMPSBackend)
        assert backend.name == "lammps"

    def test_get_backend_unknown_still_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("gaussian")


# ---------------------------------------------------------------------------
# write_input
# ---------------------------------------------------------------------------


class TestWriteInput:
    """Test LAMMPS input file generation."""

    def test_creates_data_and_input_files(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir)
        assert os.path.exists(os.path.join(calc_dir, "data.lammps"))
        assert os.path.exists(os.path.join(calc_dir, "in.lammps"))

    def test_data_file_atom_count(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir)
        with open(os.path.join(calc_dir, "data.lammps")) as f:
            content = f.read()
        assert f"{len(fe_atoms)} atoms" in content

    def test_data_file_masses(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir)
        with open(os.path.join(calc_dir, "data.lammps")) as f:
            content = f.read()
        assert "Masses" in content
        assert "Fe" in content

    def test_data_file_box_dimensions(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir)
        with open(os.path.join(calc_dir, "data.lammps")) as f:
            content = f.read()
        assert "xlo xhi" in content
        assert "ylo yhi" in content
        assert "zlo zhi" in content

    def test_input_script_pair_style(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir)
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "pair_style" in content
        assert "eam/alloy" in content

    def test_input_script_pair_coeff(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir)
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "pair_coeff" in content

    def test_input_script_nvt_fix(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir, ensemble="nvt", temperature=500)
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "nvt" in content
        assert "500.0" in content

    def test_input_script_npt_fix(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir, ensemble="npt")
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "npt" in content
        assert "iso" in content

    def test_input_script_nve_fix(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir, ensemble="nve")
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "fix             1 all nve" in content

    def test_input_script_thermo(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir)
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "thermo" in content
        assert "thermo_style" in content

    def test_input_script_dump(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir)
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "dump" in content
        assert "dump.lammpstrj" in content

    def test_input_script_run_command(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir, nsteps=50000)
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "run             50000" in content

    def test_explicit_config(self, lammps_backend, fe_atoms, calc_dir):
        config = LAMMPSInputConfig(
            pair_style="eam/fs",
            pair_coeff=["* * Fe.eam.fs Fe"],
            nsteps=10000,
        )
        lammps_backend.write_input(fe_atoms, calc_dir, config=config)
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "eam/fs" in content
        assert "run             10000" in content

    def test_minimize_first(self, lammps_backend, fe_atoms, calc_dir):
        lammps_backend.write_input(fe_atoms, calc_dir, minimize_first=True)
        with open(os.path.join(calc_dir, "in.lammps")) as f:
            content = f.read()
        assert "minimize" in content

    def test_si_diamond_data_file(self, lammps_backend, si_atoms, calc_dir):
        lammps_backend.write_input(si_atoms, calc_dir)
        with open(os.path.join(calc_dir, "data.lammps")) as f:
            content = f.read()
        assert f"{len(si_atoms)} atoms" in content
        assert "1 atom types" in content  # Si is only type

    def test_multi_element_type_count(self, lammps_backend, calc_dir):
        atoms = Atoms(
            "FeNi",
            positions=[[0, 0, 0], [1.4, 1.4, 1.4]],
            cell=[2.87, 2.87, 2.87],
            pbc=True,
        )
        lammps_backend.write_input(atoms, calc_dir)
        with open(os.path.join(calc_dir, "data.lammps")) as f:
            content = f.read()
        assert "2 atom types" in content

    def test_returns_directory(self, lammps_backend, fe_atoms, calc_dir):
        result = lammps_backend.write_input(fe_atoms, calc_dir)
        assert result == calc_dir


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------


class TestParseOutput:
    """Test LAMMPS log file parsing."""

    def _write_log(self, directory, content):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "log.lammps"), "w") as f:
            f.write(content)

    def test_parse_thermo_data(self, lammps_backend, tmp_path):
        log_content = textwrap.dedent("""\
            LAMMPS (29 Aug 2024)
            Step Temp PotEng KinEng TotEng Press Volume
            0 300.0 -4.1200 0.0388 -4.0812 100.5 23.5
            100 305.2 -4.1180 0.0394 -4.0786 102.3 23.5
            200 298.7 -4.1210 0.0386 -4.0824 99.1 23.5
            Loop time of 0.5 on 1 procs for 200 steps
        """)
        calc_dir = str(tmp_path / "parse_test")
        self._write_log(calc_dir, log_content)
        result = lammps_backend.parse_output(calc_dir)
        assert result.is_converged is True
        assert result.energy is not None
        assert abs(result.energy - (-4.0824)) < 0.001
        thermo = result.raw["thermo"]
        assert len(thermo["step"]) == 3
        assert len(thermo["temp"]) == 3
        assert abs(thermo["temp"][0] - 300.0) < 0.1

    def test_missing_log(self, lammps_backend, tmp_path):
        calc_dir = str(tmp_path / "no_log")
        os.makedirs(calc_dir, exist_ok=True)
        result = lammps_backend.parse_output(calc_dir)
        assert result.is_converged is False
        assert "not found" in result.error_log

    def test_empty_log(self, lammps_backend, tmp_path):
        calc_dir = str(tmp_path / "empty_log")
        self._write_log(calc_dir, "LAMMPS output\nNo thermo data\n")
        result = lammps_backend.parse_output(calc_dir)
        assert result.is_converged is False

    def test_thermo_with_volume(self, lammps_backend, tmp_path):
        log_content = textwrap.dedent("""\
            Step Temp PotEng KinEng TotEng Press Volume
            0 300.0 -10.5 0.04 -10.46 1.0 1000.0
            Loop time of 0.1 on 1 procs for 0 steps
        """)
        calc_dir = str(tmp_path / "vol_test")
        self._write_log(calc_dir, log_content)
        result = lammps_backend.parse_output(calc_dir)
        thermo = result.raw["thermo"]
        assert len(thermo["vol"]) == 1
        assert abs(thermo["vol"][0] - 1000.0) < 0.1


# ---------------------------------------------------------------------------
# parse_trajectory
# ---------------------------------------------------------------------------


class TestParseTrajectory:
    """Test LAMMPS dump file parsing."""

    def _write_dump(self, directory, content):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "dump.lammpstrj"), "w") as f:
            f.write(content)

    def _write_log_for_traj(self, directory, n_frames):
        """Write a minimal log.lammps for trajectory parsing."""
        os.makedirs(directory, exist_ok=True)
        lines = ["Step Temp PotEng KinEng TotEng Press Volume"]
        for i in range(n_frames):
            lines.append(f"{i*100} 300.0 -4.0 0.04 -3.96 100.0 24.0")
        lines.append("Loop time of 1.0 on 1 procs")
        with open(os.path.join(directory, "log.lammps"), "w") as f:
            f.write("\n".join(lines))

    def test_parse_single_frame(self, lammps_backend, tmp_path):
        calc_dir = str(tmp_path / "traj_test")
        dump = textwrap.dedent("""\
            ITEM: TIMESTEP
            0
            ITEM: NUMBER OF ATOMS
            2
            ITEM: BOX BOUNDS pp pp pp
            0.0 2.87
            0.0 2.87
            0.0 2.87
            ITEM: ATOMS id type x y z
            1 1 0.0 0.0 0.0
            2 1 1.435 1.435 1.435
        """)
        self._write_dump(calc_dir, dump)
        self._write_log_for_traj(calc_dir, 1)
        traj = lammps_backend.parse_trajectory(calc_dir)
        assert traj.n_frames == 1
        assert traj.n_atoms == 2
        assert traj.source == "lammps"
        assert traj.positions.shape == (1, 2, 3)

    def test_parse_multiple_frames(self, lammps_backend, tmp_path):
        calc_dir = str(tmp_path / "multi_frame")
        frames = []
        for t in [0, 100]:
            frames.append(textwrap.dedent(f"""\
                ITEM: TIMESTEP
                {t}
                ITEM: NUMBER OF ATOMS
                2
                ITEM: BOX BOUNDS pp pp pp
                0.0 2.87
                0.0 2.87
                0.0 2.87
                ITEM: ATOMS id type x y z
                1 1 0.0 0.0 0.0
                2 1 1.435 1.435 1.435
            """))
        self._write_dump(calc_dir, "\n".join(frames))
        self._write_log_for_traj(calc_dir, 2)
        traj = lammps_backend.parse_trajectory(calc_dir)
        assert traj.n_frames == 2

    def test_parse_with_velocities(self, lammps_backend, tmp_path):
        calc_dir = str(tmp_path / "vel_test")
        dump = textwrap.dedent("""\
            ITEM: TIMESTEP
            0
            ITEM: NUMBER OF ATOMS
            1
            ITEM: BOX BOUNDS pp pp pp
            0.0 10.0
            0.0 10.0
            0.0 10.0
            ITEM: ATOMS id type x y z vx vy vz
            1 1 5.0 5.0 5.0 0.1 0.2 0.3
        """)
        self._write_dump(calc_dir, dump)
        self._write_log_for_traj(calc_dir, 1)
        traj = lammps_backend.parse_trajectory(calc_dir)
        assert traj.velocities is not None
        assert abs(traj.velocities[0, 0, 0] - 0.1) < 1e-6

    def test_parse_with_forces(self, lammps_backend, tmp_path):
        calc_dir = str(tmp_path / "force_test")
        dump = textwrap.dedent("""\
            ITEM: TIMESTEP
            0
            ITEM: NUMBER OF ATOMS
            1
            ITEM: BOX BOUNDS pp pp pp
            0.0 10.0
            0.0 10.0
            0.0 10.0
            ITEM: ATOMS id type x y z vx vy vz fx fy fz
            1 1 5.0 5.0 5.0 0.0 0.0 0.0 1.5 -0.3 0.7
        """)
        self._write_dump(calc_dir, dump)
        self._write_log_for_traj(calc_dir, 1)
        traj = lammps_backend.parse_trajectory(calc_dir)
        assert traj.forces is not None
        assert abs(traj.forces[0, 0, 0] - 1.5) < 1e-6

    def test_missing_dump_raises(self, lammps_backend, tmp_path):
        calc_dir = str(tmp_path / "no_dump")
        os.makedirs(calc_dir)
        with pytest.raises(FileNotFoundError):
            lammps_backend.parse_trajectory(calc_dir)

    def test_atom_id_ordering(self, lammps_backend, tmp_path):
        """Atoms should be sorted by id, not by order in dump file."""
        calc_dir = str(tmp_path / "id_order")
        dump = textwrap.dedent("""\
            ITEM: TIMESTEP
            0
            ITEM: NUMBER OF ATOMS
            2
            ITEM: BOX BOUNDS pp pp pp
            0.0 10.0
            0.0 10.0
            0.0 10.0
            ITEM: ATOMS id type x y z
            2 1 5.0 5.0 5.0
            1 1 1.0 1.0 1.0
        """)
        self._write_dump(calc_dir, dump)
        self._write_log_for_traj(calc_dir, 1)
        traj = lammps_backend.parse_trajectory(calc_dir)
        # Atom 1 should be at (1,1,1), atom 2 at (5,5,5)
        assert abs(traj.positions[0, 0, 0] - 1.0) < 1e-6
        assert abs(traj.positions[0, 1, 0] - 5.0) < 1e-6
