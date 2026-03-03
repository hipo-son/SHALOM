"""LAMMPS classical MD backend.

Implements the ``DFTBackend`` protocol for LAMMPS molecular dynamics.
Generates LAMMPS data files and input scripts, parses log output and
dump trajectories.

LAMMPS is executed via subprocess (like QE's ``pw.x``), requiring
``lmp`` or ``lmp_mpi`` on PATH.

Usage::

    from shalom.backends import get_backend

    backend = get_backend("lammps")
    backend.write_input(atoms, "./md_run", ensemble="nvt", temperature=300)
    result = backend.parse_output("./md_run")
    trajectory = backend.parse_trajectory("./md_run")
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from shalom.backends.base import DFTResult, MDTrajectoryData
from shalom.backends.lammps_config import (
    LAMMPSInputConfig,
    detect_and_apply_lammps_hints,
    get_lammps_preset,
    resolve_potential_dir,
)

logger = logging.getLogger(__name__)


class LAMMPSBackend:
    """LAMMPS molecular dynamics backend.

    Implements the ``DFTBackend`` protocol (``name``, ``write_input``,
    ``parse_output``) plus an additional ``parse_trajectory`` method
    for MD trajectory analysis.
    """

    name: str = "lammps"

    # ------------------------------------------------------------------
    # write_input
    # ------------------------------------------------------------------

    def write_input(self, atoms: Atoms, directory: str, **params: Any) -> str:
        """Write LAMMPS data file and input script.

        Args:
            atoms: ASE Atoms object.
            directory: Target directory for input files.
            **params: Keyword arguments forwarded to LAMMPSInputConfig
                or an explicit ``config=LAMMPSInputConfig(...)`` object.

        Returns:
            The directory path where input files were written.
        """
        os.makedirs(directory, exist_ok=True)

        # Build or use provided config
        config = params.pop("config", None)
        if config is None:
            ensemble = params.pop("ensemble", "nvt")
            accuracy = params.pop("accuracy", "standard")
            config = get_lammps_preset(ensemble=ensemble, atoms=atoms, accuracy=accuracy)
            # Apply any remaining params as overrides
            for key, val in params.items():
                if hasattr(config, key):
                    setattr(config, key, val)
        else:
            # Still apply hints if pair_style not set
            detect_and_apply_lammps_hints(atoms, config)

        # Write files
        self._write_data_file(atoms, directory, config)
        self._write_input_script(atoms, directory, config)
        self._copy_potential_files(directory, config)

        return directory

    @staticmethod
    def _cell_to_lammps(cell: np.ndarray):
        """Convert ASE cell to LAMMPS box parameters and rotation matrix.

        Follows the LAMMPS triclinic convention:
          a_lammps = [lx, 0, 0]
          b_lammps = [xy, ly, 0]
          c_lammps = [xz, yz, lz]

        Ref: https://docs.lammps.org/Howto_triclinic.html

        Returns:
            (lx, ly, lz, xy, xz, yz, rotation_matrix)
        """
        a_len = np.linalg.norm(cell[0])
        b_len = np.linalg.norm(cell[1])
        c_len = np.linalg.norm(cell[2])

        # Angles between cell vectors
        cos_alpha = np.dot(cell[1], cell[2]) / (b_len * c_len)  # angle(b,c)
        cos_beta = np.dot(cell[0], cell[2]) / (a_len * c_len)   # angle(a,c)
        cos_gamma = np.dot(cell[0], cell[1]) / (a_len * b_len)  # angle(a,b)

        # LAMMPS box dimensions
        lx = a_len
        xy = b_len * cos_gamma
        ly = np.sqrt(b_len**2 - xy**2)
        xz = c_len * cos_beta
        yz = (b_len * c_len * cos_alpha - xy * xz) / ly
        lz = np.sqrt(max(c_len**2 - xz**2 - yz**2, 0.0))

        # Build LAMMPS cell matrix (rows = vectors) for coordinate transform
        lammps_cell = np.array([
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ])

        # Rotation: frac coords are cell-invariant, so convert via fractional
        # pos_lammps = frac @ lammps_cell
        return lx, ly, lz, xy, xz, yz, lammps_cell

    def _write_data_file(
        self, atoms: Atoms, directory: str, config: LAMMPSInputConfig,
    ) -> str:
        """Write LAMMPS data file (read_data compatible)."""
        filepath = os.path.join(directory, "data.lammps")
        species_order = list(dict.fromkeys(atoms.get_chemical_symbols()))
        type_map = {el: i + 1 for i, el in enumerate(species_order)}

        cell = np.array(atoms.get_cell())

        # Convert to LAMMPS convention (positive box, correct orientation)
        lx, ly, lz, xy, xz, yz, lammps_cell = self._cell_to_lammps(cell)

        # Convert positions via fractional coordinates (cell-invariant)
        frac = atoms.get_scaled_positions(wrap=True)
        positions = frac @ lammps_cell

        is_triclinic = abs(xy) > 1e-10 or abs(xz) > 1e-10 or abs(yz) > 1e-10

        with open(filepath, "w") as f:
            f.write("LAMMPS data file via SHALOM\n\n")
            f.write(f"{len(atoms)} atoms\n")
            f.write(f"{len(species_order)} atom types\n\n")
            f.write(f"0.0000000000 {lx:.10f} xlo xhi\n")
            f.write(f"0.0000000000 {ly:.10f} ylo yhi\n")
            f.write(f"0.0000000000 {lz:.10f} zlo zhi\n")
            if is_triclinic:
                f.write(f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n")
            f.write("\n")

            # Masses
            f.write("Masses\n\n")
            for el in species_order:
                z = atomic_numbers[el]
                mass = atomic_masses[z]
                f.write(f"{type_map[el]} {mass:.6f}  # {el}\n")
            f.write("\n")

            # Atoms section
            f.write(f"Atoms  # {config.atom_style}\n\n")
            symbols = atoms.get_chemical_symbols()
            for i, (sym, pos) in enumerate(zip(symbols, positions), 1):
                t = type_map[sym]
                f.write(f"{i} {t} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")

        return filepath

    def _write_input_script(
        self, atoms: Atoms, directory: str, config: LAMMPSInputConfig,
    ) -> str:
        """Write LAMMPS input script (in.lammps)."""
        filepath = os.path.join(directory, "in.lammps")

        t_start = config.temperature
        t_end = config.temperature_end if config.temperature_end is not None else t_start

        lines: List[str] = []
        lines.append("# LAMMPS input generated by SHALOM")
        lines.append(f"units           {config.units}")
        lines.append(f"atom_style      {config.atom_style}")
        lines.append(f"boundary        {config.boundary}")
        lines.append("")
        lines.append("read_data       data.lammps")
        lines.append("")

        # Pair style / coeff
        if config.pair_style:
            lines.append(f"pair_style      {config.pair_style}")
            for pc in config.pair_coeff:
                lines.append(f"pair_coeff      {pc}")
            lines.append("")

        # Optional minimization
        if config.minimize_first:
            lines.append("# Energy minimization before MD")
            lines.append("minimize        1.0e-8 1.0e-10 10000 100000")
            lines.append("reset_timestep  0")
            lines.append("")

        # Velocity initialization
        db = {}
        try:
            from shalom._config_loader import load_config
            db = load_config("lammps_potentials")
        except Exception:
            pass
        seed = db.get("simulation_defaults", {}).get("velocity_seed", 12345)
        lines.append(f"velocity        all create {t_start:.1f} {seed} dist gaussian")
        lines.append("")

        # Timestep
        lines.append(f"timestep        {config.timestep}")
        lines.append("")

        # Thermostat / Barostat fix
        ensemble = config.ensemble.lower()
        if ensemble == "nve":
            lines.append("fix             1 all nve")
        elif ensemble == "nvt":
            tdamp = config.temperature_damp * config.timestep / 100.0
            if tdamp < 1e-6:
                tdamp = config.timestep * 100.0
            lines.append(
                f"fix             1 all nvt temp {t_start:.1f} {t_end:.1f} "
                f"{config.temperature_damp:.1f}"
            )
        elif ensemble == "npt":
            lines.append(
                f"fix             1 all npt temp {t_start:.1f} {t_end:.1f} "
                f"{config.temperature_damp:.1f} "
                f"iso {config.pressure:.1f} {config.pressure:.1f} "
                f"{config.pressure_damp:.1f}"
            )
        lines.append("")

        # Thermo output
        lines.append(f"thermo          {config.thermo_interval}")
        lines.append("thermo_style    custom step temp pe ke etotal press vol")
        lines.append("")

        # Dump
        lines.append(
            f"dump            1 all custom {config.dump_interval} dump.lammpstrj "
            f"id type x y z vx vy vz fx fy fz"
        )
        lines.append("")

        # Extra commands
        for cmd in config.extra_commands:
            lines.append(cmd)
        if config.extra_commands:
            lines.append("")

        # Run
        lines.append(f"run             {config.nsteps}")
        lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        return filepath

    def _copy_potential_files(
        self, directory: str, config: LAMMPSInputConfig,
    ) -> None:
        """Copy potential files to the calculation directory."""
        if not config.potential_files:
            return

        pot_dir = resolve_potential_dir(config.potential_dir)
        for pot_file in config.potential_files:
            src = os.path.join(pot_dir, pot_file)
            dst = os.path.join(directory, pot_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                logger.warning(
                    "Potential file not found: %s (searched in %s). "
                    "The LAMMPS run will fail unless you provide it.",
                    pot_file, pot_dir,
                )

    # ------------------------------------------------------------------
    # parse_output
    # ------------------------------------------------------------------

    def parse_output(self, directory: str) -> DFTResult:
        """Parse LAMMPS log file and return a DFTResult.

        Extracts per-step thermo data (Step, Temp, PotEng, KinEng, TotEng,
        Press, Volume) from ``log.lammps``.

        Args:
            directory: Directory containing ``log.lammps``.

        Returns:
            DFTResult with final energy, temperature, and thermo history in raw.
        """
        log_path = os.path.join(directory, "log.lammps")
        if not os.path.exists(log_path):
            return DFTResult(
                is_converged=False,
                error_log=f"log.lammps not found in {directory}",
            )

        with open(log_path, "r") as f:
            content = f.read()

        thermo_data = self._parse_thermo(content)

        if not thermo_data["step"]:
            return DFTResult(
                is_converged=False,
                error_log="No thermo data found in log.lammps",
            )

        final_energy = thermo_data["etotal"][-1] if thermo_data["etotal"] else None

        return DFTResult(
            energy=final_energy,
            is_converged=True,
            raw={
                "thermo": thermo_data,
                "backend": "lammps",
            },
        )

    def _parse_thermo(self, content: str) -> Dict[str, List[float]]:
        """Parse LAMMPS thermo output from log content.

        Extracts columns between 'Step ...' header and 'Loop time ...' footer.
        """
        result: Dict[str, List[float]] = {
            "step": [], "temp": [], "pe": [], "ke": [],
            "etotal": [], "press": [], "vol": [],
        }

        # LAMMPS thermo blocks start with a header line like:
        # Step Temp PotEng KinEng TotEng Press Volume
        # and end with "Loop time of ..." or end of file
        in_thermo = False
        column_names: List[str] = []

        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("Loop time"):
                in_thermo = False
                continue

            if stripped.startswith("Step"):
                parts = stripped.split()
                if len(parts) >= 3 and all(
                    p in ("Step", "Temp", "PotEng", "KinEng", "TotEng",
                           "Press", "Volume", "v_", "c_", "Lx", "Ly", "Lz",
                           "E_pair", "E_mol", "E_long", "E_vdwl", "E_coul",
                           "Vol", "Pxx", "Pyy", "Pzz")
                    or p.startswith("v_") or p.startswith("c_")
                    for p in parts
                ):
                    in_thermo = True
                    column_names = [p.lower() for p in parts]
                    continue

            if in_thermo and column_names:
                parts = stripped.split()
                if len(parts) != len(column_names):
                    continue
                try:
                    values = [float(x) for x in parts]
                except ValueError:
                    in_thermo = False
                    continue

                col_map = dict(zip(column_names, values))
                for key in result:
                    # Map various LAMMPS column names to our keys
                    lammps_key = {
                        "step": "step", "temp": "temp",
                        "pe": "poteng", "ke": "kineng",
                        "etotal": "toteng", "press": "press", "vol": "volume",
                    }.get(key, key)
                    if lammps_key in col_map:
                        result[key].append(col_map[lammps_key])
                    elif key in col_map:
                        result[key].append(col_map[key])

        return result

    # ------------------------------------------------------------------
    # parse_trajectory
    # ------------------------------------------------------------------

    def parse_trajectory(self, directory: str) -> MDTrajectoryData:
        """Parse LAMMPS dump file into an MDTrajectoryData.

        Reads ``dump.lammpstrj`` in LAMMPS custom dump format with columns:
        ``id type x y z [vx vy vz] [fx fy fz]``.

        Args:
            directory: Directory containing ``dump.lammpstrj``.

        Returns:
            MDTrajectoryData with positions, energies (from log), and metadata.
        """
        dump_path = os.path.join(directory, "dump.lammpstrj")
        if not os.path.exists(dump_path):
            raise FileNotFoundError(f"dump.lammpstrj not found in {directory}")

        frames = self._parse_dump_file(dump_path)

        if not frames:
            raise ValueError("No frames found in dump.lammpstrj")

        n_frames = len(frames)
        n_atoms = len(frames[0]["positions"])

        positions = np.zeros((n_frames, n_atoms, 3))
        has_velocities = "velocities" in frames[0] and frames[0]["velocities"] is not None
        has_forces = "forces" in frames[0] and frames[0]["forces"] is not None
        velocities = np.zeros((n_frames, n_atoms, 3)) if has_velocities else None
        forces = np.zeros((n_frames, n_atoms, 3)) if has_forces else None
        timesteps = []

        for i, frame in enumerate(frames):
            positions[i] = frame["positions"]
            if has_velocities and velocities is not None:
                velocities[i] = frame["velocities"]
            if has_forces and forces is not None:
                forces[i] = frame["forces"]
            timesteps.append(frame.get("timestep", i))

        # Get species from type mapping (need data file for element names)
        species = frames[0].get("species", ["X"] * n_atoms)

        # Try to get thermo data from log for energies/temperatures
        log_result = self.parse_output(directory)
        thermo = log_result.raw.get("thermo", {})

        energies = np.array(thermo.get("etotal", [0.0] * n_frames)[:n_frames])
        temperatures = np.array(thermo.get("temp", [0.0] * n_frames)[:n_frames])
        times = np.array(timesteps, dtype=float)

        # Pressures
        pressures = None
        if thermo.get("press"):
            pressures = np.array(thermo["press"][:n_frames])

        return MDTrajectoryData(
            positions=positions,
            energies=energies,
            temperatures=temperatures,
            times=times,
            species=species,
            velocities=velocities,
            forces=forces,
            pressures=pressures,
            source="lammps",
        )

    def _parse_dump_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse LAMMPS custom dump file."""
        frames: List[Dict[str, Any]] = []

        with open(filepath, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "ITEM: TIMESTEP":
                i += 1
                timestep = int(lines[i].strip())
                i += 1
            elif line == "ITEM: NUMBER OF ATOMS":
                i += 1
                n_atoms = int(lines[i].strip())
                i += 1
            elif line.startswith("ITEM: BOX BOUNDS"):
                # Skip box bounds (3 lines)
                i += 1
                box_lines = []
                for _ in range(3):
                    box_lines.append(lines[i].strip())
                    i += 1
            elif line.startswith("ITEM: ATOMS"):
                # Parse column headers
                cols = line.replace("ITEM: ATOMS", "").strip().split()
                i += 1

                positions = np.zeros((n_atoms, 3))
                velocities_frame = None
                forces_frame = None

                has_v = "vx" in cols
                has_f = "fx" in cols
                if has_v:
                    velocities_frame = np.zeros((n_atoms, 3))
                if has_f:
                    forces_frame = np.zeros((n_atoms, 3))

                atom_data = []
                for _ in range(n_atoms):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split()
                    i += 1
                    if len(parts) < len(cols):
                        continue
                    atom_data.append(parts)

                # Sort by atom id
                id_idx = cols.index("id") if "id" in cols else 0
                atom_data.sort(key=lambda x: int(x[id_idx]))

                for j, parts in enumerate(atom_data):
                    col_vals = {c: parts[k] for k, c in enumerate(cols)}
                    positions[j] = [
                        float(col_vals.get("x", 0)),
                        float(col_vals.get("y", 0)),
                        float(col_vals.get("z", 0)),
                    ]
                    if has_v and velocities_frame is not None:
                        velocities_frame[j] = [
                            float(col_vals.get("vx", 0)),
                            float(col_vals.get("vy", 0)),
                            float(col_vals.get("vz", 0)),
                        ]
                    if has_f and forces_frame is not None:
                        forces_frame[j] = [
                            float(col_vals.get("fx", 0)),
                            float(col_vals.get("fy", 0)),
                            float(col_vals.get("fz", 0)),
                        ]

                frame: Dict[str, Any] = {
                    "timestep": timestep,
                    "positions": positions,
                }
                if velocities_frame is not None:
                    frame["velocities"] = velocities_frame
                if forces_frame is not None:
                    frame["forces"] = forces_frame

                frames.append(frame)
            else:
                i += 1

        return frames
