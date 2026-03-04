"""Tests for config to_summary_dict(), _detection_log, and _build_structure_analysis."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk as ase_bulk


# ---------------------------------------------------------------------------
# VASP summary & detection log
# ---------------------------------------------------------------------------


class TestVASPSummary:
    def test_basic_json_serializable(self):
        from shalom.backends.vasp_config import VASPInputConfig, CalculationType
        from shalom.backends._physics import AccuracyLevel

        cfg = VASPInputConfig(
            calc_type=CalculationType.RELAXATION,
            accuracy=AccuracyLevel.STANDARD,
            incar_settings={"ENCUT": 520, "ISMEAR": 0},
            kpoints=__import__(
                "shalom.backends.vasp_config", fromlist=["KPointsConfig"]
            ).KPointsConfig(grid=[4, 4, 4]),
        )
        d = cfg.to_summary_dict()
        # Must be JSON-serializable
        s = json.dumps(d)
        assert isinstance(json.loads(s), dict)
        assert d["calc_type"] == "relaxation"
        assert d["accuracy"] == "standard"
        assert d["kpoints_grid"] == [4, 4, 4]
        assert d["incar"]["ENCUT"] == 520

    def test_user_overrides_merged(self):
        from shalom.backends.vasp_config import VASPInputConfig

        cfg = VASPInputConfig(
            incar_settings={"ENCUT": 520, "ISMEAR": 0},
            user_incar_settings={"ENCUT": 700, "NSW": 100},
        )
        d = cfg.to_summary_dict()
        # user overrides should win
        assert d["incar"]["ENCUT"] == 700
        assert d["incar"]["NSW"] == 100

    def test_none_omitted(self):
        from shalom.backends.vasp_config import VASPInputConfig

        cfg = VASPInputConfig()
        d = cfg.to_summary_dict()
        assert "vdw_correction_IVDW" not in d
        assert "ldau_settings" not in d
        assert "initial_magmom" not in d
        assert "enmax_values_eV" not in d

    def test_ldau_included(self):
        from shalom.backends.vasp_config import VASPInputConfig

        cfg = VASPInputConfig(ldau_settings={"LDAU": True, "LDAUTYPE": 2})
        d = cfg.to_summary_dict()
        assert d["ldau_settings"]["LDAU"] is True

    def test_detection_log_magnetic(self):
        from shalom.backends.vasp_config import get_preset, CalculationType
        from shalom.backends._physics import AccuracyLevel

        fe = ase_bulk("Fe")
        cfg = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, atoms=fe)
        assert any("Magnetic" in m for m in cfg._detection_log)

    def test_detection_log_encut(self):
        from shalom.backends.vasp_config import get_preset, CalculationType
        from shalom.backends._physics import AccuracyLevel

        si = ase_bulk("Si")
        cfg = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, atoms=si)
        assert any("ENCUT=" in m for m in cfg._detection_log)

    def test_detection_log_pure_metal(self):
        from shalom.backends.vasp_config import get_preset, CalculationType
        from shalom.backends._physics import AccuracyLevel

        al = ase_bulk("Al")
        cfg = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, atoms=al)
        assert any("Pure metal" in m for m in cfg._detection_log)

    def test_detection_log_kgrid(self):
        from shalom.backends.vasp_config import get_preset, CalculationType
        from shalom.backends._physics import AccuracyLevel

        si = ase_bulk("Si")
        cfg = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, atoms=si)
        assert any("K-grid" in m for m in cfg._detection_log)


# ---------------------------------------------------------------------------
# QE summary & detection log
# ---------------------------------------------------------------------------


class TestQESummary:
    def test_basic_json_serializable(self):
        from shalom.backends.qe_config import get_qe_preset, QECalculationType
        from shalom.backends._physics import AccuracyLevel

        si = ase_bulk("Si")
        cfg = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms=si)
        d = cfg.to_summary_dict()
        s = json.dumps(d)
        assert isinstance(json.loads(s), dict)
        assert d["calc_type"] == "scf"
        assert "ecutwfc" in d.get("system", {})

    def test_pseudo_map_included(self):
        from shalom.backends.qe_config import get_qe_preset, QECalculationType
        from shalom.backends._physics import AccuracyLevel

        si = ase_bulk("Si")
        cfg = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms=si)
        d = cfg.to_summary_dict()
        assert "pseudo_map" in d
        assert "Si" in d["pseudo_map"]

    def test_detection_log_ecutwfc(self):
        from shalom.backends.qe_config import get_qe_preset, QECalculationType
        from shalom.backends._physics import AccuracyLevel

        si = ase_bulk("Si")
        cfg = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms=si)
        assert any("ecutwfc=" in m for m in cfg._detection_log)

    def test_detection_log_kgrid(self):
        from shalom.backends.qe_config import get_qe_preset, QECalculationType
        from shalom.backends._physics import AccuracyLevel

        si = ase_bulk("Si")
        cfg = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms=si)
        assert any("K-grid" in m for m in cfg._detection_log)

    def test_empty_namelists_omitted(self):
        from shalom.backends.qe_config import QEInputConfig

        cfg = QEInputConfig()
        d = cfg.to_summary_dict()
        # Empty namelists should not appear
        for ns in ("control", "system", "electrons", "ions", "cell"):
            if ns in d:
                assert d[ns]  # if present, must be non-empty


# ---------------------------------------------------------------------------
# LAMMPS summary & detection log
# ---------------------------------------------------------------------------


class TestLAMMPSSummary:
    def test_basic_json_serializable(self):
        from shalom.backends.lammps_config import LAMMPSInputConfig

        cfg = LAMMPSInputConfig(pair_style="eam/alloy", temperature=500.0)
        d = cfg.to_summary_dict()
        s = json.dumps(d)
        assert isinstance(json.loads(s), dict)
        assert d["temperature_K"] == 500.0
        assert d["pair_style"] == "eam/alloy"

    def test_npt_includes_pressure(self):
        from shalom.backends.lammps_config import LAMMPSInputConfig

        cfg = LAMMPSInputConfig(ensemble="npt", pressure=1.0)
        d = cfg.to_summary_dict()
        assert "pressure_bar" in d
        assert d["pressure_bar"] == 1.0

    def test_nvt_omits_pressure(self):
        from shalom.backends.lammps_config import LAMMPSInputConfig

        cfg = LAMMPSInputConfig(ensemble="nvt")
        d = cfg.to_summary_dict()
        assert "pressure_bar" not in d

    def test_detection_log_ff(self):
        from shalom.backends.lammps_config import get_lammps_preset

        fe = ase_bulk("Fe")
        cfg = get_lammps_preset(ensemble="nvt", atoms=fe)
        assert any("Force field" in m for m in cfg._detection_log)

    def test_detection_log_timestep(self):
        from shalom.backends.lammps_config import get_lammps_preset

        fe = ase_bulk("Fe")
        cfg = get_lammps_preset(ensemble="nvt", atoms=fe)
        assert any("Timestep" in m or "timestep" in m for m in cfg._detection_log)


# ---------------------------------------------------------------------------
# _build_structure_analysis
# ---------------------------------------------------------------------------


class TestStructureAnalysis:
    def test_fe2o3_magnetic(self):
        from shalom.direct_run import _build_structure_analysis

        # Simple Fe2O3-like structure (2 Fe + 3 O)
        atoms = Atoms(
            "Fe2O3",
            positions=[[0, 0, 0], [1, 1, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
            cell=[5, 5, 5],
            pbc=True,
        )
        d = _build_structure_analysis(atoms)
        assert d["is_magnetic"] is True
        assert "Fe" in d["magnetic_elements"]
        assert d["natoms"] == 5
        assert "Fe" in d["elements"]
        assert "O" in d["elements"]

    def test_si_nonmagnetic(self):
        from shalom.direct_run import _build_structure_analysis

        si = ase_bulk("Si")
        d = _build_structure_analysis(si)
        assert d["is_magnetic"] is False
        assert "magnetic_elements" not in d

    def test_al_metal(self):
        from shalom.direct_run import _build_structure_analysis

        al = ase_bulk("Al")
        d = _build_structure_analysis(al)
        assert d["is_metal"] is True

    def test_json_serializable(self):
        from shalom.direct_run import _build_structure_analysis

        si = ase_bulk("Si")
        d = _build_structure_analysis(si)
        s = json.dumps(d)
        assert isinstance(json.loads(s), dict)
