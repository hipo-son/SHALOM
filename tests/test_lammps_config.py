"""Tests for LAMMPS configuration, auto-detection, and presets.

Tests verify:
- detect_force_field() selects correct FF based on element composition
- detect_and_apply_lammps_hints() auto-configures all parameters from YAML
- LJ pair_coeff generation with Lorentz-Berthelot mixing
- Timestep auto-selection for light elements
- 2D boundary detection
- resolve_potential_dir() 3-tier fallback
- User override priority (pair_style set → skip auto-detection)
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms
from ase.build import bulk

from shalom.backends.lammps_config import (
    LAMMPSInputConfig,
    _generate_pair_coeff,
    detect_and_apply_lammps_hints,
    detect_force_field,
    get_lammps_preset,
    resolve_potential_dir,
)


# ---------------------------------------------------------------------------
# detect_force_field
# ---------------------------------------------------------------------------


class TestDetectForceField:
    """Test force field auto-detection from element composition."""

    def test_fe_selects_eam(self):
        atoms = bulk("Fe", "bcc", a=2.87)
        assert detect_force_field(atoms) == "eam_alloy"

    def test_si_selects_tersoff(self):
        atoms = bulk("Si", "diamond", a=5.43)
        assert detect_force_field(atoms) == "tersoff"

    def test_ar_selects_lj(self):
        atoms = Atoms("Ar", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        assert detect_force_field(atoms) == "lj_cut"

    def test_cu_selects_eam(self):
        atoms = bulk("Cu", "fcc", a=3.61)
        assert detect_force_field(atoms) == "eam_alloy"

    def test_c_selects_tersoff(self):
        atoms = bulk("C", "diamond", a=3.57)
        assert detect_force_field(atoms) == "tersoff"

    def test_unknown_element_returns_none(self):
        """Elements not in any supported_elements list → None."""
        atoms = Atoms("Ga", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        assert detect_force_field(atoms) is None

    def test_empty_atoms_returns_none(self):
        atoms = Atoms()
        assert detect_force_field(atoms) is None

    def test_alloy_fe_cu(self):
        """Multi-element alloy — both in EAM supported list."""
        atoms = Atoms(
            "FeCu",
            positions=[[0, 0, 0], [1.4, 1.4, 1.4]],
            cell=[2.87, 2.87, 2.87],
            pbc=True,
        )
        assert detect_force_field(atoms) == "eam_alloy"

    def test_si_c_compound(self):
        """SiC — both in Tersoff supported list."""
        atoms = Atoms(
            "SiC",
            positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
            cell=[4.36, 4.36, 4.36],
            pbc=True,
        )
        assert detect_force_field(atoms) == "tersoff"

    def test_priority_eam_over_others(self):
        """EAM has higher priority (90) than others."""
        atoms = bulk("Al", "fcc", a=4.05)
        assert detect_force_field(atoms) == "eam_alloy"

    def test_he_selects_lj(self):
        atoms = Atoms("He", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        result = detect_force_field(atoms)
        assert result == "lj_cut"

    def test_mixed_ff_unsupported_returns_none(self):
        """Fe + Si spans both EAM and Tersoff → no single FF covers both."""
        atoms = Atoms(
            "FeSi",
            positions=[[0, 0, 0], [1.4, 1.4, 1.4]],
            cell=[2.87, 2.87, 2.87],
            pbc=True,
        )
        assert detect_force_field(atoms) is None


# ---------------------------------------------------------------------------
# detect_and_apply_lammps_hints
# ---------------------------------------------------------------------------


class TestDetectAndApplyLammpsHints:
    """Test structure-aware auto-detection for LAMMPS configs."""

    def test_fe_eam_auto_config(self):
        atoms = bulk("Fe", "bcc", a=2.87)
        config = LAMMPSInputConfig()
        detect_and_apply_lammps_hints(atoms, config)
        assert config.pair_style == "eam/alloy"
        assert config.detected_ff == "eam_alloy"
        assert len(config.pair_coeff) == 1
        assert "Fe_mendelev.eam.fs" in config.pair_coeff[0]
        assert "Fe" in config.pair_coeff[0]
        assert config.timestep == 1.0
        assert config.units == "metal"

    def test_si_tersoff_auto_config(self):
        atoms = bulk("Si", "diamond", a=5.43)
        config = LAMMPSInputConfig()
        detect_and_apply_lammps_hints(atoms, config)
        assert config.pair_style == "tersoff"
        assert "SiCGe.tersoff" in config.pair_coeff[0]
        assert config.timestep == 0.5

    def test_ar_lj_auto_config(self):
        atoms = Atoms("Ar", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        config = LAMMPSInputConfig()
        detect_and_apply_lammps_hints(atoms, config)
        assert config.pair_style == "lj/cut 10.0"
        # LJ pair_coeff: "1 1 epsilon sigma"
        assert len(config.pair_coeff) == 1
        assert "0.01039" in config.pair_coeff[0]  # epsilon
        assert "3.405" in config.pair_coeff[0]  # sigma

    def test_lj_two_elements_mixing(self):
        """LJ with two elements → Lorentz-Berthelot mixing for cross-pair."""
        atoms = Atoms(
            "ArNe",
            positions=[[0, 0, 0], [5, 5, 5]],
            cell=[10, 10, 10],
            pbc=True,
        )
        config = LAMMPSInputConfig()
        detect_and_apply_lammps_hints(atoms, config)
        # 2 elements → 3 pair_coeff entries (1-1, 1-2, 2-2)
        assert len(config.pair_coeff) == 3

    def test_user_override_skips_autodetect(self):
        """If pair_style is already set, auto-detection is skipped."""
        atoms = bulk("Fe", "bcc", a=2.87)
        config = LAMMPSInputConfig(pair_style="eam/fs")
        detect_and_apply_lammps_hints(atoms, config)
        assert config.pair_style == "eam/fs"
        assert config.detected_ff is None  # Not auto-detected
        assert config.pair_coeff == []  # Not auto-generated

    def test_unknown_elements_warning(self):
        """Unknown elements → warning logged, no changes."""
        atoms = Atoms("Ga", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        config = LAMMPSInputConfig()
        detect_and_apply_lammps_hints(atoms, config)
        assert config.pair_style == ""  # Unchanged
        assert config.detected_ff is None

    def test_empty_atoms_unchanged(self):
        config = LAMMPSInputConfig()
        detect_and_apply_lammps_hints(Atoms(), config)
        assert config.pair_style == ""

    def test_2d_detection_boundary(self):
        """2D structure → boundary changes to 'p p f'."""
        # Create a 2D slab with large vacuum in z
        atoms = Atoms(
            "C2",
            positions=[[0, 0, 5], [1.23, 0.71, 5]],
            cell=[[2.46, 0, 0], [1.23, 2.13, 0], [0, 0, 20]],
            pbc=True,
        )
        config = LAMMPSInputConfig()
        detect_and_apply_lammps_hints(atoms, config)
        assert config.is_2d is True
        assert config.boundary == "p p f"

    def test_npt_pressure_damp_from_yaml(self):
        atoms = bulk("Fe", "bcc", a=2.87)
        config = LAMMPSInputConfig(ensemble="npt")
        detect_and_apply_lammps_hints(atoms, config)
        assert config.pressure_damp == 1000.0  # eam_alloy npt_damp

    def test_thermostat_damp_from_yaml(self):
        atoms = bulk("Si", "diamond", a=5.43)
        config = LAMMPSInputConfig()
        detect_and_apply_lammps_hints(atoms, config)
        assert config.temperature_damp == 50.0  # tersoff nvt_damp


# ---------------------------------------------------------------------------
# _generate_pair_coeff
# ---------------------------------------------------------------------------


class TestGeneratePairCoeff:
    """Test pair_coeff generation for different force field types."""

    def test_eam_single_element(self):
        from shalom._config_loader import load_config

        db = load_config("lammps_potentials")
        ff_data = db["eam_alloy"]
        pair_coeff, pot_files = _generate_pair_coeff("eam_alloy", ff_data, ["Fe"])
        assert len(pair_coeff) == 1
        assert pair_coeff[0].startswith("* *")
        assert "Fe" in pair_coeff[0]
        assert len(pot_files) >= 1

    def test_tersoff_si(self):
        from shalom._config_loader import load_config

        db = load_config("lammps_potentials")
        ff_data = db["tersoff"]
        pair_coeff, pot_files = _generate_pair_coeff("tersoff", ff_data, ["Si"])
        assert "SiCGe.tersoff" in pair_coeff[0]
        assert "Si" in pair_coeff[0]

    def test_lj_single_element(self):
        from shalom._config_loader import load_config

        db = load_config("lammps_potentials")
        ff_data = db["lj_cut"]
        pair_coeff, pot_files = _generate_pair_coeff("lj_cut", ff_data, ["Ar"])
        assert len(pair_coeff) == 1
        assert "0.01039" in pair_coeff[0]  # epsilon_Ar
        assert "3.405" in pair_coeff[0]  # sigma_Ar
        assert pot_files == []  # LJ needs no files

    def test_lj_cross_pair(self):
        from shalom._config_loader import load_config

        db = load_config("lammps_potentials")
        ff_data = db["lj_cut"]
        pair_coeff, _ = _generate_pair_coeff("lj_cut", ff_data, ["Ar", "Ne"])
        # 3 entries: 1-1, 1-2, 2-2
        assert len(pair_coeff) == 3


# ---------------------------------------------------------------------------
# resolve_potential_dir
# ---------------------------------------------------------------------------


class TestResolvePotentialDir:
    """Test 3-tier potential directory resolution."""

    def test_explicit_dir(self):
        result = resolve_potential_dir("/tmp/my_potentials")
        assert "my_potentials" in result

    def test_env_var(self):
        with patch.dict(os.environ, {"SHALOM_LAMMPS_POTENTIALS": "/opt/potentials"}):
            result = resolve_potential_dir()
            assert "potentials" in result

    def test_default_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SHALOM_LAMMPS_POTENTIALS", None)
            with patch("shalom.backends.lammps_config.Path.home", return_value=Path("/mock/home")):
                result = resolve_potential_dir()
                assert "lammps-potentials" in result

    def test_explicit_overrides_env(self):
        with patch.dict(os.environ, {"SHALOM_LAMMPS_POTENTIALS": "/opt/potentials"}):
            result = resolve_potential_dir("/my/custom/dir")
            assert "custom" in result


# ---------------------------------------------------------------------------
# get_lammps_preset
# ---------------------------------------------------------------------------


class TestGetLammpsPreset:
    """Test preset factory function."""

    def test_preset_without_atoms(self):
        config = get_lammps_preset(ensemble="nvt")
        assert config.ensemble == "nvt"
        assert config.pair_style == ""  # No atoms → no auto-detect

    def test_preset_with_atoms_fe(self):
        atoms = bulk("Fe", "bcc", a=2.87)
        config = get_lammps_preset(ensemble="nvt", atoms=atoms)
        assert config.pair_style == "eam/alloy"
        assert config.ensemble == "nvt"

    def test_preset_npt(self):
        atoms = bulk("Fe", "bcc", a=2.87)
        config = get_lammps_preset(ensemble="npt", atoms=atoms)
        assert config.ensemble == "npt"
        assert config.pressure_damp == 1000.0

    def test_preset_nve(self):
        config = get_lammps_preset(ensemble="nve")
        assert config.ensemble == "nve"


# ---------------------------------------------------------------------------
# LAMMPSInputConfig
# ---------------------------------------------------------------------------


class TestLAMMPSInputConfig:
    """Test config dataclass defaults."""

    def test_default_values(self):
        config = LAMMPSInputConfig()
        assert config.ensemble == "nvt"
        assert config.temperature == 300.0
        assert config.timestep == 1.0
        assert config.nsteps == 100000
        assert config.pair_style == ""
        assert config.pair_coeff == []
        assert config.units == "metal"
        assert config.atom_style == "atomic"
        assert config.boundary == "p p p"
        assert config.minimize_first is False

    def test_custom_values(self):
        config = LAMMPSInputConfig(
            ensemble="npt",
            temperature=1000.0,
            pressure=10.0,
            pair_style="eam/alloy",
        )
        assert config.ensemble == "npt"
        assert config.temperature == 1000.0
        assert config.pressure == 10.0
        assert config.pair_style == "eam/alloy"
