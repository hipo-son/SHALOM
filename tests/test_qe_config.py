"""Tests for shalom.backends.qe_config module."""

import pytest
from ase import Atoms
from ase.build import bulk, fcc111

from shalom.backends._physics import AccuracyLevel
from shalom.backends.qe_config import (
    QECalculationType,
    QEInputConfig,
    VASP_TO_QE_CALC_MAP,
    get_qe_preset,
    get_pseudo_filename,
    get_z_valence,
    compute_ecutwfc,
    compute_ecutrho,
    detect_and_apply_qe_hints,
)


# ---------------------------------------------------------------------------
# QECalculationType
# ---------------------------------------------------------------------------


class TestQECalculationType:
    def test_scf(self):
        assert QECalculationType.SCF.value == "scf"

    def test_vc_relax(self):
        assert QECalculationType.VC_RELAX.value == "vc-relax"


# ---------------------------------------------------------------------------
# VASP_TO_QE_CALC_MAP
# ---------------------------------------------------------------------------


class TestVASPToQECalcMap:
    def test_relaxation_maps_to_vc_relax(self):
        """Critical: VASP relaxation (ISIF=3) = QE vc-relax, NOT relax."""
        assert VASP_TO_QE_CALC_MAP["relaxation"] == QECalculationType.VC_RELAX

    def test_static_maps_to_scf(self):
        assert VASP_TO_QE_CALC_MAP["static"] == QECalculationType.SCF

    def test_band_structure_maps_to_bands(self):
        assert VASP_TO_QE_CALC_MAP["band_structure"] == QECalculationType.BANDS

    def test_dos_maps_to_nscf(self):
        assert VASP_TO_QE_CALC_MAP["dos"] == QECalculationType.NSCF


# ---------------------------------------------------------------------------
# QEInputConfig
# ---------------------------------------------------------------------------


class TestQEInputConfig:
    def test_defaults(self):
        config = QEInputConfig()
        assert config.calc_type == QECalculationType.SCF
        assert config.accuracy == AccuracyLevel.STANDARD
        assert config.pseudo_dir == "./"
        assert config.is_2d is False

    def test_get_merged_namelists_scf(self):
        config = QEInputConfig(
            control={"calculation": "scf"},
            system={"ecutwfc": 60},
            electrons={"conv_thr": 1e-7},
        )
        merged = config.get_merged_namelists()
        assert "CONTROL" in merged
        assert "SYSTEM" in merged
        assert "ELECTRONS" in merged
        assert "IONS" not in merged
        assert "CELL" not in merged

    def test_get_merged_namelists_vc_relax(self):
        config = QEInputConfig(
            calc_type=QECalculationType.VC_RELAX,
            ions={"ion_dynamics": "bfgs"},
            cell={"cell_dynamics": "bfgs"},
        )
        merged = config.get_merged_namelists()
        assert "IONS" in merged
        assert "CELL" in merged

    def test_user_settings_dot_notation(self):
        config = QEInputConfig(
            system={"ecutwfc": 60},
            user_settings={"system.ecutwfc": 80, "electrons.conv_thr": 1e-9},
        )
        merged = config.get_merged_namelists()
        assert merged["SYSTEM"]["ecutwfc"] == 80
        assert merged["ELECTRONS"]["conv_thr"] == 1e-9

    def test_user_settings_no_dot_goes_to_system(self):
        config = QEInputConfig(user_settings={"ecutwfc": 80})
        merged = config.get_merged_namelists()
        assert merged["SYSTEM"]["ecutwfc"] == 80


# ---------------------------------------------------------------------------
# SSSP metadata functions
# ---------------------------------------------------------------------------


class TestSSSPMetadata:
    def test_cu_pseudo_filename(self):
        pseudo = get_pseudo_filename("Cu")
        assert "Cu" in pseudo
        assert pseudo.endswith(".UPF") or pseudo.endswith(".upf")

    def test_unknown_element_fallback(self):
        pseudo = get_pseudo_filename("Uuo")
        assert pseudo == "Uuo.UPF"

    def test_cu_z_valence(self):
        z = get_z_valence("Cu")
        assert z == 11

    def test_fe_z_valence(self):
        z = get_z_valence("Fe")
        assert z == 16

    def test_unknown_z_valence_fallback(self):
        z = get_z_valence("Uuo")
        assert z == 1


class TestComputeEcutwfc:
    def test_cu_standard(self):
        ecutwfc = compute_ecutwfc(["Cu"], AccuracyLevel.STANDARD)
        assert ecutwfc >= 45.0  # minimum floor

    def test_fe_precise(self):
        ecutwfc = compute_ecutwfc(["Fe"], AccuracyLevel.PRECISE)
        assert ecutwfc >= 60.0  # minimum floor for precise

    def test_unknown_element_fallback(self):
        ecutwfc = compute_ecutwfc(["Uuo"], AccuracyLevel.STANDARD)
        assert ecutwfc == 45.0

    def test_multi_element_takes_max(self):
        # Fe has higher ecutwfc than Cu in SSSP
        ecutwfc = compute_ecutwfc(["Fe", "Cu"], AccuracyLevel.STANDARD)
        fe_only = compute_ecutwfc(["Fe"], AccuracyLevel.STANDARD)
        assert ecutwfc >= fe_only


class TestComputeEcutrho:
    def test_ecutrho_from_sssp(self):
        """ecutrho comes from SSSP per-element data, NOT blanket 8x."""
        ecutrho = compute_ecutrho(["Fe"], AccuracyLevel.STANDARD)
        # Fe SSSP: ecutrho=1080 (12x ecutwfc=90)
        assert ecutrho >= 1080

    def test_fallback_uses_8x(self):
        ecutrho = compute_ecutrho(["Uuo"], AccuracyLevel.STANDARD)
        ecutwfc = compute_ecutwfc(["Uuo"], AccuracyLevel.STANDARD)
        assert ecutrho == ecutwfc * 8.0


# ---------------------------------------------------------------------------
# detect_and_apply_qe_hints
# ---------------------------------------------------------------------------


class TestDetectAndApplyQEHints:
    def test_fe_magnetic_starting_mag(self):
        """Fe starting_magnetization uses z_valence (not /10)."""
        atoms = bulk("Fe", "bcc", a=2.87)
        config = QEInputConfig()
        detect_and_apply_qe_hints(atoms, config)
        assert config.system["nspin"] == 2
        # Fe: MAGMOM=5.0, z_valence=16 → 5.0/16 = 0.3125
        key = "starting_magnetization(1)"
        assert key in config.system
        assert abs(config.system[key] - 0.3125) < 0.01

    def test_cu_metal_smearing(self):
        """Pure metal uses methfessel-paxton smearing."""
        atoms = bulk("Cu", "fcc", a=3.6)
        config = QEInputConfig()
        detect_and_apply_qe_hints(atoms, config)
        assert config.system["smearing"] == "methfessel-paxton"
        assert abs(config.system["degauss"] - 0.00735) < 0.0001

    def test_ecutwfc_ecutrho_set(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = QEInputConfig()
        detect_and_apply_qe_hints(atoms, config)
        assert "ecutwfc" in config.system
        assert "ecutrho" in config.system
        assert config.system["ecutwfc"] > 0
        assert config.system["ecutrho"] > config.system["ecutwfc"]

    def test_kpoints_grid_set(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = QEInputConfig()
        detect_and_apply_qe_hints(atoms, config)
        assert config.kpoints.grid is not None
        assert len(config.kpoints.grid) == 3

    def test_pseudo_map_set(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = QEInputConfig()
        detect_and_apply_qe_hints(atoms, config)
        assert "Cu" in config.pseudo_map

    def test_ibrav_nat_ntyp(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = QEInputConfig()
        detect_and_apply_qe_hints(atoms, config)
        assert config.system["ibrav"] == 0
        assert config.system["nat"] == len(atoms)
        assert config.system["ntyp"] == 1

    def test_2d_settings(self):
        """2D slab gets assume_isolated, vdw_corr, cell_dofree."""
        atoms = fcc111("Cu", size=(1, 1, 3), vacuum=15.0)
        config = QEInputConfig(calc_type=QECalculationType.VC_RELAX)
        detect_and_apply_qe_hints(atoms, config)
        assert config.is_2d is True
        assert config.system["assume_isolated"] == "2D"
        assert config.system["vdw_corr"] == "dft-d3"
        assert config.system["dftd3_version"] == 4  # BJ damping
        assert config.cell.get("cell_dofree") == "2Dxy"
        assert config.kpoints.grid[2] == 1

    def test_2d_scf_no_cell_dofree(self):
        """SCF on 2D slab does NOT set cell_dofree."""
        atoms = fcc111("Cu", size=(1, 1, 3), vacuum=15.0)
        config = QEInputConfig(calc_type=QECalculationType.SCF)
        detect_and_apply_qe_hints(atoms, config)
        assert "cell_dofree" not in config.cell

    def test_feo_gga_u_precise(self):
        """FeO with PRECISE → lda_plus_u + Hubbard_U."""
        atoms = Atoms("FeO", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4], pbc=True)
        config = QEInputConfig(accuracy=AccuracyLevel.PRECISE)
        detect_and_apply_qe_hints(atoms, config)
        assert config.system.get("lda_plus_u") is True
        # Fe should have Hubbard_U
        assert any("Hubbard_U" in k for k in config.system)

    def test_feo_standard_no_u(self):
        """FeO with STANDARD → NO lda_plus_u."""
        atoms = Atoms("FeO", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4], pbc=True)
        config = QEInputConfig(accuracy=AccuracyLevel.STANDARD)
        detect_and_apply_qe_hints(atoms, config)
        assert "lda_plus_u" not in config.system

    def test_empty_atoms(self):
        """Empty Atoms → no crash."""
        atoms = Atoms()
        config = QEInputConfig()
        result = detect_and_apply_qe_hints(atoms, config)
        assert result is config


# ---------------------------------------------------------------------------
# get_qe_preset
# ---------------------------------------------------------------------------


class TestGetQEPreset:
    def test_scf_standard(self):
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD)
        assert config.calc_type == QECalculationType.SCF
        assert config.control.get("calculation") == "scf"
        assert config.control.get("prefix") == "shalom"

    def test_vc_relax_standard(self):
        config = get_qe_preset(QECalculationType.VC_RELAX, AccuracyLevel.STANDARD)
        assert config.calc_type == QECalculationType.VC_RELAX
        assert config.control.get("nstep") == 200

    def test_precise_conv_thr(self):
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.PRECISE)
        assert config.electrons.get("conv_thr") < 1e-8

    def test_with_atoms(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        config = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD, atoms=atoms)
        assert config.system.get("ecutwfc") is not None
        assert config.kpoints.grid is not None

    def test_relax_tighter_conv_thr(self):
        """Relaxation presets use tighter conv_thr than SCF."""
        scf = get_qe_preset(QECalculationType.SCF, AccuracyLevel.STANDARD)
        relax = get_qe_preset(QECalculationType.RELAX, AccuracyLevel.STANDARD)
        vc_relax = get_qe_preset(QECalculationType.VC_RELAX, AccuracyLevel.STANDARD)
        assert relax.electrons["conv_thr"] < scf.electrons["conv_thr"]
        assert vc_relax.electrons["conv_thr"] < scf.electrons["conv_thr"]


# ---------------------------------------------------------------------------
# Additional detect_and_apply_qe_hints tests
# ---------------------------------------------------------------------------


class TestDetectAndApplyQEHintsExtra:
    def test_non_metal_reduced_degauss(self):
        """Non-magnetic non-metal (Si) gets reduced degauss."""
        atoms = bulk("Si", "diamond", a=5.43)
        config = QEInputConfig()
        detect_and_apply_qe_hints(atoms, config)
        assert config.system["degauss"] == pytest.approx(0.00147, abs=0.0001)

    def test_metal_lower_mixing_beta(self):
        """Pure metal gets mixing_beta=0.4 for SCF stability."""
        atoms = bulk("Cu", "fcc", a=3.6)
        config = QEInputConfig()
        detect_and_apply_qe_hints(atoms, config)
        assert config.electrons["mixing_beta"] == 0.4

    def test_gga_u_standard_warning(self, caplog):
        """GGA+U skipped at STANDARD logs a warning."""
        import logging
        atoms = Atoms("FeO", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4], pbc=True)
        config = QEInputConfig(accuracy=AccuracyLevel.STANDARD)
        with caplog.at_level(logging.WARNING, logger="shalom.backends.qe_config"):
            detect_and_apply_qe_hints(atoms, config)
        assert any("GGA+U disabled" in msg for msg in caplog.messages)

    def test_multi_element_magnetic_system(self):
        """Fe+Co+O multi-element magnetic system."""
        atoms = Atoms(
            "FeCoO2",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0], [1, -1, 0]],
            cell=[4, 4, 4], pbc=True,
        )
        config = QEInputConfig(accuracy=AccuracyLevel.PRECISE)
        detect_and_apply_qe_hints(atoms, config)
        assert config.system["nspin"] == 2
        assert config.system.get("lda_plus_u") is True

    def test_user_settings_mixed_override(self):
        """Both dotted and non-dotted user_settings keys."""
        config = QEInputConfig(
            system={"ecutwfc": 60, "ecutrho": 480},
            electrons={"conv_thr": 1e-7},
            user_settings={"system.ecutwfc": 80, "degauss": 0.01},
        )
        merged = config.get_merged_namelists()
        assert merged["SYSTEM"]["ecutwfc"] == 80
        assert merged["SYSTEM"]["degauss"] == 0.01

    def test_ecutwfc_empty_elements(self):
        """Empty element list → fallback cutoffs."""
        ecutwfc = compute_ecutwfc([], AccuracyLevel.STANDARD)
        assert ecutwfc == 45.0

    def test_hubbard_u_non_pbe_warning(self, caplog):
        """GGA+U with non-PBE functional logs a warning."""
        import logging
        atoms = Atoms("FeO", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4], pbc=True)
        config = QEInputConfig(accuracy=AccuracyLevel.PRECISE, functional="TPSS")
        with caplog.at_level(logging.WARNING, logger="shalom.backends.qe_config"):
            detect_and_apply_qe_hints(atoms, config)
        assert any("PBE only" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# SSSP metadata sync test
# ---------------------------------------------------------------------------


class TestSSSPMetadataSync:
    def test_defaults_sssp_matches_yaml(self):
        """All elements in YAML sssp_metadata exist in _defaults.py."""
        from shalom._config_loader import load_config
        from shalom._defaults import CONFIGS

        yaml_cfg = load_config("sssp_metadata")
        defaults_cfg = CONFIGS["sssp_metadata"]

        yaml_elements = set(yaml_cfg["elements"].keys())
        defaults_elements = set(defaults_cfg["elements"].keys())

        assert yaml_elements == defaults_elements, (
            f"YAML vs _defaults.py mismatch: "
            f"only in YAML={yaml_elements - defaults_elements}, "
            f"only in defaults={defaults_elements - yaml_elements}"
        )


# ---------------------------------------------------------------------------
# get_band_calc_atoms
# ---------------------------------------------------------------------------


class TestGetBandCalcAtoms:
    from shalom.backends.qe_config import get_band_calc_atoms

    def test_returns_none_for_2d(self):
        from shalom.backends.qe_config import get_band_calc_atoms

        atoms = bulk("Si", "diamond", a=5.43)
        result = get_band_calc_atoms(atoms, is_2d=True)
        assert result is None

    def test_returns_none_without_seekpath(self, monkeypatch):
        """ImportError for seekpath → None (no crash)."""
        import sys
        from shalom.backends.qe_config import get_band_calc_atoms

        monkeypatch.setitem(sys.modules, "seekpath", None)
        atoms = bulk("Si", "diamond", a=5.43)
        result = get_band_calc_atoms(atoms, is_2d=False)
        assert result is None

    def test_returns_ase_atoms_with_seekpath(self, monkeypatch):
        """Mocked seekpath → Atoms with seekpath's primitive cell."""
        import sys
        import types
        from shalom.backends.qe_config import get_band_calc_atoms

        fake_seekpath = types.ModuleType("seekpath")
        fake_path_data = {
            "primitive_types": [14, 14],
            "primitive_positions": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            "primitive_lattice": [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]],
        }
        fake_seekpath.get_path = lambda *a, **kw: fake_path_data
        monkeypatch.setitem(sys.modules, "seekpath", fake_seekpath)

        atoms = bulk("Si", "diamond", a=5.43)
        result = get_band_calc_atoms(atoms, is_2d=False)
        assert result is not None
        assert len(result) == 2
        assert list(result.get_atomic_numbers()) == [14, 14]

    def test_returns_none_on_seekpath_exception(self, monkeypatch):
        """seekpath.get_path raising → None (no crash)."""
        import sys
        import types
        from shalom.backends.qe_config import get_band_calc_atoms

        fake_seekpath = types.ModuleType("seekpath")
        fake_seekpath.get_path = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("oops"))
        monkeypatch.setitem(sys.modules, "seekpath", fake_seekpath)

        atoms = bulk("Si", "diamond", a=5.43)
        result = get_band_calc_atoms(atoms, is_2d=False)
        assert result is None


# ---------------------------------------------------------------------------
# generate_band_kpath
# ---------------------------------------------------------------------------


class TestGenerateBandKpath:
    def _si(self):
        return bulk("Si", "diamond", a=5.43)

    def test_returns_kpath_config(self):
        from shalom.backends.qe_config import generate_band_kpath, QEKPointsConfig

        cfg = generate_band_kpath(self._si())
        assert isinstance(cfg, QEKPointsConfig)
        assert cfg.mode == "crystal_b"

    def test_kpath_labels_contains_gamma(self):
        from shalom.backends.qe_config import generate_band_kpath

        cfg = generate_band_kpath(self._si())
        all_labels = list(cfg.kpath_labels.values())
        # Gamma should appear at least once in any reasonable k-path
        assert any("G" in lbl or "Gamma" in lbl or "GAMMA" in lbl for lbl in all_labels)

    def test_kpath_points_populated(self):
        from shalom.backends.qe_config import generate_band_kpath

        cfg = generate_band_kpath(self._si())
        assert len(cfg.kpath_points) > 0
        for coords, npts in cfg.kpath_points:
            assert len(coords) == 3
            assert isinstance(npts, int) and npts >= 1

    def test_2d_forces_kz_zero(self):
        from shalom.backends.qe_config import generate_band_kpath

        atoms = bulk("Si", "diamond", a=5.43)
        cfg = generate_band_kpath(atoms, is_2d=True)
        for coords, _ in cfg.kpath_points:
            assert coords[2] == pytest.approx(0.0)

    def test_last_point_npts_one(self):
        from shalom.backends.qe_config import generate_band_kpath

        cfg = generate_band_kpath(self._si())
        _, last_npts = cfg.kpath_points[-1]
        assert last_npts == 1

    def test_fallback_on_seekpath_importerror(self, monkeypatch):
        """When seekpath and ASE bandpath both fail, hardcoded table still works."""
        import sys
        from shalom.backends.qe_config import generate_band_kpath, QEKPointsConfig

        monkeypatch.setitem(sys.modules, "seekpath", None)
        # Also cripple ASE bandpath by patching cell.bandpath to raise
        from unittest.mock import patch

        with patch("ase.cell.Cell.bandpath", side_effect=Exception("no bandpath")):
            cfg = generate_band_kpath(self._si())
        assert isinstance(cfg, QEKPointsConfig)
        assert len(cfg.kpath_points) > 0

    def test_npoints_applied_to_intermediate(self):
        """Non-terminal, non-break-end points must have npts == npoints."""
        from shalom.backends.qe_config import generate_band_kpath

        npoints = 55
        cfg = generate_band_kpath(self._si(), npoints=npoints)
        # The last point always has npts=1; intermediate non-break points should be npoints
        intermediate_npts = [npts for _, npts in cfg.kpath_points[:-1] if npts != 1]
        if intermediate_npts:
            assert all(n == npoints for n in intermediate_npts)
