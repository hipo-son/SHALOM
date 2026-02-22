"""Tests for shalom.backends.vasp_config — presets, POTCAR mapping, structure detection."""

import math

import pytest
from ase import Atoms
from ase.build import bulk

from shalom.backends.vasp_config import (
    AccuracyLevel,
    CalculationType,
    DEFAULT_MAGMOM,
    ENMAX_VALUES,
    HUBBARD_U_VALUES,
    KPointsConfig,
    MAGNETIC_ELEMENTS,
    VASP_RECOMMENDED_POTCARS,
    VASPInputConfig,
    compute_encut,
    compute_kpoints_grid,
    detect_2d,
    detect_and_apply_structure_hints,
    get_potcar_variant,
    get_preset,
)


# ---------------------------------------------------------------------------
# TestCalculationTypeEnum
# ---------------------------------------------------------------------------


class TestCalculationTypeEnum:
    """Tests for CalculationType and AccuracyLevel enums."""

    def test_all_calc_types_exist(self):
        assert CalculationType.RELAXATION == "relaxation"
        assert CalculationType.STATIC == "static"
        assert CalculationType.BAND_STRUCTURE == "band_structure"
        assert CalculationType.DOS == "dos"
        assert CalculationType.ELASTIC == "elastic"

    def test_all_accuracy_levels_exist(self):
        assert AccuracyLevel.STANDARD == "standard"
        assert AccuracyLevel.PRECISE == "precise"

    def test_enum_from_string(self):
        assert CalculationType("relaxation") == CalculationType.RELAXATION
        assert AccuracyLevel("precise") == AccuracyLevel.PRECISE

    def test_invalid_enum_raises(self):
        with pytest.raises(ValueError):
            CalculationType("invalid")
        with pytest.raises(ValueError):
            AccuracyLevel("ultra")


# ---------------------------------------------------------------------------
# TestPresetDefaults
# ---------------------------------------------------------------------------


class TestPresetDefaults:
    """Tests for get_preset INCAR defaults across calc_type × accuracy."""

    def test_relaxation_standard(self):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD)
        incar = config.incar_settings
        assert incar["ENCUT"] == 520
        assert incar["EDIFF"] == 1e-5
        assert incar["EDIFFG"] == -0.02
        assert incar["ISMEAR"] == 0
        assert incar["SIGMA"] == 0.05
        assert incar["NSW"] == 99
        assert incar["IBRION"] == 2
        assert incar["ISIF"] == 3

    def test_relaxation_precise(self):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.PRECISE)
        incar = config.incar_settings
        assert incar["EDIFF"] == 1e-6
        assert incar["EDIFFG"] == -0.01
        assert incar["NELM"] == 200

    def test_static_standard(self):
        config = get_preset(CalculationType.STATIC, AccuracyLevel.STANDARD)
        incar = config.incar_settings
        assert incar["NSW"] == 0
        assert incar["IBRION"] == -1
        assert "ISIF" not in incar

    def test_dos_standard(self):
        config = get_preset(CalculationType.DOS, AccuracyLevel.STANDARD)
        incar = config.incar_settings
        assert incar["ISMEAR"] == -5
        assert incar["NEDOS"] == 3001
        assert incar["ICHARG"] == 11

    def test_dos_precise(self):
        config = get_preset(CalculationType.DOS, AccuracyLevel.PRECISE)
        assert config.incar_settings["NEDOS"] == 5001

    def test_band_structure_has_icharg(self):
        config = get_preset(CalculationType.BAND_STRUCTURE, AccuracyLevel.STANDARD)
        assert config.incar_settings["ICHARG"] == 11

    def test_elastic_standard(self):
        config = get_preset(CalculationType.ELASTIC, AccuracyLevel.STANDARD)
        incar = config.incar_settings
        assert incar["IBRION"] == 6
        assert incar["EDIFF"] == 1e-6

    def test_user_incar_override(self):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD)
        config.user_incar_settings = {"ENCUT": 700, "ALGO": "Fast"}
        merged = config.get_merged_incar()
        assert merged["ENCUT"] == 700
        assert merged["ALGO"] == "Fast"
        assert merged["ISMEAR"] == 0  # preset preserved


# ---------------------------------------------------------------------------
# TestEncut
# ---------------------------------------------------------------------------


class TestEncut:
    """Tests for dynamic ENCUT calculation."""

    def test_standard_multiplier_1_3(self):
        encut = compute_encut(["O"], AccuracyLevel.STANDARD)
        # O ENMAX = 400, 400 * 1.3 = 520
        assert encut == 520

    def test_precise_multiplier_1_5(self):
        encut = compute_encut(["O"], AccuracyLevel.PRECISE)
        # O ENMAX = 400, 400 * 1.5 = 600
        assert encut == 600

    def test_minimum_520(self):
        encut = compute_encut(["H"], AccuracyLevel.STANDARD)
        # H ENMAX = 250, 250 * 1.3 = 325 → clamped to 520
        assert encut == 520

    def test_ti_sv_high_enmax(self):
        encut = compute_encut(["Ti", "O"], AccuracyLevel.PRECISE)
        # Ti ENMAX = 495, O = 400 → max = 495, 495 * 1.5 = 743
        assert encut >= 742

    def test_empty_elements_returns_520(self):
        encut = compute_encut([], AccuracyLevel.STANDARD)
        assert encut == 520

    def test_unknown_element_uses_fallback(self):
        encut = compute_encut(["Xx"], AccuracyLevel.STANDARD)
        # fallback ENMAX = 300, 300 * 1.3 = 390 → 520
        assert encut == 520

    def test_enmax_override(self):
        encut = compute_encut(["Fe"], AccuracyLevel.PRECISE, {"Fe": 500.0})
        # 500 * 1.5 = 750
        assert encut == 750


# ---------------------------------------------------------------------------
# TestPOTCARMapping
# ---------------------------------------------------------------------------


class TestPOTCARMapping:
    """Tests for POTCAR variant selection."""

    def test_fe_vasp_recommended(self):
        assert get_potcar_variant("Fe", "vasp_recommended") == "Fe_pv"

    def test_fe_mp_default(self):
        # MP default also uses Fe_pv (same as updated vasp_recommended)
        assert get_potcar_variant("Fe", "mp_default") == "Fe_pv"

    def test_ti_sv(self):
        assert get_potcar_variant("Ti", "vasp_recommended") == "Ti_sv"

    def test_w_sv_not_w_pv(self):
        assert get_potcar_variant("W", "vasp_recommended") == "W_sv"

    def test_w_mp_default(self):
        # MP still has W_pv (though deprecated in PBE_54)
        assert get_potcar_variant("W", "mp_default") == "W_pv"

    def test_mo_sv(self):
        assert get_potcar_variant("Mo", "vasp_recommended") == "Mo_sv"

    def test_ga_d(self):
        assert get_potcar_variant("Ga", "vasp_recommended") == "Ga_d"

    def test_unknown_element_returns_itself(self):
        assert get_potcar_variant("Xx", "vasp_recommended") == "Xx"

    def test_mp_default_non_differing_element(self):
        # Al is same in both presets
        assert get_potcar_variant("Al", "mp_default") == "Al"


# ---------------------------------------------------------------------------
# TestTwoDDetection
# ---------------------------------------------------------------------------


class TestTwoDDetection:
    """Tests for 2D structure detection and ISIF/KPOINTS enforcement."""

    def test_slab_with_large_vacuum_detected(self, sample_2d_slab):
        assert detect_2d(sample_2d_slab) is True

    def test_bulk_not_detected(self, sample_bulk_cu):
        assert detect_2d(sample_bulk_cu) is False

    def test_thin_vacuum_not_detected(self):
        atoms = Atoms(
            symbols=["C", "C"],
            positions=[[0, 0, 0], [0, 0, 1.4]],
            cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 5.0]],  # vacuum ~3.6 A
            pbc=True,
        )
        assert detect_2d(atoms, vacuum_threshold=5.0) is False

    def test_2d_forces_isif_4(self, sample_2d_slab):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_2d_slab)
        assert config.is_2d is True
        assert config.incar_settings["ISIF"] == 4

    def test_2d_forces_kpoints_z_1(self, sample_2d_slab):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_2d_slab)
        assert config.kpoints.grid is not None
        assert config.kpoints.grid[2] == 1

    def test_2d_forces_ivdw_12(self, sample_2d_slab):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_2d_slab)
        assert config.vdw_correction == 12
        assert config.incar_settings["IVDW"] == 12

    def test_empty_atoms_not_2d(self):
        atoms = Atoms()
        assert detect_2d(atoms) is False


# ---------------------------------------------------------------------------
# TestMagneticDetection
# ---------------------------------------------------------------------------


class TestMagneticDetection:
    """Tests for magnetic element detection and MAGMOM/ISPIN auto-setting."""

    def test_fe_detected_as_magnetic(self, sample_bulk_fe):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_fe)
        assert config.incar_settings["ISPIN"] == 2
        assert "MAGMOM" in config.incar_settings

    def test_fe_magmom_value(self, sample_bulk_fe):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_fe)
        magmom = config.incar_settings["MAGMOM"]
        assert all(m == 5.0 for m in magmom)

    def test_si_not_magnetic(self, sample_si_diamond):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_si_diamond)
        assert "ISPIN" not in config.incar_settings

    def test_gd_lanthanide_magnetic(self):
        atoms = Atoms("Gd", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, atoms)
        assert config.incar_settings["ISPIN"] == 2
        magmom = config.incar_settings["MAGMOM"]
        assert magmom[0] == 7.0

    def test_u_actinide_magnetic(self):
        atoms = Atoms("U", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, atoms)
        assert config.incar_settings["ISPIN"] == 2
        assert config.incar_settings["MAGMOM"][0] == 2.0

    def test_magnetic_elements_set_complete(self):
        """All elements in DEFAULT_MAGMOM must be in MAGNETIC_ELEMENTS."""
        assert MAGNETIC_ELEMENTS == frozenset(DEFAULT_MAGMOM.keys())
        # Must include lanthanides
        assert "Gd" in MAGNETIC_ELEMENTS
        assert "Ce" in MAGNETIC_ELEMENTS
        # Must include actinides
        assert "U" in MAGNETIC_ELEMENTS
        assert "Pu" in MAGNETIC_ELEMENTS


# ---------------------------------------------------------------------------
# TestGGAPlusU
# ---------------------------------------------------------------------------


class TestGGAPlusU:
    """Tests for automatic GGA+U detection on TMO/chalcogenides."""

    def test_feo_precise_enables_ldau(self, sample_tmo_feo):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.PRECISE, sample_tmo_feo)
        assert config.ldau_settings is not None
        assert config.ldau_settings["LDAU"] is True
        assert config.ldau_settings["LDAUTYPE"] == 2
        # LDAU params live only in ldau_settings, not duplicated in incar_settings
        assert "LDAUL" not in config.incar_settings
        assert "LDAUU" not in config.incar_settings

    def test_feo_standard_no_auto_ldau(self, sample_tmo_feo):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_tmo_feo)
        # STANDARD mode should NOT auto-enable +U
        assert config.ldau_settings is None

    def test_si_no_ldau(self, sample_si_diamond):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.PRECISE, sample_si_diamond)
        assert config.ldau_settings is None

    def test_cu_oxide_u_value(self):
        atoms = Atoms(
            symbols=["Cu", "O"],
            positions=[[0, 0, 0], [2.0, 0, 0]],
            cell=[4, 4, 4],
            pbc=True,
        )
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.PRECISE, atoms)
        assert config.ldau_settings is not None
        # Cu should have U=4.0
        elements_order = config.ldau_settings["elements_order"]
        cu_idx = elements_order.index("Cu")
        assert config.ldau_settings["LDAUU"][cu_idx] == 4.0

    def test_hubbard_u_wang_values(self):
        """Verify U values match Wang et al. PRB 73, 195107 (2006)."""
        assert HUBBARD_U_VALUES["Fe"][1] == 5.3
        assert HUBBARD_U_VALUES["Co"][1] == 3.32
        assert HUBBARD_U_VALUES["Ni"][1] == 6.2
        assert HUBBARD_U_VALUES["Mn"][1] == 3.9
        assert HUBBARD_U_VALUES["V"][1] == 3.25
        assert HUBBARD_U_VALUES["Cr"][1] == 3.7

    def test_non_pbe_functional_warns(self, sample_tmo_feo, caplog):
        """Non-PBE functional with GGA+U logs a warning."""
        import logging

        config = VASPInputConfig(
            calc_type=CalculationType.RELAXATION,
            accuracy=AccuracyLevel.PRECISE,
            incar_settings={},
            functional="r2SCAN",
        )
        with caplog.at_level(logging.WARNING, logger="shalom.backends.vasp_config"):
            detect_and_apply_structure_hints(sample_tmo_feo, config)
        assert "PBE only" in caplog.text


# ---------------------------------------------------------------------------
# TestMetalDetection
# ---------------------------------------------------------------------------


class TestMetalDetection:
    """Tests for pure metal ISMEAR/SIGMA auto-setting."""

    def test_cu_metal_ismear(self, sample_bulk_cu):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        assert config.incar_settings["ISMEAR"] == 1
        assert config.incar_settings["SIGMA"] == 0.1

    def test_sigma_is_0_1_not_0_2(self, sample_bulk_fe):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_fe)
        assert config.incar_settings["SIGMA"] == 0.1

    def test_si_not_metal(self, sample_si_diamond):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_si_diamond)
        # Si is not a pure metal → default ISMEAR=0
        assert config.incar_settings["ISMEAR"] == 0
        assert config.incar_settings["SIGMA"] == 0.05


# ---------------------------------------------------------------------------
# TestLMAXMIX
# ---------------------------------------------------------------------------


class TestLMAXMIX:
    """Tests for LMAXMIX auto-setting based on atomic numbers."""

    def test_d_electrons_lmaxmix_4(self, sample_bulk_fe):
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_fe)
        assert config.incar_settings["LMAXMIX"] == 4

    def test_f_electrons_lmaxmix_6(self):
        atoms = Atoms("Gd", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, atoms)
        assert config.incar_settings["LMAXMIX"] == 6

    def test_light_elements_no_lmaxmix(self):
        atoms = Atoms("C", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, atoms)
        assert "LMAXMIX" not in config.incar_settings


# ---------------------------------------------------------------------------
# TestKPointsGrid
# ---------------------------------------------------------------------------


class TestKPointsGrid:
    """Tests for KPOINTS grid computation."""

    def test_bulk_cu_grid_symmetric(self, sample_bulk_cu):
        grid = compute_kpoints_grid(sample_bulk_cu)
        # Cubic cell → all three grid values should be equal
        assert grid[0] == grid[1] == grid[2]
        assert grid[0] >= 1

    def test_2d_z_hardcoded_1(self, sample_2d_slab):
        grid = compute_kpoints_grid(sample_2d_slab, is_2d=True)
        assert grid[2] == 1

    def test_2d_z_not_hardcoded_when_false(self, sample_2d_slab):
        grid = compute_kpoints_grid(sample_2d_slab, is_2d=False)
        # Without 2D flag, z grid is computed normally (may be >1)
        assert grid[2] >= 1


# ---------------------------------------------------------------------------
# TestVASPInputConfig
# ---------------------------------------------------------------------------


class TestVASPInputConfig:
    """Tests for VASPInputConfig dataclass behavior."""

    def test_default_config(self):
        config = VASPInputConfig()
        assert config.calc_type == CalculationType.RELAXATION
        assert config.accuracy == AccuracyLevel.STANDARD
        assert config.potcar_preset == "vasp_recommended"
        assert config.is_2d is False

    def test_get_merged_incar(self):
        config = VASPInputConfig(
            incar_settings={"ENCUT": 520, "EDIFF": 1e-5},
            user_incar_settings={"ENCUT": 700, "ALGO": "Fast"},
        )
        merged = config.get_merged_incar()
        assert merged["ENCUT"] == 700
        assert merged["EDIFF"] == 1e-5
        assert merged["ALGO"] == "Fast"

    def test_dynamic_encut_with_atoms(self, sample_bulk_cu):
        config = get_preset(
            CalculationType.RELAXATION, AccuracyLevel.PRECISE, sample_bulk_cu,
        )
        # Cu ENMAX ≈ 295.4, 295.4 * 1.5 ≈ 444 → clamped to 520
        assert config.incar_settings["ENCUT"] >= 520

    def test_enmax_values_populated(self, sample_bulk_cu):
        config = get_preset(
            CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu,
        )
        assert config.enmax_values is not None
        assert "Cu" in config.enmax_values
