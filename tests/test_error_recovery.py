"""Tests for shalom.backends.error_recovery module.

Covers error detection, progressive escalation, false convergence detection,
ionic sloshing, and correction history tracking.
"""

import json

import pytest

from shalom.backends.error_recovery import (
    CORRECTION_STRATEGIES,
    ERROR_PATTERNS,
    ErrorRecoveryEngine,
    ErrorSeverity,
    VASPError,
    Correction,
    _check_force_oscillation,
)


# ---------------------------------------------------------------------------
# Error Detection Tests
# ---------------------------------------------------------------------------


class TestErrorDetection:
    """Tests for scan_for_errors()."""

    def test_detect_zbrent(self):
        text = "ZBRENT: fatal internal in bracketing\nsome other stuff"
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1
        assert errors[0].error_type == "ZBRENT"
        assert errors[0].severity == ErrorSeverity.CORRECTABLE

    def test_detect_brmix(self):
        text = "BRMIX: very serious problems\nthe old and the new charge density differ"
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1
        assert errors[0].error_type == "BRMIX"

    def test_detect_scf_unconverged(self):
        text = "Some output\n NELM reached\nMore output"
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1
        assert errors[0].error_type == "SCF_UNCONVERGED"

    def test_detect_posmap(self):
        text = "POSMAP internal error: some details"
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1
        assert errors[0].error_type == "POSMAP"

    def test_empty_outcar_no_errors(self):
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors("")
        assert errors == []

    def test_case_mismatch_no_detection(self):
        text = "zbrent: fatal internal in bracketing"  # lowercase
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert errors == []

    def test_multiple_errors_detected(self):
        text = "BRMIX: very serious problems\nNELM reached\n"
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert len(errors) == 2
        types = {e.error_type for e in errors}
        assert "BRMIX" in types
        assert "SCF_UNCONVERGED" in types

    def test_no_duplicate_error_types(self):
        text = "NELM reached\nsome stuff\nNELM reached again\n"
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1

    def test_fatal_error_detected(self):
        text = "VERY BAD NEWS! internal error in subroutine SGRCON"
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.FATAL

    def test_edddav_detected(self):
        text = "EDDDAV: sub-space matrix is not hermitian"
        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1
        assert errors[0].error_type == "EDDDAV"


# ---------------------------------------------------------------------------
# Correction Strategy Order Tests
# ---------------------------------------------------------------------------


class TestCorrectionStrategyOrder:
    """Tests for progressive escalation ordering."""

    def test_scf_step_0_nelm(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        corr = engine.get_correction(error)
        assert corr is not None
        assert corr.incar_updates.get("NELM") == 200
        assert corr.step == 0

    def test_scf_step_1_ismear_switch(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)  # step 0
        corr = engine.get_correction(error)  # step 1
        assert corr is not None
        assert corr.incar_updates.get("ISMEAR") == 1
        assert corr.incar_updates.get("SIGMA") == 0.1

    def test_scf_step_2_damped(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)  # step 0
        engine.get_correction(error)  # step 1
        corr = engine.get_correction(error)  # step 2
        assert corr is not None
        assert corr.incar_updates.get("ALGO") == "Damped"

    def test_scf_step_3_amix(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        for _ in range(3):
            engine.get_correction(error)
        corr = engine.get_correction(error)  # step 3
        assert corr is not None
        assert "AMIX" in corr.incar_updates

    def test_scf_step_4_all(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        for _ in range(4):
            engine.get_correction(error)
        corr = engine.get_correction(error)  # step 4
        assert corr is not None
        assert corr.incar_updates.get("ALGO") == "All"

    def test_scf_exhausted_returns_none(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        for _ in range(5):
            engine.get_correction(error)
        corr = engine.get_correction(error)
        assert corr is None

    def test_brmix_step_0_ismear_first(self):
        """BRMIX: metallization defense is step 1 (ISMEAR switch)."""
        engine = ErrorRecoveryEngine()
        error = VASPError("BRMIX", ErrorSeverity.CORRECTABLE)
        corr = engine.get_correction(error)
        assert corr is not None
        assert corr.incar_updates.get("ISMEAR") == 1

    def test_zbrent_step_0_potim_direct(self):
        """ZBRENT: goes directly to POTIM reduction (no EDIFF step)."""
        engine = ErrorRecoveryEngine()
        error = VASPError("ZBRENT", ErrorSeverity.CORRECTABLE)
        corr = engine.get_correction(error)
        assert corr is not None
        assert corr.incar_updates.get("POTIM") == 0.2
        assert "EDIFF" not in corr.incar_updates

    def test_zbrent_step_1_ibrion_1(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("ZBRENT", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)  # step 0
        corr = engine.get_correction(error)  # step 1
        assert corr is not None
        assert corr.incar_updates.get("IBRION") == 1

    def test_zbrent_step_2_damped_md(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("ZBRENT", ErrorSeverity.CORRECTABLE)
        for _ in range(2):
            engine.get_correction(error)
        corr = engine.get_correction(error)  # step 2
        assert corr is not None
        assert corr.incar_updates.get("IBRION") == 3
        assert corr.incar_updates.get("SMASS") == 0.5

    def test_ionic_sloshing_strategies(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("IONIC_SLOSHING", ErrorSeverity.CORRECTABLE)
        corr0 = engine.get_correction(error)
        assert corr0 is not None
        assert corr0.incar_updates.get("POTIM") == 0.2

    def test_fatal_error_no_correction(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SGRCON", ErrorSeverity.FATAL)
        corr = engine.get_correction(error)
        assert corr is None

    def test_unknown_error_type_no_correction(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("UNKNOWN_ERROR", ErrorSeverity.CORRECTABLE)
        corr = engine.get_correction(error)
        assert corr is None


# ---------------------------------------------------------------------------
# Multiple Errors Tests
# ---------------------------------------------------------------------------


class TestMultipleErrors:
    """Tests for handling multiple concurrent errors."""

    def test_independent_error_counters(self):
        engine = ErrorRecoveryEngine()
        scf_error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        zbrent_error = VASPError("ZBRENT", ErrorSeverity.CORRECTABLE)
        corr1 = engine.get_correction(scf_error)
        corr2 = engine.get_correction(zbrent_error)
        assert corr1.step == 0
        assert corr2.step == 0

    def test_escalation_counter_persists(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)  # step 0
        engine.get_correction(error)  # step 1
        corr = engine.get_correction(error)  # step 2
        assert corr.step == 2

    def test_reoccurrence_escalates(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("BRMIX", ErrorSeverity.CORRECTABLE)
        corr0 = engine.get_correction(error)
        corr1 = engine.get_correction(error)
        assert corr0.step == 0
        assert corr1.step == 1
        assert corr0.incar_updates != corr1.incar_updates


# ---------------------------------------------------------------------------
# False Convergence Tests
# ---------------------------------------------------------------------------


class TestFalseConvergence:
    """Tests for detect_false_convergence()."""

    def test_energy_converged_force_oscillating_is_false_convergence(self):
        # Energy barely changes, but forces oscillate around threshold
        energies = [-30.0, -30.0001, -30.00005]
        forces = [0.025, 0.018, 0.022, 0.019, 0.023, 0.017, 0.024, 0.018, 0.021, 0.020]
        assert ErrorRecoveryEngine.detect_false_convergence(energies, forces) is True

    def test_energy_converged_force_monotonic_is_normal(self):
        energies = [-30.0, -30.0001, -30.00005]
        forces = [0.5, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.003, 0.001]
        assert ErrorRecoveryEngine.detect_false_convergence(energies, forces) is False

    def test_energy_oscillating_is_not_false_convergence(self):
        energies = [-30.0, -29.5, -30.2]
        forces = [0.025, 0.018, 0.022, 0.019, 0.023]
        assert ErrorRecoveryEngine.detect_false_convergence(energies, forces) is False

    def test_insufficient_energies_returns_false(self):
        energies = [-30.0, -30.1]
        forces = [0.5, 0.3, 0.15, 0.08, 0.04]
        assert ErrorRecoveryEngine.detect_false_convergence(energies, forces) is False

    def test_insufficient_forces_returns_false(self):
        energies = [-30.0, -30.0001, -30.00005]
        forces = [0.5, 0.3, 0.15, 0.08]
        assert ErrorRecoveryEngine.detect_false_convergence(energies, forces) is False

    def test_empty_energies_returns_false(self):
        assert ErrorRecoveryEngine.detect_false_convergence([], [0.1, 0.2, 0.3, 0.4, 0.5]) is False

    def test_none_forces_returns_false(self):
        assert ErrorRecoveryEngine.detect_false_convergence([-30.0, -30.1, -30.2], None) is False

    def test_10_step_window(self):
        """Verifies 10-step window is used for force analysis."""
        energies = [-30.0, -30.0001, -30.00005]
        # Oscillating in last 10 steps
        forces = [0.5, 0.3, 0.15, 0.08,  # early: decreasing
                  0.025, 0.019, 0.023, 0.018, 0.024, 0.017, 0.022, 0.019, 0.021, 0.020]
        result = ErrorRecoveryEngine.detect_false_convergence(
            energies, forces, window=10,
        )
        assert result is True


# ---------------------------------------------------------------------------
# Ionic Sloshing Tests
# ---------------------------------------------------------------------------


class TestIonicSloshing:
    """Tests for detect_ionic_sloshing()."""

    def test_alternating_forces_detected(self):
        # Up-down-up-down pattern
        forces = [0.1, 0.05, 0.08, 0.04, 0.07, 0.03]
        assert ErrorRecoveryEngine.detect_ionic_sloshing(forces) is True

    def test_monotonic_decrease_not_sloshing(self):
        forces = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        assert ErrorRecoveryEngine.detect_ionic_sloshing(forces) is False

    def test_short_history_not_detected(self):
        forces = [0.1, 0.05, 0.08]
        assert ErrorRecoveryEngine.detect_ionic_sloshing(forces) is False

    def test_mostly_alternating(self):
        # Nearly all transitions alternate
        forces = [0.1, 0.05, 0.09, 0.04, 0.08, 0.03, 0.07]
        assert ErrorRecoveryEngine.detect_ionic_sloshing(forces, window=6) is True


# ---------------------------------------------------------------------------
# Double Relaxation Trigger Tests
# ---------------------------------------------------------------------------


class TestDoubleRelaxTrigger:
    """Tests for VASPBackend.should_trigger_step2()."""

    def test_small_volume_change_no_step2(self):
        from shalom.backends.vasp import VASPBackend
        assert VASPBackend.should_trigger_step2(100.0, 102.0) is False  # 2% < 3%

    def test_large_volume_change_triggers_step2(self):
        from shalom.backends.vasp import VASPBackend
        assert VASPBackend.should_trigger_step2(100.0, 105.0) is True  # 5% > 3%

    def test_volume_contraction_uses_abs(self):
        from shalom.backends.vasp import VASPBackend
        assert VASPBackend.should_trigger_step2(100.0, 94.0) is True  # 6% > 3%

    def test_zero_initial_volume(self):
        from shalom.backends.vasp import VASPBackend
        assert VASPBackend.should_trigger_step2(0.0, 100.0) is False

    def test_custom_threshold(self):
        from shalom.backends.vasp import VASPBackend
        assert VASPBackend.should_trigger_step2(100.0, 101.5, threshold=0.01) is True


# ---------------------------------------------------------------------------
# Correction History Tests
# ---------------------------------------------------------------------------


class TestCorrectionHistory:
    """Tests for correction history tracking and serialization."""

    def test_empty_history(self):
        engine = ErrorRecoveryEngine()
        assert engine.correction_history == []

    def test_history_after_correction(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)
        assert len(engine.correction_history) == 1
        assert engine.correction_history[0]["error_type"] == "SCF_UNCONVERGED"
        assert engine.correction_history[0]["action"] == "correction_applied"

    def test_history_json_serializable(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)
        engine.get_correction(error)
        json_str = json.dumps(engine.correction_history)
        restored = json.loads(json_str)
        assert len(restored) == 2

    def test_history_multi_step(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("ZBRENT", ErrorSeverity.CORRECTABLE)
        for _ in range(3):
            engine.get_correction(error)
        assert len(engine.correction_history) == 3
        steps = [h["step"] for h in engine.correction_history]
        assert steps == [0, 1, 2]

    def test_history_exhausted_recorded(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("ZBRENT", ErrorSeverity.CORRECTABLE)
        for _ in range(3):
            engine.get_correction(error)
        engine.get_correction(error)  # exhausted
        assert engine.correction_history[-1]["action"] == "strategies_exhausted"

    def test_history_fatal_recorded(self):
        engine = ErrorRecoveryEngine()
        error = VASPError("SGRCON", ErrorSeverity.FATAL)
        engine.get_correction(error)
        assert engine.correction_history[-1]["action"] == "fatal_no_correction"

    def test_history_is_copy(self):
        """correction_history returns a copy, not internal state."""
        engine = ErrorRecoveryEngine()
        h = engine.correction_history
        h.append({"fake": True})
        assert len(engine.correction_history) == 0


# ---------------------------------------------------------------------------
# Force Oscillation Helper Tests
# ---------------------------------------------------------------------------


class TestForceOscillationHelper:
    """Tests for _check_force_oscillation() helper."""

    def test_clearly_oscillating(self):
        forces = [0.025, 0.018, 0.023, 0.017, 0.024, 0.016, 0.022, 0.019]
        assert _check_force_oscillation(forces, threshold=0.05) is True

    def test_monotonic_not_oscillating(self):
        forces = [0.5, 0.4, 0.3, 0.2, 0.1]
        assert _check_force_oscillation(forces) is False

    def test_short_list(self):
        forces = [0.1, 0.2]
        assert _check_force_oscillation(forces) is False

    def test_zero_mean(self):
        forces = [0.0, 0.0, 0.0]
        assert _check_force_oscillation(forces) is False


# ---------------------------------------------------------------------------
# Error Pattern Registry Completeness Tests
# ---------------------------------------------------------------------------


class TestErrorPatternRegistry:
    """Tests for ERROR_PATTERNS and CORRECTION_STRATEGIES consistency."""

    def test_all_correctable_have_strategies(self):
        correctable_types = {
            pat[1] for pat in ERROR_PATTERNS
            if pat[2] == ErrorSeverity.CORRECTABLE
        }
        for error_type in correctable_types:
            assert error_type in CORRECTION_STRATEGIES, (
                f"Missing strategy for correctable error: {error_type}"
            )

    def test_strategies_are_non_empty(self):
        for error_type, strategies in CORRECTION_STRATEGIES.items():
            assert len(strategies) > 0, f"Empty strategy list for {error_type}"

    def test_all_strategies_are_dicts(self):
        for error_type, strategies in CORRECTION_STRATEGIES.items():
            for i, s in enumerate(strategies):
                assert isinstance(s, dict), (
                    f"Strategy step {i} for {error_type} is not a dict"
                )
