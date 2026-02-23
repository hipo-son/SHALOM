"""Tests for QE error recovery engine.

Covers: error detection, correction escalation, S-matrix diagnostic branching,
light atom dt safety, quality warnings, and apply_correction_to_config.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ase import Atoms
from ase.build import bulk

from shalom.backends.error_recovery import ErrorSeverity
from shalom.backends.qe_error_recovery import (
    MIN_DISTANCE_THRESHOLD,
    QE_CORRECTION_STRATEGIES,
    QE_ERROR_PATTERNS,
    QECorrection,
    QEError,
    QEErrorRecoveryEngine,
    check_atomic_overlap,
    compute_safe_dt,
)


# ---------------------------------------------------------------------------
# TestQEErrorDetection
# ---------------------------------------------------------------------------


class TestQEErrorDetection:
    """Test error pattern matching from QE pw.out text."""

    def test_scf_unconverged(self):
        engine = QEErrorRecoveryEngine()
        text = (
            "     iteration # 100\n"
            "     convergence NOT achieved after 100 iterations\n"
        )
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1
        assert errors[0].error_type == "QE_SCF_UNCONVERGED"
        assert errors[0].severity == ErrorSeverity.CORRECTABLE

    def test_bfgs_failed(self):
        engine = QEErrorRecoveryEngine()
        text = "     bfgs failed after   50 scf cycles and   25 bfgs steps\n"
        errors = engine.scan_for_errors(text)
        assert any(e.error_type == "QE_BFGS_FAILED" for e in errors)

    def test_ionic_not_converged(self):
        engine = QEErrorRecoveryEngine()
        text = "     The maximum number of steps has been reached.\n"
        errors = engine.scan_for_errors(text)
        assert any(e.error_type == "QE_IONIC_NOT_CONVERGED" for e in errors)

    def test_cdiaghg_s_matrix(self):
        engine = QEErrorRecoveryEngine()
        text = (
            "     S matrix not positive definite\n"
            "     Error in routine cdiaghg (1):\n"
        )
        errors = engine.scan_for_errors(text)
        types = {e.error_type for e in errors}
        assert "QE_S_MATRIX" in types
        assert "QE_DIAG_FAILED" in types

    def test_rdiaghg(self):
        engine = QEErrorRecoveryEngine()
        text = "     Error in routine rdiaghg (1):\n"
        errors = engine.scan_for_errors(text)
        assert any(e.error_type == "QE_DIAG_FAILED" for e in errors)

    def test_eigenvalues_not_converged(self):
        engine = QEErrorRecoveryEngine()
        text = "     eigenvalues not converged\n"
        errors = engine.scan_for_errors(text)
        assert any(e.error_type == "QE_EIGVAL_NOT_CONVERGED" for e in errors)

    def test_too_many_bands(self):
        engine = QEErrorRecoveryEngine()
        text = "     too many bands are not converged\n"
        errors = engine.scan_for_errors(text)
        assert any(e.error_type == "QE_TOO_MANY_BANDS" for e in errors)

    def test_negative_charge(self):
        engine = QEErrorRecoveryEngine()
        text = "     negative or imaginary charge in mixing\n"
        errors = engine.scan_for_errors(text)
        assert any(e.error_type == "QE_NEGATIVE_CHARGE" for e in errors)

    def test_charge_wrong(self):
        engine = QEErrorRecoveryEngine()
        text = "     charge is wrong: smearing is too large\n"
        errors = engine.scan_for_errors(text)
        assert any(e.error_type == "QE_CHARGE_WRONG" for e in errors)

    def test_cell_distorted(self):
        engine = QEErrorRecoveryEngine()
        text = "     angle between cell vectors is becoming too small\n"
        errors = engine.scan_for_errors(text)
        assert any(e.error_type == "QE_CELL_DISTORTED" for e in errors)

    def test_pseudo_not_found_fatal(self):
        engine = QEErrorRecoveryEngine()
        text = "     Error in routine readpp (1):\n"
        errors = engine.scan_for_errors(text)
        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.FATAL

    def test_out_of_memory_fatal(self):
        engine = QEErrorRecoveryEngine()
        text = "     Not enough space allocated for radial FFT\n"
        errors = engine.scan_for_errors(text)
        assert any(e.severity == ErrorSeverity.FATAL for e in errors)

    def test_io_error_fatal(self):
        engine = QEErrorRecoveryEngine()
        text = "     Error in routine davcio (10):\n"
        errors = engine.scan_for_errors(text)
        assert any(e.severity == ErrorSeverity.FATAL for e in errors)

    def test_no_errors_in_clean_output(self):
        engine = QEErrorRecoveryEngine()
        text = (
            "!    total energy              =     -31.50000000 Ry\n"
            "     convergence has been achieved in   8 iterations\n"
            "     JOB DONE.\n"
        )
        errors = engine.scan_for_errors(text)
        assert len(errors) == 0

    def test_dedup_same_type(self):
        """Same error type from different patterns should not duplicate."""
        engine = QEErrorRecoveryEngine()
        text = (
            "     Error in routine cdiaghg (1):\n"
            "     Error in routine rdiaghg (2):\n"
        )
        errors = engine.scan_for_errors(text)
        diag_errors = [e for e in errors if e.error_type == "QE_DIAG_FAILED"]
        assert len(diag_errors) == 1  # Deduplicated

    def test_pattern_count(self):
        """14 patterns should be loaded."""
        assert len(QE_ERROR_PATTERNS) == 14


# ---------------------------------------------------------------------------
# TestQECorrectionEscalation
# ---------------------------------------------------------------------------


class TestQECorrectionEscalation:
    """Test progressive correction escalation for each error type."""

    def test_scf_unconverged_escalation(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        steps = QE_CORRECTION_STRATEGIES["QE_SCF_UNCONVERGED"]
        for i in range(len(steps)):
            c = engine.get_correction(error)
            assert c is not None
            assert c.step == i
        # Exhausted
        assert engine.get_correction(error) is None

    def test_scf_step1_local_tf_fast_fail(self):
        """Step 1 (local-TF) should have electron_maxstep_cap=50."""
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)  # Step 0
        c = engine.get_correction(error)  # Step 1
        assert c is not None
        assert c.electron_maxstep_cap == 50

    def test_bfgs_failed_escalation(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_BFGS_FAILED", ErrorSeverity.CORRECTABLE)
        steps = QE_CORRECTION_STRATEGIES["QE_BFGS_FAILED"]
        for i in range(len(steps)):
            c = engine.get_correction(error)
            assert c is not None
        assert engine.get_correction(error) is None

    def test_bfgs_step2_quality_warning(self):
        """BFGS step 2 should produce 'loosely_relaxed' quality warning."""
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_BFGS_FAILED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)  # Step 0
        engine.get_correction(error)  # Step 1
        c = engine.get_correction(error)  # Step 2
        assert c is not None
        assert "loosely_relaxed" in c.quality_warnings

    def test_cell_distorted_escalation(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_CELL_DISTORTED", ErrorSeverity.CORRECTABLE)
        steps = QE_CORRECTION_STRATEGIES["QE_CELL_DISTORTED"]
        for i in range(len(steps)):
            c = engine.get_correction(error)
            assert c is not None
        assert engine.get_correction(error) is None

    def test_cell_distorted_step0_cell_factor(self):
        """Step 0 should set cell.cell_factor=2.0."""
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_CELL_DISTORTED", ErrorSeverity.CORRECTABLE)
        c = engine.get_correction(error)
        assert c is not None
        assert c.namelist_updates.get("cell.cell_factor") == 2.0

    def test_cell_distorted_step2_quality_warning(self):
        """Step 2 (relax downgrade) should produce 'vc_relax_downgraded'."""
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_CELL_DISTORTED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)  # Step 0
        engine.get_correction(error)  # Step 1
        c = engine.get_correction(error)  # Step 2
        assert c is not None
        assert "vc_relax_downgraded" in c.quality_warnings

    def test_fatal_error_returns_none(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_PSEUDO_NOT_FOUND", ErrorSeverity.FATAL)
        assert engine.get_correction(error) is None

    def test_unknown_error_type_returns_none(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_UNKNOWN_ERROR", ErrorSeverity.CORRECTABLE)
        assert engine.get_correction(error) is None


# ---------------------------------------------------------------------------
# TestQESMatrixDiagnostic
# ---------------------------------------------------------------------------


class TestQESMatrixDiagnostic:
    """Test S-matrix diagnostic branching: overlap vs basis-set path."""

    def test_overlap_path_close_atoms(self):
        """Atoms closer than 0.5 Å → overlap path."""
        atoms = Atoms("Cu2", positions=[(0, 0, 0), (0.3, 0, 0)], cell=[10, 10, 10], pbc=True)
        engine = QEErrorRecoveryEngine()
        path = engine.diagnose_s_matrix(atoms)
        assert path == "overlap"

    def test_basis_set_path_normal_atoms(self):
        """Normal structure → basis-set path."""
        atoms = bulk("Cu", "fcc", a=3.6)
        engine = QEErrorRecoveryEngine()
        path = engine.diagnose_s_matrix(atoms)
        assert path == "basis_set"

    def test_overlap_correction_has_rollback(self):
        """Overlap path corrections should have rollback_geometry=True."""
        atoms = Atoms("Cu2", positions=[(0, 0, 0), (0.3, 0, 0)], cell=[10, 10, 10], pbc=True)
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_S_MATRIX", ErrorSeverity.CORRECTABLE)
        c = engine.get_correction(error, atoms=atoms)
        assert c is not None
        assert c.rollback_geometry is True
        # Should include trust_radius reduction
        assert "ions.trust_radius_max" in c.namelist_updates

    def test_overlap_path_step1_damp(self):
        """Overlap path step 1 → damped dynamics."""
        atoms = Atoms("Cu2", positions=[(0, 0, 0), (0.3, 0, 0)], cell=[10, 10, 10], pbc=True)
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_S_MATRIX", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error, atoms=atoms)  # Step 0a
        c = engine.get_correction(error, atoms=atoms)  # Step 1a
        assert c is not None
        assert c.rollback_geometry is True
        assert c.namelist_updates.get("ions.ion_dynamics") == "damp"

    def test_basis_set_path_cg_diag(self):
        """Basis-set path step 0 → CG diagonalization."""
        atoms = bulk("Cu", "fcc", a=3.6)
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_S_MATRIX", ErrorSeverity.CORRECTABLE)
        c = engine.get_correction(error, atoms=atoms)
        assert c is not None
        assert c.rollback_geometry is False
        assert c.namelist_updates.get("electrons.diagonalization") == "cg"

    def test_basis_set_path_exhaustion(self):
        """Basis-set path should exhaust after its steps."""
        atoms = bulk("Cu", "fcc", a=3.6)
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_S_MATRIX", ErrorSeverity.CORRECTABLE)
        corrections = []
        for _ in range(10):
            c = engine.get_correction(error, atoms=atoms)
            if c is None:
                break
            corrections.append(c)
        assert len(corrections) == 2  # cg + diago_david_ndim

    def test_s_matrix_none_atoms_defaults_basis_set(self):
        """If atoms is None, default to basis-set path."""
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_S_MATRIX", ErrorSeverity.CORRECTABLE)
        c = engine.get_correction(error, atoms=None)
        assert c is not None
        assert c.rollback_geometry is False

    def test_s_matrix_history_records_path(self):
        """Correction history should record the diagnostic path."""
        atoms = bulk("Cu", "fcc", a=3.6)
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_S_MATRIX", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error, atoms=atoms)
        assert engine.correction_history[-1]["path"] == "basis_set"


# ---------------------------------------------------------------------------
# TestQELightAtomDt
# ---------------------------------------------------------------------------


class TestQELightAtomDt:
    """Test light atom dt safety for damped dynamics."""

    def test_hydrogen_dt(self):
        atoms = Atoms("H2", positions=[(0, 0, 0), (0.74, 0, 0)], cell=[10, 10, 10], pbc=True)
        assert compute_safe_dt(atoms) == 5.0

    def test_lithium_dt(self):
        atoms = Atoms("Li2", positions=[(0, 0, 0), (3.0, 0, 0)], cell=[10, 10, 10], pbc=True)
        assert compute_safe_dt(atoms) == 10.0

    def test_beryllium_dt(self):
        atoms = Atoms("Be2", positions=[(0, 0, 0), (2.0, 0, 0)], cell=[10, 10, 10], pbc=True)
        assert compute_safe_dt(atoms) == 12.0

    def test_iron_default_dt(self):
        atoms = bulk("Fe", "bcc", a=2.87)
        assert compute_safe_dt(atoms) == 20.0

    def test_mixed_h_li_takes_minimum(self):
        atoms = Atoms("HLi", positions=[(0, 0, 0), (1.6, 0, 0)], cell=[10, 10, 10], pbc=True)
        assert compute_safe_dt(atoms) == 5.0  # H dominates

    def test_bfgs_damp_with_h_atoms(self):
        """BFGS step 1 (damp) with H atoms should set dt=5.0."""
        atoms = Atoms("H2O", positions=[(0, 0, 0), (0.96, 0, 0), (0, 0.96, 0)],
                       cell=[10, 10, 10], pbc=True)
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_BFGS_FAILED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error, atoms=atoms)  # Step 0
        c = engine.get_correction(error, atoms=atoms)  # Step 1 (damp)
        assert c is not None
        assert c.namelist_updates.get("ions.ion_dynamics") == "damp"
        assert c.namelist_updates.get("ions.dt") == 5.0

    def test_bfgs_damp_with_fe_atoms(self):
        """BFGS step 1 (damp) with Fe → dt=20.0 (default)."""
        atoms = bulk("Fe", "bcc", a=2.87)
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_BFGS_FAILED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error, atoms=atoms)  # Step 0
        c = engine.get_correction(error, atoms=atoms)  # Step 1 (damp)
        assert c is not None
        assert c.namelist_updates.get("ions.dt") == 20.0


# ---------------------------------------------------------------------------
# TestQEQualityWarnings
# ---------------------------------------------------------------------------


class TestQEQualityWarnings:
    """Test quality warning tag propagation."""

    def test_loosely_relaxed_from_bfgs(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_BFGS_FAILED", ErrorSeverity.CORRECTABLE)
        for _ in range(2):
            engine.get_correction(error)
        c = engine.get_correction(error)  # Step 2
        assert c is not None
        assert "loosely_relaxed" in c.quality_warnings

    def test_vc_relax_downgraded_from_cell(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_CELL_DISTORTED", ErrorSeverity.CORRECTABLE)
        for _ in range(2):
            engine.get_correction(error)
        c = engine.get_correction(error)  # Step 2
        assert c is not None
        assert "vc_relax_downgraded" in c.quality_warnings

    def test_no_warning_on_safe_correction(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        c = engine.get_correction(error)  # Step 0 (mixing_beta reduction)
        assert c is not None
        assert len(c.quality_warnings) == 0

    def test_warnings_in_history(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_BFGS_FAILED", ErrorSeverity.CORRECTABLE)
        for _ in range(3):
            engine.get_correction(error)
        # Last entry should have quality_warnings
        last = engine.correction_history[-1]
        assert "quality_warnings" in last
        assert "loosely_relaxed" in last["quality_warnings"]


# ---------------------------------------------------------------------------
# TestQEApplyCorrection
# ---------------------------------------------------------------------------


class TestQEApplyCorrection:
    """Test apply_correction_to_config with QEInputConfig mocks."""

    def _make_mock_config(self):
        """Create a mock QEInputConfig-like object."""
        config = MagicMock()
        config.control = {"calculation": "vc-relax", "prefix": "shalom"}
        config.system = {"ecutwfc": 60, "ecutrho": 480}
        config.electrons = {"mixing_beta": 0.7, "electron_maxstep": 100}
        config.ions = {}
        config.cell = {"cell_dynamics": "bfgs"}
        return config

    def test_dot_notation_basic(self):
        config = self._make_mock_config()
        correction = QECorrection(
            error_type="QE_SCF_UNCONVERGED", step=0,
            namelist_updates={"electrons.mixing_beta": 0.3},
        )
        warnings = QEErrorRecoveryEngine.apply_correction_to_config(config, correction)
        assert config.electrons["mixing_beta"] == 0.3
        assert len(warnings) == 0

    def test_max_semantics_ecutwfc(self):
        """ecutwfc should not be decreased."""
        config = self._make_mock_config()
        config.system = {"ecutwfc": 120}  # SSSP already high
        correction = QECorrection(
            error_type="QE_S_MATRIX", step=0,
            namelist_updates={"system.ecutwfc": 80},  # Lower than current
        )
        QEErrorRecoveryEngine.apply_correction_to_config(config, correction)
        assert config.system["ecutwfc"] == 120  # Not decreased

    def test_max_semantics_increases(self):
        """ecutwfc should increase when correction is higher."""
        config = self._make_mock_config()
        correction = QECorrection(
            error_type="QE_S_MATRIX", step=0,
            namelist_updates={"system.ecutwfc": 100},
        )
        QEErrorRecoveryEngine.apply_correction_to_config(config, correction)
        assert config.system["ecutwfc"] == 100

    def test_electron_maxstep_cap(self):
        """electron_maxstep_cap should override existing value."""
        config = self._make_mock_config()
        correction = QECorrection(
            error_type="QE_SCF_UNCONVERGED", step=1,
            namelist_updates={"electrons.mixing_mode": "local-TF"},
            electron_maxstep_cap=50,
        )
        QEErrorRecoveryEngine.apply_correction_to_config(config, correction)
        assert config.electrons["electron_maxstep"] == 50
        assert config.electrons["mixing_mode"] == "local-TF"

    def test_vc_relax_downgrade(self):
        """calculation=relax should trigger vc-relax downgrade."""
        config = self._make_mock_config()
        correction = QECorrection(
            error_type="QE_CELL_DISTORTED", step=2,
            namelist_updates={"control.calculation": "relax"},
            quality_warnings=["vc_relax_downgraded"],
        )
        warnings = QEErrorRecoveryEngine.apply_correction_to_config(config, correction)
        assert config.control["calculation"] == "relax"
        assert config.cell == {}
        assert "vc_relax_downgraded" in warnings

    def test_new_namelist_key(self):
        """Adding a new key to an existing namelist should work."""
        config = self._make_mock_config()
        correction = QECorrection(
            error_type="QE_SCF_UNCONVERGED", step=1,
            namelist_updates={"electrons.mixing_mode": "local-TF"},
        )
        QEErrorRecoveryEngine.apply_correction_to_config(config, correction)
        assert config.electrons["mixing_mode"] == "local-TF"

    def test_cell_factor_applied(self):
        """cell_factor should be added to cell dict."""
        config = self._make_mock_config()
        correction = QECorrection(
            error_type="QE_CELL_DISTORTED", step=0,
            namelist_updates={"cell.cell_factor": 2.0},
        )
        QEErrorRecoveryEngine.apply_correction_to_config(config, correction)
        assert config.cell["cell_factor"] == 2.0

    def test_quality_warnings_returned(self):
        config = self._make_mock_config()
        correction = QECorrection(
            error_type="QE_BFGS_FAILED", step=2,
            namelist_updates={"control.forc_conv_thr": 1.0e-2},
            quality_warnings=["loosely_relaxed"],
        )
        warnings = QEErrorRecoveryEngine.apply_correction_to_config(config, correction)
        assert warnings == ["loosely_relaxed"]


# ---------------------------------------------------------------------------
# TestQECorrectionHistory
# ---------------------------------------------------------------------------


class TestQECorrectionHistory:
    """Test correction history tracking."""

    def test_history_empty_initially(self):
        engine = QEErrorRecoveryEngine()
        assert engine.correction_history == []

    def test_history_records_corrections(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)
        assert len(engine.correction_history) == 1
        assert engine.correction_history[0]["action"] == "correction_applied"
        assert engine.correction_history[0]["step"] == 0

    def test_history_records_fatal(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_PSEUDO_NOT_FOUND", ErrorSeverity.FATAL)
        engine.get_correction(error)
        assert engine.correction_history[-1]["action"] == "fatal_no_correction"

    def test_history_records_exhaustion(self):
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_EIGVAL_NOT_CONVERGED", ErrorSeverity.CORRECTABLE)
        steps = QE_CORRECTION_STRATEGIES["QE_EIGVAL_NOT_CONVERGED"]
        for _ in range(len(steps)):
            engine.get_correction(error)
        engine.get_correction(error)  # Exhausted
        assert engine.correction_history[-1]["action"] == "strategies_exhausted"

    def test_history_is_copy(self):
        """correction_history property should return a copy."""
        engine = QEErrorRecoveryEngine()
        h1 = engine.correction_history
        h1.append({"fake": True})
        assert len(engine.correction_history) == 0  # Not affected


# ---------------------------------------------------------------------------
# TestQEPatternRegistry
# ---------------------------------------------------------------------------


class TestQEPatternRegistry:
    """Test pattern and strategy registry consistency."""

    def test_all_correctable_types_have_strategies(self):
        """Every correctable error type should have at least one correction strategy."""
        correctable_types = {
            p[1] for p in QE_ERROR_PATTERNS if p[2] == ErrorSeverity.CORRECTABLE
        }
        for etype in correctable_types:
            assert etype in QE_CORRECTION_STRATEGIES, (
                f"No correction strategy for correctable error: {etype}"
            )

    def test_no_strategies_for_fatal(self):
        """Fatal error types should NOT have strategies."""
        fatal_types = {
            p[1] for p in QE_ERROR_PATTERNS if p[2] == ErrorSeverity.FATAL
        }
        for etype in fatal_types:
            assert etype not in QE_CORRECTION_STRATEGIES, (
                f"Fatal error {etype} should not have strategies"
            )

    def test_strategy_count(self):
        """Should have 10 error types with strategies."""
        assert len(QE_CORRECTION_STRATEGIES) == 10

    def test_pattern_severity_enum(self):
        """All severities should be valid ErrorSeverity values."""
        for _, _, severity in QE_ERROR_PATTERNS:
            assert isinstance(severity, ErrorSeverity)


# ---------------------------------------------------------------------------
# TestCheckAtomicOverlap
# ---------------------------------------------------------------------------


class TestCheckAtomicOverlap:
    """Test atomic overlap detection."""

    def test_overlapping_atoms(self):
        atoms = Atoms("Cu2", positions=[(0, 0, 0), (0.3, 0, 0)], cell=[10, 10, 10], pbc=True)
        assert check_atomic_overlap(atoms) is True

    def test_normal_structure(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        assert check_atomic_overlap(atoms) is False

    def test_single_atom(self):
        atoms = Atoms("Cu", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
        assert check_atomic_overlap(atoms) is False

    def test_periodic_overlap(self):
        """Atoms close across periodic boundary should be detected."""
        atoms = Atoms("Cu2", positions=[(0, 0, 0), (9.7, 0, 0)], cell=[10, 10, 10], pbc=True)
        assert check_atomic_overlap(atoms) is True

    def test_threshold_absolute_floor(self):
        """Distance below 0.5 Å should always be flagged."""
        assert MIN_DISTANCE_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# TestLocalTFFastFail
# ---------------------------------------------------------------------------


class TestLocalTFFastFail:
    """Test local-TF step has fast-fail cap."""

    def test_local_tf_step_has_maxstep_cap(self):
        """SCF step 1 (local-TF) should have _electron_maxstep_cap=50 in YAML."""
        strategies = QE_CORRECTION_STRATEGIES["QE_SCF_UNCONVERGED"]
        # Step 1 is the local-TF step
        step1 = strategies[1]
        assert "_electron_maxstep_cap" in step1
        assert step1["_electron_maxstep_cap"] == 50

    def test_local_tf_correction_carries_cap(self):
        """The correction object should have electron_maxstep_cap=50."""
        engine = QEErrorRecoveryEngine()
        error = QEError("QE_SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE)
        engine.get_correction(error)  # Step 0
        c = engine.get_correction(error)  # Step 1 (local-TF)
        assert c is not None
        assert c.electron_maxstep_cap == 50
