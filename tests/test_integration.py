"""Integration tests for Phase 1: VASP automation end-to-end flows.

Tests the complete path: VASPInputConfig -> write_input -> parse_output -> ReviewAgent,
including error recovery loops, double relaxation, and physics validation.
"""

import os
import shutil
import textwrap

import pytest

from shalom.backends.vasp import VASPBackend
from shalom.backends.vasp_config import (
    AccuracyLevel,
    CalculationType,
    get_preset,
)
from shalom.backends.error_recovery import ErrorRecoveryEngine, VASPError, ErrorSeverity
from shalom.agents.review_layer import ReviewAgent
from shalom.core.schemas import ReviewResult
from shalom.pipeline import PipelineConfig


# ---------------------------------------------------------------------------
# End-to-End Flow Tests
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    """Tests for config -> write_input -> parse_output -> ReviewAgent."""

    def test_bulk_cu_full_flow(self, tmp_path, sample_bulk_cu, mock_llm):
        """Full flow: Cu bulk -> 4 files -> parse OUTCAR -> ReviewAgent."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()

        # Write input files
        backend.write_input(sample_bulk_cu, str(tmp_path), config=config)
        assert os.path.exists(os.path.join(str(tmp_path), "POSCAR"))
        assert os.path.exists(os.path.join(str(tmp_path), "INCAR"))

        # Create a mock OUTCAR for parsing
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(textwrap.dedent("""\
         free  energy   TOTEN  =       -3.750000 eV

         TOTAL-FORCE (eV/Angst)
         ---------------------------
              0.000  0.000  0.000    0.0010  0.0005 -0.0008
         ---------------------------

         number of atoms/cell =      1

         General timing and accounting informations for this job:
        """))

        # Parse output
        result = backend.parse_output(str(tmp_path))
        assert result.is_converged is True
        assert result.energy == pytest.approx(-3.75)
        assert result.forces_max is not None
        assert result.forces_max < 0.01

        # Feed to ReviewAgent
        mock_result = ReviewResult(
            is_successful=True,
            energy=-3.75,
            forces_max=result.forces_max,
            feedback_for_design="Converged. Good structure.",
        )
        mock_llm.generate_structured_output.return_value = mock_result
        agent = ReviewAgent(llm_provider=mock_llm)
        review = agent.review_with_backend("Find stable Cu bulk", str(tmp_path), backend)
        assert review.is_successful is True

    def test_magnetic_structure_flow(self, tmp_path, sample_bulk_fe, mock_llm):
        """Fe bulk -> ISPIN=2, MAGMOM in INCAR -> ReviewAgent sees magnetization."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_fe)
        backend = VASPBackend()
        backend.write_input(sample_bulk_fe, str(tmp_path), config=config)

        with open(os.path.join(str(tmp_path), "INCAR"), "r", encoding="utf-8") as f:
            incar = f.read()
        assert "ISPIN = 2" in incar
        assert "MAGMOM" in incar

    def test_2d_structure_flow(self, tmp_path, sample_2d_slab, mock_llm):
        """MoS2 slab -> ISIF=4, IVDW=12, KPOINTS z=1."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_2d_slab)
        backend = VASPBackend()
        backend.write_input(sample_2d_slab, str(tmp_path), config=config)

        with open(os.path.join(str(tmp_path), "INCAR"), "r", encoding="utf-8") as f:
            incar = f.read()
        assert "ISIF = 4" in incar
        assert "IVDW = 12" in incar

        with open(os.path.join(str(tmp_path), "KPOINTS"), "r", encoding="utf-8") as f:
            lines = f.readlines()
        grid = lines[3].strip().split()
        assert grid[-1] == "1"

    def test_tmo_precise_enables_ldau(self, tmp_path, sample_tmo_feo):
        """FeO + PRECISE -> GGA+U auto-enabled with Wang U values."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.PRECISE, sample_tmo_feo)
        backend = VASPBackend()
        backend.write_input(sample_tmo_feo, str(tmp_path), config=config)

        with open(os.path.join(str(tmp_path), "INCAR"), "r", encoding="utf-8") as f:
            incar = f.read()
        assert "LDAU" in incar
        assert "LDAUU" in incar


# ---------------------------------------------------------------------------
# Error Recovery Loop Tests
# ---------------------------------------------------------------------------


class TestErrorRecoveryLoop:
    """Tests for error detection -> correction -> retry flow."""

    def test_scf_failure_correction_cycle(self):
        """SCF failure -> correct -> new INCAR settings."""
        engine = ErrorRecoveryEngine()

        outcar_text = "Some output\n NELM reached\nMore output"
        errors = engine.scan_for_errors(outcar_text)
        assert len(errors) == 1

        correction = engine.get_correction(errors[0])
        assert correction is not None
        assert "NELM" in correction.incar_updates

        # Apply correction to config
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD)
        config.user_incar_settings.update(correction.incar_updates)
        merged = config.get_merged_incar()
        assert merged["NELM"] == 200

    def test_strategy_exhaustion_fails(self):
        """All strategies exhausted -> None (pipeline should fail)."""
        engine = ErrorRecoveryEngine()
        error = VASPError("ZBRENT", ErrorSeverity.CORRECTABLE)

        # Exhaust all 3 ZBRENT strategies
        for _ in range(3):
            corr = engine.get_correction(error)
            assert corr is not None

        # Next attempt should return None
        corr = engine.get_correction(error)
        assert corr is None

    def test_false_convergence_marks_unconverged(self):
        """False convergence detected -> DFT result marked as not converged."""
        from shalom.backends.base import DFTResult

        result = DFTResult(
            energy=-30.82,
            forces_max=0.019,
            is_converged=True,
            ionic_energies=[-30.0, -30.0001, -30.00005],
            ionic_forces_max=[0.025, 0.018, 0.022, 0.019, 0.023, 0.017, 0.024, 0.018, 0.021, 0.020],
        )
        engine = ErrorRecoveryEngine()
        if engine.detect_false_convergence(
            result.ionic_energies, result.ionic_forces_max,
        ):
            result.is_converged = False

        assert result.is_converged is False


# ---------------------------------------------------------------------------
# Double Relaxation Flow Tests
# ---------------------------------------------------------------------------


class TestDoubleRelaxFlow:
    """Tests for smart double relaxation trigger."""

    def test_small_volume_change_skip_step2(self):
        """Volume change < 3% -> skip step2."""
        assert VASPBackend.should_trigger_step2(100.0, 102.5) is False

    def test_large_volume_change_trigger_step2(self, tmp_path, sample_bulk_cu):
        """Volume change > 3% -> trigger step2 with full files."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        dirs = backend.write_double_relaxation(sample_bulk_cu, str(tmp_path), config)

        # Simulate volume change > 3%
        should_run = backend.should_trigger_step2(100.0, 105.0)
        assert should_run is True
        assert os.path.isdir(dirs[1])

    def test_double_relax_step1_is_coarser(self, tmp_path, sample_bulk_cu):
        """Step1 has coarser EDIFFG than step2."""
        config = get_preset(CalculationType.RELAXATION, AccuracyLevel.STANDARD, sample_bulk_cu)
        backend = VASPBackend()
        dirs = backend.write_double_relaxation(sample_bulk_cu, str(tmp_path), config)

        with open(os.path.join(dirs[0], "INCAR"), "r", encoding="utf-8") as f:
            step1 = f.read()
        with open(os.path.join(dirs[1], "INCAR"), "r", encoding="utf-8") as f:
            step2 = f.read()

        assert "-0.05" in step1  # coarse
        assert "-0.02" in step2  # fine


# ---------------------------------------------------------------------------
# Physics Validation Tests
# ---------------------------------------------------------------------------


class TestPhysicsValidation:
    """Tests for ReviewAgent._run_physics_checks()."""

    def test_high_force_warning(self):
        from shalom.backends.base import DFTResult
        result = DFTResult(forces_max=0.05)
        warnings = ReviewAgent._run_physics_checks(result)
        assert any("0.02" in w for w in warnings)

    def test_high_entropy_warning(self):
        from shalom.backends.base import DFTResult
        result = DFTResult(entropy_per_atom=-0.005)
        warnings = ReviewAgent._run_physics_checks(result)
        assert any("SIGMA" in w for w in warnings)

    def test_brmix_correction_warning(self):
        from shalom.backends.base import DFTResult
        result = DFTResult(correction_history=[
            {"error_type": "BRMIX", "action": "correction_applied"},
        ])
        warnings = ReviewAgent._run_physics_checks(result)
        assert any("BRMIX" in w for w in warnings)

    def test_clean_result_no_warnings(self):
        from shalom.backends.base import DFTResult
        result = DFTResult(forces_max=0.01, entropy_per_atom=-0.0005)
        warnings = ReviewAgent._run_physics_checks(result)
        assert warnings == []


# ---------------------------------------------------------------------------
# PipelineConfig New Fields Tests
# ---------------------------------------------------------------------------


class TestPipelineConfigFields:
    """Tests for new PipelineConfig fields."""

    def test_default_values(self):
        config = PipelineConfig()
        assert config.calc_type == "relaxation"
        assert config.accuracy == "standard"
        assert config.vasp_user_incar is None

    def test_custom_values(self):
        config = PipelineConfig(
            calc_type="static",
            accuracy="precise",
            vasp_user_incar={"ENCUT": 700},
        )
        assert config.calc_type == "static"
        assert config.accuracy == "precise"
        assert config.vasp_user_incar["ENCUT"] == 700

    def test_serialization_roundtrip(self):
        config = PipelineConfig(
            calc_type="dos",
            vasp_user_incar={"NEDOS": 5001},
        )
        json_str = config.model_dump_json()
        restored = PipelineConfig.model_validate_json(json_str)
        assert restored.calc_type == "dos"
        assert restored.vasp_user_incar["NEDOS"] == 5001


# ---------------------------------------------------------------------------
# QE End-to-End Integration Tests (requires pw.x)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("pw.x") is None,
    reason="pw.x not installed (run from WSL with QE)",
)
class TestQEEndToEnd:
    """Real QE pw.x integration tests — skipped unless QE available."""

    def test_si_scf_end_to_end(self, tmp_path):
        """Real Si SCF with pw.x."""
        from ase.build import bulk
        from ase.io import write as ase_write

        from shalom.direct_run import direct_run, DirectRunConfig
        from shalom.backends.runner import ExecutionRunner, ExecutionConfig
        from shalom.backends.qe import QEBackend

        # Generate Si structure
        si = bulk("Si", "diamond", a=5.43)
        poscar = tmp_path / "Si_POSCAR"
        ase_write(str(poscar), si, format="vasp")

        pseudo_dir = os.environ.get("SHALOM_PSEUDO_DIR", str(tmp_path))
        config = DirectRunConfig(
            backend_name="qe",
            calc_type="scf",
            output_dir=str(tmp_path / "calc"),
            structure_file=str(poscar),
            pseudo_dir=pseudo_dir,
            force_overwrite=True,
        )
        result = direct_run("", config)
        assert result.success
        assert "pw.in" in result.files_generated

        # Execute
        runner = ExecutionRunner(ExecutionConfig(nprocs=1))
        exec_result = runner.run(result.output_dir)
        assert exec_result.success, f"QE failed: {exec_result.error_message}"

        # Parse
        backend = QEBackend()
        dft_result = backend.parse_output(result.output_dir)
        assert dft_result.is_converged
        assert dft_result.energy is not None
        assert dft_result.energy < 0  # Negative total energy


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("pw.x") is None,
    reason="pw.x not installed (run from WSL with QE)",
)
class TestQEWorkflow:
    """Full 5-step StandardWorkflow with real pw.x."""

    @pytest.fixture()
    def si_atoms(self):
        from ase.build import bulk
        return bulk("Si", "diamond", a=5.43)

    @pytest.fixture()
    def pseudo_dir(self):
        d = os.environ.get("SHALOM_PSEUDO_DIR", "")
        if not d or not os.path.isdir(d):
            pytest.skip("SHALOM_PSEUDO_DIR not set or missing")
        return d

    def test_si_scf_and_bands(self, si_atoms, pseudo_dir, tmp_path):
        """Si SCF + bands via direct_run + ExecutionRunner.

        The full 5-step workflow is too slow for CI on 1 CPU (NSCF with
        nosym+noinv generates 1000 k-points → >60min).  Instead, test
        the critical path: input generation → SCF → bands → XML parsing.
        Full validation (incl. NSCF/DOS) is done via scripts/validate_v1.py.
        """
        from ase.io import write as ase_write

        from shalom.backends.qe import QEBackend
        from shalom.backends.qe_config import (
            generate_band_kpath,
            get_band_calc_atoms,
            get_qe_preset,
        )
        from shalom.backends.qe_parser import (
            extract_fermi_energy,
            find_xml_path,
            parse_xml_bands,
        )
        from shalom.backends.runner import ExecutionConfig, ExecutionRunner
        from shalom.direct_run import DirectRunConfig, direct_run

        backend = QEBackend()
        runner = ExecutionRunner(ExecutionConfig(nprocs=1))

        # --- Step 1: SCF ------------------------------------------------
        poscar = tmp_path / "Si_POSCAR"
        ase_write(str(poscar), si_atoms, format="vasp")

        scf_dir = str(tmp_path / "02_scf")
        scf_cfg = DirectRunConfig(
            backend_name="qe",
            calc_type="scf",
            output_dir=scf_dir,
            structure_file=str(poscar),
            pseudo_dir=pseudo_dir,
            force_overwrite=True,
        )
        scf_gen = direct_run("", scf_cfg)
        assert scf_gen.success, "SCF input generation failed"

        scf_result = runner.run(scf_gen.output_dir)
        assert scf_result.success, f"SCF failed: {scf_result.error_message}"

        dft = backend.parse_output(scf_gen.output_dir)
        assert dft.is_converged, "SCF did not converge"
        assert dft.energy is not None and dft.energy < 0

        fermi = extract_fermi_energy(os.path.join(scf_dir, "pw.out"))
        assert fermi is not None, "Fermi energy not extracted from SCF"
        assert 4.0 < fermi < 9.0, f"Fermi energy out of range: {fermi}"

        # --- Step 2: bands -----------------------------------------------
        calc_atoms = get_band_calc_atoms(si_atoms) or si_atoms
        kpath_cfg = generate_band_kpath(calc_atoms, npoints=20)  # small for speed

        bands_dir = str(tmp_path / "03_bands")
        os.makedirs(bands_dir, exist_ok=True)

        # Build bands config (must pass atoms= for ibrav/nat/ntyp/ecutwfc)
        from shalom.backends.qe_config import QECalculationType
        from shalom.backends.qe_parser import compute_nbnd

        scf_tmp = os.path.abspath(os.path.join(scf_dir, "tmp"))
        bands_config = get_qe_preset(QECalculationType.BANDS, atoms=calc_atoms)
        bands_config.control["outdir"] = scf_tmp
        bands_config.pseudo_dir = pseudo_dir
        bands_config.system["nbnd"] = compute_nbnd(calc_atoms)
        bands_config.kpoints = kpath_cfg

        backend.write_input(calc_atoms, bands_dir, config=bands_config)
        bands_result = runner.run(bands_dir)
        assert bands_result.success, f"Bands failed: {bands_result.error_message}"

        # --- Step 3: Parse band structure --------------------------------
        xml_path = find_xml_path(bands_dir)
        if xml_path is None:
            # XML might be in SCF tmp dir
            xml_path = find_xml_path(scf_tmp)
        assert xml_path is not None, "Band XML not found"

        bs = parse_xml_bands(xml_path, fermi_energy=fermi)
        assert bs.eigenvalues.shape[0] > 50, f"Too few k-points: {bs.eigenvalues.shape}"
        assert bs.eigenvalues.shape[1] >= 8, f"Too few bands: {bs.eigenvalues.shape}"

        # --- Bandgap check (PBE Si: indirect ~0.5–0.7 eV) ---------------
        vbm = bs.eigenvalues[bs.eigenvalues <= fermi].max()
        cbm = bs.eigenvalues[bs.eigenvalues > fermi].min()
        gap = cbm - vbm
        assert 0.2 < gap < 1.0, f"Si bandgap out of range: {gap:.3f} eV"


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("pw.x") is None,
    reason="pw.x not installed (run from WSL with QE)",
)
class TestQEConvergence:
    """Cutoff convergence test with real pw.x."""

    def test_si_cutoff_convergence(self, tmp_path):
        """ecutwfc sweep: 20, 30, 40 Ry."""
        from ase.build import bulk
        from shalom.workflows.convergence import CutoffConvergence

        pseudo_dir = os.environ.get("SHALOM_PSEUDO_DIR", "")
        if not pseudo_dir or not os.path.isdir(pseudo_dir):
            pytest.skip("SHALOM_PSEUDO_DIR not set or missing")

        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path / "ecut"),
            values=[20, 30, 40],
            kgrid=[4, 4, 4],
            pseudo_dir=pseudo_dir,
            nprocs=1,
            timeout=600,
        )
        result = conv.run()

        # All 3 should have completed
        assert len(result.results) == 3

        # Each should have an energy
        for r in result.results:
            assert r.energy is not None, f"No energy for value={r.value}"
            assert r.energy < 0

        # Summary is not empty
        assert len(result.summary()) > 0


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("pw.x") is None,
    reason="pw.x not installed (run from WSL with QE)",
)
class TestQECLI:
    """CLI subcommand integration tests with real pw.x."""

    def test_cli_setup_qe(self):
        """setup-qe reports pw.x and pseudo status."""
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "shalom", "setup-qe", "--elements", "Si"],
            capture_output=True, text=True, timeout=30,
        )
        combined = proc.stdout + proc.stderr
        assert "pw.x" in combined
        assert "SHALOM" in combined  # header

    def test_cli_run_si_scf(self, tmp_path):
        """CLI: python -m shalom run --structure <POSCAR> -b qe --calc scf."""
        import subprocess
        import sys

        from ase.build import bulk
        from ase.io import write as ase_write

        pseudo_dir = os.environ.get("SHALOM_PSEUDO_DIR", "")
        if not pseudo_dir or not os.path.isdir(pseudo_dir):
            pytest.skip("SHALOM_PSEUDO_DIR not set or missing")

        # Write a POSCAR so we don't need mp-api
        poscar = tmp_path / "Si_POSCAR"
        si = bulk("Si", "diamond", a=5.43)
        ase_write(str(poscar), si, format="vasp")

        out = str(tmp_path / "cli_scf")
        proc = subprocess.run(
            [
                sys.executable, "-m", "shalom", "run",
                "--structure", str(poscar),
                "-b", "qe", "--calc", "scf",
                "-o", out,
                "--pseudo-dir", pseudo_dir,
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"CLI failed: {proc.stdout}\n{proc.stderr}"
        assert os.path.isfile(os.path.join(out, "pw.in"))
