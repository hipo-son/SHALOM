"""Tests for shalom.workflows.convergence and shalom.workflows.base."""

from __future__ import annotations

from typing import List

import pytest
from ase.build import bulk

from shalom.workflows.base import (
    ConvergenceResult,
    ConvergenceTestResult,
    ConvergenceWorkflow,
)
from shalom.workflows.convergence import CutoffConvergence, KpointConvergence


# ---------------------------------------------------------------------------
# Helper: concrete ConvergenceWorkflow subclass for unit testing
# ---------------------------------------------------------------------------


class _DummyConvergence(ConvergenceWorkflow):
    """Minimal subclass that injects pre-computed results."""

    _parameter_name = "dummy"

    def __init__(self, atoms, output_dir, values, results_map, **kwargs):
        super().__init__(atoms=atoms, output_dir=output_dir, **kwargs)
        self._values_list = sorted(values)
        self._results_map = results_map  # {value: ConvergenceResult}

    @property
    def _values(self) -> List[float]:
        return self._values_list

    def _run_single(self, param_value: float, calc_dir: str) -> ConvergenceResult:
        return self._results_map.get(
            param_value,
            ConvergenceResult(param_value, None, False, calc_dir),
        )


# ---------------------------------------------------------------------------
# ConvergenceWorkflow._find_converged_value
# ---------------------------------------------------------------------------


class TestFindConvergedValue:
    def _make_workflow(self, atoms, tmp_path, values, results_map):
        return _DummyConvergence(
            atoms=atoms,
            output_dir=str(tmp_path),
            values=values,
            results_map=results_map,
        )

    def test_finds_first_converged(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        # Reference energy (last value) = -100.0 eV
        # 40 Ry: diff = |(-100.000) - (-100.000)| = 0 < 1e-3 eV/atom
        # 50 Ry: diff = 0 (same as reference) → 40 Ry is the first converged
        results_map = {
            30.0: ConvergenceResult(30.0, -99.99 * 2, True, ""),  # 0.01 eV/atom diff → not converged
            40.0: ConvergenceResult(40.0, -100.00 * 2, True, ""),  # 0 diff → converged
            50.0: ConvergenceResult(50.0, -100.00 * 2, True, ""),  # reference
        }
        workflow = self._make_workflow(si, tmp_path, [30, 40, 50], results_map)
        converged = workflow._find_converged_value(list(results_map.values()))
        assert converged == pytest.approx(40.0)

    def test_returns_none_if_no_convergence(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        # All values differ by > 1e-3 eV/atom
        results_map = {
            30.0: ConvergenceResult(30.0, -99.0 * 2, True, ""),
            40.0: ConvergenceResult(40.0, -99.5 * 2, True, ""),
            50.0: ConvergenceResult(50.0, -100.0 * 2, True, ""),
        }
        workflow = self._make_workflow(si, tmp_path, [30, 40, 50], results_map)
        assert workflow._find_converged_value(list(results_map.values())) is None

    def test_returns_none_if_too_few_converged(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        results_map = {
            30.0: ConvergenceResult(30.0, None, False, ""),
            40.0: ConvergenceResult(40.0, -100.0 * 2, True, ""),
        }
        workflow = self._make_workflow(si, tmp_path, [30, 40], results_map)
        assert workflow._find_converged_value(list(results_map.values())) is None

    def test_per_atom_scaling(self, tmp_path):
        """Threshold must scale with number of atoms."""
        from ase import Atoms
        # 2-atom cell; per-atom threshold = 1e-3; absolute threshold = 2e-3 eV
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                      cell=[[0, 2.72, 2.72], [2.72, 0, 2.72], [2.72, 2.72, 0]],
                      pbc=True)
        # diff = 0.001 eV total = 0.0005 eV/atom < 1e-3 → converged
        results_map = {
            30.0: ConvergenceResult(30.0, -100.000, True, ""),
            40.0: ConvergenceResult(40.0, -100.001, True, ""),   # reference
        }
        workflow = self._make_workflow(atoms, tmp_path, [30, 40], results_map)
        assert workflow._find_converged_value(list(results_map.values())) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# ConvergenceTestResult
# ---------------------------------------------------------------------------


class TestConvergenceTestResult:
    def test_summary_converged(self):
        r = ConvergenceTestResult(
            parameter_name="ecutwfc",
            converged_value=50.0,
            reference_value=80.0,
        )
        s = r.summary()
        assert "converged at 50" in s

    def test_summary_not_converged(self):
        r = ConvergenceTestResult(parameter_name="ecutwfc")
        assert "NOT converged" in r.summary()

    def test_converged_results_filter(self):
        results = [
            ConvergenceResult(30.0, None, False),
            ConvergenceResult(40.0, -100.0, True),
            ConvergenceResult(50.0, -100.01, True),
        ]
        r = ConvergenceTestResult("ecutwfc", results=results)
        assert len(r.converged_results) == 2


# ---------------------------------------------------------------------------
# CutoffConvergence (mocked pw.x)
# ---------------------------------------------------------------------------


class TestCutoffConvergence:
    def test_instantiation(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            values=[30.0, 40.0],
            kgrid=[4, 4, 4],
        )
        assert conv._parameter_name == "ecutwfc"
        assert conv._values == [30.0, 40.0]

    def test_run_calls_run_single(self, tmp_path):
        """run() should call _run_single for each value."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            values=[30.0, 40.0],
            kgrid=[4, 4, 4],
        )
        call_count = []

        def fake_run_single(val, calc_dir):
            call_count.append(val)
            return ConvergenceResult(val, -100.0, True, calc_dir)

        conv._run_single = fake_run_single  # type: ignore[method-assign]
        result = conv.run()
        assert len(call_count) == 2
        assert isinstance(result, ConvergenceTestResult)


# ---------------------------------------------------------------------------
# KpointConvergence (mocked pw.x)
# ---------------------------------------------------------------------------


class TestKpointConvergence:
    def test_instantiation(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        conv = KpointConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            resolutions=[20.0, 30.0],
            ecutwfc=50.0,
        )
        assert conv._parameter_name == "kpoint_resolution"
        assert conv._values == [20.0, 30.0]
        assert conv.ecutwfc == pytest.approx(50.0)

    def test_run_calls_run_single(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        conv = KpointConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            resolutions=[20.0, 30.0],
            ecutwfc=50.0,
        )
        call_count = []

        def fake_run_single(val, calc_dir):
            call_count.append(val)
            return ConvergenceResult(val, -100.0, True, calc_dir)

        conv._run_single = fake_run_single  # type: ignore[method-assign]
        conv.run()
        assert len(call_count) == 2


# ---------------------------------------------------------------------------
# Convergence plot (matplotlib optional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="skip") is None,
    reason="matplotlib not installed",
)
def test_convergence_plot_creates_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")  # avoid tkinter on Windows
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    si = bulk("Si", "diamond", a=5.43)
    results = [
        ConvergenceResult(30.0, -100.0 * 2, True),
        ConvergenceResult(40.0, -100.001 * 2, True),
        ConvergenceResult(50.0, -100.002 * 2, True),
    ]
    result = ConvergenceTestResult(
        parameter_name="ecutwfc",
        results=results,
        converged_value=40.0,
        reference_value=50.0,
    )
    conv = _DummyConvergence(
        atoms=si,
        output_dir=str(tmp_path),
        values=[30, 40, 50],
        results_map={},
    )
    plot_path = conv.plot(result, output_path=str(tmp_path / "conv.png"))
    assert plot_path is not None
    assert os.path.isfile(plot_path)


import os  # noqa: E402  (needed for the test above)


# ---------------------------------------------------------------------------
# CutoffConvergence._run_single (mocked pw.x + backend)
# ---------------------------------------------------------------------------


class TestCutoffRunSingle:
    """Unit tests for CutoffConvergence._run_single without pw.x."""

    def _make_mock_runner_result(self, success=True):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.success = success
        return r

    def _make_mock_dft_result(self, energy=-100.0, converged=True):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.energy = energy
        r.is_converged = converged
        return r

    def test_run_single_success(self, tmp_path):
        """_run_single returns ConvergenceResult with energy on success."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            values=[40.0],
            kgrid=[4, 4, 4],
        )

        from unittest.mock import patch

        mock_runner_result = self._make_mock_runner_result(success=True)
        mock_dft_result = self._make_mock_dft_result(energy=-200.0, converged=True)

        calc_dir = str(tmp_path / "ecutwfc_40")

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_runner_result), \
             patch("shalom.backends.qe.QEBackend.parse_output", return_value=mock_dft_result):
            result = conv._run_single(40.0, calc_dir)

        assert result.parameter_value == pytest.approx(40.0)
        assert result.energy == pytest.approx(-200.0)
        assert result.is_converged is True

    def test_run_single_runner_failure(self, tmp_path):
        """_run_single returns is_converged=False when runner fails."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            values=[40.0],
            kgrid=[4, 4, 4],
        )

        from unittest.mock import patch

        mock_runner_result = self._make_mock_runner_result(success=False)
        mock_dft_result = self._make_mock_dft_result(energy=-200.0, converged=True)

        calc_dir = str(tmp_path / "ecutwfc_40")

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_runner_result), \
             patch("shalom.backends.qe.QEBackend.parse_output", return_value=mock_dft_result):
            result = conv._run_single(40.0, calc_dir)

        # is_converged = dft.is_converged AND runner.success → both needed
        assert result.is_converged is False

    def test_run_single_exception_caught(self, tmp_path):
        """_run_single catches exceptions and returns error result."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            values=[40.0],
            kgrid=[4, 4, 4],
        )

        from unittest.mock import patch

        calc_dir = str(tmp_path / "ecutwfc_40")

        with patch("shalom.backends.qe.QEBackend.write_input",
                   side_effect=RuntimeError("write failed")):
            result = conv._run_single(40.0, calc_dir)

        assert result.is_converged is False
        assert result.energy is None
        assert "write failed" in result.error_message

    def test_run_single_no_kgrid_uses_default(self, tmp_path):
        """When kgrid=None, _run_single computes default grid."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            values=[40.0],
            kgrid=None,  # trigger default grid path
        )

        from unittest.mock import patch

        mock_runner_result = self._make_mock_runner_result(success=True)
        mock_dft_result = self._make_mock_dft_result(energy=-100.0, converged=True)

        calc_dir = str(tmp_path / "ecutwfc_40")

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_runner_result), \
             patch("shalom.backends.qe.QEBackend.parse_output", return_value=mock_dft_result):
            result = conv._run_single(40.0, calc_dir)

        assert result.is_converged is True

    def test_run_single_pseudo_dir_applied(self, tmp_path):
        """pseudo_dir is forwarded to config when set."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            values=[40.0],
            kgrid=[4, 4, 4],
            pseudo_dir="/my/pseudos",
        )

        from unittest.mock import patch

        captured_config = {}

        def fake_write(atoms, calc_dir, config=None):
            captured_config["pseudo_dir"] = getattr(config, "pseudo_dir", None)

        mock_runner_result = self._make_mock_runner_result(success=True)
        mock_dft_result = self._make_mock_dft_result()

        calc_dir = str(tmp_path / "ecutwfc_40")

        with patch("shalom.backends.qe.QEBackend.write_input", side_effect=fake_write), \
             patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_runner_result), \
             patch("shalom.backends.qe.QEBackend.parse_output", return_value=mock_dft_result):
            conv._run_single(40.0, calc_dir)

        assert captured_config.get("pseudo_dir") == "/my/pseudos"

    def test_run_single_precise_accuracy(self, tmp_path):
        """AccuracyLevel.PRECISE is used when accuracy='precise'."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            values=[60.0],
            kgrid=[4, 4, 4],
            accuracy="precise",
        )

        from unittest.mock import patch

        mock_runner_result = self._make_mock_runner_result(success=True)
        mock_dft_result = self._make_mock_dft_result()
        calc_dir = str(tmp_path / "ecutwfc_60")

        captured_acc = {}

        def fake_preset(calc_type, accuracy, atoms=None):
            captured_acc["acc"] = accuracy
            from shalom.backends.qe_config import get_qe_preset as real_preset
            return real_preset(calc_type, accuracy, atoms=atoms)

        with patch("shalom.backends.qe_config.get_qe_preset", side_effect=fake_preset), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_runner_result), \
             patch("shalom.backends.qe.QEBackend.parse_output", return_value=mock_dft_result):
            conv._run_single(60.0, calc_dir)

        from shalom.backends._physics import AccuracyLevel
        assert captured_acc.get("acc") == AccuracyLevel.PRECISE


# ---------------------------------------------------------------------------
# KpointConvergence._run_single (mocked pw.x + backend)
# ---------------------------------------------------------------------------


class TestKpointRunSingle:
    """Unit tests for KpointConvergence._run_single without pw.x."""

    def _make_mock_runner_result(self, success=True):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.success = success
        return r

    def _make_mock_dft_result(self, energy=-100.0, converged=True):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.energy = energy
        r.is_converged = converged
        return r

    def test_run_single_success(self, tmp_path):
        """_run_single returns ConvergenceResult with energy."""
        si = bulk("Si", "diamond", a=5.43)
        conv = KpointConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            resolutions=[30.0],
            ecutwfc=50.0,
        )

        from unittest.mock import patch

        mock_runner_result = self._make_mock_runner_result(success=True)
        mock_dft_result = self._make_mock_dft_result(energy=-150.0, converged=True)

        calc_dir = str(tmp_path / "kpoint_30")

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_runner_result), \
             patch("shalom.backends.qe.QEBackend.parse_output", return_value=mock_dft_result):
            result = conv._run_single(30.0, calc_dir)

        assert result.parameter_value == pytest.approx(30.0)
        assert result.energy == pytest.approx(-150.0)
        assert result.is_converged is True

    def test_run_single_ecutwfc_applied(self, tmp_path):
        """Fixed ecutwfc is written to config.system."""
        si = bulk("Si", "diamond", a=5.43)
        conv = KpointConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            resolutions=[30.0],
            ecutwfc=55.0,
        )

        from unittest.mock import patch

        mock_runner_result = self._make_mock_runner_result(success=True)
        mock_dft_result = self._make_mock_dft_result()
        calc_dir = str(tmp_path / "kpoint_30")

        captured_system = {}

        def fake_write(atoms, calc_dir, config=None):
            if config:
                captured_system.update(config.system)

        with patch("shalom.backends.qe.QEBackend.write_input", side_effect=fake_write), \
             patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_runner_result), \
             patch("shalom.backends.qe.QEBackend.parse_output", return_value=mock_dft_result):
            conv._run_single(30.0, calc_dir)

        assert captured_system.get("ecutwfc") == pytest.approx(55.0)

    def test_run_single_no_ecutwfc_uses_preset(self, tmp_path):
        """When ecutwfc=None, default preset value is used."""
        si = bulk("Si", "diamond", a=5.43)
        conv = KpointConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            resolutions=[30.0],
            ecutwfc=None,  # don't override
        )

        from unittest.mock import patch

        mock_runner_result = self._make_mock_runner_result(success=True)
        mock_dft_result = self._make_mock_dft_result()
        calc_dir = str(tmp_path / "kpoint_30")

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_runner_result), \
             patch("shalom.backends.qe.QEBackend.parse_output", return_value=mock_dft_result):
            result = conv._run_single(30.0, calc_dir)

        assert result.is_converged is True  # no crash

    def test_run_single_exception_caught(self, tmp_path):
        """_run_single catches exceptions and returns error result."""
        si = bulk("Si", "diamond", a=5.43)
        conv = KpointConvergence(
            atoms=si,
            output_dir=str(tmp_path),
            resolutions=[30.0],
            ecutwfc=50.0,
        )

        from unittest.mock import patch

        calc_dir = str(tmp_path / "kpoint_30")

        with patch("shalom.backends.qe.QEBackend.write_input",
                   side_effect=ValueError("bad atoms")):
            result = conv._run_single(30.0, calc_dir)

        assert result.is_converged is False
        assert result.energy is None
        assert "bad atoms" in result.error_message


# ---------------------------------------------------------------------------
# ConvergenceWorkflow base helpers: _make_calc_dir, _run_sequential, parallel
# ---------------------------------------------------------------------------


class TestConvergenceBaseHelpers:
    def test_make_calc_dir_creates_directory(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si, output_dir=str(tmp_path), values=[40.0], kgrid=[4, 4, 4]
        )
        calc_dir = conv._make_calc_dir(40.0)
        assert os.path.isdir(calc_dir)
        assert "ecutwfc" in calc_dir

    def test_make_calc_dir_tag_format(self, tmp_path):
        """Directory name replaces '.' with 'p'."""
        si = bulk("Si", "diamond", a=5.43)
        conv = KpointConvergence(
            atoms=si, output_dir=str(tmp_path), resolutions=[25.5], ecutwfc=50.0
        )
        calc_dir = conv._make_calc_dir(25.5)
        assert "25p5" in calc_dir

    def test_run_sequential_calls_all(self, tmp_path):
        """_run_sequential calls _run_single for each value."""
        si = bulk("Si", "diamond", a=5.43)
        called = []

        conv = CutoffConvergence(
            atoms=si, output_dir=str(tmp_path), values=[30.0, 40.0, 50.0], kgrid=[4, 4, 4]
        )

        def fake_run_single(val, calc_dir):
            called.append(val)
            return ConvergenceResult(val, -100.0 * val, True, calc_dir)

        conv._run_single = fake_run_single  # type: ignore[method-assign]
        results = conv._run_sequential()

        assert sorted(called) == [30.0, 40.0, 50.0]
        assert len(results) == 3

    def test_run_parallel_calls_all(self, tmp_path):
        """_run_parallel calls _run_single for each value concurrently."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si, output_dir=str(tmp_path), values=[30.0, 40.0], kgrid=[4, 4, 4]
        )

        # Override _run_single to be picklable (no mock) for multiprocessing
        # Instead, patch the pool to avoid spawning actual processes
        from unittest.mock import MagicMock, patch

        results_data = [
            ConvergenceResult(30.0, -100.0, True),
            ConvergenceResult(40.0, -100.01, True),
        ]

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.starmap = MagicMock(return_value=results_data)

        with patch("multiprocessing.Pool", return_value=mock_pool):
            results = conv._run_parallel()

        assert len(results) == 2
        mock_pool.starmap.assert_called_once()

    def test_run_uses_parallel_when_flag_set(self, tmp_path):
        """run() delegates to _run_parallel when parallel=True."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si, output_dir=str(tmp_path), values=[30.0], kgrid=[4, 4, 4],
            parallel=True,
        )

        from unittest.mock import patch

        results_data = [ConvergenceResult(30.0, -100.0, True)]

        with patch.object(conv, "_run_parallel", return_value=results_data) as mock_parallel, \
             patch.object(conv, "_run_sequential") as mock_seq:
            conv.run()

        mock_parallel.assert_called_once()
        mock_seq.assert_not_called()

    def test_convergence_plot_no_converged_results(self, tmp_path):
        """plot() returns None when all results failed."""
        si = bulk("Si", "diamond", a=5.43)
        conv = CutoffConvergence(
            atoms=si, output_dir=str(tmp_path), values=[30.0], kgrid=[4, 4, 4]
        )
        result = ConvergenceTestResult(
            parameter_name="ecutwfc",
            results=[ConvergenceResult(30.0, None, False)],
        )
        # Should not crash even with no converged results
        import pytest
        pytest.importorskip("matplotlib")
        plot_path = conv.plot(result, output_path=str(tmp_path / "conv.png"))
        assert plot_path is None
