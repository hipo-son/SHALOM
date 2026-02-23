"""Tests for shalom.workflows.convergence and shalom.workflows.base."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock, patch

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
        result = conv.run()
        assert len(call_count) == 2


# ---------------------------------------------------------------------------
# Convergence plot (matplotlib optional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="skip") is None,
    reason="matplotlib not installed",
)
def test_convergence_plot_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
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
