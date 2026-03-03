"""Tests for MD trajectory analysis — RDF, MSD, VACF, diffusion, equilibration.

Tests verify:
- RDF computation for simple structures (peak at expected distance)
- MSD linear growth for constant-velocity particles
- VACF normalization and decay
- Diffusion coefficient from known MSD slope
- Equilibration detection for equilibrated and non-equilibrated trajectories
- Full analyze_md_trajectory pipeline
- Edge cases (empty trajectory, single frame, no velocities)
"""

import numpy as np
import pytest

from shalom.analysis._base import MDResult
from shalom.analysis.md import (
    analyze_md_trajectory,
    compute_diffusion_coefficient,
    compute_msd,
    compute_rdf,
    compute_vacf,
    detect_equilibration,
)
from shalom.backends.base import MDTrajectoryData


# Re-use shared trajectory factory from conftest.py
from tests.conftest import make_trajectory as _make_trajectory


# ---------------------------------------------------------------------------
# compute_rdf
# ---------------------------------------------------------------------------


class TestComputeRDF:
    """Test radial distribution function computation."""

    def test_rdf_shape(self):
        traj = _make_trajectory(n_frames=5, n_atoms=10)
        r, g = compute_rdf(traj, r_max=5.0, n_bins=50)
        assert r.shape == (50,)
        assert g.shape == (50,)

    def test_rdf_r_range(self):
        traj = _make_trajectory()
        r, g = compute_rdf(traj, r_max=5.0, n_bins=100)
        assert r[0] > 0
        assert r[-1] < 5.0

    def test_rdf_known_structure(self):
        """Two atoms at fixed distance → RDF peak at that distance."""
        n_frames = 10
        positions = np.zeros((n_frames, 2, 3))
        positions[:, 0, :] = [2.0, 2.0, 2.0]
        positions[:, 1, :] = [4.0, 2.0, 2.0]  # distance = 2.0 Å
        traj = _make_trajectory(positions=positions, box_size=10.0)
        r, g = compute_rdf(traj, r_max=5.0, n_bins=100)
        # Peak should be around r=2.0
        peak_idx = np.argmax(g)
        assert abs(r[peak_idx] - 2.0) < 0.2

    def test_rdf_pair_specific(self):
        """Pair-specific RDF for heterogeneous system."""
        n_frames = 5
        positions = np.zeros((n_frames, 4, 3))
        positions[:, 0, :] = [1.0, 1.0, 1.0]
        positions[:, 1, :] = [3.0, 1.0, 1.0]
        positions[:, 2, :] = [5.0, 5.0, 5.0]
        positions[:, 3, :] = [7.0, 5.0, 5.0]
        traj = MDTrajectoryData(
            positions=positions,
            energies=np.zeros(n_frames),
            temperatures=np.full(n_frames, 300.0),
            times=np.arange(n_frames, dtype=float),
            species=["Fe", "Fe", "O", "O"],
            cell_vectors=np.diag([10.0, 10.0, 10.0]),
            timestep_fs=1.0,
            source="test",
        )
        r, g = compute_rdf(traj, r_max=5.0, pair=("Fe", "Fe"))
        # Should have a peak (Fe-Fe at distance 2.0)
        assert np.max(g) > 0

    def test_rdf_empty_trajectory(self):
        positions = np.zeros((0, 2, 3))
        traj = _make_trajectory(positions=positions)
        r, g = compute_rdf(traj, r_max=5.0, n_bins=50)
        assert np.all(g == 0)

    def test_rdf_start_frame(self):
        traj = _make_trajectory(n_frames=10, n_atoms=4)
        r1, g1 = compute_rdf(traj, start_frame=0)
        r2, g2 = compute_rdf(traj, start_frame=5)
        # Different start frames should give different results
        # (unless identical, which is unlikely for random positions)
        assert r1.shape == r2.shape


# ---------------------------------------------------------------------------
# compute_msd
# ---------------------------------------------------------------------------


class TestComputeMSD:
    """Test mean square displacement computation."""

    def test_msd_shape(self):
        traj = _make_trajectory(n_frames=20, n_atoms=4)
        t, msd = compute_msd(traj)
        assert len(t) == 20
        assert len(msd) == 20

    def test_msd_starts_at_zero(self):
        traj = _make_trajectory(n_frames=10)
        t, msd = compute_msd(traj)
        assert abs(msd[0]) < 1e-10

    def test_msd_constant_velocity(self):
        """Atoms moving at constant velocity → MSD grows as t²."""
        n_frames = 50
        n_atoms = 1
        v = np.array([0.01, 0.0, 0.0])  # constant velocity
        positions = np.zeros((n_frames, n_atoms, 3))
        for t in range(n_frames):
            positions[t, 0, :] = v * t
        traj = _make_trajectory(positions=positions, box_size=100.0, timestep=1.0)
        t, msd = compute_msd(traj)
        # MSD should grow quadratically: |v*t|² = 0.0001 * t²
        expected_msd = 0.0001 * np.arange(n_frames) ** 2
        np.testing.assert_allclose(msd, expected_msd, atol=1e-10)

    def test_msd_species_filter(self):
        n_frames = 10
        positions = np.zeros((n_frames, 3, 3))
        # Fe atom moves, O atoms static
        for t in range(n_frames):
            positions[t, 0, :] = [t * 0.1, 0, 0]
        traj = MDTrajectoryData(
            positions=positions,
            energies=np.zeros(n_frames),
            temperatures=np.full(n_frames, 300.0),
            times=np.arange(n_frames, dtype=float),
            species=["Fe", "O", "O"],
            cell_vectors=np.diag([100.0, 100.0, 100.0]),
            timestep_fs=1.0,
            source="test",
        )
        t_fe, msd_fe = compute_msd(traj, species="Fe")
        t_o, msd_o = compute_msd(traj, species="O")
        # Fe should have growing MSD, O should be zero
        assert msd_fe[-1] > 0
        assert abs(msd_o[-1]) < 1e-10

    def test_msd_empty(self):
        positions = np.zeros((0, 2, 3))
        traj = _make_trajectory(positions=positions)
        t, msd = compute_msd(traj)
        assert len(t) == 0


# ---------------------------------------------------------------------------
# compute_vacf
# ---------------------------------------------------------------------------


class TestComputeVACF:
    """Test velocity autocorrelation function."""

    def test_vacf_normalized(self):
        traj = _make_trajectory(n_frames=50, n_atoms=4, with_velocities=True)
        t, vacf = compute_vacf(traj, max_lag=20)
        assert len(vacf) == 20
        assert abs(vacf[0] - 1.0) < 1e-10  # Normalized to 1 at t=0

    def test_vacf_no_velocities(self):
        traj = _make_trajectory(n_frames=10, with_velocities=False)
        t, vacf = compute_vacf(traj)
        assert len(t) == 0
        assert len(vacf) == 0

    def test_vacf_decays(self):
        """For random velocities, VACF should decay from 1."""
        traj = _make_trajectory(n_frames=100, n_atoms=10, with_velocities=True)
        t, vacf = compute_vacf(traj, max_lag=50)
        # First point is 1.0, later points should be smaller in magnitude
        assert abs(vacf[0]) >= abs(vacf[-1])

    def test_vacf_constant_velocity(self):
        """Constant velocity → VACF stays at 1.0."""
        n_frames = 20
        velocities = np.ones((n_frames, 2, 3)) * 0.01
        traj = _make_trajectory(
            n_frames=n_frames, n_atoms=2,
            with_velocities=True, velocities=velocities,
        )
        t, vacf = compute_vacf(traj, max_lag=10)
        np.testing.assert_allclose(vacf, 1.0, atol=1e-10)

    def test_vacf_max_lag_capped(self):
        traj = _make_trajectory(n_frames=10, with_velocities=True)
        t, vacf = compute_vacf(traj, max_lag=100)
        # Should be capped at n_frames
        assert len(vacf) == 10


# ---------------------------------------------------------------------------
# compute_diffusion_coefficient
# ---------------------------------------------------------------------------


class TestDiffusionCoefficient:
    """Test diffusion coefficient from MSD."""

    def test_known_slope(self):
        """MSD = 6D*t → D should match."""
        D_expected = 1e-5  # cm²/s
        # slope = 6D / (Å²/fs → cm²/s factor)
        # 1e-5 cm²/s = slope/6 * 0.1 → slope = 6e-4 Å²/fs
        slope = D_expected * 6.0 / 0.1  # = 6e-4
        t = np.arange(100, dtype=float)  # fs
        msd = slope * t  # Å²
        D = compute_diffusion_coefficient(t, msd, fit_start_frac=0.1, fit_end_frac=0.9)
        assert abs(D - D_expected) / D_expected < 0.05

    def test_zero_msd(self):
        t = np.arange(50, dtype=float)
        msd = np.zeros(50)
        D = compute_diffusion_coefficient(t, msd)
        assert abs(D) < 1e-15

    def test_too_few_points(self):
        D = compute_diffusion_coefficient(np.array([0, 1]), np.array([0, 0.1]))
        assert D == 0.0


# ---------------------------------------------------------------------------
# detect_equilibration
# ---------------------------------------------------------------------------


class TestDetectEquilibration:
    """Test equilibration detection."""

    def test_equilibrated_constant_energy(self):
        energies = np.full(500, -10.0) + np.random.RandomState(42).normal(0, 0.001, 500)
        frame, is_eq = detect_equilibration(energies, window=20)
        # Frame should be in the first half of the trajectory
        assert frame < 250

    def test_short_trajectory(self):
        energies = np.array([-10.0, -9.9, -9.8])
        frame, is_eq = detect_equilibration(energies, window=100)
        assert is_eq is True
        assert frame == 0

    def test_non_equilibrated_drift(self):
        # Linearly drifting energy
        energies = np.linspace(-15.0, -10.0, 1000)
        frame, is_eq = detect_equilibration(energies, window=50)
        # May or may not detect as equilibrated, but should return a frame
        assert isinstance(frame, int)


# ---------------------------------------------------------------------------
# analyze_md_trajectory (full pipeline)
# ---------------------------------------------------------------------------


class TestAnalyzeMDTrajectory:
    """Test full MD analysis pipeline."""

    def test_full_analysis_returns_md_result(self):
        traj = _make_trajectory(n_frames=50, n_atoms=8, box_size=10.0)
        result = analyze_md_trajectory(traj, r_max=5.0, n_rdf_bins=50)
        assert isinstance(result, MDResult)

    def test_full_analysis_has_rdf(self):
        traj = _make_trajectory(n_frames=20, n_atoms=4)
        result = analyze_md_trajectory(traj, r_max=5.0)
        assert result.rdf_r is not None
        assert result.rdf_g is not None
        assert result.rdf_pairs == "all"

    def test_full_analysis_has_msd(self):
        traj = _make_trajectory(n_frames=20, n_atoms=4)
        result = analyze_md_trajectory(traj)
        assert result.msd_t is not None
        assert result.msd is not None

    def test_full_analysis_has_thermodynamics(self):
        traj = _make_trajectory(n_frames=20, n_atoms=4)
        result = analyze_md_trajectory(traj)
        assert result.avg_temperature is not None
        assert result.temperature_std is not None
        assert result.avg_energy is not None

    def test_full_analysis_has_equilibration(self):
        traj = _make_trajectory(n_frames=50, n_atoms=4)
        result = analyze_md_trajectory(traj)
        assert result.equilibration_step is not None
        assert isinstance(result.is_equilibrated, bool)

    def test_full_analysis_no_vacf_without_velocities(self):
        traj = _make_trajectory(n_frames=20, with_velocities=False)
        result = analyze_md_trajectory(traj)
        assert result.vacf is None
        assert result.vacf_t is None

    def test_full_analysis_with_vacf(self):
        traj = _make_trajectory(n_frames=50, n_atoms=4, with_velocities=True)
        result = analyze_md_trajectory(traj, compute_vacf_flag=True, max_vacf_lag=20)
        assert result.vacf is not None
        assert result.vacf_t is not None
        assert abs(result.vacf[0] - 1.0) < 1e-10

    def test_full_analysis_diffusion(self):
        traj = _make_trajectory(n_frames=50, n_atoms=4)
        result = analyze_md_trajectory(traj)
        assert result.diffusion_coefficient is not None

    def test_full_analysis_energy_drift(self):
        traj = _make_trajectory(n_frames=50, n_atoms=4, timestep=1.0)
        result = analyze_md_trajectory(traj)
        assert result.energy_drift_per_atom is not None

    def test_import_from_analysis(self):
        """analyze_md_trajectory should be importable from shalom.analysis."""
        from shalom.analysis import analyze_md_trajectory as fn
        assert callable(fn)
