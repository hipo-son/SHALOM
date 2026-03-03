"""Tests for MD plotting — energy, temperature, MSD, RDF plotters.

Tests verify:
- Each plotter creates a matplotlib figure
- Output file saving works
- Edge cases (missing data, no velocities)
- Import from shalom.plotting
"""

import os

import numpy as np
import pytest

from shalom.analysis._base import MDResult
from shalom.backends.base import MDTrajectoryData
from shalom.plotting.md_plot import (
    MDEnergyPlotter,
    MDTemperaturePlotter,
    MSDPlotter,
    RDFPlotter,
)

# Use non-interactive backend for tests
import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trajectory():
    """Create a small synthetic MDTrajectoryData."""
    n_frames, n_atoms = 20, 4
    rng = np.random.RandomState(42)
    return MDTrajectoryData(
        positions=rng.uniform(0, 10, (n_frames, n_atoms, 3)),
        energies=np.linspace(-10.0, -9.5, n_frames),
        temperatures=300.0 + rng.normal(0, 10, n_frames),
        times=np.arange(n_frames, dtype=float),
        species=["Fe"] * n_atoms,
        cell_vectors=np.diag([10.0, 10.0, 10.0]),
        kinetic_energies=np.linspace(0.03, 0.04, n_frames),
        potential_energies=np.linspace(-10.03, -9.54, n_frames),
        timestep_fs=1.0,
        source="test",
    )


@pytest.fixture
def sample_md_result():
    """Create a synthetic MDResult for plotting."""
    r = np.linspace(0.1, 10.0, 100)
    g = np.exp(-((r - 2.5) ** 2) / 0.5) + 1.0  # Peak at 2.5 Å
    msd_t = np.arange(50, dtype=float)
    msd = 0.01 * msd_t  # Linear MSD
    return MDResult(
        rdf_r=r,
        rdf_g=g,
        rdf_pairs="Fe-Fe",
        msd_t=msd_t,
        msd=msd,
        diffusion_coefficient=1.67e-5,
        avg_temperature=300.0,
        temperature_std=8.5,
        avg_energy=-9.75,
        equilibration_step=5,
        is_equilibrated=True,
    )


# ---------------------------------------------------------------------------
# MDEnergyPlotter
# ---------------------------------------------------------------------------


class TestMDEnergyPlotter:
    """Test energy vs time plotter."""

    def test_creates_figure(self, sample_trajectory):
        plotter = MDEnergyPlotter(sample_trajectory)
        fig = plotter.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_to_file(self, sample_trajectory, tmp_path):
        out = str(tmp_path / "energy.png")
        plotter = MDEnergyPlotter(sample_trajectory)
        plotter.plot(output_path=out)
        assert os.path.exists(out)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_custom_title(self, sample_trajectory):
        plotter = MDEnergyPlotter(sample_trajectory)
        fig = plotter.plot(title="Fe AIMD Energy")
        ax = fig.axes[0]
        assert ax.get_title() == "Fe AIMD Energy"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_shows_components(self, sample_trajectory):
        plotter = MDEnergyPlotter(sample_trajectory)
        fig = plotter.plot(show_components=True)
        # Should have 3 lines: total, kinetic, potential
        ax = fig.axes[0]
        assert len(ax.lines) == 3
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_components(self, sample_trajectory):
        plotter = MDEnergyPlotter(sample_trajectory)
        fig = plotter.plot(show_components=False)
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# MDTemperaturePlotter
# ---------------------------------------------------------------------------


class TestMDTemperaturePlotter:
    """Test temperature vs time plotter."""

    def test_creates_figure(self, sample_trajectory):
        plotter = MDTemperaturePlotter(sample_trajectory)
        fig = plotter.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_to_file(self, sample_trajectory, tmp_path):
        out = str(tmp_path / "temp.png")
        plotter = MDTemperaturePlotter(sample_trajectory)
        plotter.plot(output_path=out)
        assert os.path.exists(out)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_target_temperature(self, sample_trajectory):
        plotter = MDTemperaturePlotter(sample_trajectory)
        fig = plotter.plot(target_temperature=300.0)
        ax = fig.axes[0]
        # Should have horizontal target line
        assert len(ax.lines) >= 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_running_average(self, sample_trajectory):
        plotter = MDTemperaturePlotter(sample_trajectory)
        fig = plotter.plot(running_avg_window=5)
        ax = fig.axes[0]
        # Should have raw + running average lines
        assert len(ax.lines) >= 2
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# MSDPlotter
# ---------------------------------------------------------------------------


class TestMSDPlotter:
    """Test MSD vs time plotter."""

    def test_creates_figure(self, sample_md_result):
        plotter = MSDPlotter(sample_md_result)
        fig = plotter.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_to_file(self, sample_md_result, tmp_path):
        out = str(tmp_path / "msd.png")
        plotter = MSDPlotter(sample_md_result)
        plotter.plot(output_path=out)
        assert os.path.exists(out)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_show_fit(self, sample_md_result):
        plotter = MSDPlotter(sample_md_result)
        fig = plotter.plot(show_fit=True)
        ax = fig.axes[0]
        # MSD line + fit line
        assert len(ax.lines) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_msd_data(self):
        result = MDResult()
        plotter = MSDPlotter(result)
        fig = plotter.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# RDFPlotter
# ---------------------------------------------------------------------------


class TestRDFPlotter:
    """Test RDF g(r) plotter."""

    def test_creates_figure(self, sample_md_result):
        plotter = RDFPlotter(sample_md_result)
        fig = plotter.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_to_file(self, sample_md_result, tmp_path):
        out = str(tmp_path / "rdf.png")
        plotter = RDFPlotter(sample_md_result)
        plotter.plot(output_path=out)
        assert os.path.exists(out)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_has_reference_line(self, sample_md_result):
        plotter = RDFPlotter(sample_md_result)
        fig = plotter.plot()
        ax = fig.axes[0]
        # Should have g(r) line + g=1 reference line
        assert len(ax.lines) >= 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_rdf_data(self):
        result = MDResult()
        plotter = RDFPlotter(result)
        fig = plotter.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Import integration
# ---------------------------------------------------------------------------


class TestPlottingImports:
    """Test plotters are importable from shalom.plotting."""

    def test_import_md_energy_plotter(self):
        from shalom.plotting import MDEnergyPlotter
        assert MDEnergyPlotter is not None

    def test_import_md_temperature_plotter(self):
        from shalom.plotting import MDTemperaturePlotter
        assert MDTemperaturePlotter is not None

    def test_import_msd_plotter(self):
        from shalom.plotting import MSDPlotter
        assert MSDPlotter is not None

    def test_import_rdf_plotter(self):
        from shalom.plotting import RDFPlotter
        assert RDFPlotter is not None
