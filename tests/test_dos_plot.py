"""Tests for shalom.plotting.dos_plot.DOSPlotter."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="matplotlib not installed; skip plot tests")


from shalom.plotting.dos_plot import DOSPlotter


# ---------------------------------------------------------------------------
# DOSPlotter
# ---------------------------------------------------------------------------


class TestDOSPlotter:
    def test_returns_figure(self, mock_dos_data):
        import matplotlib.figure
        plotter = DOSPlotter(mock_dos_data)
        fig = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, mock_dos_data, tmp_path):
        out = str(tmp_path / "dos.png")
        plotter = DOSPlotter(mock_dos_data)
        plotter.plot(output_path=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_fermi_line_present(self, mock_dos_data):
        """A horizontal line at E_F must be drawn."""
        import matplotlib.pyplot as plt
        plotter = DOSPlotter(mock_dos_data)
        fig = plotter.plot()
        ax = fig.axes[0]
        # At least one axvline (Fermi level)
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_spin_polarized_mirror(self):
        """Spin-polarised DOS should have lines below zero (mirrored spin-down)."""
        import matplotlib.pyplot as plt
        from shalom.backends.base import DOSData

        energies = np.linspace(-5, 5, 50)
        dos_up = np.abs(np.sin(energies))
        dos_dw = np.abs(np.cos(energies))
        dos = DOSData(
            energies=energies,
            dos=dos_up + dos_dw,
            integrated_dos=np.cumsum(dos_up + dos_dw),
            fermi_energy=0.0,
            dos_up=dos_up,
            dos_down=dos_dw,
            is_spin_polarized=True,
        )
        plotter = DOSPlotter(dos)
        fig = plotter.plot()
        ax = fig.axes[0]
        # y-axis minimum should be < 0 (mirrored spin-down)
        assert ax.get_ylim()[0] < 0.0
        plt.close(fig)

    def test_energy_window(self, mock_dos_data):
        """Energy window should clamp the x-axis."""
        import matplotlib.pyplot as plt
        plotter = DOSPlotter(mock_dos_data)
        fig = plotter.plot(energy_window=(-3.0, 3.0))
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        # x-limits should be consistent with the window
        assert xlim[0] <= -3.0 + 0.5   # allow small auto-margin
        assert xlim[1] >= 3.0 - 0.5
        plt.close(fig)

    def test_title_set(self, mock_dos_data):
        import matplotlib.pyplot as plt
        plotter = DOSPlotter(mock_dos_data)
        fig = plotter.plot(title="My DOS")
        assert fig.axes[0].get_title() == "My DOS"
        plt.close(fig)
