"""Tests for shalom.plotting.band_plot.BandStructurePlotter."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="matplotlib not installed; skip plot tests")


from shalom.plotting.band_plot import BandStructurePlotter, _format_label


# ---------------------------------------------------------------------------
# BandStructurePlotter
# ---------------------------------------------------------------------------


class TestBandStructurePlotter:
    def test_returns_figure(self, mock_band_data):
        import matplotlib.figure
        plotter = BandStructurePlotter(mock_band_data)
        fig = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, mock_band_data, tmp_path):
        out = str(tmp_path / "bands.png")
        plotter = BandStructurePlotter(mock_band_data)
        plotter.plot(output_path=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_energy_window_applied(self, mock_band_data):
        """y-axis limits should reflect the requested energy window."""
        import matplotlib.pyplot as plt
        plotter = BandStructurePlotter(mock_band_data)
        fig = plotter.plot(energy_window=(-3.0, 3.0))
        ax = fig.axes[0]
        ymin, ymax = ax.get_ylim()
        assert ymin == pytest.approx(-3.0)
        assert ymax == pytest.approx(3.0)
        plt.close(fig)

    def test_high_sym_labels_as_xticks(self, mock_band_data):
        """high_sym_labels (int-keyed) must produce x-tick positions."""
        import matplotlib.pyplot as plt
        plotter = BandStructurePlotter(mock_band_data)
        fig = plotter.plot()
        ax = fig.axes[0]
        ticks = ax.get_xticks()
        # 3 labels defined in fixture → 3 ticks
        assert len(ticks) == 3
        plt.close(fig)

    def test_spin_polarized_draws_two_colors(self):
        """Spin-polarised data should produce bands in two distinct colours."""
        import matplotlib.pyplot as plt
        from shalom.backends.base import BandStructureData

        nkpts, nbands = 10, 4
        rng = np.random.default_rng(0)
        evals = rng.standard_normal((nkpts, nbands))
        bs = BandStructureData(
            eigenvalues=evals,
            spin_up=evals,
            spin_down=evals + 0.1,
            kpoint_coords=np.zeros((nkpts, 3)),
            kpath_distances=np.linspace(0, 1, nkpts),
            fermi_energy=0.0,
            is_spin_polarized=True,
            nbands=nbands,
            nkpts=nkpts,
        )
        plotter = BandStructurePlotter(bs)
        fig = plotter.plot()
        ax = fig.axes[0]
        # Expect lines in both color_up (royalblue) and color_down (crimson)
        colors = {line.get_color() for line in ax.lines}
        assert len(colors) >= 2
        plt.close(fig)

    def test_title_set(self, mock_band_data):
        import matplotlib.pyplot as plt
        plotter = BandStructurePlotter(mock_band_data)
        fig = plotter.plot(title="Test Band Structure")
        assert fig.axes[0].get_title() == "Test Band Structure"
        plt.close(fig)


# ---------------------------------------------------------------------------
# _format_label helper
# ---------------------------------------------------------------------------


class TestFormatLabel:
    def test_gamma_mapping(self):
        assert _format_label("G") == "Γ"
        assert _format_label("Gamma") == "Γ"
        assert _format_label("GAMMA") == "Γ"

    def test_passthrough_unknown(self):
        assert _format_label("Z1") == "Z1"

    def test_common_labels(self):
        for raw in ("X", "M", "K", "L", "W"):
            assert _format_label(raw) == raw
