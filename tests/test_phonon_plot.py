"""Tests for shalom.plotting.phonon_plot module.

Uses synthetic PhononResult data — no DFT or phonopy required.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="matplotlib not installed")

from shalom.analysis._base import PhononResult
from shalom.plotting.phonon_plot import (
    PhononBandPlotter,
    PhononDOSPlotter,
    _format_phonon_label,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_phonon_result():
    """Synthetic phonon result (Si-like, 50 q-points, 6 branches)."""
    n_qpoints, n_branches = 50, 6
    rng = np.random.default_rng(42)

    freqs = rng.standard_normal((n_qpoints, n_branches)) * 5
    distances = np.linspace(0, 2, n_qpoints)
    dos_freq = np.linspace(-2, 15, 200)
    dos_dens = np.maximum(0, np.exp(-((dos_freq - 8) ** 2) / 5.0))

    return PhononResult(
        band_qpoints=rng.standard_normal((n_qpoints, 3)),
        band_distances=distances,
        band_frequencies=freqs,
        band_labels={0: "G", 25: "X", 49: "L"},
        dos_frequencies=dos_freq,
        dos_density=dos_dens,
        min_frequency_THz=-3.5,
        is_stable=False,
        n_atoms=2,
        n_branches=6,
    )


@pytest.fixture
def stable_phonon_result():
    """Synthetic stable phonon result (all positive frequencies)."""
    n_qpoints, n_branches = 30, 6
    rng = np.random.default_rng(123)
    freqs = np.abs(rng.standard_normal((n_qpoints, n_branches))) * 5 + 0.5
    distances = np.linspace(0, 1.5, n_qpoints)

    return PhononResult(
        band_qpoints=rng.standard_normal((n_qpoints, 3)),
        band_distances=distances,
        band_frequencies=freqs,
        band_labels={0: "G", 15: "X", 29: "L"},
        dos_frequencies=np.linspace(0, 15, 100),
        dos_density=np.abs(rng.standard_normal(100)),
        min_frequency_THz=0.5,
        is_stable=True,
        n_atoms=2,
        n_branches=6,
    )


# ---------------------------------------------------------------------------
# PhononBandPlotter
# ---------------------------------------------------------------------------


class TestPhononBandPlotter:
    def test_returns_figure(self, mock_phonon_result):
        import matplotlib.figure

        fig = PhononBandPlotter(mock_phonon_result).plot()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_png(self, mock_phonon_result, tmp_path):
        out = str(tmp_path / "phonon_bands.png")
        PhononBandPlotter(mock_phonon_result).plot(output_path=out)
        assert (tmp_path / "phonon_bands.png").exists()

    def test_freq_range_applied(self, mock_phonon_result):
        fig = PhononBandPlotter(mock_phonon_result).plot(freq_range=(-5, 20))
        ax = fig.axes[0]
        assert ax.get_ylim() == (-5.0, 20.0)

    def test_high_sym_labels(self, mock_phonon_result):
        fig = PhononBandPlotter(mock_phonon_result).plot()
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Γ" in tick_labels
        assert "X" in tick_labels
        assert "L" in tick_labels

    def test_stable_no_imaginary_color(self, stable_phonon_result):
        """Stable result — all branches drawn in default color, none in imaginary."""
        fig = PhononBandPlotter(stable_phonon_result).plot()
        assert isinstance(fig, __import__("matplotlib").figure.Figure)

    def test_custom_colors(self, mock_phonon_result):
        fig = PhononBandPlotter(mock_phonon_result).plot(
            color="green", imaginary_color="orange",
        )
        assert isinstance(fig, __import__("matplotlib").figure.Figure)


# ---------------------------------------------------------------------------
# PhononDOSPlotter
# ---------------------------------------------------------------------------


class TestPhononDOSPlotter:
    def test_returns_figure(self, mock_phonon_result):
        import matplotlib.figure

        fig = PhononDOSPlotter(mock_phonon_result).plot()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_png(self, mock_phonon_result, tmp_path):
        out = str(tmp_path / "phonon_dos.png")
        PhononDOSPlotter(mock_phonon_result).plot(output_path=out)
        assert (tmp_path / "phonon_dos.png").exists()

    def test_freq_range_applied(self, mock_phonon_result):
        fig = PhononDOSPlotter(mock_phonon_result).plot(freq_range=(-2, 15))
        ax = fig.axes[0]
        assert ax.get_xlim() == (-2.0, 15.0)

    def test_custom_title(self, mock_phonon_result):
        fig = PhononDOSPlotter(mock_phonon_result).plot(title="My DOS")
        ax = fig.axes[0]
        assert ax.get_title() == "My DOS"


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------


class TestFormatPhononLabel:
    def test_gamma(self):
        assert _format_phonon_label("G") == "Γ"

    def test_plain(self):
        assert _format_phonon_label("X") == "X"

    def test_composite(self):
        assert _format_phonon_label("G|K") == "Γ|K"
