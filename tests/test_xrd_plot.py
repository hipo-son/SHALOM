"""Tests for shalom.plotting.xrd_plot module.

Uses synthetic XRDResult data -- no DFT or pymatgen required for plotting.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="matplotlib not installed")

from shalom.analysis._base import XRDResult
from shalom.plotting.xrd_plot import XRDPlotter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_xrd_result():
    """Synthetic XRD result (Si-like, 5 peaks)."""
    return XRDResult(
        two_theta=np.array([28.4, 47.3, 56.1, 69.1, 76.4]),
        intensities=np.array([100.0, 55.0, 30.0, 6.0, 11.0]),
        hkl_indices=[(1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (3, 3, 1)],
        d_spacings=np.array([3.14, 1.92, 1.64, 1.36, 1.25]),
        wavelength="CuKa",
        wavelength_angstrom=1.5406,
        n_peaks=5,
    )


# ---------------------------------------------------------------------------
# XRDPlotter
# ---------------------------------------------------------------------------


class TestXRDPlotter:
    def test_returns_figure(self, mock_xrd_result):
        import matplotlib.figure

        fig = XRDPlotter(mock_xrd_result).plot()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_png(self, mock_xrd_result, tmp_path):
        out = str(tmp_path / "xrd_pattern.png")
        XRDPlotter(mock_xrd_result).plot(output_path=out)
        assert (tmp_path / "xrd_pattern.png").exists()

    def test_theta_range(self, mock_xrd_result):
        fig = XRDPlotter(mock_xrd_result).plot(theta_range=(20, 60))
        ax = fig.axes[0]
        assert ax.get_xlim() == (20.0, 60.0)

    def test_custom_title(self, mock_xrd_result):
        fig = XRDPlotter(mock_xrd_result).plot(title="My XRD Pattern")
        ax = fig.axes[0]
        assert ax.get_title() == "My XRD Pattern"

    def test_hkl_labels_present(self, mock_xrd_result):
        """Peaks above threshold should have hkl annotation text."""
        fig = XRDPlotter(mock_xrd_result).plot(label_threshold=5.0)
        ax = fig.axes[0]
        # Collect annotation texts
        texts = [child.get_text() for child in ax.get_children()
                 if hasattr(child, "get_text") and child.get_text()]
        # Filter for hkl-style labels "(h k l)"
        hkl_labels = [t for t in texts if t.startswith("(") and t.endswith(")")]
        # Intensities above 5.0: 100, 55, 30, 6, 11 -> all 5 peaks
        assert len(hkl_labels) >= 4  # at least 4 peaks above threshold

    def test_label_threshold(self, mock_xrd_result):
        """High threshold should suppress most labels."""
        fig = XRDPlotter(mock_xrd_result).plot(label_threshold=90.0)
        ax = fig.axes[0]
        texts = [child.get_text() for child in ax.get_children()
                 if hasattr(child, "get_text") and child.get_text()]
        hkl_labels = [t for t in texts if t.startswith("(") and t.endswith(")")]
        # Only 100.0 is >= 90.0, so at most 1 label
        assert len(hkl_labels) <= 1

    def test_custom_color(self, mock_xrd_result):
        """Custom color should not raise."""
        fig = XRDPlotter(mock_xrd_result).plot(color="crimson")
        assert isinstance(fig, __import__("matplotlib").figure.Figure)


# ---------------------------------------------------------------------------
# Package imports (Phase 2)
# ---------------------------------------------------------------------------


class TestPackageImports:
    @pytest.mark.skip(reason="Phase 2: XRDPlotter not yet exported from plotting __init__")
    def test_plotting_exports(self):
        from shalom.plotting import XRDPlotter as XP
        assert XP is not None
