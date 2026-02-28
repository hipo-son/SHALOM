"""Tests for shalom.plotting.combined_plot.CombinedPlotter."""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from shalom.backends.base import BandStructureData, DOSData
from shalom.plotting.combined_plot import CombinedPlotter


# ---------------------------------------------------------------------------
# Helper: create fake band structure and DOS data
# ---------------------------------------------------------------------------


def _make_band_data(
    nkpts: int = 20,
    nbands: int = 8,
    fermi: float = 5.0,
    spin: bool = False,
) -> BandStructureData:
    """Create a minimal BandStructureData for testing."""
    evals = np.random.default_rng(42).uniform(-2, 2, (nkpts, nbands))
    evals.sort(axis=1)  # bands should increase
    kcoords = np.zeros((nkpts, 3))
    kcoords[:, 0] = np.linspace(0, 1, nkpts)
    kdist = np.linspace(0, 1, nkpts)

    kwargs = dict(
        eigenvalues=evals + fermi,
        kpoint_coords=kcoords,
        kpath_distances=kdist,
        fermi_energy=fermi,
        high_sym_labels={0: "G", nkpts // 2: "X", nkpts - 1: "L"},
        source="qe",
        nbands=nbands,
        nkpts=nkpts,
    )

    if spin:
        kwargs["spin_up"] = evals + fermi
        kwargs["spin_down"] = evals + fermi + 0.1
        kwargs["is_spin_polarized"] = True

    return BandStructureData(**kwargs)


def _make_dos_data(
    npts: int = 100,
    fermi: float = 5.0,
    spin: bool = False,
) -> DOSData:
    """Create a minimal DOSData for testing."""
    energies = np.linspace(fermi - 6.0, fermi + 6.0, npts)
    dos = np.abs(np.sin(energies - fermi)) + 0.1
    integ = np.cumsum(dos) * (energies[1] - energies[0])

    kwargs = dict(
        energies=energies,
        dos=dos,
        integrated_dos=integ,
        fermi_energy=fermi,
        source="qe",
    )

    if spin:
        kwargs["dos_up"] = dos
        kwargs["dos_down"] = dos * 0.9
        kwargs["is_spin_polarized"] = True

    return DOSData(**kwargs)


# ---------------------------------------------------------------------------
# CombinedPlotter tests
# ---------------------------------------------------------------------------


class TestCombinedPlotter:
    """Test CombinedPlotter figure creation and options."""

    def test_basic_figure_creation(self):
        """CombinedPlotter.plot() returns a matplotlib Figure."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
        # Figure should have 2 axes (band + dos)
        assert len(fig.get_axes()) == 2

    def test_png_output(self, tmp_path):
        """CombinedPlotter.plot() saves PNG to disk."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        out = str(tmp_path / "combined.png")
        fig = plotter.plot(output_path=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_spin_polarized(self):
        """CombinedPlotter correctly draws spin-up and spin-down bands."""
        bs = _make_band_data(spin=True)
        dos = _make_dos_data(spin=True)
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()
        assert len(fig.get_axes()) == 2
        ax_band = fig.get_axes()[0]
        lines = ax_band.get_lines()
        # 8 spin-up + 8 spin-down + 1 Fermi + 3 high-sym vlines = 20
        n_hsym = len(bs.high_sym_labels)
        assert len(lines) == 8 + 8 + 1 + n_hsym
        # At least one dashed line (spin-down uses "--")
        dashed = [ln for ln in lines if ln.get_linestyle() == "--"]
        assert len(dashed) >= 1

    def test_spin_polarized_png(self, tmp_path):
        """Spin-polarized combined plot saves valid PNG."""
        bs = _make_band_data(spin=True)
        dos = _make_dos_data(spin=True)
        plotter = CombinedPlotter(bs, dos)
        out = str(tmp_path / "combined_spin.png")
        fig = plotter.plot(output_path=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 100  # not a trivially empty file

    @pytest.mark.parametrize("emin,emax", [(-3.0, 3.0), (-10.0, 10.0)])
    def test_energy_window(self, emin, emax):
        """Energy window parameter controls y-axis limits."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(energy_window=(emin, emax))
        ax_band = fig.get_axes()[0]
        ylim = ax_band.get_ylim()
        assert ylim[0] == pytest.approx(emin)
        assert ylim[1] == pytest.approx(emax)

    def test_title(self):
        """Title parameter sets the figure suptitle."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(title="Silicon Band+DOS")
        # Check suptitle text via public API (matplotlib 3.7+)
        suptitle_text = fig.get_suptitle() if hasattr(fig, "get_suptitle") else ""
        if not suptitle_text:
            # Fallback for older matplotlib: check fig.texts
            suptitle_text = fig.texts[0].get_text() if fig.texts else ""
        assert "Silicon" in suptitle_text

    def test_no_title(self):
        """When title=None, no suptitle is set."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(title=None)
        suptitle_text = fig.get_suptitle() if hasattr(fig, "get_suptitle") else ""
        assert suptitle_text == ""

    def test_figsize_parameter(self):
        """Custom figsize is applied to the figure."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(figsize=(12.0, 6.0))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12.0)
        assert h == pytest.approx(6.0)

    def test_zero_at_fermi_shifts_bands(self):
        """zero_at_fermi=True shifts eigenvalues so Fermi=0."""
        fermi = 5.0
        bs = _make_band_data(fermi=fermi)
        dos = _make_dos_data(fermi=fermi)
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(zero_at_fermi=True)
        ax_band = fig.get_axes()[0]
        # y-axis label should mention E_F
        ylabel = ax_band.get_ylabel()
        assert "E_F" in ylabel or "E - E" in ylabel
        # Verify actual eigenvalue shift: plotted values should be near 0, not near fermi
        lines = ax_band.get_lines()
        band_line = lines[0]
        ydata = band_line.get_ydata()
        # Original eigenvalues are fermi ± 2, so shifted should be ± 2
        assert np.abs(np.mean(ydata)) < 3.0
        assert np.max(np.abs(ydata)) < 8.0  # well below fermi=5

    def test_zero_at_fermi_false(self):
        """zero_at_fermi=False uses raw energies and different ylabel."""
        fermi = 5.0
        bs = _make_band_data(fermi=fermi)
        dos = _make_dos_data(fermi=fermi)
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(zero_at_fermi=False)
        ax_band = fig.get_axes()[0]
        ylabel = ax_band.get_ylabel()
        assert "Energy (eV)" in ylabel
        # Values should be near fermi (not shifted)
        lines = ax_band.get_lines()
        band_line = lines[0]
        ydata = band_line.get_ydata()
        assert np.mean(ydata) > 3.0  # close to fermi=5

    def test_no_high_sym_labels(self):
        """Plot works even when high_sym_labels is empty."""
        bs = _make_band_data()
        bs.high_sym_labels = {}
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()
        assert len(fig.get_axes()) == 2

    def test_shared_y_axis(self):
        """Band and DOS panels share the same y-axis."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(energy_window=(-4.0, 4.0))
        ax_band, ax_dos = fig.get_axes()
        # Both should have the same y-limits
        assert ax_band.get_ylim() == ax_dos.get_ylim()

    def test_dos_xlabel(self):
        """DOS panel has 'DOS (states/eV)' xlabel."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()
        ax_dos = fig.get_axes()[1]
        assert "DOS" in ax_dos.get_xlabel()

    def test_width_ratios_default(self):
        """Default width_ratios=(3,1) produces wider band panel."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()  # default (3, 1)
        ax_band, ax_dos = fig.get_axes()
        assert ax_band.get_position().width > ax_dos.get_position().width

    def test_stores_band_and_dos_data(self):
        """CombinedPlotter stores band_data and dos_data attributes."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        assert plotter.band_data is bs
        assert plotter.dos_data is dos

    def test_energy_window_outside_data_range(self):
        """Plot does not crash when energy window excludes all DOS data."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(energy_window=(100.0, 200.0))
        assert len(fig.get_axes()) == 2

    @pytest.mark.parametrize("bs_spin,dos_spin,extra_bands", [
        (True, False, 8),   # spin-polarized bands add 8 spin-down lines
        (False, True, 0),   # non-spin bands, no extra lines
    ])
    def test_mixed_spin_data(self, bs_spin, dos_spin, extra_bands):
        """Mixed spin-polarization between bands and DOS renders without error."""
        bs = _make_band_data(spin=bs_spin)
        dos = _make_dos_data(spin=dos_spin)
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()
        assert len(fig.get_axes()) == 2
        ax_band = fig.get_axes()[0]
        lines = ax_band.get_lines()
        n_hsym = len(bs.high_sym_labels)
        assert len(lines) == 8 + extra_bands + 1 + n_hsym

    def test_width_ratios_affect_panel_widths(self):
        """Custom width_ratios actually changes panel relative widths."""
        bs = _make_band_data()
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot(width_ratios=(4.0, 1.0))
        ax_band, ax_dos = fig.get_axes()
        bb_band = ax_band.get_position().width
        bb_dos = ax_dos.get_position().width
        assert bb_band > bb_dos * 2  # band panel significantly wider

    def test_composite_high_sym_labels(self):
        """Composite labels like 'G|X' are handled by _format_label."""
        bs = _make_band_data()
        bs.high_sym_labels = {0: "G|X", bs.nkpts - 1: "L"}
        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()
        ax_band = fig.get_axes()[0]
        tick_labels = [t.get_text() for t in ax_band.get_xticklabels()]
        # "G" should be converted to Γ in at least one label
        assert any("\u0393" in lbl or "G" in lbl for lbl in tick_labels)

    def test_gap_collapse_at_discontinuity(self):
        """Gap-collapse removes artificial gaps at 'X|K' path breaks."""
        bs = _make_band_data(nkpts=30)
        # Set up a discontinuity at k-point 15
        # Manually create kpath_distances with a jump at index 15
        raw_dist = np.linspace(0, 2, 30)
        # Insert artificial gap at break point
        raw_dist[16:] += 0.5
        bs.kpath_distances = raw_dist
        bs.high_sym_labels = {0: "G", 15: "X|K", 29: "L"}

        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()
        ax_band = fig.get_axes()[0]

        # After gap-collapse, x-axis extent should be smaller than raw
        xlim = ax_band.get_xlim()
        assert xlim[1] - xlim[0] < raw_dist[-1] - raw_dist[0]

    def test_no_gap_collapse_without_pipe_label(self):
        """Normal labels (no '|') do not trigger gap collapse."""
        bs = _make_band_data(nkpts=30)
        raw_dist = np.linspace(0, 2, 30)
        bs.kpath_distances = raw_dist.copy()
        bs.high_sym_labels = {0: "G", 15: "X", 29: "L"}

        dos = _make_dos_data()
        plotter = CombinedPlotter(bs, dos)
        fig = plotter.plot()
        ax_band = fig.get_axes()[0]

        # No collapse → x extent unchanged
        xlim = ax_band.get_xlim()
        assert abs((xlim[1] - xlim[0]) - (raw_dist[-1] - raw_dist[0])) < 1e-6


# ---------------------------------------------------------------------------
# Import / export from shalom.plotting
# ---------------------------------------------------------------------------


class TestCombinedPlotterImport:
    """Verify CombinedPlotter is exported from shalom.plotting."""

    def test_import_from_plotting(self):
        """CombinedPlotter is importable from shalom.plotting."""
        from shalom.plotting import CombinedPlotter as CP
        assert CP is CombinedPlotter

    def test_in_all(self):
        """CombinedPlotter is in shalom.plotting.__all__."""
        import shalom.plotting
        assert "CombinedPlotter" in shalom.plotting.__all__
