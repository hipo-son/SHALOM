"""Combined band structure + DOS plot for publication-quality figures.

Creates a side-by-side figure with band structure on the left (wide panel)
and density of states on the right (narrow panel, rotated 90 degrees),
sharing a common energy axis.  This is the standard figure format in
computational materials science publications.

Usage::

    from shalom.backends import parse_xml_bands, parse_dos_file, extract_fermi_energy
    from shalom.plotting.combined_plot import CombinedPlotter

    fermi = extract_fermi_energy("nscf/pw.out") or 0.0
    bs = parse_xml_bands("bands/data-file-schema.xml", fermi)
    dos = parse_dos_file("nscf/pwscf.dos")
    dos.fermi_energy = fermi

    plotter = CombinedPlotter(bs, dos)
    plotter.plot(output_path="combined.png")

Requires the ``[plotting]`` optional dependency::

    pip install shalom[plotting]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.figure
    from shalom.backends.base import BandStructureData, DOSData


class CombinedPlotter:
    """Side-by-side band structure (left) + DOS (right) plot.

    Args:
        band_data: ``BandStructureData`` instance.
        dos_data: ``DOSData`` instance.
    """

    def __init__(
        self,
        band_data: "BandStructureData",
        dos_data: "DOSData",
    ) -> None:
        self.band_data = band_data
        self.dos_data = dos_data

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        energy_window: Tuple[float, float] = (-6.0, 6.0),
        figsize: Tuple[float, float] = (10.0, 8.0),
        dpi: int = 150,
        zero_at_fermi: bool = True,
        color_up: str = "royalblue",
        color_down: str = "crimson",
        band_lw: float = 1.2,
        dos_lw: float = 1.5,
        dos_fill_alpha: float = 0.25,
        width_ratios: Tuple[float, float] = (3.0, 1.0),
    ) -> "matplotlib.figure.Figure":
        """Generate combined band + DOS figure.

        Args:
            output_path: Save path (PNG/PDF). If ``None``, only returns fig.
            title: Figure suptitle.
            energy_window: ``(E_min, E_max)`` in eV (relative to Fermi).
            figsize: ``(width, height)`` in inches.
            dpi: Resolution for raster output.
            zero_at_fermi: Shift eigenvalues/energies so Fermi = 0.
            color_up: Colour for spin-up (or non-spin) bands/DOS.
            color_down: Colour for spin-down bands/DOS.
            band_lw: Line width for band curves.
            dos_lw: Line width for DOS curves.
            dos_fill_alpha: Fill opacity for DOS area.
            width_ratios: ``(band_width, dos_width)`` panel ratio.

        Returns:
            ``matplotlib.figure.Figure`` object.
        """
        _mpl, plt = _require_matplotlib()
        import matplotlib.gridspec as gridspec
        import numpy as np
        from shalom.plotting.band_plot import _format_label

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(
            1, 2, width_ratios=list(width_ratios), wspace=0.05,
        )
        ax_band = fig.add_subplot(gs[0])
        ax_dos = fig.add_subplot(gs[1], sharey=ax_band)

        bd = self.band_data
        dd = self.dos_data
        shift_band = bd.fermi_energy if zero_at_fermi else 0.0
        shift_dos = dd.fermi_energy if zero_at_fermi else 0.0

        # ── Left panel: Band structure ──────────────────────────────────
        x = bd.kpath_distances
        evals = bd.eigenvalues - shift_band

        if (bd.is_spin_polarized
                and bd.spin_up is not None and bd.spin_down is not None):
            evals_up = bd.spin_up - shift_band
            evals_dn = bd.spin_down - shift_band
            for b in range(evals_up.shape[1]):
                ax_band.plot(x, evals_up[:, b], color=color_up, lw=band_lw,
                             label="up" if b == 0 else "")
            for b in range(evals_dn.shape[1]):
                ax_band.plot(x, evals_dn[:, b], color=color_down, lw=band_lw,
                             linestyle="--", label="down" if b == 0 else "")
        else:
            for b in range(evals.shape[1]):
                ax_band.plot(x, evals[:, b], color=color_up, lw=band_lw)

        # Fermi level
        ax_band.axhline(0.0 if zero_at_fermi else bd.fermi_energy,
                        color="grey", lw=0.8, ls="--", alpha=0.7)

        # High-symmetry labels
        if bd.high_sym_labels:
            ticks, labels = [], []
            for idx, lbl in sorted(bd.high_sym_labels.items()):
                if 0 <= idx < len(x):
                    ticks.append(x[idx])
                    labels.append(_format_label(lbl))
                    ax_band.axvline(x[idx], color="grey", lw=0.6,
                                    ls=":", alpha=0.6)
            ax_band.set_xticks(ticks)
            ax_band.set_xticklabels(labels, fontsize=12)
        else:
            ax_band.set_xticks([])

        ax_band.set_xlim(x[0], x[-1])
        ax_band.set_ylim(energy_window)
        ax_band.set_ylabel(
            "$E - E_F$ (eV)" if zero_at_fermi else "Energy (eV)",
            fontsize=12,
        )
        ax_band.set_xlabel("k-path", fontsize=11)

        # ── Right panel: DOS (rotated 90 degrees) ──────────────────────
        energies = dd.energies - shift_dos

        if (dd.is_spin_polarized
                and dd.dos_up is not None and dd.dos_down is not None):
            ax_dos.plot(dd.dos_up, energies, color=color_up, lw=dos_lw)
            ax_dos.fill_betweenx(energies, 0, dd.dos_up,
                                 color=color_up, alpha=dos_fill_alpha)
            ax_dos.plot(-dd.dos_down, energies, color=color_down, lw=dos_lw)
            ax_dos.fill_betweenx(energies, 0, -dd.dos_down,
                                 color=color_down, alpha=dos_fill_alpha)
        else:
            ax_dos.plot(dd.dos, energies, color=color_up, lw=dos_lw)
            ax_dos.fill_betweenx(energies, 0, dd.dos,
                                 color=color_up, alpha=dos_fill_alpha)

        # Fermi level on DOS panel
        ax_dos.axhline(0.0 if zero_at_fermi else dd.fermi_energy,
                       color="grey", lw=0.8, ls="--", alpha=0.7)

        ax_dos.set_xlabel("DOS (states/eV)", fontsize=11)
        plt.setp(ax_dos.get_yticklabels(), visible=False)

        # Auto-scale DOS x-axis to visible energy window
        mask = (energies >= energy_window[0]) & (energies <= energy_window[1])
        if np.any(mask):
            if (dd.is_spin_polarized
                    and dd.dos_up is not None and dd.dos_down is not None):
                peak = max(dd.dos_up[mask].max(), dd.dos_down[mask].max())
                ax_dos.set_xlim(-peak * 1.15, peak * 1.15)
            else:
                peak = dd.dos[mask].max()
                ax_dos.set_xlim(0, peak * 1.15)

        if title:
            fig.suptitle(title, fontsize=13, y=0.98)

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

        return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_matplotlib():
    """Lazy-import matplotlib; raise a friendly error if missing."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with:\n"
            "    pip install shalom[plotting]\n"
            "or: pip install matplotlib"
        ) from exc
