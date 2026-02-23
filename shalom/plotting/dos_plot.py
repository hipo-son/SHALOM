"""Density of states (DOS) plotter for SHALOM DFT results.

Renders total DOS from ``DOSData`` (from QE ``dos.x`` or a future VASP parser)
as a publication-quality DOS diagram using matplotlib.

Usage::

    from shalom.backends import parse_dos_file, extract_fermi_energy
    from shalom.plotting.dos_plot import DOSPlotter

    dos = parse_dos_file("nscf/pwscf.dos")
    dos.fermi_energy = extract_fermi_energy("nscf/pw.out") or 0.0
    plotter = DOSPlotter(dos)
    plotter.plot(output_path="dos.png")

Requires the ``[plotting]`` optional dependency::

    pip install shalom[plotting]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.figure
    from shalom.backends.base import DOSData


class DOSPlotter:
    """Plot density of states as a function of energy.

    Args:
        data: ``DOSData`` instance produced by a backend parser
            (e.g. ``parse_dos_file`` for QE or a future VASP equivalent).
            ``data.fermi_energy`` should be set before plotting; use
            ``extract_fermi_energy`` if it was not already set by the parser.

    Attributes:
        data: The DOS data being plotted.
    """

    def __init__(self, data: "DOSData") -> None:
        self.data = data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        energy_window: Optional[Tuple[float, float]] = (-6.0, 6.0),
        dos_max: Optional[float] = None,
        figsize: Tuple[float, float] = (6.0, 5.0),
        dpi: int = 150,
        zero_at_fermi: bool = True,
        fill_alpha: float = 0.25,
        color_up: str = "royalblue",
        color_down: str = "crimson",
        lw: float = 1.5,
        show_integrated: bool = False,
    ) -> "matplotlib.figure.Figure":
        """Generate and optionally save a DOS plot.

        The Fermi energy is shown as a dashed grey vertical line.

        For spin-polarised calculations, spin-up DOS is plotted on the positive
        y-axis and spin-down DOS is mirrored onto the negative y-axis, following
        the standard 'mirror DOS' convention.

        Args:
            output_path: If given, the figure is saved to this path (PNG, PDF,
                etc. — inferred from extension).
            title: Figure title.  Defaults to ``"Density of States"``.
            energy_window: ``(E_min, E_max)`` in eV relative to the Fermi
                energy (when ``zero_at_fermi=True``).  ``None`` disables
                clipping so the full energy range is shown.
            dos_max: Upper limit for the DOS axis.  Computed automatically if
                ``None``.
            figsize: ``(width, height)`` in inches.
            dpi: Resolution for rasterised output formats.
            zero_at_fermi: If ``True`` (default), the energy axis is shifted so
                the Fermi level is at zero.
            fill_alpha: Alpha (transparency) for the filled area under the DOS
                curve.
            color_up: Colour for spin-up (or non-spin) DOS.
            color_down: Colour for spin-down DOS (mirrored below zero).
            lw: Line width for DOS curves.
            show_integrated: If ``True``, overlay the integrated DOS on a
                secondary y-axis.

        Returns:
            The ``matplotlib.figure.Figure`` object.

        Raises:
            ImportError: If matplotlib is not installed
                (``pip install shalom[plotting]``).
        """
        _mpl, plt = _require_matplotlib()

        data = self.data
        shift = data.fermi_energy if zero_at_fermi else 0.0
        energies = data.energies - shift

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # ------------------------------------------------------------------
        # Draw DOS
        # ------------------------------------------------------------------
        if data.is_spin_polarized and data.dos_up is not None and data.dos_down is not None:
            # Mirror convention: spin-up positive, spin-down negative
            ax.plot(energies, data.dos_up, color=color_up, lw=lw, label="↑")
            ax.fill_between(energies, 0, data.dos_up, color=color_up, alpha=fill_alpha)
            ax.plot(energies, -data.dos_down, color=color_down, lw=lw, label="↓")
            ax.fill_between(energies, 0, -data.dos_down, color=color_down, alpha=fill_alpha)
            ax.axhline(0.0, color="black", lw=0.5)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
        else:
            ax.plot(energies, data.dos, color=color_up, lw=lw)
            ax.fill_between(energies, 0, data.dos, color=color_up, alpha=fill_alpha)

        # ------------------------------------------------------------------
        # Integrated DOS overlay (secondary axis)
        # ------------------------------------------------------------------
        if show_integrated and data.integrated_dos is not None:
            ax2 = ax.twinx()
            ax2.plot(energies, data.integrated_dos, color="grey",
                     lw=0.8, linestyle=":", alpha=0.7, label="IDOS")
            ax2.set_ylabel("Integrated DOS (electrons)", fontsize=10, color="grey")
            ax2.tick_params(axis="y", labelcolor="grey")

        # ------------------------------------------------------------------
        # Fermi level line
        # ------------------------------------------------------------------
        ax.axvline(0.0 if zero_at_fermi else data.fermi_energy,
                   color="grey", lw=0.8, linestyle="--", alpha=0.7)

        # ------------------------------------------------------------------
        # Axis limits / labels
        # ------------------------------------------------------------------
        if energy_window is not None:
            ax.set_xlim(energy_window)
            # Auto-scale DOS axis to the visible window
            mask = (energies >= energy_window[0]) & (energies <= energy_window[1])
            if mask.any():
                if data.is_spin_polarized and data.dos_up is not None and data.dos_down is not None:
                    peak = max(data.dos_up[mask].max(), data.dos_down[mask].max())
                else:
                    peak = data.dos[mask].max()
                _dos_max = dos_max if dos_max is not None else peak * 1.15
                if data.is_spin_polarized:
                    ax.set_ylim(-_dos_max, _dos_max)
                else:
                    ax.set_ylim(0, _dos_max)

        ax.set_xlabel("$E - E_F$ (eV)" if zero_at_fermi else "Energy (eV)", fontsize=12)
        ax.set_ylabel("DOS (states/eV)", fontsize=12)

        if title is None:
            title = "Density of States"
        ax.set_title(title, fontsize=13, pad=8)

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
