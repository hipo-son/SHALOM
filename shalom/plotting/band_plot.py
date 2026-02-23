"""Band structure plotter for SHALOM DFT results.

Renders eigenvalues from ``BandStructureData`` (from QE or VASP parsers) as a
publication-quality band structure diagram using matplotlib.

Usage::

    from shalom.backends import parse_xml_bands, extract_fermi_energy
    from shalom.plotting.band_plot import BandStructurePlotter

    fermi = extract_fermi_energy("nscf/pw.out") or 0.0
    bs = parse_xml_bands("bands/tmp/shalom.save/data-file-schema.xml", fermi)
    plotter = BandStructurePlotter(bs)
    plotter.plot(output_path="bands.png")

Requires the ``[plotting]`` optional dependency::

    pip install shalom[plotting]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.figure
    from shalom.backends.base import BandStructureData


class BandStructurePlotter:
    """Plot band structure eigenvalues relative to the Fermi energy.

    Args:
        data: ``BandStructureData`` instance produced by a backend parser
            (e.g. ``parse_xml_bands`` for QE or a future VASP equivalent).

    Attributes:
        data: The band structure data being plotted.
    """

    def __init__(self, data: "BandStructureData") -> None:
        self.data = data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        energy_window: Tuple[float, float] = (-6.0, 6.0),
        figsize: Tuple[float, float] = (6.0, 8.0),
        dpi: int = 150,
        zero_at_fermi: bool = True,
        color_up: str = "royalblue",
        color_down: str = "crimson",
        lw: float = 1.2,
    ) -> "matplotlib.figure.Figure":
        """Generate and optionally save a band structure plot.

        The Fermi energy is shown as a dashed grey horizontal line.
        High-symmetry k-point labels are drawn as vertical grey dotted lines
        if ``data.high_sym_labels`` is populated.

        For spin-polarised calculations, spin-up bands are drawn with
        ``color_up`` (solid) and spin-down with ``color_down`` (dashed).

        Args:
            output_path: If given, the figure is saved to this path (PNG, PDF,
                etc. — inferred from extension). The figure object is also
                returned for further manipulation.
            title: Figure title.  Defaults to ``"Band Structure"`` (or
                ``"Band Structure (spin polarised)"`` for LSDA).
            energy_window: ``(E_min, E_max)`` in eV relative to the Fermi
                energy (when ``zero_at_fermi=True``).
            figsize: ``(width, height)`` in inches.
            dpi: Resolution for rasterised output formats.
            zero_at_fermi: If ``True`` (default), eigenvalues are shifted by
                ``-data.fermi_energy`` so the Fermi level is at zero.
            color_up: Matplotlib colour string for spin-up (or non-spin) bands.
            color_down: Matplotlib colour string for spin-down bands.
            lw: Line width for band curves.

        Returns:
            The ``matplotlib.figure.Figure`` object.

        Raises:
            ImportError: If matplotlib is not installed
                (``pip install shalom[plotting]``).
        """
        mpl, plt = _require_matplotlib()

        data = self.data
        shift = data.fermi_energy if zero_at_fermi else 0.0
        x = data.kpath_distances  # shape (nkpts,)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # ------------------------------------------------------------------
        # Draw bands
        # ------------------------------------------------------------------
        evals = data.eigenvalues - shift  # (nkpts, nbands)

        if data.is_spin_polarized and data.spin_up is not None and data.spin_down is not None:
            evals_up = data.spin_up - shift
            evals_dn = data.spin_down - shift
            for b in range(evals_up.shape[1]):
                ax.plot(x, evals_up[:, b], color=color_up, lw=lw, label="↑" if b == 0 else "")
            for b in range(evals_dn.shape[1]):
                ax.plot(x, evals_dn[:, b], color=color_down, lw=lw,
                        linestyle="--", label="↓" if b == 0 else "")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
        else:
            for b in range(evals.shape[1]):
                ax.plot(x, evals[:, b], color=color_up, lw=lw)

        # ------------------------------------------------------------------
        # Fermi level line
        # ------------------------------------------------------------------
        ax.axhline(0.0 if zero_at_fermi else data.fermi_energy,
                   color="grey", lw=0.8, linestyle="--", alpha=0.7)

        # ------------------------------------------------------------------
        # High-symmetry k-point labels & vertical lines
        # ------------------------------------------------------------------
        if data.high_sym_labels:
            tick_positions = []
            tick_labels: list[str] = []
            for idx, label in sorted(data.high_sym_labels.items()):
                if 0 <= idx < len(x):
                    tick_positions.append(x[idx])
                    tick_labels.append(_format_label(label))
                    ax.axvline(x[idx], color="grey", lw=0.6, linestyle=":", alpha=0.6)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=12)
        else:
            ax.set_xticks([])

        # ------------------------------------------------------------------
        # Axis limits / labels
        # ------------------------------------------------------------------
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(energy_window)
        ax.set_ylabel("$E - E_F$ (eV)" if zero_at_fermi else "Energy (eV)", fontsize=12)
        ax.set_xlabel("k-path", fontsize=11)

        if title is None:
            title = "Band Structure (spin polarised)" if data.is_spin_polarized else "Band Structure"
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


def _format_label(raw: str) -> str:
    """Convert common k-point label strings to Unicode/LaTeX-friendly form.

    Composite labels (e.g. ``"U|K"`` at band-path discontinuities) are split
    on ``"|"`` and each part is formatted independently, then re-joined.

    Examples: ``"G"`` → ``"Γ"``, ``"Gamma"`` → ``"Γ"``, ``"G|K"`` → ``"Γ|K"``.
    """
    if "|" in raw:
        return "|".join(_format_label(p) for p in raw.split("|"))
    mapping = {
        "G": "Γ", "Gamma": "Γ", "GAMMA": "Γ",
        "X": "X", "M": "M", "K": "K", "A": "A", "L": "L", "H": "H",
        "W": "W", "U": "U", "R": "R", "Z": "Z", "S": "S", "T": "T",
        "Y": "Y", "N": "N", "P": "P",
    }
    return mapping.get(raw, raw)
