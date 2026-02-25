"""Phonon dispersion and DOS plotter for SHALOM phonon results.

Renders phonon band structure and DOS from ``PhononResult`` as
publication-quality diagrams using matplotlib.

Usage::

    from shalom.analysis import analyze_phonon
    from shalom.plotting import PhononBandPlotter, PhononDOSPlotter

    result = analyze_phonon(atoms, supercell_matrix, force_sets)
    PhononBandPlotter(result).plot(output_path="phonon_bands.png")
    PhononDOSPlotter(result).plot(output_path="phonon_dos.png")

Requires the ``[plotting]`` optional dependency::

    pip install shalom[plotting]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.figure
    from shalom.analysis._base import PhononResult


class PhononBandPlotter:
    """Plot phonon dispersion (frequency vs q-path).

    Args:
        data: ``PhononResult`` with populated ``band_*`` fields.
    """

    def __init__(self, data: "PhononResult") -> None:
        self.data = data

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (6.0, 5.0),
        dpi: int = 150,
        color: str = "royalblue",
        imaginary_color: str = "crimson",
        lw: float = 1.2,
    ) -> "matplotlib.figure.Figure":
        """Generate and optionally save a phonon dispersion plot.

        Imaginary frequencies (below zero) are drawn in *imaginary_color*.
        A dashed grey line at 0 THz marks the acoustic baseline.
        High-symmetry q-point labels are drawn as vertical dotted lines.

        Args:
            output_path: Save path (PNG/PDF).
            title: Plot title.  Defaults to ``"Phonon Dispersion"``.
            freq_range: ``(freq_min, freq_max)`` in THz.  Auto if None.
            figsize: Figure size in inches.
            dpi: Resolution for rasterised output.
            color: Colour for positive-frequency branches.
            imaginary_color: Colour for negative (imaginary) frequencies.
            lw: Line width.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        _, plt = _require_matplotlib()
        import numpy as np

        data = self.data
        x = data.band_distances
        freqs = data.band_frequencies

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Draw each branch
        n_branches = freqs.shape[1]
        for b in range(n_branches):
            branch = freqs[:, b]
            pos_mask = branch >= 0
            neg_mask = ~pos_mask

            # Positive frequencies
            if np.any(pos_mask):
                branch_pos = np.where(pos_mask, branch, np.nan)
                ax.plot(x, branch_pos, color=color, lw=lw)

            # Imaginary frequencies
            if np.any(neg_mask):
                branch_neg = np.where(neg_mask, branch, np.nan)
                ax.plot(x, branch_neg, color=imaginary_color, lw=lw)

        # Zero line
        ax.axhline(0.0, color="grey", lw=0.8, linestyle="--", alpha=0.7)

        # High-symmetry labels
        if data.band_labels:
            tick_positions = []
            tick_labels = []
            for idx, label in sorted(data.band_labels.items()):
                if label and 0 <= idx < len(x):
                    tick_positions.append(x[idx])
                    tick_labels.append(_format_phonon_label(label))
                    ax.axvline(x[idx], color="grey", lw=0.6, linestyle=":", alpha=0.6)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=12)
        else:
            ax.set_xticks([])

        # Axis limits
        ax.set_xlim(x[0], x[-1])
        if freq_range is not None:
            ax.set_ylim(freq_range)

        ax.set_ylabel("Frequency (THz)", fontsize=12)
        ax.set_xlabel("q-path", fontsize=11)

        if title is None:
            title = "Phonon Dispersion"
        ax.set_title(title, fontsize=13, pad=8)

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

        return fig


class PhononDOSPlotter:
    """Plot phonon density of states.

    Args:
        data: ``PhononResult`` with populated ``dos_*`` fields.
    """

    def __init__(self, data: "PhononResult") -> None:
        self.data = data

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (6.0, 5.0),
        dpi: int = 150,
        color: str = "royalblue",
        fill_alpha: float = 0.25,
        lw: float = 1.5,
    ) -> "matplotlib.figure.Figure":
        """Generate and optionally save a phonon DOS plot.

        Args:
            output_path: Save path (PNG/PDF).
            title: Plot title.  Defaults to ``"Phonon DOS"``.
            freq_range: ``(freq_min, freq_max)`` in THz.  Auto if None.
            figsize: Figure size in inches.
            dpi: Resolution.
            color: Line and fill colour.
            fill_alpha: Opacity of the shaded region under the curve.
            lw: Line width.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        _, plt = _require_matplotlib()

        data = self.data
        freq = data.dos_frequencies
        dos = data.dos_density

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.plot(freq, dos, color=color, lw=lw)
        ax.fill_between(freq, dos, alpha=fill_alpha, color=color)

        # Zero frequency line
        ax.axvline(0.0, color="grey", lw=0.8, linestyle="--", alpha=0.7)

        if freq_range is not None:
            ax.set_xlim(freq_range)

        ax.set_xlabel("Frequency (THz)", fontsize=12)
        ax.set_ylabel("DOS (states/THz)", fontsize=12)

        if title is None:
            title = "Phonon DOS"
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


def _format_phonon_label(raw: str) -> str:
    """Convert phonon k-point labels to display format.

    ``"G"`` → ``"Γ"``, ``"|"`` separators preserved for discontinuities.
    """
    if "|" in raw:
        return "|".join(_format_phonon_label(p) for p in raw.split("|"))
    mapping = {
        "G": "Γ", "Gamma": "Γ", "GAMMA": "Γ",
    }
    return mapping.get(raw, raw)
