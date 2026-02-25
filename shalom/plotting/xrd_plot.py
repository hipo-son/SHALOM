"""X-ray diffraction pattern plotter for SHALOM XRD results.

Renders powder XRD patterns from ``XRDResult`` as publication-quality
stick diagrams with Miller index labels using matplotlib.

Usage::

    from shalom.analysis.xrd import calculate_xrd
    from shalom.plotting.xrd_plot import XRDPlotter

    result = calculate_xrd(atoms)
    XRDPlotter(result).plot(output_path="xrd_pattern.png")

Requires the ``[plotting]`` optional dependency::

    pip install shalom[plotting]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.figure
    from shalom.analysis.xrd import XRDResult


class XRDPlotter:
    """Plot powder X-ray diffraction pattern (intensity vs 2-theta).

    Args:
        data: ``XRDResult`` with populated ``two_theta``, ``intensities``,
            and ``hkl_indices`` fields.
    """

    def __init__(self, data: "XRDResult") -> None:
        self.data = data

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        theta_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (8.0, 5.0),
        dpi: int = 150,
        color: str = "royalblue",
        label_threshold: float = 5.0,
        lw: float = 1.5,
    ) -> "matplotlib.figure.Figure":
        """Generate and optionally save an XRD pattern plot.

        Peaks are rendered as vertical lines (stem plot) at each 2-theta
        position with height proportional to intensity.  Peaks with
        intensity above ``label_threshold`` are annotated with their
        Miller indices.

        Args:
            output_path: Save path (PNG/PDF).  If None, the figure is
                returned without saving.
            title: Plot title.  Defaults to
                ``"X-ray Diffraction ({wavelength})"``.
            theta_range: ``(theta_min, theta_max)`` in degrees for x-axis.
                If None, auto-scaled to data range.
            figsize: Figure size in inches ``(width, height)``.
            dpi: Resolution for rasterised output.
            color: Colour for diffraction peaks.
            label_threshold: Minimum intensity (0-100 scale) for a peak
                to receive an hkl annotation label.
            lw: Line width for peak stems.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        _, plt = _require_matplotlib()
        import numpy as np

        data = self.data
        two_theta = np.asarray(data.two_theta)
        intensities = np.asarray(data.intensities)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Draw peaks as vertical lines from baseline to intensity
        markerline, stemlines, baseline = ax.stem(
            two_theta,
            intensities,
            linefmt="-",
            markerfmt=" ",
            basefmt="-",
        )
        plt.setp(stemlines, color=color, linewidth=lw)
        plt.setp(baseline, color="grey", linewidth=0.5)

        # Annotate peaks above label_threshold with Miller indices
        hkl_list = data.hkl_indices if data.hkl_indices else []
        for i, intensity in enumerate(intensities):
            if intensity >= label_threshold and i < len(hkl_list):
                hkl = hkl_list[i]
                label_text = f"({hkl[0]} {hkl[1]} {hkl[2]})"
                ax.annotate(
                    label_text,
                    xy=(two_theta[i], intensity),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        # Axis labels and formatting
        ax.set_xlabel(r"2$\theta$ (degrees)", fontsize=12)
        ax.set_ylabel("Intensity (a.u.)", fontsize=12)

        if theta_range is not None:
            ax.set_xlim(theta_range)

        ax.set_ylim(bottom=0)

        if title is None:
            title = f"X-ray Diffraction ({data.wavelength})"
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
