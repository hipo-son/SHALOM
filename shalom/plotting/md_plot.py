"""MD trajectory plotters — energy, temperature, MSD, RDF.

Publication-quality plots for MD trajectory analysis results using matplotlib.
Works with ``MDTrajectoryData`` (raw trajectory) and ``MDResult`` (analysis).

Requires the ``[plotting]`` optional dependency::

    pip install shalom[plotting]

Usage::

    from shalom.plotting.md_plot import MDEnergyPlotter, RDFPlotter
    from shalom.analysis.md import analyze_md_trajectory

    result = analyze_md_trajectory(trajectory)
    RDFPlotter(result).plot(output_path="rdf.png")
    MDEnergyPlotter(trajectory).plot(output_path="energy.png")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.figure


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _MDPlotterBase:
    """Shared base for MD analysis plotters."""

    def __init__(self, data) -> None:
        self.data = data

    @staticmethod
    def _new_figure(
        figsize: Tuple[float, float] = (8.0, 5.0), dpi: int = 150,
    ):
        """Create a new figure and axes."""
        import matplotlib.pyplot as plt
        return plt.subplots(figsize=figsize, dpi=dpi)

    @staticmethod
    def _finalize(fig, ax, output_path: Optional[str] = None, dpi: int = 150):
        """Apply common formatting, legend, and save."""
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

class MDEnergyPlotter(_MDPlotterBase):
    """Plot energy vs time from an MD trajectory.

    Args:
        data: ``MDTrajectoryData`` with energies and times.
    """

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (8.0, 5.0),
        dpi: int = 150,
        show_components: bool = True,
        color_total: str = "black",
        color_kinetic: str = "crimson",
        color_potential: str = "royalblue",
        lw: float = 1.2,
    ) -> "matplotlib.figure.Figure":
        """Generate energy vs time plot.

        Args:
            output_path: Save path (PNG, PDF). None for display only.
            title: Figure title. Defaults to "MD Energy".
            figsize: Figure size in inches.
            dpi: Resolution.
            show_components: Show kinetic/potential if available.
            color_total: Color for total energy line.
            color_kinetic: Color for kinetic energy line.
            color_potential: Color for potential energy line.
            lw: Line width.

        Returns:
            matplotlib Figure.
        """
        import numpy as np

        fig, ax = self._new_figure(figsize, dpi)

        times = self.data.times
        if times is None:
            times = np.arange(self.data.n_frames) * self.data.timestep_fs

        ax.plot(times, self.data.energies, color=color_total, lw=lw, label="Total")

        if show_components:
            if self.data.kinetic_energies is not None:
                ax.plot(
                    times, self.data.kinetic_energies,
                    color=color_kinetic, lw=lw * 0.8, ls="--", label="Kinetic",
                )
            if self.data.potential_energies is not None:
                ax.plot(
                    times, self.data.potential_energies,
                    color=color_potential, lw=lw * 0.8, ls="--", label="Potential",
                )

        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(title or "MD Energy")
        return self._finalize(fig, ax, output_path, dpi)


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

class MDTemperaturePlotter(_MDPlotterBase):
    """Plot temperature vs time with running average.

    Args:
        data: ``MDTrajectoryData`` with temperatures and times.
    """

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        target_temperature: Optional[float] = None,
        running_avg_window: int = 50,
        figsize: Tuple[float, float] = (8.0, 5.0),
        dpi: int = 150,
        color: str = "royalblue",
        color_avg: str = "crimson",
        color_target: str = "grey",
        lw: float = 0.8,
        alpha: float = 0.5,
    ) -> "matplotlib.figure.Figure":
        """Generate temperature vs time plot.

        Args:
            output_path: Save path (PNG, PDF). None for display only.
            title: Figure title. Defaults to "MD Temperature".
            target_temperature: Show target T as horizontal dashed line.
            running_avg_window: Window size for running average.
            figsize: Figure size in inches.
            dpi: Resolution.
            color: Color for raw temperature.
            color_avg: Color for running average.
            color_target: Color for target temperature line.
            lw: Line width.
            alpha: Alpha for raw temperature line.

        Returns:
            matplotlib Figure.
        """
        import numpy as np

        fig, ax = self._new_figure(figsize, dpi)

        times = self.data.times
        if times is None:
            times = np.arange(self.data.n_frames) * self.data.timestep_fs
        temps = self.data.temperatures

        ax.plot(times, temps, color=color, lw=lw, alpha=alpha, label="T(t)")

        if len(temps) >= running_avg_window:
            kernel = np.ones(running_avg_window) / running_avg_window
            avg = np.convolve(temps, kernel, mode="valid")
            t_avg = times[running_avg_window - 1:]
            ax.plot(t_avg, avg, color=color_avg, lw=lw * 2, label=f"Avg ({running_avg_window})")

        if target_temperature is not None:
            ax.axhline(target_temperature, color=color_target, ls="--", lw=1.0, label="Target")

        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(title or "MD Temperature")
        return self._finalize(fig, ax, output_path, dpi)


# ---------------------------------------------------------------------------
# MSD
# ---------------------------------------------------------------------------

class MSDPlotter(_MDPlotterBase):
    """Plot MSD vs time from MD analysis result.

    Args:
        data: ``MDResult`` with msd_t and msd arrays.
    """

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        show_fit: bool = True,
        figsize: Tuple[float, float] = (7.0, 5.0),
        dpi: int = 150,
        color: str = "royalblue",
        color_fit: str = "crimson",
        lw: float = 1.5,
    ) -> "matplotlib.figure.Figure":
        """Generate MSD vs time plot.

        Args:
            output_path: Save path (PNG, PDF). None for display only.
            title: Figure title. Defaults to "Mean Square Displacement".
            show_fit: Show linear fit line and diffusion coefficient.
            figsize: Figure size in inches.
            dpi: Resolution.
            color: Color for MSD curve.
            color_fit: Color for fit line.
            lw: Line width.

        Returns:
            matplotlib Figure.
        """
        fig, ax = self._new_figure(figsize, dpi)

        msd_t = self.data.msd_t
        msd = self.data.msd
        if msd_t is None or msd is None:
            ax.text(0.5, 0.5, "No MSD data", transform=ax.transAxes, ha="center")
            return fig

        ax.plot(msd_t, msd, color=color, lw=lw, label="MSD")

        if show_fit and self.data.diffusion_coefficient is not None:
            D = self.data.diffusion_coefficient
            slope = 6.0 * D * 10.0  # cm²/s → Å²/fs
            fit_line = slope * msd_t
            ax.plot(msd_t, fit_line, color=color_fit, ls="--", lw=1.0, label=f"D={D:.2e} cm²/s")

        ax.set_xlabel("Time (fs)")
        ax.set_ylabel(r"MSD ($\AA^2$)")
        ax.set_title(title or "Mean Square Displacement")
        return self._finalize(fig, ax, output_path, dpi)


# ---------------------------------------------------------------------------
# RDF
# ---------------------------------------------------------------------------

class RDFPlotter(_MDPlotterBase):
    """Plot radial distribution function g(r) from MD analysis result.

    Args:
        data: ``MDResult`` with rdf_r and rdf_g arrays.
    """

    def plot(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 5.0),
        dpi: int = 150,
        color: str = "royalblue",
        lw: float = 1.5,
        fill_alpha: float = 0.2,
    ) -> "matplotlib.figure.Figure":
        """Generate RDF g(r) plot.

        Args:
            output_path: Save path (PNG, PDF). None for display only.
            title: Figure title. Defaults to "Radial Distribution Function".
            figsize: Figure size in inches.
            dpi: Resolution.
            color: Color for g(r) curve.
            lw: Line width.
            fill_alpha: Alpha for fill under curve.

        Returns:
            matplotlib Figure.
        """
        fig, ax = self._new_figure(figsize, dpi)

        r = self.data.rdf_r
        g = self.data.rdf_g
        if r is None or g is None:
            ax.text(0.5, 0.5, "No RDF data", transform=ax.transAxes, ha="center")
            return fig

        pair_label = self.data.rdf_pairs or "all"
        ax.plot(r, g, color=color, lw=lw, label=f"g(r) [{pair_label}]")
        ax.fill_between(r, 0, g, color=color, alpha=fill_alpha)
        ax.axhline(1.0, color="grey", ls="--", lw=0.8)

        ax.set_xlabel(r"r ($\AA$)")
        ax.set_ylabel("g(r)")
        ax.set_title(title or "Radial Distribution Function")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        return self._finalize(fig, ax, output_path, dpi)
