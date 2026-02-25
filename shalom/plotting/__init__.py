"""Band structure, DOS, and phonon plotting utilities.

Requires the ``[plotting]`` optional dependency group::

    pip install shalom[plotting]

Usage::

    from shalom.plotting import BandStructurePlotter, DOSPlotter
    from shalom.plotting import PhononBandPlotter, PhononDOSPlotter
"""

from shalom.plotting.band_plot import BandStructurePlotter
from shalom.plotting.dos_plot import DOSPlotter
from shalom.plotting.phonon_plot import PhononBandPlotter, PhononDOSPlotter

__all__ = [
    "BandStructurePlotter",
    "DOSPlotter",
    "PhononBandPlotter",
    "PhononDOSPlotter",
]
