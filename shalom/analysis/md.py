"""MD trajectory analysis — RDF, MSD, VACF, diffusion, equilibration.

Pure numpy implementation (no external MD analysis libraries required).
Works with any ``MDTrajectoryData`` regardless of source (VASP, LAMMPS, QE).

Usage::

    from shalom.analysis.md import analyze_md_trajectory
    from shalom.backends.base import MDTrajectoryData

    traj = backend.parse_trajectory(calc_dir)
    result = analyze_md_trajectory(traj, r_max=10.0)
    print(f"Diffusion: {result.diffusion_coefficient:.2e} cm²/s")
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from shalom.analysis._base import MDResult
from shalom.backends.base import MDTrajectoryData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Radial Distribution Function (RDF)
# ---------------------------------------------------------------------------


def compute_rdf(
    trajectory: MDTrajectoryData,
    r_max: float = 10.0,
    n_bins: int = 200,
    pair: Optional[Tuple[str, str]] = None,
    start_frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the radial distribution function g(r).

    Uses minimum image convention for orthorhombic cells.

    Args:
        trajectory: MD trajectory data.
        r_max: Maximum distance in Angstrom.
        n_bins: Number of histogram bins.
        pair: Element pair to compute (e.g. ``("Fe", "O")``).
            None computes total RDF for all pairs.
        start_frame: Skip frames before this index (equilibration).

    Returns:
        (r, g_r) — radial distance grid and pair correlation function.
    """
    positions = trajectory.positions[start_frame:]
    n_frames, n_atoms, _ = positions.shape

    if n_frames == 0 or n_atoms < 2:
        r = np.linspace(0, r_max, n_bins)
        return r, np.zeros(n_bins)

    # Get cell dimensions (orthorhombic approximation)
    cell = trajectory.cell_vectors
    if cell is not None:
        if cell.ndim == 2:
            box = np.diag(cell)
        else:
            box = np.diag(cell[start_frame])
    else:
        box = np.array([r_max * 3] * 3)

    dr = r_max / n_bins
    r = np.linspace(dr / 2, r_max - dr / 2, n_bins)
    hist = np.zeros(n_bins)

    # Atom selection for pair-specific RDF
    species = trajectory.species
    if pair is not None:
        idx_a = [i for i, s in enumerate(species) if s == pair[0]]
        idx_b = [i for i, s in enumerate(species) if s == pair[1]]
    else:
        idx_a = list(range(n_atoms))
        idx_b = list(range(n_atoms))

    n_a = len(idx_a)
    n_b = len(idx_b)
    if n_a == 0 or n_b == 0:
        return r, np.zeros(n_bins)

    idx_a_arr = np.array(idx_a)
    idx_b_arr = np.array(idx_b)
    same_species = pair is None or pair[0] == pair[1]

    for frame_idx in range(n_frames):
        pos = positions[frame_idx]
        pos_a = pos[idx_a_arr]  # (n_a, 3)
        pos_b = pos[idx_b_arr]  # (n_b, 3)
        # Vectorized pairwise distances via broadcasting
        delta = pos_b[np.newaxis, :, :] - pos_a[:, np.newaxis, :]  # (n_a, n_b, 3)
        # Minimum image convention
        delta -= np.round(delta / box) * box
        dists = np.sqrt(np.sum(delta ** 2, axis=2))  # (n_a, n_b)
        if same_species:
            # Exclude self-pairs and double-counting (i >= j)
            ii, jj = np.meshgrid(idx_a_arr, idx_b_arr, indexing="ij")
            dists = np.where(ii < jj, dists, 0.0)
        mask = (dists > 0) & (dists < r_max)
        if np.any(mask):
            hist += np.histogram(dists[mask], bins=n_bins, range=(0, r_max))[0]

    # Normalize (vectorized)
    volume = np.prod(box)
    if pair is not None and pair[0] == pair[1]:
        n_pairs = n_a * (n_a - 1) / 2
    elif pair is not None:
        n_pairs = n_a * n_b
    else:
        n_pairs = n_atoms * (n_atoms - 1) / 2

    rho = n_atoms / volume if pair is None else n_b / volume
    shell_vols = 4.0 / 3.0 * np.pi * ((r + dr / 2) ** 3 - (r - dr / 2) ** 3)
    if n_frames > 0 and n_pairs > 0:
        if pair is not None and pair[0] != pair[1]:
            hist /= n_frames * n_a * rho * shell_vols
        else:
            hist /= n_frames * n_pairs * (n_atoms / volume) * shell_vols
            hist *= n_atoms / 2

    return r, hist


# ---------------------------------------------------------------------------
# Mean Square Displacement (MSD)
# ---------------------------------------------------------------------------


def compute_msd(
    trajectory: MDTrajectoryData,
    start_frame: int = 0,
    species: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean square displacement (MSD) vs time.

    Uses the Einstein relation: MSD(t) = <|r(t) - r(0)|²>.

    Args:
        trajectory: MD trajectory data.
        start_frame: Reference frame (t=0).
        species: Compute MSD for a specific element only.

    Returns:
        (t, msd) — time in fs and MSD in Angstrom².
    """
    positions = trajectory.positions[start_frame:]
    n_frames = len(positions)

    if n_frames == 0:
        return np.array([]), np.array([])

    # Select atoms by species
    if species is not None:
        idx = [i for i, s in enumerate(trajectory.species) if s == species]
        if not idx:
            return np.array([]), np.array([])
        positions = positions[:, idx, :]

    # Unwrap PBC not implemented (assumes unwrapped or short trajectory)
    r0 = positions[0]
    msd = np.zeros(n_frames)
    for t in range(n_frames):
        displacement = positions[t] - r0
        msd[t] = np.mean(np.sum(displacement ** 2, axis=1))

    times = trajectory.times[start_frame:start_frame + n_frames]
    if times is None or len(times) == 0:
        times = np.arange(n_frames) * trajectory.timestep_fs

    return times, msd


# ---------------------------------------------------------------------------
# Velocity Autocorrelation Function (VACF)
# ---------------------------------------------------------------------------


def compute_vacf(
    trajectory: MDTrajectoryData,
    max_lag: int = 500,
    start_frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized velocity autocorrelation function.

    VACF(t) = <v(0)·v(t)> / <v(0)·v(0)>

    Args:
        trajectory: MD trajectory data (must have velocities).
        max_lag: Maximum lag in frames.
        start_frame: Skip equilibration frames.

    Returns:
        (t, vacf) — time lag in fs and normalized VACF.
        Returns empty arrays if no velocities available.
    """
    if trajectory.velocities is None:
        return np.array([]), np.array([])

    velocities = trajectory.velocities[start_frame:]
    n_frames = len(velocities)
    max_lag = min(max_lag, n_frames)

    if n_frames == 0 or max_lag == 0:
        return np.array([]), np.array([])

    vacf = np.zeros(max_lag)
    for lag in range(max_lag):
        n_avg = n_frames - lag
        if n_avg <= 0:
            break
        # <v(t)·v(t+lag)> averaged over atoms and origins
        dot_products = np.sum(
            velocities[:n_avg] * velocities[lag:lag + n_avg], axis=2
        )  # (n_avg, n_atoms)
        vacf[lag] = np.mean(dot_products)

    # Normalize
    if abs(vacf[0]) > 1e-30:
        vacf /= vacf[0]

    times = np.arange(max_lag) * trajectory.timestep_fs
    return times, vacf


# ---------------------------------------------------------------------------
# Diffusion Coefficient
# ---------------------------------------------------------------------------


def compute_diffusion_coefficient(
    msd_t: np.ndarray,
    msd: np.ndarray,
    fit_start_frac: float = 0.3,
    fit_end_frac: float = 0.9,
) -> float:
    """Compute diffusion coefficient from MSD via Einstein relation.

    D = lim(t→∞) MSD(t) / (6t)

    Fits a line to the linear regime of MSD vs time.

    Args:
        msd_t: Time array in fs.
        msd: MSD array in Angstrom².
        fit_start_frac: Start of linear fit region (fraction of total).
        fit_end_frac: End of linear fit region (fraction of total).

    Returns:
        Diffusion coefficient in cm²/s.
    """
    n = len(msd_t)
    if n < 3:
        return 0.0

    i_start = max(1, int(n * fit_start_frac))
    i_end = int(n * fit_end_frac)
    if i_end <= i_start:
        return 0.0

    t_fit = msd_t[i_start:i_end]
    msd_fit = msd[i_start:i_end]

    if len(t_fit) < 2:
        return 0.0

    # Linear fit: MSD = 6D*t + b
    coeffs = np.polyfit(t_fit, msd_fit, 1)
    slope = coeffs[0]  # Å²/fs

    # D = slope / 6, convert Å²/fs → cm²/s
    # 1 Å² = 1e-16 cm², 1 fs = 1e-15 s
    D = slope / 6.0 * 1e-16 / 1e-15  # = slope / 6 * 0.1
    D = slope / 6.0 * 1e-1
    return D


# ---------------------------------------------------------------------------
# Equilibration Detection
# ---------------------------------------------------------------------------


def detect_equilibration(
    energies: np.ndarray,
    window: int = 100,
) -> Tuple[int, bool]:
    """Estimate equilibration cutoff using block averaging.

    Compares energy mean of sliding windows; when the drift drops below
    threshold, the system is considered equilibrated.

    Args:
        energies: Energy array (n_frames,).
        window: Block averaging window size.

    Returns:
        (equilibration_frame, is_equilibrated).
    """
    n = len(energies)
    if n < 2 * window:
        return 0, True

    # Compute running mean
    running_mean = np.convolve(energies, np.ones(window) / window, mode="valid")

    if len(running_mean) < 2:
        return 0, True

    # Find where the drift becomes small relative to fluctuations
    std = np.std(running_mean)
    if std < 1e-10:
        return 0, True

    # Look for where the derivative of running mean becomes small
    derivative = np.abs(np.diff(running_mean))
    threshold = np.median(derivative)

    # Find first point where derivative stays below threshold
    eq_frame = 0
    consecutive = 0
    target = max(10, window // 5)
    for i, d in enumerate(derivative):
        if d <= threshold:
            consecutive += 1
            if consecutive >= target:
                eq_frame = max(0, i - target + window)
                break
        else:
            consecutive = 0

    is_equilibrated = eq_frame < n // 2
    return eq_frame, is_equilibrated


# ---------------------------------------------------------------------------
# Full Analysis
# ---------------------------------------------------------------------------


def analyze_md_trajectory(
    trajectory: MDTrajectoryData,
    r_max: float = 10.0,
    n_rdf_bins: int = 200,
    compute_vacf_flag: bool = True,
    max_vacf_lag: int = 500,
) -> MDResult:
    """Run full MD trajectory analysis.

    Computes RDF, MSD, VACF (if velocities available), diffusion coefficient,
    thermodynamic averages, energy drift, and equilibration detection.

    Args:
        trajectory: MD trajectory data from any backend.
        r_max: Maximum distance for RDF in Angstrom.
        n_rdf_bins: Number of RDF histogram bins.
        compute_vacf_flag: Whether to compute VACF.
        max_vacf_lag: Maximum lag for VACF in frames.

    Returns:
        MDResult with all computed properties.
    """
    # Input validation
    if trajectory.positions is None or trajectory.positions.size == 0:
        logger.warning("Empty trajectory — returning default MDResult.")
        return MDResult()
    if np.any(np.isnan(trajectory.positions)):
        logger.warning("NaN detected in trajectory positions.")
    if np.any(np.isinf(trajectory.positions)):
        logger.warning("Inf detected in trajectory positions.")

    # Equilibration detection
    eq_frame, is_eq = detect_equilibration(trajectory.energies)

    # RDF
    rdf_r, rdf_g = compute_rdf(
        trajectory, r_max=r_max, n_bins=n_rdf_bins, start_frame=eq_frame,
    )

    # MSD
    msd_t, msd = compute_msd(trajectory, start_frame=eq_frame)

    # Diffusion
    diff_coeff = 0.0
    if len(msd_t) > 5:
        diff_coeff = compute_diffusion_coefficient(msd_t, msd)

    # VACF
    vacf_t, vacf = None, None
    if compute_vacf_flag and trajectory.velocities is not None:
        vacf_t, vacf = compute_vacf(
            trajectory, max_lag=max_vacf_lag, start_frame=eq_frame,
        )

    # Thermodynamic averages (post-equilibration) — slice once
    eq_slice = slice(eq_frame, None)
    temps = trajectory.temperatures[eq_slice] if trajectory.temperatures is not None else None
    energies = trajectory.energies[eq_slice] if trajectory.energies is not None else None
    times_eq = trajectory.times[eq_slice] if trajectory.times is not None else None

    avg_temp = float(np.mean(temps)) if temps is not None and len(temps) > 0 else None
    temp_std = float(np.std(temps)) if temps is not None and len(temps) > 0 else None
    avg_energy = float(np.mean(energies)) if energies is not None and len(energies) > 0 else None

    # Pressure average
    avg_pressure = None
    if trajectory.pressures is not None:
        p = trajectory.pressures[eq_slice]
        if len(p) > 0:
            avg_pressure = float(np.mean(p))

    # Energy drift per atom (NVE quality check)
    energy_drift = None
    if energies is not None and len(energies) > 2 and trajectory.n_atoms > 0:
        if times_eq is not None and len(times_eq) > 2:
            dt_ps = (times_eq[-1] - times_eq[0]) / 1000.0  # fs → ps
            if dt_ps > 0:
                energy_drift = (energies[-1] - energies[0]) / trajectory.n_atoms / dt_ps

    return MDResult(
        rdf_r=rdf_r,
        rdf_g=rdf_g,
        rdf_pairs="all",
        msd_t=msd_t if len(msd_t) > 0 else None,
        msd=msd if len(msd) > 0 else None,
        diffusion_coefficient=diff_coeff,
        vacf_t=vacf_t,
        vacf=vacf,
        avg_temperature=avg_temp,
        temperature_std=temp_std,
        avg_pressure=avg_pressure,
        avg_energy=avg_energy,
        energy_drift_per_atom=energy_drift,
        equilibration_step=eq_frame,
        is_equilibrated=is_eq,
    )
