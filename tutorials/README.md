# SHALOM DFT Tutorials

Hands-on Jupyter notebooks that run real Quantum ESPRESSO calculations and
generate publication-quality analyses and figures using the SHALOM framework.

## Notebooks

| Notebook | Material | Features | Time |
|----------|----------|----------|------|
| `01_silicon_complete_study.ipynb` | Si (diamond) | Convergence, bands, DOS, phonons, XRD | ~30 min |
| `02_fe2o3_magnetic_oxide.ipynb` | Fe2O3 (hematite) | Spin-polarized DOS, GGA+U, magnetic analysis | ~45-60 min |

## Prerequisites

1. **conda environment** with Python 3.11+:
   ```bash
   conda activate shalom-env
   ```

2. **Quantum ESPRESSO 7.1+** installed (pw.x, dos.x):
   - Linux/macOS: native installation
   - Windows: install inside WSL (`wsl --install`)
   - QE 7.1+ is required for the HUBBARD card syntax used by SHALOM

3. **SSSP pseudopotentials** downloaded:
   ```bash
   python -m shalom setup-qe --elements Si,Fe,O --download
   ```

4. **Python packages** (install all optional dependencies):
   ```bash
   pip install -e ".[all]"      # plotting + analysis + phonon + symmetry
   pip install jupyter           # Jupyter notebook
   ```

5. **Verify** QE environment:
   ```bash
   python -m shalom setup-qe
   ```

## Running

```bash
cd tutorials/
jupyter notebook
```

Open the desired notebook and run cells sequentially. Each notebook includes
estimated runtimes per step and can be interrupted/resumed.

## Outputs

Each notebook saves all plots and data under `~/Desktop/shalom-tutorials/`:

```
~/Desktop/shalom-tutorials/
  si_study/
    si_xrd.png
    convergence_ecutwfc/ecutwfc_convergence.png
    convergence_kpoints/kpoint_resolution_convergence.png
    workflow/{01_vc_relax,02_scf,03_bands,04_nscf,bands.png,dos.png}
    si_combined_band_dos.png
    si_phonon_bands.png
    si_phonon_dos.png
    si_thermal_properties.png
  fe2o3_study/
    fe2o3_xrd.png
    workflow/{02_scf,03_bands,04_nscf,dos.png}
    fe2o3_dos_spin.png
    fe2o3_dos_shalom.png
```

## Configuration

At the top of each notebook, adjust these variables:

```python
# Windows (WSL):
PSEUDO_DIR = r"C:\Users\<username>\pseudopotentials"
WSL = True
NPROCS = 2

# Linux / macOS (native QE):
PSEUDO_DIR = "/home/<username>/pseudopotentials"
WSL = False
NPROCS = 4
```

Or set the `SHALOM_PSEUDO_DIR` environment variable globally:
```bash
export SHALOM_PSEUDO_DIR=~/pseudopotentials
```
