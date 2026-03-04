# SHALOM Tutorials

Hands-on Jupyter notebooks demonstrating computational materials science workflows
with publication-quality analyses and figures using the SHALOM framework.

## Tutorials

```
tutorials/
├── 01_silicon/                    # Si: convergence, bands, DOS, phonons, XRD
│   ├── notebook.ipynb
│   └── expected_output/           # Reference results for validation
│       ├── bands.png
│       ├── dos.png
│       ├── xrd.png
│       ├── results_summary.json
│       ├── electronic_results.json
│       ├── symmetry_results.json
│       └── xrd_results.json
├── 02_fe2o3/                      # Fe2O3: spin-polarized, GGA+U, magnetic
│   ├── notebook.ipynb
│   └── expected_output/
│       ├── dos.png
│       ├── dos_spin.png
│       └── xrd.png
├── 03_multiscale_md/              # Fe/Si/Ar: LAMMPS MD, VASP AIMD, analysis
│   ├── notebook.ipynb
│   └── expected_output/
│       ├── md_energy.png
│       ├── md_temperature.png
│       ├── md_rdf.png
│       ├── md_msd.png
│       └── md_analysis_results.json
└── README.md
```

| Tutorial | Material | Features | Time |
|----------|----------|----------|------|
| `01_silicon/` | Si (diamond) | Convergence, bands, DOS, phonons, XRD | ~30 min |
| `02_fe2o3/` | Fe2O3 (hematite) | Spin-polarized DOS, GGA+U, magnetic analysis | ~45-60 min |
| `03_multiscale_md/` | Fe + Si + Ar | LAMMPS MD, VASP AIMD, RDF, MSD, diffusion | ~5 min |

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
cd tutorials/01_silicon/
jupyter notebook notebook.ipynb
```

Open the desired notebook and run cells sequentially. Each notebook includes
estimated runtimes per step and can be interrupted/resumed.

## Expected Output

Each tutorial folder contains an `expected_output/` directory with reference
results (plots and JSON). Compare your results against these to verify
correctness.

> **Note**: Tutorial 03 does not require external software (QE, VASP, LAMMPS).
> It generates input files and uses synthetic trajectories for analysis/plotting.

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
