# SHALOM Tutorials

Hands-on Jupyter notebooks demonstrating computational materials science workflows
with publication-quality analyses and figures using the SHALOM framework.

## Tutorials

```
tutorials/
├── 01_silicon/                    # Si: convergence, bands, DOS, phonons, XRD
│   ├── notebook.ipynb
│   └── expected_output/           # Reference results for validation
│       ├── si_combined_band_dos.png
│       ├── si_xrd.png
│       ├── si_phonon_bands.png
│       ├── si_phonon_dos.png
│       ├── si_thermal_properties.png
│       ├── results_summary.json     # v3 workflow report
│       ├── electronic_results.json
│       ├── symmetry_results.json
│       └── xrd_results.json
├── 02_fe2o3/                      # Fe2O3: spin-polarized, GGA+U, magnetic
│   ├── notebook.ipynb
│   └── expected_output/
│       ├── fe2o3_dos_spin.png
│       ├── fe2o3_xrd.png
│       ├── results_summary.json     # v3 workflow report
│       ├── symmetry_results.json
│       ├── xrd_results.json
│       └── magnetic_results.json
├── 03_multiscale_md/              # Fe/Si/Ar: LAMMPS MD, VASP AIMD, analysis
│   ├── notebook.ipynb
│   └── expected_output/
│       ├── fe_md_energy.png
│       ├── fe_md_temperature.png
│       ├── fe_md_rdf.png
│       ├── fe_md_msd.png
│       ├── fe_md_vacf.png
│       ├── md_analysis_results.json
│       ├── run_info_fe.json         # v2 structured run info
│       ├── run_info_si.json
│       └── run_info_ar.json
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

### Structured Report Files

Each tutorial demonstrates SHALOM's structured calculation reporting:

- **`results_summary.json`** (v3) — Written by `StandardWorkflow.run()`. Contains
  structure analysis, per-step timing, detection log (auto-detected parameters),
  Fermi energy, and plot paths.
- **`run_info.json`** (v2) — Written by `direct_run()`. Contains structure analysis,
  auto-detected settings, detection log, and backend/calc_type metadata.
- **`*_results.json`** — Analysis results from `save_result_json()` (symmetry,
  electronic, XRD, magnetic, MD).

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
