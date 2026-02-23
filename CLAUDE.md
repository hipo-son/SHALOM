# SHALOM Project Guide

## Project Overview
SHALOM (System of Hierarchical Agents for Logical Orchestration of Materials) — a multi-agent framework for computational materials science. Uses LLM-driven agents for autonomous material discovery with DFT (VASP/QE) validation.

## Architecture
```
shalom/
├── agents/           # LLM agent layers (Design, Simulation, Review)
├── backends/         # DFT backends (VASP, QE) + error recovery + execution
│   ├── _physics.py   # Shared physics constants (AccuracyLevel, MAGMOM, detect_2d, etc.)
│   ├── _compression.py # Token-aware error log compression (compress_error_log, truncate_to_tokens)
│   ├── base.py       # Backend-agnostic dataclasses: DFTResult, BandStructureData, DOSData
│   ├── vasp.py       # VASP backend (write_input, parse_output)
│   ├── vasp_config.py # VASP config (VASPInputConfig, get_preset)
│   ├── qe.py         # QE backend (write_input pw.in, parse_output pw.out, parse_output_bands)
│   ├── qe_config.py  # QE config (QEInputConfig, get_qe_preset, generate_band_kpath, SSSP metadata)
│   ├── qe_parser.py  # QE output parsers (parse_xml_bands, parse_dos_file, find_xml_path,
│   │                 #   extract_fermi_energy, compute_nbnd; HA_TO_EV=27.2114, QE_XML_NS)
│   ├── qe_error_recovery.py # QE error recovery (progressive correction, S-matrix diagnostic)
│   └── runner.py     # DFT execution runner (subprocess pw.x, error recovery loop)
├── plotting/         # Matplotlib visualisation (optional: pip install shalom[plotting])
│   ├── band_plot.py  # BandStructurePlotter — band structure with high-sym labels, spin
│   └── dos_plot.py   # DOSPlotter — total/spin-polarised DOS with Fermi level marker
├── workflows/        # High-level multi-step workflows
│   ├── base.py       # ConvergenceWorkflow ABC, ConvergenceResult, ConvergenceTestResult
│   ├── convergence.py # CutoffConvergence (ecutwfc sweep), KpointConvergence (k-mesh sweep)
│   └── standard.py   # StandardWorkflow — 5-step QE pipeline (vc-relax→scf→bands→nscf→dos.x→plot)
├── core/             # LLMProvider, schemas, sandbox
├── tools/            # ASE builder utilities
├── prompts/          # LLM system prompts (.md files)
├── config/           # Physics/DFT settings (.yaml files)
│   ├── sssp_metadata.yaml  # SSSP Efficiency v1.3.0 pseudopotential metadata
│   ├── qe_presets.yaml     # QE pw.x presets (scf/relax/vc-relax/bands/nscf)
│   ├── qe_error_patterns.yaml     # QE error patterns (14 verified against QE 7.x)
│   └── qe_correction_strategies.yaml  # Progressive correction strategies (10 error types)
├── _config_loader.py # load_prompt(), load_config()
├── _config_schemas.py # Pydantic validation for YAML configs
├── _defaults.py      # Hardcoded fallback values
├── mp_client.py      # Materials Project API client (optional: pip install mp-api)
├── direct_run.py     # Direct material run (structure -> DFT input files)
├── __main__.py       # CLI: python -m shalom run/plot/workflow/converge
└── pipeline.py       # End-to-end LLM pipeline orchestrator
```

## Key Conventions

### Configuration System
- **Prompts**: `shalom/prompts/*.md` loaded via `load_prompt("name")`
- **Configs**: `shalom/config/*.yaml` loaded via `load_config("name")`
- **Fallbacks**: `_defaults.py` has identical copies (update both when changing)
- `load_config()` returns deep copies (safe to mutate)
- YAML syntax errors are Fail-Fast (never silently fall back)

### When Adding a New Prompt
1. Create `shalom/prompts/new_name.md`
2. Add the same text to `_defaults.py` PROMPTS dict
3. Use `load_prompt("new_name")` in agent code

### When Adding a New Config
1. Create `shalom/config/new_name.yaml`
2. Add the same data to `_defaults.py` CONFIGS dict
3. (Optional) Add Pydantic schema in `_config_schemas.py` if critical
4. Use `load_config("new_name")` in consuming code

### Testing
- `pytest tests/ -x` — run all tests (922 passed, 1 skipped; 83% coverage on Windows)
- `pytest tests/ --cov=shalom --cov-fail-under=85` — with coverage (85% threshold; integration paths require pw.x to reach full coverage)
- All existing tests must pass unchanged when refactoring
- Mock LLM calls with `unittest.mock` (no real API calls in tests)
- Known: 19 VASP OUTCAR parse tests fail with pymatgen>=2025.10 (upstream `IndexError` in `Outcar.__init__`); QE/agent/CLI tests unaffected
- QE integration test (`test_si_scf_end_to_end`) requires pw.x; auto-skipped on Windows
- Run via `call conda activate shalom-env` in a batch script (direct conda Python path fails from bash on Windows)

### Code Quality
- `ruff check shalom/` — linting
- `mypy shalom/ --ignore-missing-imports` — type checking
- Line length: 100 chars
- Target: Python 3.9+

### Physics Constants
- POTCAR mappings: PBE_54 dataset (version metadata in YAML)
- SSSP pseudopotentials: SSSP Efficiency v1.3.0 (PBE), per-element ecutwfc/ecutrho/z_valence
- Hubbard U: Dudarev scheme, Wang et al. PRB 73 (2006), PBE-fitted only
- MAGMOM: high-spin ferromagnetic initialization defaults
- ANION_ELEMENTS: O, S, Se, Te (triggers GGA+U with magnetic TM)

### QE Config Conventions
- Unit: ecutwfc/ecutrho in Ry, conv_thr in Ry, degauss in Ry, forc_conv_thr in Ry/Bohr
- degauss = VASP SIGMA(eV) / 13.6057 (exact conversion, not approximation)
- conv_thr = VASP EDIFF(eV) / 13.6057
- starting_magnetization = MAGMOM / z_valence (from SSSP, NOT /10)
- VASP "relaxation" (ISIF=3) maps to QE "vc-relax" (not "relax")
- ecutrho from SSSP per-element metadata (not blanket 8x ecutwfc)
- 2D: assume_isolated='2D', vdw_corr='dft-d3', dftd3_version=4 (BJ), cell_dofree='2Dxy'

### CLI Usage
```bash
python -m shalom run mp-19717                              # VASP vc-relax → ~/Desktop/shalom-runs/
python -m shalom run Fe2O3 --backend qe --calc scf         # QE SCF
python -m shalom run --structure POSCAR --backend vasp      # Local file
python -m shalom run mp-19717 --set ENCUT=600               # VASP override
python -m shalom run mp-19717 --backend qe --set ecutwfc=80 # QE override
python -m shalom run Fe2O3 -b qe --execute                 # QE execute locally
python -m shalom run Fe2O3 -b qe -x -np 4 --timeout 7200   # QE execute, 4 MPI procs
python -m shalom run Si -p silicon_study                    # project sub-folder grouping
python -m shalom run Si -w /data/runs                      # custom workspace root
python -m shalom setup-qe                                   # QE environment check
python -m shalom setup-qe --elements Si,Fe --download       # Download missing pseudos

# ── Band/DOS plotting (from completed QE calculation) ─────────────────────────
python -m shalom plot ./03_bands --bands                   # Plot band structure
python -m shalom plot ./04_nscf --dos                      # Plot DOS
python -m shalom plot ./03_bands --bands --fermi-from ./04_nscf  # Use NSCF Fermi energy
python -m shalom plot ./03_bands --bands --emin -8 --emax 6 --title "Si"

# ── Sequential 5-step workflow ────────────────────────────────────────────────
python -m shalom workflow Si -o ./si_wf -np 4             # Full vc-relax→scf→bands→nscf→dos
python -m shalom workflow Si --skip-relax -np 4           # Skip vc-relax, start from SCF
python -m shalom workflow mp-19717 -b qe -np 8 --dos-emin -20

# ── Convergence tests (run ecutwfc first, then kpoints) ──────────────────────
python -m shalom converge Si --test cutoff --values 30,40,50,60,80 -np 2
python -m shalom converge Si --test kpoints --values 20,30,40,50 --ecutwfc 60
```

### Workspace / Output Directory
- Default root: `~/Desktop/shalom-runs/` (Desktop present) or `~/shalom-runs/`
- Override with `$SHALOM_WORKSPACE` env var or `-w/--workspace` CLI flag
- Project grouping: `-p/--project NAME` → `workspace/NAME/auto_name/`
- Explicit path: `-o/--output PATH` → bypasses workspace logic entirely
- Each output folder contains a `README.md` with next steps and reproduce command

### Git
- Commit messages in English
- Do not commit `.env`, credentials, or large binary files
- `.gitattributes` enforces LF line endings
