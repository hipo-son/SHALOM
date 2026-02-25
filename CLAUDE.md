# SHALOM Project Guide

## Project Overview
SHALOM (System of Hierarchical Agents for Logical Orchestration of Materials) — a multi-agent framework for computational materials science. Uses LLM-driven agents for autonomous material discovery with DFT (VASP/QE) validation.

## Architecture
```
shalom/
├── agents/           # LLM agent layers (Design, Simulation, Review)
├── backends/         # DFT backends (VASP, QE) + error recovery + execution
│   ├── _physics.py   # Shared constants index (unit conversions, DFT thresholds, AccuracyLevel, detect_2d)
│   ├── _compression.py # Token-aware error log compression + postprocess_parse_result helper
│   ├── base.py       # Backend-agnostic dataclasses: DFTResult, BandStructureData, DOSData
│   ├── vasp.py       # VASP backend (write_input, parse_output)
│   ├── vasp_config.py # VASP config (VASPInputConfig, get_preset)
│   ├── qe.py         # QE backend (write_input pw.in, parse_output pw.out, parse_output_bands)
│   ├── qe_config.py  # QE config (QEInputConfig, get_qe_preset, generate_band_kpath, SSSP metadata)
│   ├── qe_parser.py  # QE output parsers (parse_xml_bands, parse_dos_file, find_xml_path,
│   │                 #   extract_fermi_energy, compute_nbnd, QE_XML_NS)
│   ├── qe_error_recovery.py # QE error recovery (progressive correction, S-matrix diagnostic)
│   ├── runner.py     # DFT execution runner (subprocess pw.x, error recovery loop, create_runner)
│   └── slurm.py      # Slurm HPC job submission/monitoring (SlurmConfig, SlurmRunner)
├── analysis/         # Post-DFT analysis wrapping external libraries
│   ├── _base.py      # Result dataclasses (Elastic/Phonon/Electronic/XRD/Symmetry/MagneticResult)
│   ├── elastic.py    # Elastic tensor analysis via pymatgen (bulk/shear/Young's modulus, stability)
│   ├── phonon.py     # Phonon analysis via phonopy (band structure, DOS, thermal props, stability)
│   ├── electronic.py # Electronic structure (band gap, VBM/CBM, effective mass, DOS@Ef)
│   ├── xrd.py        # XRD pattern calculation via pymatgen XRDCalculator
│   ├── symmetry.py   # Crystal symmetry analysis via spglib (space/point group, Wyckoff)
│   └── magnetic.py   # Magnetic/charge analysis from QE pw.out (site moments, Löwdin charges)
├── plotting/         # Matplotlib visualisation (optional: pip install shalom[plotting])
│   ├── band_plot.py  # BandStructurePlotter — band structure with high-sym labels, spin
│   ├── dos_plot.py   # DOSPlotter — total/spin-polarised DOS with Fermi level marker
│   ├── phonon_plot.py # PhononBandPlotter, PhononDOSPlotter — phonon dispersion & DOS
│   └── xrd_plot.py   # XRDPlotter — XRD stem plot with hkl labels
├── workflows/        # High-level multi-step workflows
│   ├── base.py       # ConvergenceWorkflow ABC, ConvergenceResult, ConvergenceTestResult
│   ├── convergence.py # CutoffConvergence (ecutwfc sweep), KpointConvergence (k-mesh sweep)
│   └── standard.py   # StandardWorkflow — 5-step QE pipeline (vc-relax→scf→bands→nscf→dos.x→plot)
├── core/             # Core infrastructure
│   ├── llm_provider.py  # LLMProvider — OpenAI/Anthropic + local LLM (base_url) support
│   ├── schemas.py       # Pydantic schemas (MaterialCandidate, PipelineResult, etc.)
│   ├── sandbox.py       # SafeExecutor — hardened sandbox with whitelist-only builtins
│   └── audit.py         # Audit logging (JSON-line, opt-in via $SHALOM_AUDIT_LOG)
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
├── mcp_server.py     # MCP server for Claude Code integration (16 tools)
├── __main__.py       # CLI: python -m shalom run/plot/workflow/converge/analyze/pipeline
└── pipeline.py       # End-to-end LLM pipeline orchestrator (supports base_url for local LLMs)
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

**Quick tests** (mock-based, ~20s, no external deps):
```bash
pytest tests/                          # default: 1390 tests, coverage ≥85%
pytest tests/ -x --no-cov             # fast, stop on first failure
conda run -n shalom-env python -m pytest tests/   # Windows/bash
```

**Integration tests** (real pw.x in WSL, ~20min):
```bash
# From WSL with QE conda env:
export SHALOM_PSEUDO_DIR=/root/pseudopotentials
pytest tests/ -m integration --no-cov -v          # 5 QE integration tests
```

**Physics validation** (full 5-step workflow, ~30-50min):
```bash
python scripts/validate_v1.py --output-dir ./validation_output --nprocs 4
python scripts/validate_v1.py --skip-relax --nprocs 4   # skip vc-relax
```

- Default `pytest` excludes integration tests (`addopts = -m 'not integration'`)
- All existing tests must pass unchanged when refactoring
- Mock LLM calls with `unittest.mock` (no real API calls in tests)
- Known: 19 VASP OUTCAR parse tests fail with pymatgen>=2025.10 (upstream `IndexError`)
- Integration tests require WSL2 + conda-forge QE (`conda create -n qe python=3.12 qe`)

### Code Quality
- `ruff check shalom/` — linting
- `mypy shalom/ --ignore-missing-imports` — type checking
- Line length: 100 chars
- Target: Python 3.9+

### Named Constants Architecture
All magic numbers are named constants. `_physics.py` docstring has the full index.

| File | Owns | Examples |
|------|------|---------|
| `backends/_physics.py` | Shared cross-backend | `DEFAULT_KPR`, `FORCE_CONVERGENCE_THRESHOLD`, `DEFAULT_TIMEOUT_SECONDS`, unit conversions |
| `backends/vasp_config.py` | VASP-specific | `ENCUT_MULTIPLIER_STANDARD/PRECISE`, `ENCUT_MINIMUM`, `METAL_SIGMA` |
| `backends/qe_config.py` | QE-specific | `ECUTWFC_MIN_*`, `DEGAUSS_METAL_RY`, `NBND_MULTIPLIER` |
| `backends/runner.py` | Execution | `MIN_RECOVERY_TIMEOUT`, `STDERR_TAIL_CHARS` |

**Rules for adding new constants:**
1. Used by 2+ files → `_physics.py`
2. Single backend only → that backend's `*_config.py`
3. Never inline a physics/computation threshold — always use a named constant
4. Run `help(shalom.backends._physics)` to see the full list

### Analysis Module (External Library Wrappers)
- `shalom/analysis/` wraps mature external libraries for post-DFT property analysis
- **Elastic** (pymatgen): `pip install shalom[analysis]` — bulk/shear/Young's modulus, Poisson ratio, stability
- **Phonon** (phonopy): `pip install shalom[phonon]` — band structure, DOS, thermal properties, stability
- **Electronic** (numpy): no extra deps — band gap, VBM/CBM, effective mass, DOS at Fermi
- **XRD** (pymatgen): `pip install shalom[analysis]` — powder diffraction pattern from structure
- **Symmetry** (spglib): `pip install shalom[symmetry]` — space/point group, crystal system, Wyckoff
- **Magnetic** (builtin): no extra deps — site magnetization & Löwdin charges from QE pw.out
- Pattern: `_AVAILABLE` flag + `_ensure_available()` guard (same as `mp_client.py`)
- Result dataclasses in `analysis/_base.py`: `ElasticResult`, `PhononResult`, `ElectronicResult`, `XRDResult`, `SymmetryResult`, `MagneticResult`
- Import: `from shalom.analysis import analyze_elastic_tensor, analyze_phonon, analyze_band_structure, calculate_xrd, analyze_symmetry, analyze_magnetism`

**When adding a new analysis module:**
1. Create `shalom/analysis/<property>.py` with optional dep guard
2. Add result dataclass to `_base.py`
3. Export from `__init__.py`
4. Add MCP tool to `mcp_server.py`
5. Add CLI subcommand to `__main__.py` (under `analyze` parser)
6. Add `[<property>]` optional dep group in `pyproject.toml`
7. Write tests in `tests/test_analysis_<property>.py`

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

# ── Post-DFT analysis ─────────────────────────────────────────────────────────
python -m shalom analyze elastic --tensor '[[165.7,63.9,...],...]'  # Elastic from JSON string
python -m shalom analyze elastic --file tensor.json                 # Elastic from JSON file
python -m shalom analyze phonon --structure POSCAR --supercell 2x2x2 --generate-displacements -o ./phonon/
python -m shalom analyze phonon --structure POSCAR --supercell 2x2x2 --force-sets FORCE_SETS
python -m shalom analyze phonon --structure POSCAR --supercell 2x2x2 --force-constants FORCE_CONSTANTS
python -m shalom analyze electronic --bands-xml ./03_bands/    # Band gap, VBM/CBM, effective mass
python -m shalom analyze electronic --calc-dir ./03_bands/ --fermi-energy 5.2
python -m shalom analyze xrd --structure POSCAR                # XRD pattern (CuKa default)
python -m shalom analyze xrd --structure POSCAR --wavelength MoKa -o xrd.png
python -m shalom analyze symmetry --structure POSCAR           # Space group, point group
python -m shalom analyze symmetry --structure POSCAR --symprec 1e-3
python -m shalom analyze magnetic --pw-out ./02_scf/pw.out     # Site magnetization, Löwdin
python -m shalom analyze magnetic --pw-out pw.out --structure POSCAR

# ── Slurm HPC execution ──────────────────────────────────────────────────────
python -m shalom run Si -b qe -x --slurm --partition=compute --account=mat_sci
python -m shalom workflow Si -o ./si_wf --slurm --nodes=4 --ntasks-per-node=32
python -m shalom converge Si --test cutoff --values 30,40,50 --slurm --walltime=02:00:00

# ── LLM-driven autonomous pipeline ──────────────────────────────────────────
python -m shalom pipeline "Find a 2D HER catalyst"              # Full pipeline (OpenAI)
python -m shalom pipeline "Stable cathode" --provider anthropic  # Use Claude
python -m shalom pipeline "MoS2 band structure" --material MoS2  # Skip Design layer
python -m shalom pipeline "Find catalyst" -b qe -x -np 4        # Execute DFT too

# ── Local LLM (no API key needed) ─────────────────────────────────────────
python -m shalom pipeline "Find HER catalyst" --base-url http://localhost:11434/v1  # Ollama
python -m shalom pipeline "MoS2 bands" --base-url http://localhost:8000/v1          # vLLM
export SHALOM_LLM_BASE_URL=http://localhost:11434/v1  # Or set via env var
```

### MCP Server (Claude Code Integration)
SHALOM can be used as an MCP tool server for Claude Code, enabling natural
language interaction with DFT workflows using your Claude subscription (no API key needed
for deterministic tools).

```bash
# Install MCP support
pip install "shalom[mcp]"

# Register in Claude Code (one-time setup)
claude mcp add shalom -- python -m shalom.mcp_server

# Or use project-scoped .mcp.json (already in repo root)
# Claude Code auto-detects this file when opening the project.

# Verify connection inside Claude Code
/mcp
```

After setup, tell Claude Code things like:
- "Si의 SCF 계산 입력 파일을 만들어줘"
- "mp-1040425 그래핀 밴드 구조 계산해줘"
- "QE 환경이 설정되어 있는지 확인해줘"

**16 MCP tools** (15 deterministic + 1 LLM-driven):

| Tool | Description | Requires API Key? |
|------|-------------|:-----------------:|
| `search_material` | Materials Project search by MP ID or formula | No |
| `generate_dft_input` | Generate QE/VASP input files | No |
| `run_workflow` | 5-step QE workflow (vc-relax→scf→bands→nscf→dos) | No |
| `execute_dft` | Execute pw.x with error recovery | No |
| `parse_dft_output` | Parse QE/VASP output files | No |
| `plot_bands` | Band structure plot | No |
| `plot_dos` | Density of states plot | No |
| `run_convergence` | Cutoff/k-point convergence test | No |
| `check_qe_setup` | Verify QE environment | No |
| `analyze_elastic` | Elastic tensor analysis (bulk/shear/Young's modulus) | No |
| `analyze_phonon_properties` | Phonon analysis (band structure, DOS, thermal, stability) | No |
| `analyze_electronic_structure` | Band gap, VBM/CBM, effective mass, DOS at Fermi | No |
| `analyze_xrd_pattern` | Powder XRD pattern from crystal structure | No |
| `analyze_symmetry_properties` | Space group, point group, crystal system, Wyckoff | No |
| `analyze_magnetic_properties` | Site magnetization and Löwdin charges from QE pw.out | No |
| `run_pipeline` | Full multi-agent LLM pipeline | Yes (or `base_url`) |

The `run_pipeline` tool supports `base_url` for local LLM servers (Ollama, vLLM, etc.)
as an alternative to external API keys.

### Local LLM Support
- `LLMProvider` accepts `base_url` parameter for OpenAI-compatible local servers
- Env var fallback: `SHALOM_LLM_BASE_URL` (used by CLI, MCP, and Python API)
- When `base_url` is set, API key check is skipped (dummy key `"local"` used)
- Compatible servers: Ollama (`localhost:11434/v1`), vLLM (`localhost:8000/v1`), llama.cpp, LM Studio
- Anthropic provider also supports `base_url` for self-hosted proxies

### Security & Audit
- **SafeExecutor sandbox**: whitelist-only builtins (eval/exec/compile/open/breakpoint blocked)
  - `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `type` also blocked
  - Override with `SHALOM_ALLOW_UNSAFE_EXEC=1` for development
- **Audit logging**: opt-in via `SHALOM_AUDIT_LOG` env var (JSON-line format)
  - Records `llm_call`, `pipeline_start` events with timestamps
  - Example: `export SHALOM_AUDIT_LOG=~/.shalom/audit.log`
  - Module: `shalom/core/audit.py` — never blocks execution on failure

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
