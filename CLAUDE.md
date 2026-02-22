# SHALOM Project Guide

## Project Overview
SHALOM (System of Hierarchical Agents for Logical Orchestration of Materials) — a multi-agent framework for computational materials science. Uses LLM-driven agents for autonomous material discovery with DFT (VASP/QE) validation.

## Architecture
```
shalom/
├── agents/           # LLM agent layers (Design, Simulation, Review)
├── backends/         # DFT backends (VASP, QE) + error recovery
│   ├── _physics.py   # Shared physics constants (AccuracyLevel, MAGMOM, detect_2d, etc.)
│   ├── vasp.py       # VASP backend (write_input, parse_output)
│   ├── vasp_config.py # VASP config (VASPInputConfig, get_preset)
│   ├── qe.py         # QE backend (write_input pw.in, parse_output pw.out)
│   └── qe_config.py  # QE config (QEInputConfig, get_qe_preset, SSSP metadata)
├── core/             # LLMProvider, schemas, sandbox
├── tools/            # ASE builder utilities
├── prompts/          # LLM system prompts (.md files)
├── config/           # Physics/DFT settings (.yaml files)
│   ├── sssp_metadata.yaml  # SSSP Efficiency v1.3.0 pseudopotential metadata
│   └── qe_presets.yaml     # QE pw.x presets (scf/relax/vc-relax/bands/nscf)
├── _config_loader.py # load_prompt(), load_config()
├── _config_schemas.py # Pydantic validation for YAML configs
├── _defaults.py      # Hardcoded fallback values
├── mp_client.py      # Materials Project API client (optional: pip install mp-api)
├── direct_run.py     # Direct material run (structure -> DFT input files)
├── __main__.py       # CLI: python -m shalom run mp-19717 --backend qe
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
- `pytest tests/ -x` — run all tests (643 total)
- `pytest tests/ --cov=shalom --cov-fail-under=85` — with coverage
- All existing tests must pass unchanged when refactoring
- Mock LLM calls with `unittest.mock` (no real API calls in tests)

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
python -m shalom run mp-19717                              # VASP vc-relax (default)
python -m shalom run Fe2O3 --backend qe --calc scf         # QE SCF
python -m shalom run --structure POSCAR --backend vasp      # Local file
python -m shalom run mp-19717 --set ENCUT=600               # VASP override
python -m shalom run mp-19717 --backend qe --set ecutwfc=80 # QE override
```

### Git
- Commit messages in English
- Do not commit `.env`, credentials, or large binary files
- `.gitattributes` enforces LF line endings
