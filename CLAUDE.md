# SHALOM Project Guide

## Project Overview
SHALOM (System of Hierarchical Agents for Logical Orchestration of Materials) — a multi-agent framework for computational materials science. Uses LLM-driven agents for autonomous material discovery with DFT (VASP/QE) validation.

## Architecture
```
shalom/
├── agents/           # LLM agent layers (Design, Simulation, Review)
├── backends/         # DFT backends (VASP, QE) + error recovery
├── core/             # LLMProvider, schemas, sandbox
├── tools/            # ASE builder utilities
├── prompts/          # LLM system prompts (.md files)
├── config/           # Physics/VASP settings (.yaml files)
├── _config_loader.py # load_prompt(), load_config()
├── _config_schemas.py # Pydantic validation for YAML configs
├── _defaults.py      # Hardcoded fallback values
└── pipeline.py       # End-to-end orchestrator
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
- `pytest tests/ -x` — run all tests (346 total)
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
- Hubbard U: Dudarev scheme, Wang et al. PRB 73 (2006), PBE-fitted only
- MAGMOM: high-spin ferromagnetic initialization defaults
- ANION_ELEMENTS: O, S, Se, Te (triggers GGA+U with magnetic TM)

### Git
- Commit messages in English
- Do not commit `.env`, credentials, or large binary files
- `.gitattributes` enforces LF line endings
