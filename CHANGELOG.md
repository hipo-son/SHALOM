# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Token-aware error log compression** (`shalom/backends/_compression.py`)
  - `compress_error_log()`: keyword-preserving + tail extraction with token budget enforcement
  - `truncate_to_tokens()`, `truncate_list()`, `estimate_tokens()` utilities
  - Loads VASP error keywords from `error_patterns.yaml` (single source of truth)
- **QE error_log support** — `qe.py` now attaches compressed error log for unconverged runs
- **Agent Guidelines** (`AGENT_GUIDELINES.md`) with QE-specific domain knowledge (Rydberg units, SSSP, magnetism, 2D)
- 44 new tests (643 total, 95.6% coverage): compression utilities, CLI cmd_run/main, QE error_log, review physics checks
- **Configuration externalization system** (`_config_loader.py`, `_config_schemas.py`, `_defaults.py`)
  - `load_prompt(name)` loads LLM system prompts from `shalom/prompts/*.md`
  - `load_config(name)` loads physics/VASP settings from `shalom/config/*.yaml`
  - Built-in fallback defaults in `_defaults.py` ensure operation without external files
  - Deep-copy protection on mutable config returns prevents cache poisoning
  - Fail-Fast on YAML syntax errors (corrupted files never silently fall back)
  - CRLF normalization for Windows cross-platform compatibility
  - Pydantic schema validation for critical configs (POTCAR mapping, Hubbard U)
- 11 LLM prompt files in `shalom/prompts/` (`.md` format, version-tagged)
- 9 physics config files in `shalom/config/` (`.yaml` format with literature references)
- `ShalomConfigurationError` custom exception for config/prompt loading failures
- `test_config_loader.py` with 25 tests covering loader, schemas, caching, and fallback
- `.gitattributes` for line-ending normalization
- `pyyaml>=6.0` dependency
- POTCAR dataset version metadata (`potcar_version: "54"`)
- Hubbard U functional dependency tag (`functional: "PBE"`)

### Changed
- `vasp.py` error log uses `compress_error_log()` (replaces raw `lines[-100:]` slice)
- `review_layer.py` caps `correction_history` at 20 entries, token-budget-aware error log insertion (80K default)
- `base.py` `DFTResult.error_log` documented as compressed (not for `scan_for_errors()`)
- Ionic step history (`energies`, `forces_history`, `magmoms`) capped at 50 entries in VASP/QE backends
- Agent system prompts externalized from Python strings to `shalom/prompts/*.md`:
  - `design_layer.py`: CoarseSelector, FineSelector prompts
  - `evaluators.py`: 6 specialist evaluator prompts + confidence rule + weights/thresholds
  - `simulation_layer.py`: GeometryGenerator prompt
  - `review_layer.py`: ReviewAgent prompt
- Physics constants externalized from Python dicts to `shalom/config/*.yaml`:
  - `vasp_config.py`: POTCAR mapping, ENMAX values, MAGMOM defaults, Hubbard U, metallic elements, INCAR presets
  - `error_recovery.py`: error patterns, correction strategies
- **Net code reduction: -404 lines hardcoded data, +53 lines loader calls** (7 modified files)
- All existing tests pass unchanged (zero behavioral change)

### Fixed
- `llm_provider.py`: removed duplicate `raise ValueError` (dead code on line 113)

### Previous [Unreleased] changes
- GitHub Actions CI/CD workflows for testing, linting, and PyPI release
- Cross-platform `SafeExecutor` sandbox with `ThreadPoolExecutor` timeout fallback for Windows
- `ReviewAgent` with OUTCAR parsing and LLM-based result evaluation
- `ReviewResult` Pydantic schema for closed-loop feedback
- Comprehensive test suites: `test_sandbox.py`, `test_llm_provider.py`, expanded FormFiller edge cases
- Reproducibility infrastructure with explicit `seed` and `temperature=0.0` in LLM calls
- Dockerfile and `environment.yml` for HPC/SLURM containerization
- Sphinx documentation with `napoleon`, `nbsphinx`, `intersphinx` extensions
- Community files: `CITATION.cff`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`
- `[all]` optional dependency group in `pyproject.toml`
- Pydantic `score` field validation (`ge=0.0, le=1.0`)
- All agent system prompts and feedback messages translated to English
- All Pydantic `Field` descriptions translated to English
- `ruff` and `mypy` target versions aligned to `py39` (minimum supported)
- `anthropic` minimum version raised to `>=0.25.0`
- Coverage threshold unified to 85% across CI, `pyproject.toml`, and `CONTRIBUTING.md`

## [0.1.0] - 2026-02-22

### Added
- Core framework: `LLMProvider` with OpenAI and Anthropic structured output support
- Pydantic schemas: `MaterialCandidate`, `RankedMaterial`, `StructureReviewForm`, `AgentMessage`
- Design Layer: `CoarseSelector` (3-5 candidate screening) and `FineSelector` (precision ranking)
- Simulation Layer: `GeometryGenerator`, `FormFiller` (rule-based validation), `GeometryReviewer` (feedback loop)
- Tools: `ASEBuilder` with `construct_bulk`, `construct_surface`, `save_poscar`, `analyze_structure`
- Unit tests with mock-based LLM testing
- Master design document and Sphinx documentation scaffolding
