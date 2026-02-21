# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
- All agent system prompts and feedback messages translated to English
- All Pydantic `Field` descriptions translated to English
- `sphinxcontrib.napoleon` replaced with built-in `sphinx.ext.napoleon`
- `sphinx_rtd_theme` removed from extensions list (used only as theme)
- `ruff` and `mypy` target versions aligned to `py39` (minimum supported)
- `anthropic` minimum version raised to `>=0.25.0`
- Coverage threshold unified to 85% across CI, `pyproject.toml`, and `CONTRIBUTING.md`
- `print()` logging replaced with `logging` module in simulation layer
- Integration test refactored to use `pytest.fail()` instead of bare `return`

### Fixed
- Dockerfile `.[all]` extra now resolves correctly
- Array shape guard added in `FormFiller.evaluate_atoms` for non-3D cells
- `__import__` explicitly blocked in `SafeExecutor` whitelist
- Unreachable code after `pytest.skip()` removed from integration test
- "AST" typo corrected to "ASE" in `architecture.rst`
- PyPI badge link corrected (was pointing to python-poetry.org)
- Zenodo DOI placeholder removed until actual integration

## [0.1.0] - 2026-02-22

### Added
- Core framework: `LLMProvider` with OpenAI and Anthropic structured output support
- Pydantic schemas: `MaterialCandidate`, `RankedMaterial`, `StructureReviewForm`, `AgentMessage`
- Design Layer: `CoarseSelector` (3-5 candidate screening) and `FineSelector` (precision ranking)
- Simulation Layer: `GeometryGenerator`, `FormFiller` (rule-based validation), `GeometryReviewer` (feedback loop)
- Tools: `ASEBuilder` with `construct_bulk`, `construct_surface`, `save_poscar`, `analyze_structure`
- Unit tests with mock-based LLM testing
- Master design document and Sphinx documentation scaffolding
