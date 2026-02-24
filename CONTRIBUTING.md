# Contributing to SHALOM

Thank you for your interest in contributing to SHALOM! We welcome contributions to our LLM-based multi-agent framework for computational materials science.

## Development Environment Setup

1. Fork and clone the repository.
2. Create a virtual environment and install the development dependencies:
   ```bash
   # Option A — venv (built-in)
   python -m venv .venv
   source .venv/bin/activate          # Linux / macOS
   # .venv\Scripts\activate           # Windows
   pip install -e ".[dev]"

   # Option B — conda
   conda env create -f environment.yml
   conda activate shalom-env
   ```
3. Install the pre-commit hooks to ensure code quality before commits:
   ```bash
   pre-commit install
   ```

## Coding Rules and Standards

- **Code Style**: We strictly enforce `black` for formatting and `ruff` for linting.
- **Type Hinting**: All code must include Python type hints, checked using `mypy`.
- **Docstrings**: We use the *Google Style* for docstrings (in English) across all classes and methods.
- **Tests**: All new features and bug fixes must include tests. High test coverage (>85%) is expected. We use `pytest`. Current baseline: 1124 tests.

## Adding New Components

### New Prompt
1. Create `shalom/prompts/your_name.md` with the prompt text.
2. Add the **identical** text to `shalom/_defaults.py` in the `PROMPTS` dict.
3. Use `load_prompt("your_name")` in agent code.

### New Config (YAML)
1. Create `shalom/config/your_name.yaml`.
2. Add the **identical** data to `shalom/_defaults.py` in the `CONFIGS` dict.
3. (Optional) Add a Pydantic validation schema in `shalom/_config_schemas.py` if the config is safety-critical.
4. Use `load_config("your_name")` in consuming code.

### New DFT Backend
1. Create a new module in `shalom/backends/` (e.g., `lammps.py`).
2. Implement the `DFTBackend` protocol defined in `shalom/backends/base.py`:
   - `name: str` — backend identifier
   - `write_input(atoms, directory, **params) -> str`
   - `parse_output(directory) -> DFTResult`
3. Add an `elif` branch in `shalom/backends/__init__.py` `get_backend()`.
4. Add tests in `tests/test_<backend_name>.py`.

### New Evaluator (Specialist Perspective)
1. Create prompt: `shalom/prompts/eval_new.md` + add to `_defaults.py` PROMPTS dict.
2. Add `load_prompt("eval_new")` as a module-level variable in `evaluators.py`.
3. Add the variable to `_DEFAULT_PROMPTS` dict in `evaluators.py`.
4. Add weight + veto_threshold in `config/evaluator_weights.yaml` + `_defaults.py`.
5. Add to `create_default_evaluators()` factory in `evaluators.py`.

### New MCP Tool
1. Add a new `@mcp.tool()` function in `shalom/mcp_server.py`.
2. Wrap existing SHALOM library functions — do not add new logic in the tool itself.
3. Return structured `dict` with `"success"` key. Catch exceptions and return `{"success": False, "error": "..."}`.
4. **Never use `print()`** — stdout is reserved for JSON-RPC protocol (use `logging` to stderr).
5. Add tests in `tests/test_mcp_server.py` (mock internal dependencies, test input/output contract).

### Security Considerations
- **SafeExecutor**: whitelist-only builtins. When adding new builtins, justify in PR description.
- **Audit logging**: security-sensitive operations (LLM calls, DFT execution) should call `log_event()` from `shalom/core/audit.py`.
- **Never commit** API keys, `.env` files, or audit logs.

## Pull Request Workflow

1. Create a descriptive branch (e.g., `feature/add-new-agent`, `fix/import-bug`).
2. Implement your changes along with tests.
3. Ensure all tests and linting pass locally by running:
   ```bash
   pytest
   ruff check shalom/ tests/
   black --check shalom/ tests/
   mypy shalom/ --ignore-missing-imports
   ```
4. Open a Pull Request referencing any related issues.
5. Wait for CI workflow checks (GitHub Actions) to pass and for team review.
