# Contributing to SHALOM

Thank you for your interest in contributing to SHALOM! We welcome contributions to our LLM-based multi-agent framework for computational materials science.

## Development Environment Setup

1. Fork and clone the repository.
2. Create a virtual environment and install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install the pre-commit hooks to ensure code quality before commits:
   ```bash
   pre-commit install
   ```

## Coding Rules and Standards

- **Code Style**: We strictly enforce `black` for formatting and `ruff` for linting.
- **Type Hinting**: All code must include Python type hints, checked using `mypy`.
- **Docstrings**: We use the *Google Style* for docstrings (in English) across all classes and methods.
- **Tests**: All new features and bug fixes must include tests. High test coverage (>85%) is expected. We use `pytest`.

## Pull Request Workflow

1. Create a descriptive branch (e.g., `feature/add-new-agent`, `fix/import-bug`).
2. Implement your changes along with tests.
3. Ensure all tests and linting pass locally by running:
   ```bash
   pytest
   ruff check shalom/ tests/
   black --check shalom/ tests/
   mypy shalom/
   ```
4. Open a Pull Request referencing any related issues.
5. Wait for CI workflow checks (GitHub Actions) to pass and for team review.
