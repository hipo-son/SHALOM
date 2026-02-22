import pytest

from shalom.core.sandbox import SafeExecutor, TimeoutException


class TestSafeExecutor:
    """Tests for the SafeExecutor sandbox environment."""

    def test_basic_execution(self):
        """Simple code executes and returns local state."""
        result = SafeExecutor.execute("x = 1 + 2")
        assert result["x"] == 3

    def test_local_vars_passed_through(self):
        """Pre-populated local_vars are accessible in executed code."""
        result = SafeExecutor.execute("y = x * 10", local_vars={"x": 5})
        assert result["y"] == 50

    def test_whitelist_allows_builtins(self):
        """Whitelisted builtins (len, range, list, etc.) are available."""
        result = SafeExecutor.execute("result = len(list(range(5)))")
        assert result["result"] == 5

    def test_whitelist_allows_print(self):
        """print() is allowed in the sandbox."""
        # Should not raise
        SafeExecutor.execute("print('hello from sandbox')")

    def test_import_blocked(self):
        """__import__ is explicitly None, so import statements should fail."""
        with pytest.raises(TypeError):
            SafeExecutor.execute("import os")

    def test_import_builtin_blocked(self):
        """Direct __import__ call is blocked."""
        with pytest.raises(TypeError):
            SafeExecutor.execute("__import__('os')")

    def test_empty_code(self):
        """Empty code string executes without error."""
        result = SafeExecutor.execute("")
        assert isinstance(result, dict)

    def test_syntax_error_propagates(self):
        """SyntaxError in executed code is propagated."""
        with pytest.raises(SyntaxError):
            SafeExecutor.execute("def (invalid")

    def test_runtime_error_propagates(self):
        """Runtime errors in executed code are propagated."""
        with pytest.raises(ZeroDivisionError):
            SafeExecutor.execute("x = 1 / 0")

    def test_unsafe_exec_mode(self, monkeypatch):
        """SHALOM_ALLOW_UNSAFE_EXEC=1 uses globals() instead of whitelist."""
        monkeypatch.setenv("SHALOM_ALLOW_UNSAFE_EXEC", "1")
        # In unsafe mode, regular imports should work
        result = SafeExecutor.execute("import math; pi = math.pi")
        assert abs(result["pi"] - 3.14159) < 0.001

    def test_timeout_exception_class(self):
        """TimeoutException is a proper Exception subclass."""
        exc = TimeoutException("test")
        assert isinstance(exc, Exception)
        assert str(exc) == "test"

    def test_atoms_construction_in_sandbox(self):
        """ASE Atoms can be constructed when passed via local_vars."""
        from ase.build import bulk

        result = SafeExecutor.execute(
            "atoms = bulk_fn('Cu', 'fcc', a=3.6)",
            local_vars={"bulk_fn": bulk},
        )
        assert "atoms" in result
        assert len(result["atoms"]) == 1

    def test_none_local_vars_default(self):
        """Passing None for local_vars creates empty dict."""
        result = SafeExecutor.execute("a = 42", local_vars=None)
        assert result["a"] == 42

    def test_timeout_via_thread_pool(self):
        """Long-running code triggers TimeoutException via ThreadPoolExecutor."""
        import time

        with pytest.raises(TimeoutException, match="timed out"):
            SafeExecutor.execute(
                "sleep(10)", local_vars={"sleep": time.sleep}, timeout_seconds=1,
            )

    def test_unsafe_exec_globals_scope(self, monkeypatch):
        """SHALOM_ALLOW_UNSAFE_EXEC=1 allows access to full globals."""
        monkeypatch.setenv("SHALOM_ALLOW_UNSAFE_EXEC", "1")
        result = SafeExecutor.execute("import os; cwd = os.getcwd()")
        assert "cwd" in result
        assert isinstance(result["cwd"], str)
