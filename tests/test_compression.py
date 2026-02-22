"""Tests for shalom.backends._compression module."""

from shalom.backends._compression import (
    compress_error_log,
    estimate_tokens,
    truncate_list,
    truncate_to_tokens,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # "hello" = 5 chars -> 5/4 = 1.25 -> int(1.25) = 1
        assert estimate_tokens("hello") >= 1

    def test_longer_string(self):
        text = "a" * 400  # 400 chars -> ~100 tokens
        assert estimate_tokens(text) == 100

    def test_returns_nonnegative(self):
        assert estimate_tokens("x") >= 0


# ---------------------------------------------------------------------------
# truncate_to_tokens
# ---------------------------------------------------------------------------


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        text = "short text"
        result = truncate_to_tokens(text, 1000)
        assert result == text

    def test_long_text_truncated(self):
        text = "\n".join(f"line {i}" for i in range(500))
        result = truncate_to_tokens(text, 50)
        assert "truncated" in result.lower()
        assert estimate_tokens(result) <= 60  # some margin for header

    def test_empty_text(self):
        assert truncate_to_tokens("", 100) == ""

    def test_zero_budget(self):
        assert truncate_to_tokens("hello", 0) == ""

    def test_keeps_tail(self):
        """Truncation keeps the end (tail) of the text, not the beginning."""
        lines = [f"line_{i}" for i in range(100)]
        text = "\n".join(lines)
        result = truncate_to_tokens(text, 20)
        # Should contain lines from the end
        assert "line_99" in result
        # Should NOT contain lines from the beginning (too far)
        assert "line_0" not in result


# ---------------------------------------------------------------------------
# truncate_list
# ---------------------------------------------------------------------------


class TestTruncateList:
    def test_none_returns_none(self):
        assert truncate_list(None, 10) is None

    def test_short_list_unchanged(self):
        data = [1, 2, 3]
        assert truncate_list(data, 10) == [1, 2, 3]

    def test_exact_limit(self):
        data = [1, 2, 3]
        assert truncate_list(data, 3) == [1, 2, 3]

    def test_over_limit_keeps_tail(self):
        data = [1, 2, 3, 4, 5]
        assert truncate_list(data, 3) == [3, 4, 5]

    def test_empty_list(self):
        assert truncate_list([], 5) == []


# ---------------------------------------------------------------------------
# compress_error_log
# ---------------------------------------------------------------------------


class TestCompressErrorLog:
    def test_empty_text(self):
        assert compress_error_log("") == ""

    def test_short_text_unchanged(self):
        text = "ZBRENT: error\nsome detail\n"
        result = compress_error_log(text, max_tokens=10000)
        assert result == text

    def test_preserves_keyword_lines(self):
        """Lines matching important patterns are preserved."""
        lines = [f"normal line {i}" for i in range(200)]
        lines[50] = "WARNING: something bad happened"
        lines[150] = "ERROR: critical failure"
        text = "\n".join(lines)
        result = compress_error_log(
            text, max_tokens=500, important_patterns=["WARNING", "ERROR"]
        )
        assert "WARNING: something bad happened" in result
        assert "ERROR: critical failure" in result

    def test_preserves_tail(self):
        """Last N lines are always preserved."""
        lines = [f"line {i}" for i in range(200)]
        text = "\n".join(lines)
        result = compress_error_log(
            text, max_tokens=200, important_patterns=[], tail_lines=10
        )
        assert "line 199" in result
        assert "line 190" in result

    def test_omission_marker(self):
        """Non-contiguous regions show '... N lines omitted ...' marker."""
        lines = [f"line {i}" for i in range(200)]
        lines[10] = "WARNING: early warning"
        text = "\n".join(lines)
        result = compress_error_log(
            text, max_tokens=200, important_patterns=["WARNING"], tail_lines=10
        )
        assert "lines omitted" in result

    def test_context_lines_preserved(self):
        """Lines around keyword matches are preserved (context_lines)."""
        lines = [f"line {i}" for i in range(200)]
        lines[50] = "ZBRENT: error here"
        text = "\n".join(lines)
        result = compress_error_log(
            text, max_tokens=500, important_patterns=["ZBRENT"],
            tail_lines=5, context_lines=2,
        )
        assert "ZBRENT: error here" in result
        assert "line 48" in result  # context before
        assert "line 52" in result  # context after

    def test_token_budget_enforced(self):
        """Output stays within the token budget."""
        text = "ZBRENT error\n" * 5000  # Very long text
        result = compress_error_log(text, max_tokens=100)
        assert estimate_tokens(result) <= 120  # small margin for truncation header

    def test_auto_loads_vasp_patterns(self):
        """When important_patterns=None, loads from error_patterns.yaml."""
        lines = [f"line {i}" for i in range(200)]
        lines[50] = "ZBRENT: something"
        text = "\n".join(lines)
        # Should not raise — loads patterns automatically
        result = compress_error_log(text, max_tokens=500)
        assert "ZBRENT" in result
