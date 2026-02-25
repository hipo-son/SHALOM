"""Token-aware context compression utilities for DFT output logs.

Provides heuristic token estimation and smart truncation that preserves
error-relevant lines (keywords, warnings) while respecting a token budget.

Also provides :func:`postprocess_parse_result`, a shared post-processing
helper called by both VASP and QE ``parse_output()`` methods.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:
    from shalom.backends.base import DFTResult

logger = logging.getLogger(__name__)

# Default token budget for error_log field.
DEFAULT_ERROR_LOG_MAX_TOKENS: int = 2000

# Characters per token heuristic.  Conservative (over-estimates) for safety.
CHARS_PER_TOKEN: float = 4.0


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a text string.

    Uses ~4 characters per token, which is conservative for DFT output
    (over-estimates slightly, preventing context window overflow).

    Returns:
        Estimated token count (always >= 0).
    """
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


_CACHED_VASP_PATTERNS: Optional[List[str]] = None


def _load_vasp_patterns() -> List[str]:
    """Load VASP error patterns from error_patterns.yaml (single source of truth).

    Results are cached after the first successful load.
    """
    global _CACHED_VASP_PATTERNS
    if _CACHED_VASP_PATTERNS is not None:
        return _CACHED_VASP_PATTERNS
    try:
        from shalom._config_loader import load_config
        patterns_cfg = load_config("error_patterns")
        _CACHED_VASP_PATTERNS = [p["pattern"] for p in patterns_cfg]  # type: ignore[index]
    except Exception as e:
        logger.warning("Failed to load error_patterns config: %s. Using minimal fallback.", e)
        _CACHED_VASP_PATTERNS = [
            "ZBRENT", "BRMIX", "EDDDAV", "NELM",
            "WARNING", "ERROR", "VERY BAD NEWS",
        ]
    return _CACHED_VASP_PATTERNS


def compress_error_log(
    full_text: str,
    max_tokens: int = DEFAULT_ERROR_LOG_MAX_TOKENS,
    important_patterns: Optional[Sequence[str]] = None,
    tail_lines: int = 50,
    context_lines: int = 3,
) -> str:
    """Compress DFT output text to fit within a token budget.

    Strategy:
    1. Extract lines matching important_patterns (with ±context_lines).
    2. Extract the last ``tail_lines`` lines.
    3. Merge (deduplicated, order-preserved), truncate to max_tokens.

    If the full text already fits, returns it unchanged.

    Args:
        full_text: Complete DFT output text (e.g., OUTCAR or pw.out).
        max_tokens: Maximum token budget for the compressed output.
        important_patterns: Keyword substrings to preserve.
            None loads from error_patterns.yaml automatically.
        tail_lines: Number of lines from the end to always include.
        context_lines: Lines before/after each keyword match to preserve.
    """
    if not full_text:
        return ""

    # Fast path: text already fits
    if estimate_tokens(full_text) <= max_tokens:
        return full_text

    if important_patterns is None:
        important_patterns = _load_vasp_patterns()

    all_lines = full_text.splitlines(keepends=True)
    total = len(all_lines)

    # Phase 1: Collect important line indices (with context)
    selected: set = set()
    for idx, line in enumerate(all_lines):
        for pattern in important_patterns:
            if pattern in line:
                start = max(0, idx - context_lines)
                end = min(total, idx + context_lines + 1)
                selected.update(range(start, end))
                break

    # Phase 2: Collect tail lines
    tail_start = max(0, total - tail_lines)
    selected.update(range(tail_start, total))

    # Phase 3: Merge with order preservation
    sorted_indices = sorted(selected)

    # Build output with markers for non-contiguous regions
    parts: List[str] = []
    prev_idx = -2
    for idx in sorted_indices:
        if idx != prev_idx + 1 and prev_idx >= 0:
            gap = idx - prev_idx - 1
            parts.append(f"  [... {gap} lines omitted ...]\n")
        parts.append(all_lines[idx])
        prev_idx = idx

    compressed = "".join(parts)

    # Phase 4: Final token-budget enforcement (keep tail if still too large)
    if estimate_tokens(compressed) > max_tokens:
        compressed = truncate_to_tokens(compressed, max_tokens)

    return compressed


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget, keeping the tail.

    Removes lines from the beginning until token budget is met.
    Prepends a truncation marker.
    """
    if not text or max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    header = "[... truncated to fit token budget ...]\n"
    target_chars = int((max_tokens - estimate_tokens(header)) * CHARS_PER_TOKEN)
    if target_chars <= 0:
        return header

    # Keep the tail
    truncated = text[-target_chars:]

    # Clean cut at line boundary
    first_newline = truncated.find("\n")
    if first_newline > 0:
        truncated = truncated[first_newline + 1:]

    return header + truncated


def truncate_list(data: Optional[List], max_items: int) -> Optional[List]:
    """Keep only the last ``max_items`` entries of a list.

    Returns None if input was None.
    """
    if data is None:
        return None
    if len(data) <= max_items:
        return data
    return data[-max_items:]


def postprocess_parse_result(
    result: DFTResult,
    output_path: str,
    important_patterns: Optional[Sequence[str]] = None,
    max_ionic_items: int = 50,
) -> None:
    """Attach compressed error log and truncate ionic history on a DFTResult.

    Shared post-processing called by both VASP and QE ``parse_output()``.
    Modifies *result* in-place.

    Args:
        result: DFTResult to post-process.
        output_path: Path to the DFT output file (OUTCAR or pw.out).
        important_patterns: Error keywords to preserve in compression.
            Defaults to VASP patterns if None.
        max_ionic_items: Maximum ionic history entries to keep.
    """
    if not result.is_converged:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            result.error_log = compress_error_log(
                full_text,
                important_patterns=important_patterns,
            )
        except Exception:
            logger.debug("Error log extraction failed for %s", output_path)

    result.ionic_energies = truncate_list(result.ionic_energies, max_ionic_items)
    result.ionic_forces_max = truncate_list(result.ionic_forces_max, max_ionic_items)
