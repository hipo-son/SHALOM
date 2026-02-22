"""SHALOM Demo — interactive material discovery demonstration.

Requires the ``rich`` library (``pip install shalom[demo]``).
"""


def _check_demo_dependencies() -> None:
    """Verify that demo-only dependencies are installed."""
    try:
        import rich  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The 'rich' library is required for the demo.\n"
            "Install with: pip install shalom[demo]"
        ) from None
