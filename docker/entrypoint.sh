#!/bin/bash
set -e

case "$1" in
    test)
        # Run test suite
        shift
        exec python -m pytest tests/ -x --no-cov "$@"
        ;;
    mcp-server)
        # Run MCP server (stdio transport for Claude Code)
        shift
        exec python -m shalom.mcp_server "$@"
        ;;
    run|workflow|converge|plot|setup-qe|pipeline)
        # Pass directly to SHALOM CLI
        exec python -m shalom "$@"
        ;;
    -*)
        # Flags (e.g., --help) -> pass to SHALOM CLI
        exec python -m shalom "$@"
        ;;
    *)
        # Unknown command -> pass to SHALOM CLI
        exec python -m shalom "$@"
        ;;
esac
