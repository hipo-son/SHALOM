"""CLI entry point for ``python -m shalom.demo``."""

import argparse
import logging
import os
import sys

from shalom.demo import _check_demo_dependencies
from shalom.demo.scenarios import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHALOM Material Discovery Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m shalom.demo --dry-run                    # offline preview\n"
            "  python -m shalom.demo --scenario smoke_test         # real API (~$0.03)\n"
            "  python -m shalom.demo --scenario her_catalyst       # multi-agent (~$0.12)\n"
            "  python -m shalom.demo --list                        # list all scenarios\n"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="smoke_test",
        help="Demo scenario to run (default: smoke_test).",
    )
    parser.add_argument(
        "--objective",
        help="Custom objective string (overrides scenario objective).",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="LLM provider (auto-detected from API keys if omitted).",
    )
    parser.add_argument(
        "--model",
        help="LLM model name (default: provider-specific).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with canned responses (no API calls, $0 cost).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_scenarios",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output (results only).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output.",
    )
    parser.add_argument(
        "--save-report",
        default="demo_report.json",
        help="Path for JSON report (default: demo_report.json).",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip cost confirmation prompt.",
    )
    args = parser.parse_args()

    # Check demo dependencies (rich)
    _check_demo_dependencies()

    # List scenarios
    if args.list_scenarios:
        print("\nAvailable SHALOM Demo Scenarios:\n")
        for key, sc in SCENARIOS.items():
            print(f"  {key:20s}  {sc.name:25s}  ~{sc.estimated_api_calls} calls  "
                  f"~${sc.estimated_cost_usd:.2f}  ({sc.selector_mode})")
            print(f"  {'':20s}  {sc.description}")
            print()
        sys.exit(0)

    # Logging
    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve scenario
    scenario = SCENARIOS[args.scenario]

    # Override objective if provided
    if args.objective:
        from dataclasses import replace
        scenario = replace(scenario, objective=args.objective)

    # API key validation (skip for dry-run)
    provider = args.provider
    model = args.model

    if not args.dry_run:
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

        if not provider:
            if openai_key:
                provider = "openai"
            elif anthropic_key:
                provider = "anthropic"
            else:
                print(
                    "Error: No API key found.\n\n"
                    "  export OPENAI_API_KEY='sk-...'\n"
                    "  export ANTHROPIC_API_KEY='ant-...'\n\n"
                    "Or use --dry-run for offline demo."
                )
                sys.exit(1)

        if not model:
            model = "gpt-4o" if provider == "openai" else "claude-sonnet-4-6"
    else:
        provider = provider or "canned"
        model = model or "dry-run"

    # Cost confirmation
    if not args.dry_run and not args.yes:
        from shalom.demo.console import DemoConsole
        display = DemoConsole(no_color=args.no_color)
        if not display.print_cost_confirmation(scenario):
            print("Aborted.")
            sys.exit(0)

    # Run
    from shalom.demo.runner import DemoRunner
    from shalom.demo.report import save_report

    runner = DemoRunner(
        scenario=scenario,
        provider_type=provider,
        model_name=model,
        dry_run=args.dry_run,
        quiet=args.quiet,
        no_color=args.no_color,
        verbose=args.verbose,
    )
    result = runner.run()

    # Save report
    report_path = save_report(result, runner.tracker, args.save_report)
    if report_path:
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
