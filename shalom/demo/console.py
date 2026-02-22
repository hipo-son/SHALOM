"""Rich console output for the SHALOM demo."""

from typing import TYPE_CHECKING, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from shalom.core.schemas import (
        EvaluationDetails,
        MaterialCandidate,
        PipelineResult,
        RankedMaterial,
    )
    from shalom.demo.cost_tracker import CostTracker
    from shalom.demo.scenarios import Scenario


class DemoConsole:
    """Rich-formatted console output for the demo pipeline."""

    def __init__(self, quiet: bool = False, no_color: bool = False) -> None:
        self.console = Console(no_color=no_color)
        self.quiet = quiet

    def print_header(
        self,
        scenario: "Scenario",
        provider: str,
        model: str,
        dry_run: bool = False,
    ) -> None:
        mode_tag = "[bold yellow]DRY-RUN[/]" if dry_run else "[bold green]LIVE[/]"
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]SHALOM Material Discovery Demo[/]\n\n"
                f"Scenario: [cyan]{scenario.name}[/]\n"
                f"Objective: {scenario.objective[:100]}...\n"
                f"Provider: [magenta]{provider}[/] / {model}\n"
                f"Mode: {mode_tag}  |  Est. API calls: {scenario.estimated_api_calls}  "
                f"|  Est. cost: ${scenario.estimated_cost_usd:.2f}",
                title="[bold blue]SHALOM[/]",
                border_style="blue",
            )
        )

    def print_candidates(
        self,
        candidates: List["MaterialCandidate"],
        elapsed: float,
        tracker: "CostTracker",
    ) -> None:
        if self.quiet:
            return
        table = Table(title="Coarse Selection Results", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Material", style="cyan bold")
        table.add_column("Elements", style="green")
        table.add_column("Reasoning", max_width=60)

        for i, c in enumerate(candidates, 1):
            table.add_row(
                str(i),
                c.material_name,
                ", ".join(c.elements),
                c.reasoning[:120] + ("..." if len(c.reasoning) > 120 else ""),
            )

        self.console.print()
        self.console.print(table)
        self.console.print(
            f"  [dim]{elapsed:.1f}s | {tracker.call_count} API calls "
            f"| {tracker.total_tokens} tokens | ${tracker.total_cost:.4f}[/]"
        )

    def print_evaluation_matrix(
        self,
        details: "EvaluationDetails",
        candidates: List["MaterialCandidate"],
    ) -> None:
        if self.quiet or not details.evaluations:
            return

        if self.console.width < 100:
            self._print_eval_cards(details)
        else:
            self._print_eval_table(details, candidates)

    def _print_eval_table(
        self,
        details: "EvaluationDetails",
        candidates: List["MaterialCandidate"],
    ) -> None:
        table = Table(title="Multi-Agent Evaluation Matrix", show_lines=True)
        table.add_column("Material", style="cyan bold")

        perspectives = [ev.perspective for ev in details.evaluations]
        for p in perspectives:
            table.add_column(p.replace("_", " ").title(), justify="center")
        table.add_column("Status", justify="center")

        # Build score lookup
        scores = {}
        for ev in details.evaluations:
            for cs in ev.scores:
                scores.setdefault(cs.material_name, {})[ev.perspective] = cs.score

        vetoed_names = set()
        for reason in details.veto_reasons:
            name = reason.split(":")[0].strip()
            vetoed_names.add(name)

        for cand in candidates:
            name = cand.material_name
            row = [name]
            for p in perspectives:
                s = scores.get(name, {}).get(p)
                if s is not None:
                    color = "green" if s >= 0.5 else ("yellow" if s >= 0.3 else "red")
                    row.append(f"[{color}]{s:.2f}[/]")
                else:
                    row.append("[dim]N/A[/]")
            status = "[red]VETOED[/]" if name in vetoed_names else "[green]PASS[/]"
            row.append(status)
            table.add_row(*row)

        self.console.print()
        self.console.print(table)

    def _print_eval_cards(self, details: "EvaluationDetails") -> None:
        """Narrow terminal fallback: card layout."""
        for ev in details.evaluations:
            lines = [f"[bold]{ev.perspective.replace('_', ' ').title()}[/]"]
            for cs in ev.scores:
                color = "green" if cs.score >= 0.5 else ("yellow" if cs.score >= 0.3 else "red")
                lines.append(
                    f"  {cs.material_name}: [{color}]{cs.score:.2f}[/] "
                    f"(conf={cs.confidence:.2f})"
                )
            self.console.print("\n".join(lines))

    def print_veto_info(self, veto_reasons: List[str]) -> None:
        if self.quiet or not veto_reasons:
            return
        self.console.print()
        self.console.print("[bold yellow]Veto Reasons:[/]")
        for reason in veto_reasons:
            self.console.print(f"  [yellow]- {reason[:150]}[/]")

    def print_winner(self, ranked: "RankedMaterial") -> None:
        if self.quiet:
            return
        self.console.print()
        self.console.print(
            Panel(
                f"[bold green]{ranked.candidate.material_name}[/]\n"
                f"Score: [bold]{ranked.score:.2f}[/]\n"
                f"Elements: {', '.join(ranked.candidate.elements)}\n"
                f"Justification: {ranked.ranking_justification[:200]}",
                title="[bold]Winner Material[/]",
                border_style="green",
            )
        )

    def print_structure_info(self, path: str) -> None:
        if self.quiet:
            return
        self.console.print()
        self.console.print(f"[bold]Structure generated:[/] [cyan]{path}[/]")

    def print_cost_confirmation(self, scenario: "Scenario") -> bool:
        """Show estimated cost and ask for confirmation. Returns True to proceed."""
        self.console.print()
        self.console.print(
            f"[bold yellow]This will make ~{scenario.estimated_api_calls} "
            f"API calls (est. ${scenario.estimated_cost_usd:.2f}).[/]"
        )
        try:
            answer = input("Proceed? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("", "y", "yes")

    def print_error(self, msg: str, verbose: bool = False) -> None:
        self.console.print(f"\n[bold red]Error:[/] {msg}")

    def print_final_report(
        self,
        result: "PipelineResult",
        tracker: "CostTracker",
        elapsed: float,
    ) -> None:
        status_color = "green" if result.status.value in ("completed", "awaiting_dft") else "red"
        self.console.print()
        self.console.print(
            Panel(
                f"Status: [{status_color}]{result.status.value.upper()}[/]\n"
                f"Objective: {result.objective[:80]}...\n"
                f"Steps: {' -> '.join(result.steps_completed)}\n\n"
                f"API calls: {tracker.call_count}\n"
                f"Total tokens: {tracker.total_tokens:,}\n"
                f"Estimated cost: ${tracker.total_cost:.4f}\n"
                f"Elapsed: {elapsed:.1f}s",
                title="[bold]Pipeline Complete[/]",
                border_style="blue",
            )
        )
        if result.structure_path:
            self.console.print(
                f"\n[bold]DFT input files:[/] [cyan]{result.structure_path}[/]"
            )
        if result.error_message:
            self.console.print(f"\n[bold red]Error:[/] {result.error_message}")
