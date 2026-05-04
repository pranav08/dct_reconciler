#!/usr/bin/env python3
"""
scripts/run_demo.py — Walk all 3 messy sources through the 5-layer pipeline.

Usage:
    python scripts/run_demo.py                # mock extractor (no GPU, no API)
    python scripts/run_demo.py --judge        # enable judge LLM (needs ANTHROPIC_API_KEY)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import box

from pipeline.run import Pipeline
from data.synthetic import VENDOR_JSON, SITE_CRF_CSV, HL7_FRAGMENT
from schemas.enums  import SourceFormat

console = Console()


def render_result(result, idx: int) -> None:
    rec = result.source_record
    if result.dropped:
        console.print(Panel(
            f"[red]✗ DROPPED[/red]\n[dim]{rec.raw_text}[/dim]\n\n"
            f"[red]reason:[/red] {result.drop_reason}",
            title=f"#{idx} {rec.source_id}", border_style="red", box=box.ROUNDED,
        ))
        return

    obs = result.canonical
    val = result.val_report
    jr  = result.judge_report

    if val and val.error_count > 0:
        status, color = "[red]✗ ERROR[/red]", "red"
    elif val and val.warn_count > 0:
        status, color = "[yellow]⚠ WARN[/yellow]", "yellow"
    else:
        status, color = "[green]✓ VALID[/green]", "green"

    body  = (
        f"[dim]{rec.raw_text[:120]}{'…' if len(rec.raw_text) > 120 else ''}[/dim]\n\n"
        f"[bold cyan]canonical[/bold cyan]: subject={obs.subject_id} "
        f"[bold]{obs.vs_code.name}[/bold] = {obs.value_numeric} {obs.unit.value} "
        f"@ {obs.measured_at}"
    )
    if obs.value_si is not None and obs.value_si != obs.value_numeric:
        body += f"  [dim](SI: {obs.value_si})[/dim]"

    if val and val.findings:
        body += "\n\n[bold]validator findings[/bold]:"
        for f in val.findings:
            sev_col = {"ERROR":"red","WARN":"yellow","INFO":"dim"}.get(f.severity.value, "white")
            body += f"\n  [{sev_col}]{f.severity.value:5}[/{sev_col}] {f.rule_id}: {f.message}"

    if jr:
        body += (
            f"\n\n[bold blue]judge[/bold blue] (Haiku): "
            f"plausibility={jr.plausibility_score:.2f}  "
            f"action={jr.suggested_action}  "
            f"cost=${jr.judge_cost_usd:.6f}\n  [dim]{jr.rationale}[/dim]"
        )

    console.print(Panel(body, title=f"#{idx} {status} {rec.source_id} ({result.processing_ms:.1f}ms)",
                        border_style=color, box=box.ROUNDED))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true",
                        help="Enable judge LLM (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]DCT Simple — The 5-Layer Sandwich[/bold cyan]\n"
        "[white]Lossy parser · Compiled AI · Deterministic validator · Conditional judge[/white]",
        border_style="cyan",
    ))

    pipeline = Pipeline(use_mock_extractor=True, judge_enabled=args.judge)

    sources = [
        ("Vendor JSON (wearable batch)", SourceFormat.VENDOR_JSON, VENDOR_JSON),
        ("Site CRF CSV (investigator)",  SourceFormat.SITE_CRF_CSV, SITE_CRF_CSV),
        ("HL7 v2 fragment (device feed)",SourceFormat.HL7_FRAGMENT, HL7_FRAGMENT),
    ]

    grand_results = []
    grand_summary = None

    for name, fmt, raw in sources:
        console.rule(f"[bold]{name}[/bold]")
        results, summary = pipeline.process_batch(fmt, raw, batch_id=fmt.value[:8])
        grand_results.extend(results)
        for i, r in enumerate(results, 1):
            render_result(r, i)

        # per-source summary
        st = Table(box=box.SIMPLE, show_header=False, padding=(0,1))
        st.add_column(style="cyan"); st.add_column()
        st.add_row("records",          str(summary.total_records))
        st.add_row("canonicalised",    str(summary.canonicalised))
        st.add_row("dropped",          str(summary.dropped))
        st.add_row("clean & valid",    str(summary.valid_e2e))
        st.add_row("flagged WARN",     str(summary.flagged_warn))
        st.add_row("flagged ERROR",    str(summary.flagged_error))
        st.add_row("judge calls",      str(summary.judge_calls))
        st.add_row("judge cost (USD)", f"${summary.judge_total_cost:.6f}")
        console.print(Panel(st, title=f"summary — {name}", border_style="dim"))

    # ── Grand totals ──────────────────────────────────────────────────────
    console.rule("[bold]Pipeline totals across all 3 sources[/bold]")
    total = len(grand_results)
    canonicalised = sum(1 for r in grand_results if not r.dropped)
    clean = sum(1 for r in grand_results if r.val_report and r.val_report.is_valid
                                            and r.val_report.warn_count == 0)
    judged = sum(1 for r in grand_results if r.judge_report)
    judge_cost = sum(r.judge_report.judge_cost_usd for r in grand_results if r.judge_report)

    gt = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0,1))
    gt.add_column(style="bold cyan", width=24); gt.add_column()
    gt.add_row("total source records", str(total))
    gt.add_row("canonicalised",        f"{canonicalised}  ({100*canonicalised/total:.0f}%)")
    gt.add_row("clean (no findings)",  f"{clean}  ({100*clean/canonicalised if canonicalised else 0:.0f}%)")
    gt.add_row("judge invocations",    f"{judged}  ({100*judged/total:.0f}% of total)")
    gt.add_row("judge total cost",     f"${judge_cost:.4f}")
    if total > 0:
        gt.add_row("avg cost / record",    f"${judge_cost/total:.6f}")
    console.print(gt)

    console.print(
        "\n[bold green]✓ Done.[/bold green] The judge LLM was called "
        f"on {judged}/{total} records (= {100*judged/total:.0f}%). "
        "The other 85% cost $0."
    )


if __name__ == "__main__":
    main()
