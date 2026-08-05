"""CLI: show counterfactual output-token reduction."""

from __future__ import annotations

import click

from .main import main


@main.command(name="output-savings")
def output_savings() -> None:
    """Show estimated/measured output-token reduction from the shaper.

    Output tokens are counterfactual — we never see what the model *would* have
    emitted unshaped. This reports the honest estimate:

      * "measured" — from an A/B holdout (set HEADROOM_OUTPUT_HOLDOUT>0), the
        unbiased difference between unshaped and shaped arms.
      * "estimated" — synthetic control: shaped output vs. the per-stratum
        baseline learned by `headroom learn --verbosity`.

    Both are shown with a 95% confidence band so the uncertainty is explicit.
    """
    from ..paths import workspace_dir
    from ..proxy.output_savings import SavingsLedger

    path = workspace_dir() / "output_savings.json"
    if not path.exists():
        click.echo("No output-savings data yet.")
        click.echo("Run `headroom learn --verbosity --apply` to seed the baseline,")
        click.echo("then enable the shaper (HEADROOM_OUTPUT_SHAPER=1) and send traffic.")
        return

    ledger = SavingsLedger.load(path)
    est = ledger.best_estimate()

    click.echo(f"\n{'=' * 56}")
    click.echo("Output-token reduction")
    click.echo(f"{'=' * 56}")
    if est.n_requests == 0:
        click.echo("  No shaped requests recorded yet.")
        click.echo(
            f"  Baseline: {ledger.baseline.total_samples} samples, "
            f"{len(ledger.baseline.strata)} strata."
        )
        return

    label = "MEASURED (A/B holdout)" if est.kind == "measured" else "ESTIMATED (synthetic control)"
    click.echo(f"  Method:    {label}")
    click.echo(f"  Requests:  {est.n_requests:,} shaped")
    click.echo(f"  Baseline:  {est.baseline_tokens:,.0f} output tokens expected")
    click.echo(f"  Saved:     {est.tokens_saved:,.0f} output tokens")
    click.echo(
        f"  Reduction: {est.pct:.1f}%   (95% CI {est.ci_low_pct:.1f}% … {est.ci_high_pct:.1f}%)"
    )
    if est.kind == "estimated":
        click.echo(
            "\n  Note: estimated vs the learned baseline. For a measured number,"
            "\n  set HEADROOM_OUTPUT_HOLDOUT=0.1 to leave 10% of conversations"
            "\n  unshaped as a control arm."
        )
