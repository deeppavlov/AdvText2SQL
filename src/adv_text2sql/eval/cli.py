"""
eval/cli — `text2sql eval` команда.

Пример:
    text2sql eval \
        --profile data/profiles/card_games/profile.json \
        --gold data/bird_train_full_pg.json \
        --model-url http://localhost:8000/v1 \
        --model-name card_games \
        --db-url postgresql+psycopg://user:pass@localhost:5444/card_games \
        --limit 50
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..profiler.profile import Profile
from .reporter import write_report
from .runner import load_bird_subset, run_eval

console = Console()


def run_eval_cmd(
    profile_path: str,
    gold_path: str,
    model_url: str,
    model_name: str,
    db_url: str,
    limit: int | None = 50,
    use_evidence: bool = False,
    out_dir: str = "experiments",
) -> dict:
    profile = Profile.load_json(profile_path)
    gold_items = load_bird_subset(gold_path, db_id=profile.db_id, limit=limit)

    if not gold_items:
        console.print(f"[red]No gold items for db_id={profile.db_id}[/red]")
        raise SystemExit(1)

    console.print(
        f"[bold cyan]Eval[/bold cyan] {profile.db_id} on {len(gold_items)} items\n"
        f"  model  = {model_name} @ {model_url}\n"
        f"  db     = {profile.db_id}\n"
        f"  gold   = {gold_path}"
    )

    report = run_eval(
        profile=profile,
        api_url=model_url,
        model_name=model_name,
        db_url=db_url,
        gold_items=gold_items,
        use_evidence=use_evidence,
    )

    out_path = write_report(report, out_dir=out_dir)
    summary = report.summary()

    console.print(
        f"\n[bold green]✓ Eval done[/bold green]\n"
        f"  overall  = [bold]{summary['overall_accuracy_pct']}%[/bold]\n"
        f"  by diff  = {summary['by_difficulty']}\n"
        f"  errors   = {summary['error_count']} / {summary['total_items']}\n"
        f"  report   = {out_path}"
    )
    return summary
