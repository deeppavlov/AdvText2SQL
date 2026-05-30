"""
eval/reporter — markdown-report по результатам eval.

Output: `experiments/<db_id>_eval.md` с таблицами accuracy + примеры ошибок.
"""
from __future__ import annotations

from pathlib import Path

from .runner import EvalReport


def render_markdown(report: EvalReport) -> str:
    """Полный отчёт в markdown для дефенса/PR."""
    config = report.config
    summary = report.summary()

    md = [
        f"# Eval report — {config.get('db_id', 'unknown')}",
        "",
        f"**Started at**: {report.started_at}  ",
        f"**Model**: `{config.get('model_name')}` @ `{config.get('api_url')}`  ",
        f"**Items**: {summary['total_items']}  ",
        f"**Total latency**: {summary['total_latency_s']} s "
        f"(avg {summary['total_latency_s'] / max(1, summary['total_items']) * 1000:.0f} ms/q)",
        "",
        "## Overall",
        "",
        f"- **Execution accuracy**: **{summary['overall_accuracy_pct']}%**",
        f"- Errors (predicted SQL crashed at execute): {summary['error_count']} / {summary['total_items']}",
        "",
        "## By difficulty",
        "",
        "| Difficulty | Items | Accuracy |",
        "|---|---|---|",
    ]
    for diff, acc_pct in sorted(summary["by_difficulty"].items()):
        n = sum(1 for i in report.items if i.difficulty == diff)
        md.append(f"| {diff} | {n} | {acc_pct}% |")

    md.extend(["", "## Sample failures (first 10)", ""])
    fails = [i for i in report.items if not i.score][:10]
    if not fails:
        md.append("_No failures 🎉_")
    else:
        for f in fails:
            md.extend([
                f"### q_id={f.question_id} ({f.difficulty})",
                "",
                f"**Question**: {f.question}",
                "",
                "**Gold SQL**:",
                "```sql",
                f.gold_sql.strip(),
                "```",
                "",
                "**Predicted SQL**:",
                "```sql",
                f.predicted_sql.strip() or "(empty)",
                "```",
                "",
            ])
            if f.error:
                md.append(f"**Error**: `{f.error[:150]}`")
                md.append("")

    md.extend([
        "## Reproducibility",
        "",
        "```json",
        _format_config(report.config),
        "```",
        "",
    ])
    return "\n".join(md)


def write_report(report: EvalReport, out_dir: str | Path = "experiments") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_id = report.config.get("db_id", "report")
    path = out_dir / f"{db_id}_eval.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def _format_config(config: dict) -> str:
    import json as _json
    return _json.dumps(config, ensure_ascii=False, indent=2)
