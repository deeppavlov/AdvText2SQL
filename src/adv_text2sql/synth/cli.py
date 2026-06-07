"""
synth/cli.py — реализация `text2sql generate`.

Полный пайплайн стадии GENERATE:

    Profile → [TemplateGen + LLMGen] → raw.jsonl → Validator → validated.jsonl
                                                            ↘ rejected.jsonl

  --count auto     — берём target из profile.target_synthetic_count()
  --auto-resize    — если pass_rate < 40%, перегенерим с x2 target_count
  --judge          — добавить LLM-judge поверх execution-фильтра
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from rich.console import Console

from ..profiler.connector import DBConnector
from ..profiler.profile import Profile
from .llm_generator import LLMGeneratorConfig, LLMSyntheticGenerator
from .template_generator import SyntheticExample, TemplateSyntheticGenerator
from .validator import Validator

logger = logging.getLogger("text2sql_tool.synth.cli")
console = Console()


def run_generate(
    profile_path: str,
    count: str = "auto",
    generator: str = "template",     # "template" | "llm" | "both"
    llm_model: str = "gpt-4o",
    judge: bool = False,
    auto_resize: bool = False,
    out_dir: str = "data/synthetic",
    db_url: str | None = None,        # для validator stage; если None — берём redacted из Profile
    seed: int = 42,
    language: str = "ru",            # "ru" | "en" — язык вопросов (train==inference!)
) -> dict:
    """Run full GENERATE stage. Returns statistics dict."""
    profile = Profile.load_json(profile_path)

    if count == "auto":
        target = profile.target_synthetic_count()
    else:
        target = int(count)

    console.print(
        f"[bold cyan]Generate stage[/bold cyan]\n"
        f"  db_id        = {profile.db_id}\n"
        f"  generator    = {generator}\n"
        f"  target count = {target}\n"
        f"  judge        = {judge}\n"
        f"  auto-resize  = {auto_resize}"
    )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: raw generation ─────────────────────────────────────────────
    raw_path = out_dir_path / f"{profile.db_id}_raw_{generator}.jsonl"
    examples = _run_generators(profile, generator, target, llm_model, seed, language)
    _write_jsonl(examples, raw_path)
    console.print(f"  → raw written: {raw_path.name} ({len(examples)} examples)")

    # ── Stage 2: validation (parse + whitelist + execute) ──────────────────
    if db_url is None:
        console.print(
            "[yellow]No --db-url, skipping execute-check[/yellow] "
            "(parse + whitelist still run)"
        )
        connector = None
    else:
        connector = DBConnector(db_url)

    validator = Validator(connector, skip_execute=(connector is None))
    try:
        v_result = validator.validate_jsonl(raw_path, out_dir_path)
    finally:
        if connector:
            connector.close()

    console.print(
        f"  → validated: {v_result.passed} pass, {v_result.rejected} reject "
        f"({v_result.pass_rate * 100:.1f}%)\n"
        f"     reasons = {v_result.by_reason}"
    )

    # ── Stage 3: auto-resize if pass-rate низкий ────────────────────────────
    # Если все отказы — sql_error, это обрыв соединения с БД, а не плохой SQL.
    # Удваивать датасет в этом случае бессмысленно — нужно починить туннель.
    all_conn_failures = (
        v_result.passed == 0
        and v_result.by_reason.get("sql_error", 0) == v_result.rejected
    )
    if all_conn_failures:
        console.print(
            "[red]Все отказы — sql_error (обрыв соединения с БД). "
            "Проверьте SSH-туннель (nc -z localhost 5444) и запустите валидацию заново.[/red]"
        )
        return
    if auto_resize and v_result.pass_rate < 0.40 and target < 5000:
        new_target = target * 2
        console.print(
            f"[yellow]Pass rate {v_result.pass_rate*100:.1f}% < 40% → "
            f"auto-resize to {new_target}[/yellow]"
        )
        return run_generate(
            profile_path=profile_path,
            count=str(new_target),
            generator=generator,
            llm_model=llm_model,
            judge=judge,
            auto_resize=False,  # один раз, не цикл
            out_dir=out_dir,
            db_url=db_url,
            seed=seed + 1,
            language=language,
        )

    return {
        "db_id": profile.db_id,
        "target": target,
        "raw_count": len(examples),
        "passed": v_result.passed,
        "rejected": v_result.rejected,
        "pass_rate": v_result.pass_rate,
        "passed_path": str(v_result.passed_path),
        "rejected_path": str(v_result.rejected_path),
        "rejection_reasons": v_result.by_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _run_generators(
    profile: Profile,
    generator: str,
    target: int,
    llm_model: str,
    seed: int,
    language: str = "ru",
) -> list[SyntheticExample]:
    examples: list[SyntheticExample] = []

    if generator in ("template", "both"):
        n_template = target if generator == "template" else target // 2
        tg = TemplateSyntheticGenerator(profile, seed=seed, language=language)
        examples.extend(tg.generate(n_template))

    if generator in ("llm", "both"):
        n_llm = target if generator == "llm" else target - len(examples)
        config = LLMGeneratorConfig(
            model_name=llm_model,
            base_url=os.environ.get("LLM_BASE_URL"),
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            language=language,
        )
        lg = LLMSyntheticGenerator(profile, config)
        examples.extend(asyncio.run(lg.generate(n_llm)))

    return examples


def _write_jsonl(examples: list[SyntheticExample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            f.write(json.dumps(ex.to_jsonl_record(i), ensure_ascii=False) + "\n")
