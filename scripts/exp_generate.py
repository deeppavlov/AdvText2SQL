#!/usr/bin/env python3
"""
exp_generate — ЭТАП 1 (локально): генерация + валидация синтетики для ВСЕХ
экспериментов из реестра, по одному скрипту.

Для каждого эксперимента: для каждого языка генерирует count_per_lang Q-SQL пар
(LLM), валидирует на реальной БД, затем объединяет валидные всех языков в
data/exp/<name>/validated.jsonl.

Требует: SSH-туннель на localhost:5444, LLM_* env vars, DB_USER/DB_PASS.

Запуск:
    uv run --env-file .env python scripts/exp_generate.py
    uv run --env-file .env python scripts/exp_generate.py --only gpt41_en   # один эксперимент
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _registry import REGISTRY_PATH, Experiment, load_registry  # noqa: E402

from adv_text2sql.profiler.profile import Profile  # noqa: E402
from adv_text2sql.synth.cli import run_generate  # noqa: E402


def _db_url() -> str:
    return (
        f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PASS']}"
        f"@localhost:5444/card_games"
    )


def generate_one(exp: Experiment, profile: Profile, profile_path: str, db_url: str) -> int:
    """Сгенерировать + валидировать все языки эксперимента, объединить. Возврат: N валидных."""
    exp.dir.mkdir(parents=True, exist_ok=True)
    merged: list[dict] = []

    count_per_lang = exp.resolved_count_per_lang(profile)
    mode = "auto" if exp.is_adaptive else "fixed"

    for lang in exp.languages:
        print(
            f"\n  ── {exp.name} / {lang} "
            f"({count_per_lang} пар [{mode}], {exp.llm_model}) ──"
        )
        lang_dir = exp.dir / f"_{lang}"
        stats = run_generate(
            profile_path=profile_path,
            count=str(count_per_lang),
            generator="llm",          # LLM двуязычный; template — только ru
            llm_model=exp.llm_model,
            judge=False,
            auto_resize=False,
            out_dir=str(lang_dir),
            db_url=db_url,
            language=lang,
        )
        passed_path = Path(stats["passed_path"])
        if passed_path.exists():
            for line in passed_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    merged.append(json.loads(line))
        print(f"     валидных {lang}: {stats['passed']} ({stats['pass_rate']*100:.0f}%)")

    # Объединённый validated.jsonl
    with exp.validated_path.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ {exp.name}: {len(merged)} валидных → {exp.validated_path}")
    return len(merged)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="имя одного эксперимента из реестра")
    ap.add_argument(
        "--registry", default=str(REGISTRY_PATH),
        help="путь к registry.json (для отдельных наборов экспериментов)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="пересчитать даже те, у кого уже есть validated.jsonl",
    )
    args = ap.parse_args()

    reg = load_registry(args.registry)
    db_url = _db_url()
    profile = Profile.load_json(reg.profile_path)
    exps = [e for e in reg.experiments if not args.only or e.name == args.only]
    if not exps:
        print(f"Нет эксперимента '{args.only}' в реестре")
        return 1

    print(f"Генерация {len(exps)} экспериментов, db={reg.db_id}")
    for exp in exps:
        # resume: пропускаем уже сгенерированные, чтобы не тратить токены заново
        if not args.force and exp.validated_path.exists():
            n = sum(1 for _ in exp.validated_path.open())
            print(f"\n  ⏭  {exp.name}: уже есть {n} валидных — пропускаю (--force чтобы пересчитать)")
            continue
        generate_one(exp, profile, reg.profile_path, db_url)

    print("\n✓ Готово. Дальше: scripts/exp_build.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
