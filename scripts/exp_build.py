#!/usr/bin/env python3
"""
exp_build — ЭТАП 2 (локально): сборка train/val датасетов для ВСЕХ экспериментов.

Берёт data/exp/<name>/validated.jsonl → строит chat-format train.jsonl/val.jsonl
в той же папке (один и тот же системный промпт = инвариант train==inference).

Запуск:
    uv run --env-file .env python scripts/exp_build.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _registry import load_registry  # noqa: E402

from adv_text2sql.profiler.profile import Profile  # noqa: E402
from adv_text2sql.training.dataset_builder import DatasetBuilder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="имя одного эксперимента")
    ap.add_argument(
        "--force", action="store_true",
        help="пересобрать даже те, у кого уже есть train.jsonl",
    )
    args = ap.parse_args()

    reg = load_registry()
    profile = Profile.load_json(reg.profile_path)
    exps = [e for e in reg.experiments if not args.only or e.name == args.only]

    for exp in exps:
        if not exp.validated_path.exists():
            print(f"⚠  {exp.name}: нет {exp.validated_path} — сначала exp_generate.py")
            continue
        if not args.force and exp.train_path.exists():
            print(f"⏭  {exp.name}: уже собран train.jsonl — пропускаю (--force чтобы пересобрать)")
            continue
        # DatasetBuilder кладёт в out_dir/<db_id>/ — кладём прямо в exp.dir
        builder = DatasetBuilder(profile)
        stats = builder.build(exp.validated_path, exp.dir)
        # builder создаёт подпапку <db_id>/ — переносим файлы на уровень выше
        sub = exp.dir / reg.db_id
        for fname in ("train.jsonl", "val.jsonl"):
            src = sub / fname
            if src.exists():
                src.replace(exp.dir / fname)
        if sub.exists() and not any(sub.iterdir()):
            sub.rmdir()
        print(f"✓ {exp.name}: train={stats.train_count} val={stats.val_count} → {exp.dir}")

    print("\n✓ Датасеты собраны. Дальше: загрузи data/exp/<name>/ на Drive и запусти обучение.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
