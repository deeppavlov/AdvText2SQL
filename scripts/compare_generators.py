#!/usr/bin/env python3
"""
compare_generators — финальное сравнение генераторов синтетики.

На вход: predictions JSON от каждого эксперимента (созданы в Colab инференсом
FT-адаптера на BIRD card_games subset). Каждый файл:
    [{"question_id", "question", "gold_sql", "predicted_sql", "difficulty"}, ...]

Что делает: исполняет gold + predicted SQL на реальной PG (через туннель),
сравнивает result sets, считает execution accuracy. Переиспользует
`compare_result_sets` из eval/runner — та же логика что в `text2sql eval`.

Запуск:
    uv run --env-file .env python scripts/compare_generators.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _registry import load_registry  # noqa: E402

from adv_text2sql.eval.runner import compare_result_sets  # noqa: E402

# Имена экспериментов берём из реестра + baseline (база без FT).
# Predictions ищем в data/exp/<name>/eval_predictions.json (единое соглашение).
REPORT_PATH = Path("experiments/generator_comparison.md")


def _experiments() -> dict[str, str]:
    """{имя эксперимента → человекочитаемая метка}. baseline всегда первый.

    Также подхватывает RU-эксперименты (папки с суффиксом _ru) — это прогоны
    старых run1-адаптеров на русском BIRD для проверки языковой гипотезы.
    """
    reg = load_registry()
    out = {"baseline": "Qwen-Coder-7B (без FT)"}
    for e in reg.experiments:
        out[e.name] = f"{e.llm_model} [{'+'.join(e.languages)}]"
    # Авто-подхват RU-папок: data/exp/<X>_ru/eval_predictions.json
    exp_root = Path("data/exp")
    if exp_root.exists():
        for sub in sorted(exp_root.iterdir()):
            if sub.is_dir() and sub.name.endswith("_ru") and (sub / "eval_predictions.json").exists():
                if sub.name not in out:
                    out[sub.name] = f"{sub.name} (RU BIRD)"
    return out


def _predictions_path(exp: str) -> Path:
    """Сначала ищем в data/exp/<exp>/, потом fallback в experiments/predictions/."""
    p1 = Path("data/exp") / exp / "eval_predictions.json"
    if p1.exists():
        return p1
    return Path("experiments/predictions") / f"{exp}.json"


def _db_url() -> str:
    return (
        f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PASS']}"
        f"@localhost:5444/card_games"
    )


def evaluate_one(db_url: str, predictions: list[dict]) -> dict:
    """Прогнать один эксперимент: вернуть accuracy + разбивку по difficulty."""
    n_total = len(predictions)
    n_correct = 0
    n_error = 0
    by_diff: dict[str, list[bool]] = {}

    for p in predictions:
        gold = p.get("gold_sql", "")
        pred = (p.get("predicted_sql", "") or "").strip()
        passed, err = compare_result_sets(db_url, gold, pred)
        if err and pred not in ("", "error"):
            n_error += 1
        if passed:
            n_correct += 1
        diff = p.get("difficulty", "unknown")
        by_diff.setdefault(diff, []).append(passed)

    return {
        "total": n_total,
        "correct": n_correct,
        "accuracy": round(n_correct / n_total * 100, 1) if n_total else 0.0,
        "errors": n_error,
        "by_difficulty": {
            d: round(sum(v) / len(v) * 100, 1) for d, v in sorted(by_diff.items())
        },
    }


def main() -> int:
    db_url = _db_url()
    results: dict[str, dict] = {}

    for exp, gen_name in _experiments().items():
        pred_path = _predictions_path(exp)
        if not pred_path.exists():
            print(f"⚠  {exp}: нет {pred_path} — пропускаю (скачай из Colab/Drive)")
            continue
        predictions = json.loads(pred_path.read_text(encoding="utf-8"))
        print(f"\n=== {gen_name} ({exp}) — {len(predictions)} вопросов ===")
        res = evaluate_one(db_url, predictions)
        results[gen_name] = res
        print(
            f"  accuracy = {res['accuracy']}%  "
            f"({res['correct']}/{res['total']}), errors={res['errors']}"
        )
        print(f"  by difficulty: {res['by_difficulty']}")

    if not results:
        print("\nНет ни одного predictions-файла. Сначала прогони инференс в Colab.")
        return 1

    _write_report(results)
    print(f"\n✓ Отчёт записан → {REPORT_PATH}")
    return 0


def _write_report(results: dict[str, dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Сравнение генераторов синтетики",
        "",
        "Каждый генератор создал синтетический Q-SQL датасет для `card_games`, "
        "на котором дообучен LoRA-адаптер поверх Qwen2.5-Coder-7B. "
        "Метрика — execution accuracy на BIRD card_games subset (реальные вопросы).",
        "",
        "| Генератор | Accuracy | Correct/Total | Errors |",
        "|---|---|---|---|",
    ]
    for gen, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        lines.append(
            f"| {gen} | **{r['accuracy']}%** | {r['correct']}/{r['total']} | {r['errors']} |"
        )

    lines += ["", "## Accuracy по сложности", "", "| Генератор | " ]
    all_diffs = sorted({d for r in results.values() for d in r["by_difficulty"]})
    lines[-1] += " | ".join(all_diffs) + " |"
    lines.append("|---|" + "|".join(["---"] * len(all_diffs)) + "|")
    for gen, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        cells = [str(r["by_difficulty"].get(d, "—")) for d in all_diffs]
        lines.append(f"| {gen} | " + " | ".join(cells) + " |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
