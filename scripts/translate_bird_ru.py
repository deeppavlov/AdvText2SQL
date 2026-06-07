#!/usr/bin/env python3
"""
translate_bird_ru — переводит BIRD card_games вопросы на русский.

Зачем: проверить гипотезу языкового рассинхрона из run1. Старые адаптеры
обучались на русской синтетике → должны лучше работать на русских вопросах.
SQL (gold) не меняем — он PostgreSQL-валидный независимо от языка вопроса.

Запуск:
    uv run --env-file .env python scripts/translate_bird_ru.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

SRC = Path("data/bird_large.json")
DST = Path("data/bird_large_ru.json")
DB_ID = "card_games"

SYSTEM = """Ты — профессиональный переводчик технических вопросов.
Переведи вопрос аналитика к БД с английского на русский язык.

Требования:
- Сохрани термины domain (cards, sets, foil, mythic, Commander, MTGO и т.п.)
- Имена собственных (John Avon, Coldsnap, Tendo Ice Bridge) — НЕ переводи
- Технические термины SQL не упрощай
- Естественный русский, как от живого аналитика

Верни ТОЛЬКО перевод, без пояснений."""


def translate(client: OpenAI, model: str, text: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip().strip('"').strip("'")


def main() -> int:
    if not SRC.exists():
        print(f"✗ нет {SRC}")
        return 1

    client = OpenAI(
        base_url=os.environ.get("LLM_BASE_URL"),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    # gpt-4.1 умеет chat.completions; LLM_MODEL_NAME из .env может быть codex-mini
    # (только Responses API), поэтому фиксируем gpt-4.1 как переводчик.
    model = "gpt-4.1"

    bird = json.loads(SRC.read_text(encoding="utf-8"))
    out = []
    target = [q for q in bird if q.get("db_id") == DB_ID]
    others = [q for q in bird if q.get("db_id") != DB_ID]
    print(f"Переводим {len(target)} вопросов {DB_ID} через {model}…")

    for i, q in enumerate(target, 1):
        q_ru = translate(client, model, q["question"])
        ev_ru = translate(client, model, q["evidence"]) if q.get("evidence") else ""
        new_q = dict(q)
        new_q["question"] = q_ru
        new_q["evidence"] = ev_ru
        new_q["question_en"] = q["question"]  # сохраняем оригинал для дебага
        out.append(new_q)
        print(f"  [{i}/{len(target)}] {q_ru[:60]}")

    # сохраняем переведённые + остальные db_id без изменений
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(
        json.dumps(out + others, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✓ {len(out)} переведено → {DST}")
    print(f"  Загрузи на Drive: MyDrive/text2sql/bird_large_ru.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
