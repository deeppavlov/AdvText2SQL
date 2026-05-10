"""
Smoke-test для локального LM Studio inference.

Зачем: после Phase 4 (загрузка GGUF в LM Studio + Local Server) хочется
за 5 секунд проверить, что:
  1) LM Studio API живой по http://localhost:1234/v1
  2) Модель грузится и отвечает
  3) Chat template применяется правильно (Qwen использует ChatML)
  4) Тот же интерфейс, что autogen_ext.OpenAIChatCompletionClient использует
     в основном пайплайне — то есть если smoke-test прошёл, то bird_benchmark.py
     тоже заведётся

Запуск:
  uv run --env-file .env scripts/lm_studio_smoke_test.py

Учебный нюанс: тестируем именно через openai-клиент, а не через requests/curl.
Потому что финальный пайплайн использует OpenAI SDK через autogen — если бы
LM Studio неправильно эмулировал спеку, мы бы это словили на бенчмарке.
Здесь же ловим за один маленький запрос.
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI


def main() -> None:
    base_url = os.environ["LLM_BASE_URL"]
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL_NAME"]
    print(f"→ base_url = {base_url}")
    print(f"→ model    = {model}")

    if "localhost" not in base_url and "127.0.0.1" not in base_url:
        print("⚠  base_url не локальный. Точно ли запустил LM Studio Local Server?")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # 1) Список доступных моделей
    try:
        models = client.models.list()
        names = [m.id for m in models.data]
        print(f"✓ available models: {names}")
        if model not in names:
            print(f"⚠  '{model}' не в списке. LM Studio видит только то, что Loaded.")
    except Exception as e:
        print(f"✗ models.list() упало: {e}")
        sys.exit(1)

    # 2) Простой chat completion — проверка ChatML / роли работают
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты лаконичный SQL-ассистент. Отвечай только запросом без markdown."},
                {"role": "user", "content": "Напиши SQL: посчитай число строк в таблице users."},
            ],
            temperature=0,
            max_tokens=64,
        )
        out = resp.choices[0].message.content
        print(f"✓ chat reply: {out!r}")
        if "select" not in (out or "").lower():
            print("⚠  ответ не похож на SQL — возможно, chat template не применился")
    except Exception as e:
        print(f"✗ chat.completions.create упало: {e}")
        sys.exit(1)

    # 3) Token-count sanity check (LM Studio возвращает usage)
    if resp.usage:
        print(f"✓ usage: prompt={resp.usage.prompt_tokens}, "
              f"completion={resp.usage.completion_tokens}")
    else:
        print("⚠  usage отсутствует в ответе LM Studio")

    print("\n✓ smoke-test passed. Можно запускать bird_benchmark.py / ambrosia_benchmark.py")


if __name__ == "__main__":
    main()
