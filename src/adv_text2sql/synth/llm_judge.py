"""
LLMJudge — семантическая проверка пары (вопрос, SQL).

Execution-фильтр в `validator.py` отлавливает синтаксис и имена колонок, но
не семантику. Пример проблемы которую execution не ловит:

    Вопрос: "Сколько карт типа Creature?"
    SQL:    SELECT COUNT(*) FROM cards  -- считает ВСЕ карты, не только Creature
    Execute: возвращает 56822, без ошибки

Чтобы поймать такие случаи, отдаём пару LLM (GPT-4 или сильной локальной)
и спрашиваем: "правильно ли SQL отвечает на вопрос?". Это медленный шаг —
включается опциональным флагом `--judge` в CLI.

Стратегия: даём judge'у "доступ" к execution-результату — если SQL вернул
осмысленные данные (не 0 строк, не пустые ячейки), это сигнал в пользу
правильности. Иначе говоря: execution > judge при конфликте.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger("text2sql_tool.synth.judge")


JUDGE_SYSTEM_PROMPT = """Ты — независимый эксперт по Text2SQL.

Тебе показывают вопрос пользователя и SQL-запрос. Твоя задача — оценить,
действительно ли этот SQL отвечает на вопрос.

Критерии:
  - Правильные таблицы и колонки выбраны
  - Правильные фильтры (WHERE) и агрегации
  - Корректные JOIN-условия
  - Не упущены ли важные ограничения из вопроса

Отвечай СТРОГО в JSON:
{"verdict": "good" | "bad", "reason": "одна короткая фраза"}
"""


JUDGE_USER_PROMPT = """Вопрос: {question}

SQL:
{sql}

{execution_hint}

Соответствует ли SQL вопросу?
"""


@dataclass
class JudgeConfig:
    model_name: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0


@dataclass
class JudgeVerdict:
    is_good: bool
    reason: str


class LLMJudge:
    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        self.client = AsyncOpenAI(
            api_key=self.config.api_key or "dummy",
            base_url=self.config.base_url,
        )

    async def judge(
        self,
        question: str,
        sql: str,
        execution_result_summary: str | None = None,
    ) -> JudgeVerdict:
        """Один вызов LLM → verdict.

        execution_result_summary — если есть, передаётся judge'у как подсказка
        (напр. "запрос вернул 56822 строки"). Помогает отличить
        семантически правильный SQL от формально валидного но бессмысленного.
        """
        execution_hint = (
            f"Результат исполнения: {execution_result_summary}"
            if execution_result_summary
            else "(результат исполнения недоступен)"
        )
        user = JUDGE_USER_PROMPT.format(
            question=question, sql=sql, execution_hint=execution_hint
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=self.config.temperature,
            )
            content = resp.choices[0].message.content or ""
            parsed = _parse_judge_json(content)
            return JudgeVerdict(
                is_good=parsed.get("verdict", "bad") == "good",
                reason=parsed.get("reason", "no reason"),
            )
        except Exception as e:
            logger.exception(f"Judge call failed: {e}")
            # Fail-open: если judge упал, не блокируем пример
            return JudgeVerdict(is_good=True, reason=f"judge_error: {e}")


def _parse_judge_json(text: str) -> dict:
    """Извлечь {"verdict": ..., "reason": ...} из ответа judge'а."""
    # Снять markdown-фенсы
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"```\s*$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in judge response: {text[:200]}")
    return json.loads(match.group(0))
