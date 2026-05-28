"""
heal/sql_generator — корректирующая генерация для failed-запросов.

Для каждого FailedQuery:
  1. Спрашиваем у GPT-4 правильный SQL (с полным контекстом схемы)
  2. Парсим/валидируем/исполняем на реальной БД
  3. Если SQL прошёл → добавляем в corrections.jsonl как "hard example" для FT

corrections.jsonl сливается с предыдущим train.jsonl и запускается retraining.
Это и есть self-healing loop: модель учится на своих собственных ошибках.

Стратегия: используем максимально сильную модель (GPT-4) для коррекции —
смысл в том чтобы получить ВЫСОКОКАЧЕСТВЕННЫЙ обучающий сигнал поверх плохих
исходных предсказаний нашей FT-модели.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from ..profiler.connector import DBConnector
from ..profiler.profile import Profile
from ..synth.template_generator import SyntheticExample
from ..synth.validator import check_execute, check_parse, check_whitelist
from .collector import FailedQuery

logger = logging.getLogger("text2sql_tool.heal.sql_generator")


CORRECTION_SYSTEM_PROMPT = """Ты — эксперт-аналитик PostgreSQL. Пользователь
прислал вопрос на естественном языке к БД с известной схемой. Модель
сгенерировала неверный SQL и он упал.

Твоя задача — написать ПРАВИЛЬНЫЙ SQL для вопроса. Используй схему
буквально (имена таблиц/колонок — точь-в-точь из схемы). Помни про PostgreSQL:
::numeric для приведения типов, NULLIF(x, 0) перед делением.

Отвечай ТОЛЬКО SQL-кодом, без markdown-фенсов и объяснений.
"""


CORRECTION_USER_PROMPT = """### Схема БД ({db_id}):
{schema_str}

### Отношения:
{relationships_str}

### Статистика:
{column_stats_str}

### Вопрос пользователя:
{question}

### Что было сгенерировано (СЛОМАНО):
{broken_sql}

### Ошибка от БД:
{error}

### Правильный SQL:
"""


@dataclass
class HealConfig:
    model_name: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    execute_validation: bool = True


@dataclass
class HealingResult:
    total_input: int
    healed: int
    still_broken: int
    corrections_path: Path

    @property
    def heal_rate(self) -> float:
        return self.healed / self.total_input if self.total_input else 0.0


class HealSQLGenerator:
    """Принимает FailedQuery → генерирует исправленный SQL → валидирует на БД."""

    def __init__(
        self,
        profile: Profile,
        db_url: str,
        config: HealConfig | None = None,
    ) -> None:
        self.profile = profile
        self.db_url = db_url
        self.config = config or HealConfig()
        self.client = AsyncOpenAI(
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            base_url=self.config.base_url,
        )

    async def heal(
        self,
        failed_queries: list[FailedQuery],
        out_path: str | Path,
    ) -> HealingResult:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Параллельные heal-вызовы (но не больше 5 одновременно)
        sem = asyncio.Semaphore(5)

        async def heal_one(fq: FailedQuery) -> SyntheticExample | None:
            async with sem:
                return await self._heal_single(fq)

        results = await asyncio.gather(*[heal_one(fq) for fq in failed_queries])
        healed = [r for r in results if r is not None]

        # Записываем corrections.jsonl
        with out_path.open("w", encoding="utf-8") as f:
            for i, ex in enumerate(healed):
                f.write(json.dumps(ex.to_jsonl_record(i), ensure_ascii=False) + "\n")

        return HealingResult(
            total_input=len(failed_queries),
            healed=len(healed),
            still_broken=len(failed_queries) - len(healed),
            corrections_path=out_path,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    async def _heal_single(self, fq: FailedQuery) -> SyntheticExample | None:
        user_prompt = CORRECTION_USER_PROMPT.format(
            db_id=self.profile.db_id,
            schema_str=self.profile.schema_str,
            relationships_str=self.profile.relationships_str,
            column_stats_str=self.profile.column_stats_str or "(нет)",
            question=fq.question,
            broken_sql=fq.generated_sql or "(пусто)",
            error=fq.error_message or "(нет деталей)",
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
            )
        except Exception as e:
            logger.exception(f"Correction LLM-call failed for question={fq.question!r}: {e}")
            return None

        sql = _strip_sql_fences(resp.choices[0].message.content or "")

        # Валидация: parse + whitelist + execute
        ok, reason = check_parse(sql)
        if not ok:
            logger.debug(f"Healed SQL doesn't parse: {reason}")
            return None
        ok, reason = check_whitelist(sql)
        if not ok:
            logger.debug(f"Healed SQL not whitelist-passing: {reason}")
            return None

        if self.config.execute_validation:
            with DBConnector(self.db_url) as conn:
                ok, reason = check_execute(conn, sql, min_rows=1)
            if not ok:
                logger.debug(f"Healed SQL execute failed: {reason}")
                return None

        return SyntheticExample(
            question=fq.question,
            sql=sql,
            db_id=self.profile.db_id,
            template_id="heal_correction",
            difficulty="challenging",   # это были failed-кейсы, явно не simple
        )


def _strip_sql_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        body = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        s = "\n".join(body)
    return s.strip()
