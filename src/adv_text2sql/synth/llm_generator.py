"""
LLMSyntheticGenerator — Q-SQL пары через LLM.

Что это даёт по сравнению с TemplateSyntheticGenerator:
  - **Натуральные** вопросы (человеческий язык, без шаблонных конструкций)
  - **Семантическое разнообразие** — модель сама придумывает разные углы
  - **Сложные конструкции** — CTE, оконные функции, подзапросы, которых нет
    в шаблонах

Что хуже:
  - Дороже (требует API-вызовы)
  - Галлюцинации (имена колонок, синтаксические ошибки) → нужна валидация
  - Зависимость от качества генератора (см. эксперимент с разными моделями)

Стратегия батчинга: просим за один вызов сразу 20 пар → быстрее, дешевле,
LLM лучше разнообразит ответы видя другие свои выходы.

Формат ответа: JSON-array, парсится через json.loads. При невалидном JSON —
повторяем 2 раза, потом дроп батча.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from ..profiler.profile import Profile
from .template_generator import SyntheticExample

logger = logging.getLogger("text2sql_tool.synth.llm")


# Models that ONLY work with /v1/responses, NOT /v1/chat/completions
# Source: OpenAI API docs — Codex and select newer models use the Responses API
_RESPONSES_API_MODELS: frozenset[str] = frozenset(
    {
        "codex-mini-latest",
        "gpt-5.1-codex-mini",
        "o3-mini",
    }
)


def _needs_responses_api(model: str) -> bool:
    """Return True if this model requires /v1/responses instead of /v1/chat/completions."""
    return model in _RESPONSES_API_MODELS or "codex" in model.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────


# Язык вопросов — критичный параметр: модель должна обучаться на том же языке,
# на котором будет тестироваться (инвариант train==inference по языку).
_LANGUAGE_RULE = {
    "ru": "Вопросы — на РУССКОМ языке, естественные, как от реального аналитика",
    "en": "Questions — in ENGLISH, natural, as from a real data analyst",
}


SYNTHESIS_SYSTEM_PROMPT = """Ты — генератор обучающих данных для Text2SQL модели.

Тебе даётся схема PostgreSQL БД с колонками, отношениями и статистикой.
Твоя задача — сгенерировать {batch_size} разнообразных пар (вопрос, SQL).

ТРЕБОВАНИЯ:
1. {language_rule}
2. SQL — корректный PostgreSQL (НЕ SQLite!): используй ::numeric, ::date, NULLIF
3. Все имена таблиц и колонок — ТОЛЬКО из предоставленной схемы
4. Используй реальные значения из low_cardinality_values где это уместно
5. Разнообразие сложности: ~40% simple, ~40% moderate, ~20% challenging
6. Разнообразие категорий: lookup, count, top-N, JOIN, GROUP BY, агрегаты, даты

ФОРМАТ ВЫВОДА — JSON array, без markdown-фенсов:
[
  {{"question": "...", "sql": "...", "difficulty": "simple"}},
  ...
]
"""


SYNTHESIS_USER_PROMPT = """### Схема БД ({db_id}):
{schema_str}

### Отношения:
{relationships_str}

### Статистика колонок:
{column_stats_str}

### Sample-значения для grounding:
{low_cardinality_block}

Сгенерируй {batch_size} разнообразных пар (вопрос, SQL).
"""


# ─────────────────────────────────────────────────────────────────────────────
# Generator class
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMGeneratorConfig:
    model_name: str = "gpt-4o"
    base_url: str | None = None  # None → дефолт OpenAI; иначе — LM Studio, OpenRouter и т.д.
    api_key: str | None = None
    temperature: float = 0.8
    batch_size: int = 20
    max_retries_per_batch: int = 2
    language: str = "ru"  # "ru" | "en" — язык генерируемых вопросов
    # Параллельных запросов. 1 = последовательно (чистые логи, не упирается в TPM
    # лимит gpt-4.1 ~30K/мин при ~5K токенов/батч). Подними для моделей с большим лимитом.
    max_concurrency: int = 1


class LLMSyntheticGenerator:
    """Async LLM-генератор. Использует тот же OpenAI-клиент, что и BenchmarkBase."""

    def __init__(
        self,
        profile: Profile,
        config: LLMGeneratorConfig | None = None,
    ) -> None:
        self.profile = profile
        self.config = config or LLMGeneratorConfig()
        self.client = AsyncOpenAI(
            api_key=self.config.api_key or "dummy",
            base_url=self.config.base_url,
        )

    async def generate(self, target_count: int) -> list[SyntheticExample]:
        """Сгенерировать ≥target_count примеров, батчами по batch_size."""
        n_batches = (target_count + self.config.batch_size - 1) // self.config.batch_size
        logger.info(
            f"LLM generation: {target_count} target, {n_batches} batches "
            f"of {self.config.batch_size} via {self.config.model_name}"
        )

        results: list[SyntheticExample] = []
        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def run_batch(batch_idx: int) -> list[SyntheticExample]:
            async with semaphore:
                return await self._generate_one_batch(batch_idx)

        batches = await asyncio.gather(
            *[run_batch(i) for i in range(n_batches)], return_exceptions=True
        )
        for b in batches:
            if isinstance(b, Exception):
                logger.exception("Batch failed", exc_info=b)
                continue
            results.extend(b)

        logger.info(f"LLM generation done: {len(results)} examples produced")
        return results

    # ── Internals ────────────────────────────────────────────────────────────

    async def _generate_one_batch(self, batch_idx: int) -> list[SyntheticExample]:
        lang_rule = _LANGUAGE_RULE.get(self.config.language, _LANGUAGE_RULE["ru"])
        system_prompt = SYNTHESIS_SYSTEM_PROMPT.format(
            batch_size=self.config.batch_size, language_rule=lang_rule
        )
        user_prompt = SYNTHESIS_USER_PROMPT.format(
            db_id=self.profile.db_id,
            schema_str=self.profile.schema_str,
            relationships_str=self.profile.relationships_str,
            column_stats_str=self.profile.column_stats_str or "(статистика недоступна)",
            low_cardinality_block=_format_low_cardinality(
                self.profile.low_cardinality_values
            ),
            batch_size=self.config.batch_size,
        )

        use_responses = _needs_responses_api(self.config.model_name)

        for attempt in range(self.config.max_retries_per_batch + 1):
            try:
                content = await self._call_llm(system_prompt, user_prompt, use_responses)
                parsed = _parse_json_array(content)
                return [
                    SyntheticExample(
                        question=item["question"],
                        sql=item["sql"],
                        db_id=self.profile.db_id,
                        template_id=f"llm_batch_{batch_idx}",
                        difficulty=item.get("difficulty", "moderate"),
                    )
                    for item in parsed
                    if "question" in item and "sql" in item
                ]
            except Exception as e:
                err_str = str(e)
                # Auto-detect Responses API requirement from the error message
                if "only supported in v1/responses" in err_str and not use_responses:
                    logger.info(
                        f"Model {self.config.model_name!r} requires Responses API — switching"
                    )
                    use_responses = True
                    continue  # retry immediately with right API
                if attempt == self.config.max_retries_per_batch:
                    logger.exception(f"Batch {batch_idx} failed after retries: {e}")
                    return []
                # 429 rate-limit: parse wait time from error message, sleep
                wait = 15.0  # default backoff
                if "rate_limit_exceeded" in err_str or "429" in err_str:
                    import re as _re
                    m = _re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                    wait = float(m.group(1)) + 2.0 if m else 20.0
                    logger.warning(
                        f"Batch {batch_idx} rate-limited, sleeping {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.warning(
                        f"Batch {batch_idx} attempt {attempt} failed: {e}, retrying"
                    )

        return []

    async def _call_llm(
        self, system_prompt: str, user_prompt: str, use_responses_api: bool
    ) -> str:
        """Call the right OpenAI endpoint and return the text content."""
        if use_responses_api:
            # /v1/responses — used by Codex and select newer OpenAI models
            resp = await self.client.responses.create(
                model=self.config.model_name,
                instructions=system_prompt,
                input=user_prompt,
            )
            # Output is in resp.output_text (convenience attr) or nested structure
            if hasattr(resp, "output_text"):
                return resp.output_text or ""
            # Fallback: traverse output → content
            for block in getattr(resp, "output", []):
                for part in getattr(block, "content", []):
                    if getattr(part, "type", None) == "output_text":
                        return getattr(part, "text", "") or ""
            return ""
        else:
            # /v1/chat/completions — standard path for most models
            resp = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
            )
            return resp.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Извлечь JSON-array из ответа LLM. Терпимо к markdown-фенсам и тексту вокруг."""
    # Снять markdown-фенсы если есть
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"```\s*$", "", cleaned)
    # Найти первый [...] блок
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in LLM response: {text[:200]}")
    return json.loads(match.group(0))


def _format_low_cardinality(
    low_card: dict[str, list[Any]], max_columns: int = 20, max_values: int = 8
) -> str:
    """Компактная сериализация enum-колонок для промпта.

    Без ограничения промпт раздувается на больших схемах. 20 enum-колонок
    хватает для grounding-а большинства генерируемых примеров.
    """
    if not low_card:
        return "(нет low-cardinality колонок)"

    parts: list[str] = []
    for i, (key, values) in enumerate(low_card.items()):
        if i >= max_columns:
            parts.append(f"... и ещё {len(low_card) - max_columns} колонок")
            break
        truncated = values[:max_values]
        suffix = "..." if len(values) > max_values else ""
        formatted_vals = ", ".join(repr(v) for v in truncated)
        parts.append(f"  {key}: [{formatted_vals}{suffix}]")
    return "\n".join(parts)
