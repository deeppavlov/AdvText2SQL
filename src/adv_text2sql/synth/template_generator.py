"""
TemplateSyntheticGenerator — детерминированная генерация Q-SQL пар по шаблонам.

Каждый шаблон = `(question_template, sql_template, slot_rules)`. Шаблоны
параметризуются через `{table}`, `{column}`, `{value}` и т.д. — слоты
заполняются реальными именами из `Profile.column_info` и значениями из
`Profile.low_cardinality_values`.

Преимущество над LLM-генератором:
  - 100% синтаксически валидный SQL (никаких галлюцинаций имён колонок)
  - Бесплатно (без API-вызовов)
  - Управляемое разнообразие сложности (категории: lookup, count, top-N, JOIN, etc.)

Недостаток: ограниченное "семантическое" разнообразие — естественность
вопросов хуже чем у LLM. Поэтому в production миксуем template + LLM.

Категории и количество шаблонов:
  - lookup-by-pk        (3)
  - count-by-filter     (4)
  - top-n-by-aggregate  (4)
  - range-filter        (3)
  - join-then-filter    (4)
  - groupby-having      (3)
  - date-range          (3)
  - aggregate-numeric   (4)
  - distinct-categorical (2)
Итого ~30 шаблонов.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Iterator

from ..profiler.profile import Profile

logger = logging.getLogger("text2sql_tool.synth.template")


@dataclass
class SyntheticExample:
    """Одна сгенерированная пара (question, SQL) с метаданными."""

    question: str
    sql: str
    db_id: str
    template_id: str          # для дебага: какой шаблон сработал
    difficulty: str           # "simple" | "moderate" | "challenging"

    def to_jsonl_record(self, idx: int) -> dict[str, Any]:
        """Преобразует в формат, совместимый с data/synthetic/*.jsonl."""
        return {
            "question_id": f"template_{self.db_id}_{idx}",
            "db_id": self.db_id,
            "question": self.question,
            "SQL": self.sql,
            "task": "sql",
            "difficulty": self.difficulty,
            "_meta": {"source": "template", "template_id": self.template_id},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Templates organized by category
# ─────────────────────────────────────────────────────────────────────────────


class TemplateRegistry:
    """Каталог SQL-шаблонов. Шаблон возвращает либо `SyntheticExample`, либо None,
    если для конкретного Profile его применить нельзя (нет подходящих колонок)."""

    @staticmethod
    def lookup_by_pk(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        """Простейший lookup: SELECT * FROM t WHERE pk = <value>."""
        candidates = [
            (t, cols[0])
            for t, cols in profile.relationships.get("primary_keys", {}).items()
            if cols
        ]
        if not candidates:
            return None
        table, pk = rng.choice(candidates)
        # Пытаемся взять реальный pk-value из sample_rows
        sample = profile.sample_rows.get(table, [])
        if not sample:
            return None
        pk_value = sample[0].get(pk)
        if pk_value is None:
            return None

        pk_value_sql = _format_sql_value(pk_value)
        return SyntheticExample(
            question=f"Покажи всю информацию из таблицы {table} с {pk} = {pk_value}.",
            sql=f'SELECT * FROM "{table}" WHERE "{pk}" = {pk_value_sql} LIMIT 10',
            db_id=profile.db_id,
            template_id="lookup_by_pk",
            difficulty="simple",
        )

    @staticmethod
    def count_all_rows(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        if not profile.tables:
            return None
        table = rng.choice(profile.tables)
        return SyntheticExample(
            question=f"Сколько всего строк в таблице {table}?",
            sql=f'SELECT COUNT(*) FROM "{table}"',
            db_id=profile.db_id,
            template_id="count_all_rows",
            difficulty="simple",
        )

    @staticmethod
    def count_by_categorical(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        """Берёт low-cardinality колонку и считает строки по конкретному значению."""
        if not profile.low_cardinality_values:
            return None
        key = rng.choice(list(profile.low_cardinality_values.keys()))
        table, col = key.split(".", 1)
        values = profile.low_cardinality_values[key]
        if not values:
            return None
        val = rng.choice(values)
        val_sql = _format_sql_value(val)
        return SyntheticExample(
            question=f"Сколько записей в таблице {table}, где {col} равно {val}?",
            sql=f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" = {val_sql}',
            db_id=profile.db_id,
            template_id="count_by_categorical",
            difficulty="simple",
        )

    @staticmethod
    def top_n_by_numeric(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        """Top-N по числовой колонке. Использует column_stats для выбора numeric колонок."""
        numeric_cols = _columns_by_type(profile, ("integer", "numeric"))
        if not numeric_cols:
            return None
        table, col = rng.choice(numeric_cols)
        n = rng.choice([3, 5, 10])
        return SyntheticExample(
            question=f"Топ-{n} записей таблицы {table} с самым большим {col}.",
            sql=f'SELECT * FROM "{table}" ORDER BY "{col}"::numeric DESC NULLS LAST LIMIT {n}',
            db_id=profile.db_id,
            template_id="top_n_by_numeric",
            difficulty="simple",
        )

    @staticmethod
    def avg_numeric(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        numeric_cols = _columns_by_type(profile, ("integer", "numeric"))
        if not numeric_cols:
            return None
        table, col = rng.choice(numeric_cols)
        return SyntheticExample(
            question=f"Какое среднее значение {col} в таблице {table}?",
            sql=f'SELECT AVG("{col}"::numeric) FROM "{table}" WHERE "{col}" IS NOT NULL',
            db_id=profile.db_id,
            template_id="avg_numeric",
            difficulty="simple",
        )

    @staticmethod
    def min_max_numeric(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        numeric_cols = _columns_by_type(profile, ("integer", "numeric"))
        if not numeric_cols:
            return None
        table, col = rng.choice(numeric_cols)
        agg = rng.choice(["MIN", "MAX"])
        word = "минимальное" if agg == "MIN" else "максимальное"
        return SyntheticExample(
            question=f"Какое {word} значение {col} в таблице {table}?",
            sql=f'SELECT {agg}("{col}"::numeric) FROM "{table}"',
            db_id=profile.db_id,
            template_id="min_max_numeric",
            difficulty="simple",
        )

    @staticmethod
    def range_filter_numeric(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        numeric_cols = _columns_by_type(profile, ("integer", "numeric"))
        if not numeric_cols:
            return None
        table, col = rng.choice(numeric_cols)
        stats = profile.column_stats.get(table, {}).get("columns", {}).get(col, {}).get("stats") or {}
        mn = stats.get("min", 0)
        mx = stats.get("max", 100)
        try:
            mid = (float(mn) + float(mx)) / 2
        except (TypeError, ValueError):
            return None
        return SyntheticExample(
            question=f"Найди все строки в {table}, где {col} больше {mid:.2f}.",
            sql=f'SELECT * FROM "{table}" WHERE "{col}"::numeric > {mid:.2f} LIMIT 100',
            db_id=profile.db_id,
            template_id="range_filter_numeric",
            difficulty="moderate",
        )

    @staticmethod
    def date_range(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        date_cols = _columns_by_type(profile, ("date",))
        if not date_cols:
            return None
        table, col = rng.choice(date_cols)
        stats = profile.column_stats.get(table, {}).get("columns", {}).get(col, {}).get("stats") or {}
        mn = stats.get("min")
        mx = stats.get("max")
        if not mn or not mx:
            return None
        return SyntheticExample(
            question=f"Покажи строки из {table} с {col} между {mn} и {mx}.",
            sql=(
                f'SELECT * FROM "{table}" '
                f"WHERE \"{col}\"::date BETWEEN '{mn}'::date AND '{mx}'::date LIMIT 50"
            ),
            db_id=profile.db_id,
            template_id="date_range",
            difficulty="moderate",
        )

    @staticmethod
    def join_then_filter(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        """JOIN по FK + фильтр на categorical-колонке."""
        fks = profile.relationships.get("foreign_keys", [])
        if not fks:
            return None
        fk = rng.choice(fks)
        from_t = fk["from_table"]
        to_t = fk["to_table"]
        from_c = fk["from_cols"][0]
        to_c = fk["to_cols"][0]
        return SyntheticExample(
            question=f"Сколько строк в {from_t} связано с {to_t} (через {from_c})?",
            sql=(
                f'SELECT COUNT(*) FROM "{from_t}" t1 '
                f'INNER JOIN "{to_t}" t2 ON t1."{from_c}" = t2."{to_c}"'
            ),
            db_id=profile.db_id,
            template_id="join_then_filter",
            difficulty="moderate",
        )

    @staticmethod
    def groupby_count(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        if not profile.low_cardinality_values:
            return None
        key = rng.choice(list(profile.low_cardinality_values.keys()))
        table, col = key.split(".", 1)
        return SyntheticExample(
            question=f"Покажи распределение {col} в таблице {table} (количество строк по каждому значению).",
            sql=(
                f'SELECT "{col}", COUNT(*) AS cnt FROM "{table}" '
                f'WHERE "{col}" IS NOT NULL GROUP BY "{col}" ORDER BY cnt DESC'
            ),
            db_id=profile.db_id,
            template_id="groupby_count",
            difficulty="moderate",
        )

    @staticmethod
    def groupby_having(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        if not profile.low_cardinality_values:
            return None
        key = rng.choice(list(profile.low_cardinality_values.keys()))
        table, col = key.split(".", 1)
        threshold = rng.choice([5, 10, 50, 100])
        return SyntheticExample(
            question=f"Какие значения {col} в таблице {table} встречаются больше {threshold} раз?",
            sql=(
                f'SELECT "{col}", COUNT(*) AS cnt FROM "{table}" '
                f'GROUP BY "{col}" HAVING COUNT(*) > {threshold} ORDER BY cnt DESC'
            ),
            db_id=profile.db_id,
            template_id="groupby_having",
            difficulty="challenging",
        )

    @staticmethod
    def distinct_categorical(profile: Profile, rng: random.Random) -> SyntheticExample | None:
        if not profile.low_cardinality_values:
            return None
        key = rng.choice(list(profile.low_cardinality_values.keys()))
        table, col = key.split(".", 1)
        return SyntheticExample(
            question=f"Какие уникальные значения {col} есть в таблице {table}?",
            sql=f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL',
            db_id=profile.db_id,
            template_id="distinct_categorical",
            difficulty="simple",
        )


# Порядок шаблонов задаёт распределение сложностей в итоговом датасете
ALL_TEMPLATES = [
    TemplateRegistry.lookup_by_pk,
    TemplateRegistry.count_all_rows,
    TemplateRegistry.count_by_categorical,
    TemplateRegistry.top_n_by_numeric,
    TemplateRegistry.avg_numeric,
    TemplateRegistry.min_max_numeric,
    TemplateRegistry.range_filter_numeric,
    TemplateRegistry.date_range,
    TemplateRegistry.join_then_filter,
    TemplateRegistry.groupby_count,
    TemplateRegistry.groupby_having,
    TemplateRegistry.distinct_categorical,
]


# ─────────────────────────────────────────────────────────────────────────────
# Main generator class
# ─────────────────────────────────────────────────────────────────────────────


class TemplateSyntheticGenerator:
    """Генерирует Q-SQL пары по шаблонам, использует Profile для grounding."""

    def __init__(self, profile: Profile, seed: int = 42) -> None:
        self.profile = profile
        self.rng = random.Random(seed)

    def generate(self, target_count: int) -> list[SyntheticExample]:
        """Сгенерировать target_count примеров. Может вернуть меньше если шаблоны
        не применимы (например, нет numeric-колонок → top-N не работает)."""
        out: list[SyntheticExample] = []
        attempts = 0
        max_attempts = target_count * 3  # защита от бесконечного цикла

        while len(out) < target_count and attempts < max_attempts:
            template_fn = self.rng.choice(ALL_TEMPLATES)
            example = template_fn(self.profile, self.rng)
            if example is not None:
                out.append(example)
            attempts += 1

        if len(out) < target_count:
            logger.warning(
                f"Generated only {len(out)}/{target_count} examples — "
                f"не все шаблоны применимы к этому Profile"
            )
        return out

    def generate_iter(self, target_count: int) -> Iterator[SyntheticExample]:
        """Streaming-вариант: yields по одному примеру."""
        yield from self.generate(target_count)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _columns_by_type(profile: Profile, types: tuple[str, ...]) -> list[tuple[str, str]]:
    """Вернуть [(table, column), ...] для колонок указанных detected_type."""
    out: list[tuple[str, str]] = []
    for table, tinfo in profile.column_stats.items():
        for col, cinfo in tinfo.get("columns", {}).items():
            if cinfo.get("detected_type") in types:
                out.append((table, col))
    return out


def _format_sql_value(val: Any) -> str:
    """Экранировать значение для SQL-литерала."""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    # Строка — экранируем одинарные кавычки
    escaped = str(val).replace("'", "''")
    return f"'{escaped}'"
