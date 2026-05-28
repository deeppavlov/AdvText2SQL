"""
schema_extractor — pure-functions версия логики извлечения схемы из БД.

Источник: `Text2SQLGenerator._get_db_schema_light`, `._explore_db_relationships`,
`._format_relationships`. Здесь они переписаны как функции без `self`, чтобы
быть переиспользуемыми из:

    1. `Text2SQLGenerator.build()` (минимально модифицированного — может теперь
       вызывать эти функции вместо своих _методов, либо оставаться как есть).
    2. `text2sql profile` CLI — для создания `profile.json`.
    3. Unit-теста, который проверяет byte-identical инвариант:
       schema_str(profile) == schema_str(Text2SQLGenerator.build())

КРИТИЧЕСКИЙ ИНВАРИАНТ: вывод `extract_schema_light()` должен byte-to-byte
совпадать с `Text2SQLGenerator._get_db_schema_light()`. Если изменяешь формат
здесь — обязательно обнови и оригинал, иначе train==inference сломается.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

from sqlalchemy.engine.reflection import Inspector

from .connector import DBConnector

logger = logging.getLogger("text2sql_tool.profiler.schema")


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────


class ForeignKey(TypedDict):
    from_table: str
    from_cols: list[str]
    to_table: str
    to_cols: list[str]


class Relationships(TypedDict):
    foreign_keys: list[ForeignKey]
    primary_keys: dict[str, list[str]]


# ─────────────────────────────────────────────────────────────────────────────
# Pure functions (без self) — берутся 1-в-1 из Text2SQLGenerator
# ─────────────────────────────────────────────────────────────────────────────


def extract_schema_light(inspector: Inspector, schema: str = "public") -> str:
    """Lightweight schema → строка для system prompt.

    Формат byte-identical с `Text2SQLGenerator._get_db_schema_light()`:

        TABLE table_name
          - col1 (TYPE1)
          - col2 (TYPE2)
        <blank line>
        TABLE next_table
          ...
    """
    tables = inspector.get_table_names(schema=schema)
    if not tables:
        raise RuntimeError(
            f"No tables found in schema '{schema}' — пустая БД или нет доступа"
        )

    schema_parts: list[str] = []
    for table in tables:
        schema_parts.append(f"TABLE {table}")
        columns = inspector.get_columns(table, schema=schema)
        for col in columns:
            col_type = str(col["type"])
            schema_parts.append(f"  - {col['name']} ({col_type})")
        schema_parts.append("")

    return "\n".join(schema_parts).strip()


def extract_relationships(
    inspector: Inspector, schema: str = "public"
) -> Relationships:
    """FK + PK через SQLAlchemy inspector. Безопасно игнорирует таблицы без PK/FK."""
    tables = inspector.get_table_names(schema=schema)

    foreign_keys: list[ForeignKey] = []
    primary_keys: dict[str, list[str]] = {}

    for table in tables:
        try:
            pk = inspector.get_pk_constraint(table, schema=schema)
            if pk and pk.get("constrained_columns"):
                primary_keys[table] = list(pk["constrained_columns"])
        except Exception:
            logger.exception(f"Could not get PK for table {table}")

        try:
            for fk in inspector.get_foreign_keys(table, schema=schema):
                foreign_keys.append(
                    ForeignKey(
                        from_table=table,
                        from_cols=list(fk["constrained_columns"]),
                        to_table=fk["referred_table"],
                        to_cols=list(fk["referred_columns"]),
                    )
                )
        except Exception:
            logger.exception(f"Could not get FK for table {table}")

    logger.info(f"Relationships: {len(primary_keys)} PKs, {len(foreign_keys)} FKs")
    return Relationships(foreign_keys=foreign_keys, primary_keys=primary_keys)


def format_relationships(relationships: Relationships) -> str:
    """Форматирует FK/PK в DDL-стиль для system prompt.

    Формат byte-identical с `Text2SQLGenerator._format_relationships()`.
    """
    parts: list[str] = []

    pks = relationships.get("primary_keys", {})
    if pks:
        parts.append("-- Primary Keys:")
        for table, cols in pks.items():
            parts.append(f"-- {table}: {', '.join(cols)}")
        parts.append("")

    fks = relationships.get("foreign_keys", [])
    if fks:
        parts.append("-- Foreign Keys:")
        for fk in fks:
            from_str = ", ".join(f'{fk["from_table"]}.{c}' for c in fk["from_cols"])
            to_str = ", ".join(f'{fk["to_table"]}.{c}' for c in fk["to_cols"])
            parts.append(f"-- {from_str} -> {to_str}")

    return "\n".join(parts) if parts else "-- No relationships found"


# ─────────────────────────────────────────────────────────────────────────────
# Class wrapper — для удобного использования из CLI
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SchemaExtractionResult:
    schema_str: str               # для system_prompt; byte-identical с генератором
    relationships: Relationships  # структурированно
    relationships_str: str        # для system_prompt
    tables: list[str]             # для удобства downstream
    column_info: dict[str, list[dict]]  # {table: [{name, type, nullable, ...}]}


class SchemaExtractor:
    """High-level API над pure functions выше.

    Используется в CLI `text2sql profile` как первая стадия:

        with DBConnector(uri) as conn:
            schema = SchemaExtractor(conn).extract()
            print(schema.schema_str)        # для system prompt
    """

    def __init__(self, connector: DBConnector) -> None:
        self.connector = connector

    def extract(self) -> SchemaExtractionResult:
        inspector = self.connector.make_inspector()
        schema = self.connector.default_schema

        schema_str = extract_schema_light(inspector, schema=schema)
        relationships = extract_relationships(inspector, schema=schema)
        relationships_str = format_relationships(relationships)

        tables = inspector.get_table_names(schema=schema)
        column_info = {
            table: [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                }
                for col in inspector.get_columns(table, schema=schema)
            ]
            for table in tables
        }

        return SchemaExtractionResult(
            schema_str=schema_str,
            relationships=relationships,
            relationships_str=relationships_str,
            tables=tables,
            column_info=column_info,
        )
