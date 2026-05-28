"""
DBConnector — тонкая обёртка над SQLAlchemy engine для PostgreSQL.

Хотя мы целимся только на PG (по решению дня 0), оборачиваем create_engine в класс
чтобы централизовать настройки connection pool и не разбрасывать `create_engine`
вызовы по pipeline. Также удобно держать здесь же `db_id`-извлечение из URI —
оно нужно всем стадиям (для пути артефактов).
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector


SUPPORTED_DIALECTS = ("postgresql",)


@dataclass
class DBConnector:
    """SQLAlchemy-обёртка для PostgreSQL.

    Пример:
        conn = DBConnector("postgresql+psycopg://user:pass@host:5444/card_games")
        inspector = conn.make_inspector()
        tables = inspector.get_table_names(schema=conn.default_schema)
        conn.close()
    """

    db_uri: str
    pool_pre_ping: bool = True
    default_schema: str = "public"
    _engine: Engine | None = None

    def __post_init__(self) -> None:
        # ленивая инициализация — engine создаётся при первом обращении
        # это позволяет конструировать Connector без побочных эффектов
        pass

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.db_uri, pool_pre_ping=self.pool_pre_ping)
            self._validate_dialect()
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> "DBConnector":
        # триггерит engine init для context-manager использования
        _ = self.engine
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Inspectors ───────────────────────────────────────────────────────────

    def make_inspector(self) -> Inspector:
        return inspect(self.engine)

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    @property
    def db_id(self) -> str:
        """Извлечь имя БД из URI — оно же `db_id` для путей артефактов.

        Пример: 'postgresql+psycopg://user:pw@h:5444/card_games' → 'card_games'.
        """
        path = urlparse(self.db_uri).path  # '/card_games'
        return path.lstrip("/") or "unknown_db"

    def redacted_uri(self) -> str:
        """URI без пароля — для логов и profile.json."""
        parsed = urlparse(self.db_uri)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}", ":***")
            return parsed._replace(netloc=netloc).geturl()
        return self.db_uri

    # ── Internal ─────────────────────────────────────────────────────────────

    def _validate_dialect(self) -> None:
        dialect = self.engine.dialect.name
        if dialect not in SUPPORTED_DIALECTS:
            raise NotImplementedError(
                f"Dialect '{dialect}' not supported. "
                f"Supported: {SUPPORTED_DIALECTS}. "
                f"Adding new dialects requires reviewing stats_collector regex queries."
            )
