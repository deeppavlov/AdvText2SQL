"""Mandatory PostgreSQL execution boundary, independent of feature flags.

AST checks reject statements outside the answer-query contract. PostgreSQL
read-only transactions enforce database write restrictions as a second layer.
Use a least-privileged role as well: read-only mode is not a sandbox for arbitrary
user-defined functions with external side effects.
"""

from contextlib import contextmanager

from sqlglot import exp, parse
from sqlglot.errors import SqlglotError


class UnsafeSQL(ValueError):
    """The submitted SQL does not satisfy the read-only answer contract."""


class ReadOnlySetupError(RuntimeError):
    """Protection could not be established; stop rather than label a model answer."""


def validate_read_query(sql: str, *, dialect: str = "postgres") -> None:
    """Validate in the input dialect; execution always uses PostgreSQL below."""
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQL("Expected a non-empty SQL query")
    try:
        statements = [statement for statement in parse(sql, read=dialect)
                      if statement is not None and not isinstance(statement, exp.Semicolon)]
    except SqlglotError:
        raise UnsafeSQL(f"Cannot parse SQL as a {dialect} query") from None
    if len(statements) != 1 or not isinstance(statements[0], (exp.Query, exp.Values)):
        raise UnsafeSQL("Exactly one SELECT/VALUES query is allowed")
    for node in statements[0].walk():
        if isinstance(node, (exp.DML, exp.DDL, exp.Into, exp.Lock, exp.Command)):
            raise UnsafeSQL("Writes, commands and locking queries are not allowed")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if (
                name in {"set_config", "setseed", "nextval", "setval"}
                or name.startswith(("dblink", "lo_"))
                or (name.startswith("pg_") and name not in {"pg_typeof", "pg_column_size"})
            ):
                raise UnsafeSQL("Session-changing or privileged functions are not allowed")


@contextmanager
def read_only_connection(engine, *statements: str):
    """Validate before connecting; establish READ ONLY before executing any query.

SQL is returned to the driver unchanged. Always roll back, including on errors;
never fall back to an unprotected connection if setup fails.
"""
    for statement in statements:
        validate_read_query(statement)
    if engine.dialect.name != "postgresql":
        raise UnsafeSQL("Read-only execution requires PostgreSQL")
    with engine.connect() as connection:
        # SQLAlchemy begin() alone does not start a DB transaction in AUTOCOMMIT.
        connection.execution_options(isolation_level="READ COMMITTED")
        transaction = connection.begin()
        try:
            try:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                connection.exec_driver_sql("SET LOCAL statement_timeout = '30s'")
            except Exception as error:
                raise ReadOnlySetupError(
                    f"Read-only transaction setup failed ({type(error).__name__})"
                ) from None
            yield connection
        finally:
            transaction.rollback()
