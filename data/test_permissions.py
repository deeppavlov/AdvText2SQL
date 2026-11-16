"""Manual, non-mutating privilege inventory; importing this module performs no IO.

Run explicitly with BENCHMARK_DB_URL, DB_USER, DB_PASS and optionally DB_NAME.
No CREATE/INSERT/UPDATE/DELETE/ALTER probes are executed, even on manual runs.
"""

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

from adv_text2sql.sql_safety import read_only_connection


def main() -> int:
    # Environment access and engine construction are deliberately inside main.
    server = make_url("postgresql+psycopg://" + os.environ["BENCHMARK_DB_URL"])
    uri = URL.create(
        "postgresql+psycopg", username=os.environ["DB_USER"],
        password=os.environ["DB_PASS"], host=server.host, port=server.port,
        database=os.getenv("DB_NAME", "california_schools"),
    )
    engine = create_engine(uri, pool_pre_ping=True)
    try:
        with read_only_connection(engine) as conn:
            role = conn.execute(text("""
                SELECT rolsuper, rolcreatedb, rolcreaterole
                FROM pg_roles WHERE rolname = current_user
            """)).one()
            privileges = conn.execute(text("""
                SELECT c.relname, p.privilege
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                CROSS JOIN (VALUES ('INSERT'), ('UPDATE'), ('DELETE'),
                                   ('TRUNCATE'), ('TRIGGER')) AS p(privilege)
                WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'f')
                  AND has_table_privilege(current_user, c.oid, p.privilege)
            """)).fetchall()
            can_create = conn.execute(text("""
                SELECT has_schema_privilege(current_user, 'public', 'CREATE')
                    OR has_database_privilege(current_user, current_database(), 'CREATE')
            """)).scalar_one()
        restricted = not (any(role) or privileges or can_create)
        print("Privilege inventory:", "no checked write privileges" if restricted else "write privileges detected")
        print("Read-only transaction used; no mutation probes executed.")
        return 0 if restricted else 1
    except Exception as error:
        print(f"Privilege inventory failed ({type(error).__name__})")
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())

