#!/usr/bin/env python3
"""
preflight — проверка окружения перед запуском autonomous Text2SQL pipeline.

Что проверяется:
  - env vars: DB_USER, DB_PASS (для подключения к БД)
  - env vars: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME (для LLM-генератора/инференса)
  - SSH-туннель на localhost:5444 (для PG-БД через bastion)
  - PostgreSQL connectivity (опц., если задан --db-url)
  - Python deps: typer, pydantic, sqlalchemy, sqlglot, openai

Запуск:
    uv run python scripts/preflight.py
    uv run python scripts/preflight.py --db-url postgresql+psycopg://.../card_games
"""
from __future__ import annotations

import argparse
import importlib
import os
import socket
import sys
from typing import Callable


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[33m⚠\033[0m {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────


def check_env_vars() -> bool:
    print("\n[1] Environment variables:")
    required_db = ["DB_USER", "DB_PASS"]
    required_llm = ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL_NAME"]
    all_ok = True

    for name in required_db:
        if os.environ.get(name):
            _ok(f"{name} = {'***' if 'PASS' in name else os.environ[name][:10]}...")
        else:
            _fail(f"{name} not set — нужен для подключения к PostgreSQL")
            all_ok = False

    for name in required_llm:
        if os.environ.get(name):
            _ok(f"{name} = {os.environ[name][:30]}...")
        else:
            _warn(f"{name} not set — нужен для LLM-генератора и инференса")

    return all_ok


def check_ssh_tunnel(port: int = 5444) -> bool:
    print(f"\n[2] SSH tunnel (localhost:{port}):")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect(("localhost", port))
        _ok(f"port {port} accepts connections")
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError):
        _fail(f"port {port} closed — поднимите SSH-туннель:")
        print(f"      ssh -N -L {port}:10.11.1.6:{port} user@lnsigo.mipt.ru -p2278")
        return False


def check_db_connection(db_url: str | None) -> bool:
    if not db_url:
        return True
    print(f"\n[3] PostgreSQL connection ({db_url[:50]}...):")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT current_database(), version()"))
            db_name, ver = res.fetchone()
        _ok(f"connected to {db_name}, server: {ver[:60]}")
        engine.dispose()
        return True
    except Exception as e:
        _fail(f"connection failed: {str(e)[:200]}")
        return False


def check_python_deps() -> bool:
    print("\n[4] Python dependencies:")
    deps: dict[str, Callable[[], bool]] = {
        "typer": lambda: importlib.import_module("typer") is not None,
        "pydantic": lambda: importlib.import_module("pydantic") is not None,
        "sqlalchemy": lambda: importlib.import_module("sqlalchemy") is not None,
        "sqlglot": lambda: importlib.import_module("sqlglot") is not None,
        "openai": lambda: importlib.import_module("openai") is not None,
        "psycopg": lambda: importlib.import_module("psycopg") is not None,
        "fastmcp": lambda: importlib.import_module("fastmcp") is not None,
    }
    all_ok = True
    for name, check in deps.items():
        try:
            check()
            _ok(name)
        except Exception as e:
            _fail(f"{name}: {e}")
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url", help="PG URL для тестового connect. Например postgresql+psycopg://user:pass@localhost:5444/card_games"
    )
    parser.add_argument(
        "--port", type=int, default=5444, help="Port для проверки SSH-туннеля"
    )
    args = parser.parse_args()

    print("\033[1mPreflight check — autonomous Text2SQL pipeline\033[0m")

    results = [
        check_python_deps(),
        check_env_vars(),
        check_ssh_tunnel(args.port),
        check_db_connection(args.db_url),
    ]

    print()
    if all(results):
        print("\033[32mAll checks passed.\033[0m Можно запускать `text2sql init --db-url ...`")
        return 0
    else:
        print("\033[31mНекоторые проверки не прошли.\033[0m Исправьте выше и повторите.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
