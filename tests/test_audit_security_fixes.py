"""AUD-001/003/006: exercise real entry points with fake SQL connections only."""

import asyncio
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import pytest
from sqlalchemy import text
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from adv_text2sql.sql_safety import UnsafeSQL, read_only_connection, validate_read_query  # noqa: E402
from adv_text2sql.safe_logging import database_label, register_database_url  # noqa: E402
from adv_text2sql.mcp_servers.text2sql_tool.src import text2sql_implementation as impl  # noqa: E402
from benchmarks import evaluate_bird  # noqa: E402
import prompt_pilot_runtime as pilot  # noqa: E402


def fake_engine():
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = [(1,)]
    conn.execute.return_value.keys.return_value = ["value"]
    return engine, conn


def load_permission_inventory():
    spec = importlib.util.spec_from_file_location("permission_inventory", ROOT / "data/test_permissions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_permission_module_import_needs_no_env_and_performs_no_sql():
    with patch.dict("os.environ", {}, clear=True), patch("sqlalchemy.create_engine") as create:
        module = load_permission_inventory()
    create.assert_not_called()
    assert callable(module.main)


def test_manual_permission_inventory_uses_catalog_reads_and_rolls_back():
    module = load_permission_inventory()
    engine, conn = fake_engine()
    conn.execute.side_effect = [
        SimpleNamespace(one=lambda: (False, False, False)),
        SimpleNamespace(fetchall=lambda: []),
        SimpleNamespace(scalar_one=lambda: False),
    ]
    with patch.dict("os.environ", {"BENCHMARK_DB_URL": "localhost:5432",
                                  "DB_USER": "reader", "DB_PASS": "dummy"}), \
            patch.object(module, "create_engine", return_value=engine):
        assert module.main() == 0
    for call in conn.execute.call_args_list:
        validate_read_query(str(call.args[0]))
    conn.begin.return_value.rollback.assert_called_once()
    engine.dispose.assert_called_once()


@pytest.mark.parametrize("sql", [
    "SELECT 1", "VALUES (1)", "(SELECT 1)",
    "SELECT 1; -- trailing comment",
    "SELECT '-- ; DROP TABLE t' AS x, '/* intact */' AS y;",
    'SELECT "DESC" FROM "CamelCase"',
    "WITH x AS (SELECT 1 AS id) SELECT id FROM x",
    "WITH RECURSIVE x(n) AS (VALUES (1) UNION ALL SELECT n+1 FROM x WHERE n<3) SELECT * FROM x",
    "SELECT 1 UNION ALL SELECT 2", "SELECT COUNT(*), ROUND(AVG(x), 2) FROM t",
])
def test_queries_are_accepted_without_rewriting(sql):
    engine, conn = fake_engine()
    with read_only_connection(engine, sql) as active:
        active.execute(text(sql))
    assert str(conn.execute.call_args.args[0]) == sql
    assert [c.args[0] for c in conn.exec_driver_sql.call_args_list] == [
        "SET TRANSACTION READ ONLY", "SET LOCAL statement_timeout = '30s'",
    ]
    assert conn.mock_calls[0][0] == "execution_options"
    assert conn.mock_calls[0].kwargs == {"isolation_level": "READ COMMITTED"}
    assert conn.mock_calls[1][0] == "begin"
    assert conn.mock_calls[2][0] == "exec_driver_sql"
    conn.begin.return_value.rollback.assert_called_once()
    conn.begin.return_value.commit.assert_not_called()


UNSAFE = [
    "", "-- comment", "SELECT FROM", "DELETE FROM t", "UPDATE t SET a=1",
    "INSERT INTO t SELECT 1", "CREATE TEMP TABLE x AS SELECT 1", "DROP TABLE t",
    "SELECT 1; SELECT 2", "SELECT 1; DELETE FROM t", "SELECT 1; SET TRANSACTION READ WRITE",
    "SELECT 1; /* separator */ DELETE FROM t",
    "WITH x AS (DELETE FROM t RETURNING id) SELECT id FROM x",
    "WITH x AS (INSERT INTO t VALUES (1) RETURNING id) SELECT id FROM x",
    "SELECT * INTO TEMP x FROM t", "SELECT * FROM t FOR UPDATE",
    "EXPLAIN ANALYZE DELETE FROM t", "COPY t TO '/tmp/result'", "COMMIT",
    "SELECT pg_catalog.set_config('transaction_read_only', 'off', false)",
    "SELECT nextval('seq')", "SELECT setval('seq', 1)",
    "SELECT pg_catalog.pg_notify('channel', 'payload')", "SELECT lo_export(1, '/tmp/file')",
    "SELECT dblink_exec('conn', 'DELETE FROM t')",
]


@pytest.mark.parametrize("sql", UNSAFE)
def test_unsafe_sql_never_reaches_connection(sql):
    engine, _ = fake_engine()
    with pytest.raises(UnsafeSQL), read_only_connection(engine, sql):
        pytest.fail("Unsafe SQL reached the connection")
    engine.connect.assert_not_called()


def test_read_only_setup_failure_does_not_execute_query_or_fall_back():
    engine, conn = fake_engine()
    conn.exec_driver_sql.side_effect = RuntimeError("read-only setup denied")
    with pytest.raises(RuntimeError), read_only_connection(engine, "SELECT 1") as active:
        active.execute(text("SELECT 1"))
    conn.execute.assert_not_called()
    conn.begin.return_value.rollback.assert_called_once()
    engine.connect.assert_called_once()


def test_query_failure_rolls_back_and_non_postgresql_is_rejected():
    engine, conn = fake_engine()
    with pytest.raises(RuntimeError), read_only_connection(engine, "SELECT 1"):
        raise RuntimeError("query failed")
    conn.begin.return_value.rollback.assert_called_once()
    other, _ = fake_engine()
    other.dialect.name = "sqlite"
    with pytest.raises(UnsafeSQL), read_only_connection(other, "SELECT 1"):
        pytest.fail("Unsupported dialect")
    other.connect.assert_not_called()


@pytest.mark.parametrize("sql", ["DELETE FROM t", "SELECT 1; DELETE FROM t",
                                 "WITH x AS (DELETE FROM t RETURNING id) SELECT * FROM x"])
def test_execute_safe_cannot_disable_execution_guard_with_features(sql):
    tool = object.__new__(impl.Text2SQLGenerator)
    tool.engine, _ = fake_engine()
    with patch.multiple(impl, FEAT_15=False, FEAT_16=False):
        assert tool.execute_safe(sql)["status"] == "error"
    tool.engine.connect.assert_not_called()


def test_execute_safe_success_has_read_only_transaction():
    tool = object.__new__(impl.Text2SQLGenerator)
    tool.engine, conn = fake_engine()
    with patch.multiple(impl, FEAT_15=False, FEAT_16=False):
        result = tool.execute_safe("SELECT 1")
    assert result["results"] == [{"value": 1}]
    assert conn.exec_driver_sql.call_args_list[0].args == ("SET TRANSACTION READ ONLY",)
    conn.begin.return_value.rollback.assert_called_once()


@pytest.mark.parametrize("prediction", ["SELECT 1", "SELECT 1; DELETE FROM t"])
def test_base_bird_evaluator_enforces_boundary(tmp_path, monkeypatch, prediction):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DB_USER", "reader")
    monkeypatch.setenv("DB_PASS", "dummy")
    answer_file = tmp_path / "gold.json"
    answer_file.write_text(json.dumps([{"question_id": 1, "SQL": "SELECT 1",
                                       "db_id": "dummy", "difficulty": "simple"}]))
    engine, conn = fake_engine()
    with patch.object(evaluate_bird, "create_engine", return_value=engine):
        report = evaluate_bird.run_evaluation({"1": prediction}, answer_file, "localhost:5432")
    assert report["overall_accuracy"] == (100 if prediction == "SELECT 1" else 0)
    if prediction == "SELECT 1":
        assert conn.exec_driver_sql.call_args_list[0].args == ("SET TRANSACTION READ ONLY",)
        conn.begin.return_value.rollback.assert_called_once()
    else:
        engine.connect.assert_not_called()


def test_prompt_evaluator_rejects_unsafe_sql_and_protects_valid_execution():
    session = object.__new__(pilot.Session)
    engine, conn = fake_engine()
    session.engines = {"dummy": engine}
    session.golds = {"1": {(1,)}}
    item = {"question_id": 1, "db_id": "dummy"}
    assert session.score(item, "SELECT 1; DELETE FROM t", "bird") == (0.0, "unsafe_sql")
    engine.connect.assert_not_called()
    assert session.score(item, "SELECT 1", "bird") == (1.0, None)
    assert conn.exec_driver_sql.call_args_list[0].args == ("SET TRANSACTION READ ONLY",)
    conn.begin.return_value.rollback.assert_called_once()


def test_prompt_reference_preflight_rejects_multiple_statements_before_execution():
    session = object.__new__(pilot.Session)
    item = {"question_id": 1, "db_id": "dummy", "SQL": "SELECT 1; DELETE FROM t"}
    session.bird_items, session.items, session.phase = [item], [item], "search"
    session.manifest = {"schema_audit": {"schemas": {"bird": {}, "ambrosia": {}}}}
    session.engines = {"dummy": fake_engine()[0]}
    session.golds = {}
    session.progress = lambda update: None
    with patch("resilient_hybrid_runtime.validate_live_schemas"):
        with pytest.raises(RuntimeError, match="Reference preflight failed"):
            session.preflight()
    session.engines["dummy"].connect.assert_not_called()


def test_credentials_redacted_in_console_file_parent_and_exception(tmp_path):
    secret = "fictional-p@ss/word"
    uri = URL.create("postgresql+psycopg", username="reader", password=secret,
                     host="localhost", database="dummy")
    register_database_url(uri)
    assert secret not in database_label(uri)
    assert "reader" not in database_label(uri)
    assert "localhost" in database_label(uri)
    console, inherited = io.StringIO(), io.StringIO()
    log_path = tmp_path / "debug.log"
    logger = impl.logger
    original = (logger.handlers[:], logger.level, logger.propagate)
    parent = logging.getLogger("audit_test_parent")
    parent_handler = logging.StreamHandler(inherited)
    parent.addHandler(parent_handler)
    prior_parent = logger.parent
    try:
        logger.handlers = []
        logger.parent = parent
        logger.propagate = True
        with patch.object(impl, "FEAT_11", True):
            impl.setup_logging(str(log_path))
        logger.handlers[0].setStream(console)
        try:
            raise RuntimeError(f"Connection rejected {uri.render_as_string(hide_password=False)} / {secret}")
        except RuntimeError:
            logger.exception("Credentials %s and URI %s", secret, uri.render_as_string(hide_password=False))
        for handler in logger.handlers:
            handler.flush()
        outputs = [console.getvalue(), inherited.getvalue(), log_path.read_text(encoding="utf-8")]
        for output in outputs:
            assert secret not in output
            assert quote(secret, safe="") not in output
            assert "RuntimeError" in output
            assert "***" in output
        assert "exception" in json.loads(outputs[2])
        logging.getLogger("text2sql_tool.child").warning("Child secret: %s", secret)
        for handler in logger.handlers:
            handler.flush()
        for output in (console.getvalue(), inherited.getvalue(), log_path.read_text(encoding="utf-8")):
            assert secret not in output
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers, logger.level, logger.propagate = original
        logger.parent = prior_parent
        parent.removeHandler(parent_handler)


def test_empty_schema_errors_never_expose_credentials():
    tool = object.__new__(impl.Text2SQLGenerator)
    secret = "fictional-empty-schema-password"
    tool.db_uri = f"postgresql+psycopg://reader:{secret}@localhost/dummy"
    inspector = SimpleNamespace(get_table_names=lambda **kwargs: [])
    with patch.object(impl, "inspect", return_value=inspector):
        tool.engine = MagicMock()
        with pytest.raises(RuntimeError) as caught:
            tool._get_db_schema_tsv()
    assert secret not in str(caught.value)


def test_build_logging_with_failing_introspection_hides_password():
    tool = object.__new__(impl.Text2SQLGenerator)
    secret = "fictional-build-password"
    tool.db_uri = f"postgresql+psycopg://reader:{secret}@localhost/dummy"
    tool._get_db_schema_tsv = MagicMock(side_effect=RuntimeError("catalog unavailable"))
    with patch.object(impl.logger, "info") as info, patch.object(impl, "FEAT_35", True):
        with pytest.raises(RuntimeError):
            asyncio.run(tool.build())
    assert secret not in str(info.call_args_list)
