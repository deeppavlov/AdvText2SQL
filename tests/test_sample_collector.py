"""
Тесты для profiler.sample_collector — bounded low-cardinality сбор.

Без реальной БД: подменяем Connection фейком, который записывает выполненный
SQL/параметры и отдаёт заданные строки.
"""
from __future__ import annotations

from adv_text2sql.profiler.sample_collector import collect_low_cardinality_values


class _FakeResult:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.calls: list[tuple] = []

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        return _FakeResult(self.rows)

    def rollback(self) -> None:
        pass


def test_low_cardinality_query_is_bounded() -> None:
    """Запрос содержит LIMIT = threshold+1 (early-exit на high-cardinality)."""
    conn = _FakeConn(rows=[("black",), ("white",), ("gold",)])
    out = collect_low_cardinality_values(conn, "cards", [{"name": "bordercolor"}], threshold=10)

    sql, params = conn.calls[0]
    assert "LIMIT" in sql.upper()
    assert params == {"lim": 11}
    assert out == {"bordercolor": ["black", "white", "gold"]}


def test_high_cardinality_column_excluded() -> None:
    """Если вернулось threshold+1 значений → колонка отбрасывается."""
    rows = [(f"v{i}",) for i in range(11)]  # 11 > threshold=10
    conn = _FakeConn(rows=rows)
    out = collect_low_cardinality_values(conn, "cards", [{"name": "name"}], threshold=10)
    assert out == {}
