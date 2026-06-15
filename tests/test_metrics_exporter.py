"""
Тесты для serve.metrics_exporter — парсинг и инкрементальный tailing queries.jsonl.

Без prometheus_client: он импортируется только внутри main(), поэтому чистая
логика (parse_event, JsonlTailer) тестируется напрямую.
"""
from __future__ import annotations

import json

from adv_text2sql.serve.metrics_exporter import JsonlTailer, parse_event


def _line(status: str, latency: float | None = 12.0) -> str:
    return json.dumps({"status": status, "latency_ms": latency}, ensure_ascii=False) + "\n"


def test_parse_event_valid_and_invalid() -> None:
    m = parse_event(_line("success", 42.0))
    assert m is not None and m.status == "success" and m.latency_ms == 42.0
    assert parse_event("") is None
    assert parse_event("not-json") is None


def test_tailer_reads_only_complete_lines(tmp_path) -> None:
    p = tmp_path / "queries.jsonl"
    p.write_text(_line("success") + _line("error_execute"), encoding="utf-8")
    tailer = JsonlTailer(p)

    events = tailer.read_new()
    assert [e.status for e in events] == ["success", "error_execute"]
    # повторный вызов без новых данных → пусто (offset сохранён)
    assert tailer.read_new() == []


def test_tailer_ignores_partial_tail_until_complete(tmp_path) -> None:
    p = tmp_path / "queries.jsonl"
    p.write_text(_line("success"), encoding="utf-8")
    tailer = JsonlTailer(p)
    assert len(tailer.read_new()) == 1

    # writer дописал неполную строку (без '\n') — не должна читаться
    with p.open("a", encoding="utf-8") as f:
        f.write('{"status": "success", "latency')
    assert tailer.read_new() == []

    # дописал остаток строки + перевод строки — теперь читается
    with p.open("a", encoding="utf-8") as f:
        f.write('_ms": 5.0}\n')
    events = tailer.read_new()
    assert len(events) == 1 and events[0].status == "success"


def test_tailer_handles_rotation(tmp_path) -> None:
    p = tmp_path / "queries.jsonl"
    p.write_text(_line("success") + _line("success"), encoding="utf-8")
    tailer = JsonlTailer(p)
    assert len(tailer.read_new()) == 2

    # файл усечён/пересоздан (стал короче offset) → читаем с начала
    p.write_text(_line("error_parse"), encoding="utf-8")
    events = tailer.read_new()
    assert len(events) == 1 and events[0].status == "error_parse"
