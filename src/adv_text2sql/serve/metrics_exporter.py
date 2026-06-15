"""
serve/metrics_exporter.py — Prometheus exporter продуктовых метрик Text2SQL.

vLLM уже отдаёт инфра-метрики (throughput, токены, очередь) на /metrics.
Здесь — *продуктовые* метрики качества, которых vLLM не знает: доля успешных
запросов, число failed, распределение latency. Источник — `queries.jsonl`,
который пишет `query_logger.QueryLogger` из MCP-сервера.

Дизайн: standalone-скрипт без импортов из пакета (чтобы запускаться в тонком
python-контейнере, примонтировав только этот файл). Инкрементально дочитывает
append-only jsonl от сохранённого offset, на каждый новый event обновляет
prometheus-метрики (Counter'ы кумулятивны — идеально для tailing).

Запуск:
    python metrics_exporter.py --log-file data/logs/queries.jsonl --port 9105
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

# Гистограммные корзины latency (мс): покрывают быстрый кэш-хит … медленный
# multi-join + execute. inf-корзина обязательна для prometheus histogram.
LATENCY_BUCKETS_MS = (50, 100, 250, 500, 1000, 2000, 5000, 10000, float("inf"))


@dataclass
class QueryMetric:
    status: str
    latency_ms: float | None


def parse_event(raw: str) -> QueryMetric | None:
    """Распарсить одну строку queries.jsonl. None — если строка не event."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return QueryMetric(
        status=str(obj.get("status", "unknown")),
        latency_ms=obj.get("latency_ms"),
    )


class JsonlTailer:
    """Инкрементальное чтение append-only jsonl от сохранённого byte-offset.

    Читает только *полные* строки (до последнего '\\n'); неполный хвост (writer
    в процессе append) не трогает до следующего опроса. Корректно переживает
    усечение/ротацию файла (offset сбрасывается, если файл стал короче).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0

    def read_new(self) -> list[QueryMetric]:
        if not self.path.exists():
            return []
        if self.path.stat().st_size < self._offset:
            self._offset = 0  # файл усечён/пересоздан
        with self.path.open("rb") as f:
            f.seek(self._offset)
            data = f.read()
        last_nl = data.rfind(b"\n")
        if last_nl == -1:
            return []  # ни одной полной строки
        complete = data[: last_nl + 1]
        self._offset += len(complete)  # '\n' = 1 байт → безопасная граница в utf-8
        events: list[QueryMetric] = []
        for raw in complete.splitlines():
            m = parse_event(raw.decode("utf-8", "replace"))
            if m is not None:
                events.append(m)
        return events


def main() -> None:
    ap = argparse.ArgumentParser(description="Text2SQL product-metrics exporter")
    ap.add_argument("--log-file", default="data/logs/queries.jsonl")
    ap.add_argument("--port", type=int, default=9105)
    ap.add_argument("--poll-interval", type=float, default=5.0)
    args = ap.parse_args()

    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    queries = Counter("text2sql_queries_total", "Запросы по статусу", ["status"])
    failed = Counter("text2sql_failed_total", "Запросы со статусом != success")
    latency = Histogram(
        "text2sql_query_latency_ms", "Latency запроса (мс)", buckets=LATENCY_BUCKETS_MS
    )
    success_rate = Gauge("text2sql_success_rate", "Доля успешных запросов [0..1]")

    tailer = JsonlTailer(args.log_file)
    n_success = 0
    n_total = 0

    start_http_server(args.port)
    print(f"[metrics_exporter] :{args.port}/metrics ← {args.log_file}", flush=True)

    while True:
        for m in tailer.read_new():
            queries.labels(status=m.status).inc()
            n_total += 1
            if m.status == "success":
                n_success += 1
            else:
                failed.inc()
            if m.latency_ms is not None:
                latency.observe(float(m.latency_ms))
        if n_total:
            success_rate.set(n_success / n_total)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
