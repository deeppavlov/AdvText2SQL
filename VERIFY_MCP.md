# Проверка готовности MCP-сервера — подробный пошаговый план

## Статус (проверено вживую в этой сессии)

В ходе живого прогона нашлись и **исправлены 3 бага**, без которых MCP-сервер не
работал:
1. `main.py` импортировал отсутствующий модуль `generate_tool_description`
   (`ModuleNotFoundError`) → заменено на статическое схемо-осведомлённое описание.
2. Tool вызывал `query(..., check_ambiguity=, check_sql_query=)` — таких
   аргументов нет → убрано.
3. Tool разбирал ответ `query()` по несуществующей вложенной структуре и не
   исполнял SQL → переписан под реальный контракт `{"status","query"}` +
   добавлено исполнение через `execute_safe()`.

После фиксов MCP-сервер **доказан как «условно готов» end-to-end**: поднимается,
tool виден, по вопросу генерируется корректный SQL, всё логируется. Полное
исполнение упирается только в SSH-туннель к БД (конфиг, не код).

## Две конфиг-ловушки (НЕ баги кода — но без них «успеха» не будет)

1. **Модель должна работать через `/v1/chat/completions`.** В `.env` сейчас
   `LLM_MODEL_NAME=gpt-5.1-codex-mini` — он на OpenAI доступен только через
   `/v1/responses`, и chat-completions отдаёт `404`. Для проверки задай
   chat-совместимую модель (`gpt-4.1`, `gpt-4o-mini` и т.п.).
2. **Исполнение SQL требует SSH-туннеля** на `localhost:5444`. Без него SQL
   генерируется, но не исполняется (это и есть граница «условно» vs «полностью»).

---

## Что понадобится
- **2 терминала** (Терминал-1 — сервер, Терминал-2 — клиент). Опционально
  **Терминал-3** для SSH-туннеля, если хочешь полное исполнение.
- Всё из корня репозитория:
  `/Users/vlad/Desktop/uni/maga/2_quarter_project/AdvText2SQL`

---

## ШАГ 1 — поднять сервер (Терминал-1)

```bash
TEXT2SQL_PROFILE_PATH=data/profiles/card_games/profile.json \
LLM_MODEL_NAME=gpt-4o-mini \
  uv run --env-file .env python -m adv_text2sql.mcp_servers.text2sql_tool.main
```
- `TEXT2SQL_PROFILE_PATH` → hot-start (<50мс, без обращения к БД на старте).
- `LLM_MODEL_NAME=gpt-4o-mini` → переопределяем несовместимую модель из `.env`.

**Что увидишь (признак готовности):**
```
INFO  Starting MCP server 'text2sql_tool_server' with transport 'http' on http://0.0.0.0:8000/mcp
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
Сервер слушает `:8000`. **Оставь терминал открытым.**

> Порт занят? Добавь `MCP_PORT=8011` в начало команды и используй `:8011` ниже.

---

## ШАГ 2 — увидеть tool по MCP (Терминал-2)

```bash
uv run python -c "
import asyncio
from fastmcp import Client
async def main():
    async with Client('http://127.0.0.1:8000/mcp/') as c:
        tools = await c.list_tools()
        print('TOOLS:', [t.name for t in tools])
        print('DESC:', tools[0].description.splitlines()[1] if tools else '')
asyncio.run(main())
"
```
**Ожидаемый вывод:**
```
TOOLS: ['text2sql']
DESC: ## Описание   (или 'Средний' — первые строки описания)
```
Если видишь `['text2sql']` — сервер виден по MCP, tool зарегистрирован. ✅

---

## ШАГ 3 — вызвать tool вопросом (Терминал-2)

```bash
uv run python -c "
import asyncio
from fastmcp import Client
async def main():
    async with Client('http://127.0.0.1:8000/mcp/') as c:
        res = await c.call_tool('text2sql',
              {'user_query_text': 'Сколько всего карт в таблице cards?'}, timeout=150)
        out = res.data if getattr(res,'data',None) is not None else res.content
        print(out)
asyncio.run(main())
"
```

**Исход без туннеля (УСЛОВНО готов) — реальный вывод из сессии:**
```
SQL сгенерирован, но исполнение не удалось (нет доступа к БД?).

```sql
SELECT COUNT(*) AS total_cards
FROM "cards";
```

_Ошибка_: (psycopg.OperationalError) connection ... port 5444 ... Connection refused
```
→ Это **успех проверки «условно готов»**: связка MCP↔LLM работает, SQL корректен,
не хватает только доступа к БД.

**Исход с туннелём (ПОЛНОСТЬЮ готов) — реальный вывод из сессии:**
```
|   total_cards |
|--------------:|
|         56822 |
```
А на «Покажи топ-5 художников по числу карт»:
```
| artist       |   card_count |
|:-------------|-------------:|
| John Avon    |         1128 |
| Kev Walker   |          997 |
| Dan Frazier  |          682 |
| Mark Tedin   |          674 |
| Greg Staples |          597 |
```

---

## ШАГ 4 — проверить логирование (Терминал-2)

```bash
python3 -c "
import json
e = json.loads(open('data/logs/queries.jsonl').read().strip().splitlines()[-1])
for k in ('status','question','generated_sql','latency_ms'):
    print(f'{k:13}:', str(e.get(k))[:90])
"
```
**Ожидаемый вывод (реальный из сессии):**
```
status       : error_execute        # или 'success' с туннелём
question     : Сколько всего карт в таблице cards?
generated_sql: SELECT COUNT(*) AS total_cards FROM "cards";
latency_ms   : 2539.4
```
Это тот же `queries.jsonl`, который читает exporter из пункта 1.1 → запрос к MCP
сразу виден на дашборде Grafana.

---

## ШАГ 5 (опционально) — полное исполнение через туннель

Терминал-3 (держать открытым):
```bash
ssh -N -L 5444:10.11.1.6:5444 <user>@lnsigo.mipt.ru -p2278
```
Убедись, что в `.env` есть `DB_USER`/`DB_PASS` (БД `card_games` по умолчанию).
Повтори ШАГ 3 — теперь вернётся таблица, а в логе `status: success`.

---

## Остановить сервер
В Терминале-1 нажми `Ctrl-C`. Если запускал в фоне:
```bash
lsof -tiTCP:8000 -sTCP:LISTEN | xargs kill
```

---

## Итоговый чек-лист

| Критерий | Шаг | Статус (в сессии) |
|---|---|---|
| Импорт/бутстрап (boot) | — | ✅ проверено |
| Сервер поднимается по HTTP `:8000/mcp/` | 1 | ✅ проверено |
| Tool `text2sql` виден по MCP | 2 | ✅ проверено |
| NL-вопрос → корректный SQL (LLM) | 3 | ✅ проверено (gpt-4o-mini) |
| Исполнение SQL на БД | 3+5 | ✅ проверено (туннель + 56822 карт, топ-5 художников) |
| Логирование в `queries.jsonl` | 4 | ✅ проверено (`status=success`) |

**Вердикт:** MCP-сервер **ПОЛНОСТЬЮ ГОТОВ** — доказано end-to-end в сессии:
boot → discovery → NL→SQL → исполнение на реальной БД (вернул таблицы) →
логирование `status=success`. Условия эксплуатации: chat-совместимая модель в
`.env` (не `gpt-5.1-codex-mini`) и поднятый SSH-туннель.

---

## Troubleshooting (реальные ошибки)
- `404 ... only supported in v1/responses` → модель в `.env` несовместима с
  chat/completions. Задай `LLM_MODEL_NAME=gpt-4o-mini` (или `gpt-4.1`).
- `Connection refused ... port 5444` → нет SSH-туннеля (ШАГ 5). Для «условно
  готов» это ожидаемо.
- `KeyError: 'LLM_MODEL_NAME'` → запускай с `--env-file .env`.
- Старт висит ~30с и лезет в БД → не задан `TEXT2SQL_PROFILE_PATH` (cold-start).
- Порт занят → `MCP_PORT=8011` при старте + `:8011` в URL клиента.
